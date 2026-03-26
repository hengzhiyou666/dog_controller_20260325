#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Vbot机器狗路径跟踪节点（增强版 v2）
ROS2 Humble | Ubuntu22.04

订阅：
  /odometry               (nav_msgs/msg/Odometry)    — 当前位姿/速度
  /dog_output_local_path  (nav_msgs/msg/Path)        — 局部规划路径
  /lidar_points_filtered  (sensor_msgs/msg/PointCloud2) — 过滤后2D障碍点云(head_init系)

发布：
  /vel_cmd                (geometry_msgs/msg/Twist)   — 适配Vbot速度控制

功能：
  1. 纯追踪路径跟踪（自适应速度+预瞄距离）
  2. Vbot速度死区处理（线速度/角速度最小值）
  3. 前方障碍物紧急检测（基于过滤后点云）
  4. 小障碍物：原地转向或稍微后退再绕行
  5. 大障碍物/快速移动物体(疑似车辆)：原地暂停等待
  6. 路径超时检测：长时间无新路径则停车
"""
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import Twist
from sensor_msgs.msg import PointCloud2
import struct
import time
import math

# ============================================================================
# Vbot机器狗速度限制（严格遵循官方文档，RL Locomotion模式）
# ============================================================================
VBOT_LINEAR_X_FWD_MIN = 0.3    # 最小前进线速度 m/s
VBOT_LINEAR_X_FWD_MAX = 1.5    # 最大前进线速度 m/s
VBOT_LINEAR_X_BWD_MIN = -0.3   # 最小后退线速度（绝对值最小）m/s
VBOT_LINEAR_X_BWD_MAX = -1.5   # 最大后退线速度（绝对值最大）m/s
VBOT_ANGULAR_Z_MIN_ABS = 0.5   # 角速度死区绝对值 rad/s
VBOT_ANGULAR_Z_MAX_ABS = 3.0   # 最大角速度绝对值 rad/s

# ============================================================================
# 机器狗物理尺寸
# ============================================================================
DOG_LENGTH = 0.70   # 机器狗长度 m
DOG_WIDTH = 0.40    # 机器狗宽度 m
DOG_DIAG_RADIUS = math.sqrt((DOG_LENGTH / 2) ** 2 + (DOG_WIDTH / 2) ** 2)  # ≈0.403m

# ============================================================================
# 控制参数
# ============================================================================
CONTROL_FREQ = 10.0             # 控制发布频率 Hz
LOOKAHEAD_BASE = 0.6            # 基础预瞄距离 m
LOOKAHEAD_SPEED_GAIN = 0.5      # 速度-预瞄距离增益：L = base + gain*v
LOOKAHEAD_MAX = 2.0             # 最大预瞄距离 m
PATH_TIMEOUT_SEC = 2.0          # 路径超时（秒），超过此时间未收到新路径则停车
GOAL_ARRIVE_DIST = 0.3          # 认为到达路径终点的距离 m

# ============================================================================
# 紧急避障参数
# ============================================================================
EMERGENCY_STOP_DIST = 0.5       # 紧急停车距离 m（前方此距离内有障碍则紧急处理）
SLOW_DOWN_DIST = 1.5            # 减速距离 m（前方此距离内有障碍则减速）
FRONT_SECTOR_HALF_ANGLE = 45.0  # 前方检测扇形半角（度）
SIDE_SECTOR_HALF_ANGLE = 90.0   # 侧方检测角度（用于判断绕行方向）

# 障碍物分类阈值
OBSTACLE_CLUSTER_DIST = 0.3     # 障碍点聚类距离 m
LARGE_OBSTACLE_WIDTH = 1.5      # 大障碍物宽度阈值 m（超过视为车辆等大型障碍）
OBSTACLE_SPEED_THRESHOLD = 0.5  # 障碍物运动速度阈值 m/s（超过视为移动障碍）

# 行为状态
STATE_NORMAL = 0                # 正常路径跟踪
STATE_EMERGENCY_STOP = 1       # 紧急停车
STATE_TURN_IN_PLACE = 2        # 原地转向
STATE_BACK_UP = 3              # 后退
STATE_WAIT_FOR_PASS = 4        # 等待大障碍物通过


def quaternion_to_yaw(x, y, z, w):
    """四元数转偏航角yaw（避免依赖tf_transformations）"""
    norm = math.sqrt(x ** 2 + y ** 2 + z ** 2 + w ** 2)
    if norm < 1e-12:
        return 0.0
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def normalize_angle(angle):
    """将角度归一化到 [-pi, pi]"""
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def apply_vbot_deadzone(linear_x, angular_z):
    """
    适配Vbot速度死区和限幅。
    Vbot有速度死区：线速度[0, 0.3)无效，角速度[0, 0.5)无效。
    对于需要停车的情况(linear_x=0且angular_z=0)，直接发0。
    """
    # 停车命令：直接发零
    if abs(linear_x) < 1e-6 and abs(angular_z) < 1e-6:
        return 0.0, 0.0

    # 线速度处理
    if linear_x > 0:
        # 前进：限制在 [VBOT_LINEAR_X_FWD_MIN, VBOT_LINEAR_X_FWD_MAX]
        if linear_x < VBOT_LINEAR_X_FWD_MIN:
            linear_x = VBOT_LINEAR_X_FWD_MIN
        linear_x = min(linear_x, VBOT_LINEAR_X_FWD_MAX)
    elif linear_x < 0:
        # 后退：限制在 [VBOT_LINEAR_X_BWD_MAX, VBOT_LINEAR_X_BWD_MIN]
        if linear_x > VBOT_LINEAR_X_BWD_MIN:
            linear_x = VBOT_LINEAR_X_BWD_MIN
        linear_x = max(linear_x, VBOT_LINEAR_X_BWD_MAX)
    # linear_x == 0 时保持0（原地转向场景）

    # 角速度处理
    if abs(angular_z) > 1e-6:
        # 有转向需求时，确保超过死区
        if abs(angular_z) < VBOT_ANGULAR_Z_MIN_ABS:
            angular_z = math.copysign(VBOT_ANGULAR_Z_MIN_ABS, angular_z)
        # 限幅
        angular_z = max(-VBOT_ANGULAR_Z_MAX_ABS, min(VBOT_ANGULAR_Z_MAX_ABS, angular_z))
    else:
        angular_z = 0.0

    return linear_x, angular_z


def read_pointcloud2_xy(msg):
    """
    从PointCloud2消息中读取所有(x, y)坐标。
    返回列表 [(x1,y1), (x2,y2), ...]
    仅处理float32 xyz字段。
    """
    points = []
    if not msg or not msg.data:
        return points

    # 查找x和y字段的偏移
    x_offset = None
    y_offset = None
    for field in msg.fields:
        if field.name == 'x':
            x_offset = field.offset
        elif field.name == 'y':
            y_offset = field.offset
    if x_offset is None or y_offset is None:
        return points

    point_step = msg.point_step
    data = msg.data
    n_points = msg.width * msg.height
    for i in range(n_points):
        base = i * point_step
        if base + max(x_offset, y_offset) + 4 > len(data):
            break
        px = struct.unpack_from('<f', data, base + x_offset)[0]
        py = struct.unpack_from('<f', data, base + y_offset)[0]
        if math.isfinite(px) and math.isfinite(py):
            points.append((px, py))
    return points


class VbotPathFollower(Node):
    def __init__(self):
        super().__init__('vbot_path_follower_node')
        qos_be = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # 数据缓存
        self.current_odom = None
        self.local_path = None
        self.odom_received = False
        self.path_received = False
        self.last_path_time = 0.0       # 最后收到路径的时间戳
        self.obstacle_points = []       # 过滤后障碍点(head_init坐标系) [(x,y), ...]
        self.prev_obstacle_points = []  # 上一帧障碍点（用于估计障碍物运动）
        self.prev_obstacle_time = 0.0
        self.cloud_received = False

        # 状态机
        self.state = STATE_NORMAL
        self.state_start_time = 0.0     # 进入当前状态的时间
        self.backup_start_time = 0.0    # 后退开始时间
        self.turn_target_yaw = 0.0      # 原地转向目标角度
        self.wait_start_time = 0.0      # 等待开始时间

        # 订阅器
        self.odom_sub = self.create_subscription(
            Odometry, '/odometry', self.odom_callback, qos_be)
        self.path_sub = self.create_subscription(
            Path, '/dog_output_local_path', self.path_callback, qos_be)
        self.cloud_sub = self.create_subscription(
            PointCloud2, '/lidar_points_filtered', self.cloud_callback, qos_be)

        # 发布器
        self.vel_pub = self.create_publisher(Twist, '/vel_cmd', qos_be)

        # 控制定时器
        self.control_timer = self.create_timer(1.0 / CONTROL_FREQ, self.control_callback)

        self.get_logger().info("Vbot路径跟踪节点(v2增强版)已启动，等待话题数据...")

    # ========================================================================
    # 回调函数
    # ========================================================================
    def odom_callback(self, msg):
        self.current_odom = msg
        self.odom_received = True

    def path_callback(self, msg):
        if len(msg.poses) > 1:
            self.local_path = msg
            self.path_received = True
            self.last_path_time = time.time()
        elif len(msg.poses) <= 1:
            # 规划器发送空路径或单点路径：到达目标/规划失败，停车
            self.local_path = msg
            self.last_path_time = time.time()

    def cloud_callback(self, msg):
        """接收过滤后的障碍点云(已在head_init坐标系下)"""
        new_points = read_pointcloud2_xy(msg)
        self.prev_obstacle_points = self.obstacle_points
        self.prev_obstacle_time = getattr(self, '_last_cloud_time', 0.0)
        self.obstacle_points = new_points
        self._last_cloud_time = time.time()
        self.cloud_received = True

    # ========================================================================
    # 辅助函数
    # ========================================================================
    def get_robot_pose(self):
        """返回 (x, y, yaw) 或 (None, None, None)"""
        if not self.current_odom:
            return None, None, None
        p = self.current_odom.pose.pose.position
        q = self.current_odom.pose.pose.orientation
        yaw = quaternion_to_yaw(q.x, q.y, q.z, q.w)
        return p.x, p.y, yaw

    def get_robot_speed(self):
        """返回当前线速度（前进方向为正）"""
        if not self.current_odom:
            return 0.0
        return self.current_odom.twist.twist.linear.x

    def scan_front_obstacles(self, robot_x, robot_y, robot_yaw):
        """
        扫描机器人前方扇形区域的障碍物。
        返回:
          front_min_dist: 前方最近障碍距离
          front_points:   前方扇形内的障碍点列表
          left_count:     左侧障碍点数
          right_count:    右侧障碍点数
        """
        front_min_dist = float('inf')
        front_points = []
        left_count = 0
        right_count = 0

        for (ox, oy) in self.obstacle_points:
            dx = ox - robot_x
            dy = oy - robot_y
            dist = math.hypot(dx, dy)

            # 忽略自身范围内的点
            if dist < DOG_DIAG_RADIUS:
                continue

            # 计算障碍点相对机器人的方位角
            angle_to_obs = math.atan2(dy, dx)
            relative_angle = normalize_angle(angle_to_obs - robot_yaw)
            relative_angle_deg = math.degrees(relative_angle)

            # 检查是否在前方扇形内
            if abs(relative_angle_deg) <= FRONT_SECTOR_HALF_ANGLE:
                front_points.append((ox, oy, dist))
                if dist < front_min_dist:
                    front_min_dist = dist

            # 统计左右障碍（用于决定转向方向）
            if abs(relative_angle_deg) <= SIDE_SECTOR_HALF_ANGLE:
                if relative_angle_deg > 0:
                    left_count += 1
                else:
                    right_count += 1

        return front_min_dist, front_points, left_count, right_count

    def estimate_obstacle_cluster_width(self, front_points, robot_x, robot_y, robot_yaw):
        """
        估计前方障碍物聚类的横向宽度。
        将前方障碍点投影到机器人的左右方向（body y轴），计算宽度。
        """
        if not front_points:
            return 0.0
        min_lateral = float('inf')
        max_lateral = float('-inf')
        cos_yaw = math.cos(robot_yaw)
        sin_yaw = math.sin(robot_yaw)
        for (ox, oy, _dist) in front_points:
            dx = ox - robot_x
            dy = oy - robot_y
            # body frame lateral (y方向)
            lateral = -dx * sin_yaw + dy * cos_yaw
            min_lateral = min(min_lateral, lateral)
            max_lateral = max(max_lateral, lateral)
        if max_lateral > min_lateral:
            return max_lateral - min_lateral
        return 0.0

    def estimate_obstacle_motion(self, front_points, robot_x, robot_y, robot_yaw):
        """
        粗略估计前方障碍物的径向运动速度（靠近为正）。
        比较当前帧与上一帧最近障碍距离。
        """
        if not self.prev_obstacle_points or not front_points:
            return 0.0
        dt = self._last_cloud_time - self.prev_obstacle_time if hasattr(self, '_last_cloud_time') else 0.0
        if dt < 0.01:
            return 0.0

        # 当前帧前方最近距离
        cur_min = min(p[2] for p in front_points)

        # 上一帧前方最近距离
        prev_min = float('inf')
        for (ox, oy) in self.prev_obstacle_points:
            dx = ox - robot_x
            dy = oy - robot_y
            dist = math.hypot(dx, dy)
            if dist < DOG_DIAG_RADIUS:
                continue
            angle = math.atan2(dy, dx)
            rel_angle = abs(math.degrees(normalize_angle(angle - robot_yaw)))
            if rel_angle <= FRONT_SECTOR_HALF_ANGLE:
                prev_min = min(prev_min, dist)

        if prev_min == float('inf'):
            return 0.0

        # 正值表示障碍在靠近（距离变小）
        approach_speed = (prev_min - cur_min) / dt
        return approach_speed

    def choose_turn_direction(self, left_count, right_count, robot_yaw, path_poses):
        """
        选择原地转向方向。优先转向路径方向，其次转向障碍少的一侧。
        返回目标yaw。
        """
        # 尝试用路径第一个远点的方向
        if path_poses and len(path_poses) >= 2:
            # 用路径的中间点作为参考方向
            mid_idx = min(len(path_poses) - 1, max(1, len(path_poses) // 3))
            target_x = path_poses[mid_idx].pose.position.x
            target_y = path_poses[mid_idx].pose.position.y
            rx, ry, _ = self.get_robot_pose()
            if rx is not None:
                desired_yaw = math.atan2(target_y - ry, target_x - rx)
                return desired_yaw

        # 备选：转向障碍少的一侧（转90度）
        if right_count <= left_count:
            return normalize_angle(robot_yaw - math.pi / 2)  # 右转
        else:
            return normalize_angle(robot_yaw + math.pi / 2)  # 左转

    def publish_stop(self):
        """发布零速度停车命令"""
        vel = Twist()
        self.vel_pub.publish(vel)

    def publish_vel(self, linear_x, angular_z):
        """发布速度命令（经死区处理）"""
        lx, az = apply_vbot_deadzone(linear_x, angular_z)
        vel = Twist()
        vel.linear.x = lx
        vel.angular.z = az
        self.vel_pub.publish(vel)
        return lx, az

    # ========================================================================
    # 纯追踪路径跟踪（增强版）
    # ========================================================================
    def pure_pursuit(self, robot_x, robot_y, robot_yaw, path_poses, speed_limit=None):
        """
        纯追踪算法 + 自适应速度/预瞄距离。
        speed_limit: 外部速度上限（如减速区域）
        返回 (linear_x, angular_z)
        """
        current_speed = abs(self.get_robot_speed())

        # 自适应预瞄距离
        lookahead = LOOKAHEAD_BASE + LOOKAHEAD_SPEED_GAIN * current_speed
        lookahead = max(LOOKAHEAD_BASE, min(LOOKAHEAD_MAX, lookahead))

        # 查找预瞄点
        target_x, target_y = None, None
        target_dist = 0.0
        for pose in path_poses:
            px = pose.pose.position.x
            py = pose.pose.position.y
            dist = math.hypot(px - robot_x, py - robot_y)
            if dist >= lookahead:
                target_x = px
                target_y = py
                target_dist = dist
                break

        # 无有效预瞄点：检查是否到达终点
        if target_x is None:
            last_pose = path_poses[-1]
            dist_to_end = math.hypot(last_pose.pose.position.x - robot_x,
                                     last_pose.pose.position.y - robot_y)
            if dist_to_end < GOAL_ARRIVE_DIST:
                return 0.0, 0.0
            # 用最后一个点作为预瞄点
            target_x = last_pose.pose.position.x
            target_y = last_pose.pose.position.y
            target_dist = dist_to_end

        # 计算预瞄点在机器人局部坐标系中的位置
        dx = target_x - robot_x
        dy = target_y - robot_y
        local_dx = dx * math.cos(robot_yaw) + dy * math.sin(robot_yaw)
        local_dy = -dx * math.sin(robot_yaw) + dy * math.cos(robot_yaw)

        # 如果预瞄点在身后（local_dx < 0），需要先转向
        if local_dx < -0.1:
            target_angle = math.atan2(dy, dx)
            angle_diff = normalize_angle(target_angle - robot_yaw)
            # 原地转向朝向路径方向
            angular_z = 2.0 * angle_diff  # P控制
            angular_z = max(-VBOT_ANGULAR_Z_MAX_ABS, min(VBOT_ANGULAR_Z_MAX_ABS, angular_z))
            return 0.0, angular_z

        # 纯追踪公式：ω = 2*v*Ly / L²
        L_actual = math.hypot(local_dx, local_dy)
        if L_actual < 1e-6:
            return 0.0, 0.0

        # 根据曲率自适应前进速度
        # 曲率 κ = 2*|Ly| / L²
        curvature = 2.0 * abs(local_dy) / (L_actual ** 2) if L_actual > 0.01 else 0.0

        # 速度策略：曲率大时减速，曲率小时加速
        if curvature < 0.3:
            desired_speed = 0.8  # 直线段较快
        elif curvature < 1.0:
            desired_speed = 0.5  # 中等曲率
        else:
            desired_speed = VBOT_LINEAR_X_FWD_MIN  # 急弯最慢

        # 应用外部速度限制
        if speed_limit is not None:
            desired_speed = min(desired_speed, speed_limit)

        desired_speed = max(VBOT_LINEAR_X_FWD_MIN, min(VBOT_LINEAR_X_FWD_MAX, desired_speed))

        angular_z = (2.0 * local_dy * desired_speed) / (L_actual ** 2)
        angular_z = max(-VBOT_ANGULAR_Z_MAX_ABS, min(VBOT_ANGULAR_Z_MAX_ABS, angular_z))

        return desired_speed, angular_z

    # ========================================================================
    # 状态机控制
    # ========================================================================
    def transition_to(self, new_state):
        """状态转换"""
        if self.state != new_state:
            state_names = {
                STATE_NORMAL: "正常跟踪",
                STATE_EMERGENCY_STOP: "紧急停车",
                STATE_TURN_IN_PLACE: "原地转向",
                STATE_BACK_UP: "后退",
                STATE_WAIT_FOR_PASS: "等待通过"
            }
            self.get_logger().info(
                f"状态切换: {state_names.get(self.state, '?')} -> {state_names.get(new_state, '?')}")
            self.state = new_state
            self.state_start_time = time.time()

    def control_callback(self):
        """控制主回调：状态机 + 紧急避障 + 路径跟踪"""
        # 等待数据
        if not self.odom_received:
            self.get_logger().info("等待 /odometry 话题...", throttle_duration_sec=2.0)
            return
        if not self.path_received:
            self.get_logger().info("等待 /dog_output_local_path 话题...", throttle_duration_sec=2.0)
            return

        robot_x, robot_y, robot_yaw = self.get_robot_pose()
        if robot_x is None:
            return

        # 路径超时检测：任何状态下路径超时都必须停车
        if time.time() - self.last_path_time > PATH_TIMEOUT_SEC:
            self.get_logger().warn("路径超时，停车等待新路径", throttle_duration_sec=2.0)
            self.publish_stop()
            self.transition_to(STATE_EMERGENCY_STOP)
            return

        # 空路径或单点路径：停车
        if self.local_path is None or len(self.local_path.poses) < 2:
            self.publish_stop()
            return

        path_poses = self.local_path.poses

        # ====================================================================
        # 前方障碍物检测
        # ====================================================================
        front_min_dist = float('inf')
        front_points = []
        left_count = 0
        right_count = 0
        obstacle_width = 0.0
        approach_speed = 0.0

        if self.cloud_received and self.obstacle_points:
            front_min_dist, front_points, left_count, right_count = \
                self.scan_front_obstacles(robot_x, robot_y, robot_yaw)
            if front_points:
                obstacle_width = self.estimate_obstacle_cluster_width(
                    front_points, robot_x, robot_y, robot_yaw)
                approach_speed = self.estimate_obstacle_motion(
                    front_points, robot_x, robot_y, robot_yaw)

        is_large_obstacle = obstacle_width > LARGE_OBSTACLE_WIDTH
        is_fast_approaching = approach_speed > OBSTACLE_SPEED_THRESHOLD

        # ====================================================================
        # 状态机决策
        # ====================================================================
        now = time.time()
        state_elapsed = now - self.state_start_time

        if self.state == STATE_NORMAL:
            # 正常跟踪模式中检查紧急情况
            if front_min_dist < EMERGENCY_STOP_DIST:
                if is_large_obstacle and is_fast_approaching:
                    # 大型快速障碍物（疑似车辆）：暂停等待
                    self.transition_to(STATE_WAIT_FOR_PASS)
                    self.wait_start_time = now
                    self.publish_stop()
                    self.get_logger().warn(
                        f"前方{front_min_dist:.2f}m处检测到大型移动障碍(宽{obstacle_width:.2f}m, "
                        f"接近速度{approach_speed:.2f}m/s)，暂停等待通过")
                    return
                elif is_large_obstacle:
                    # 大型静止障碍物：紧急停车（等规划器重规划绕行）
                    self.transition_to(STATE_EMERGENCY_STOP)
                    self.publish_stop()
                    self.get_logger().warn(
                        f"前方{front_min_dist:.2f}m处检测到大型静止障碍(宽{obstacle_width:.2f}m)，紧急停车")
                    return
                else:
                    # 小障碍物：尝试后退一点再转向绕过
                    self.transition_to(STATE_BACK_UP)
                    self.backup_start_time = now
                    self.get_logger().warn(
                        f"前方{front_min_dist:.2f}m处检测到小障碍(宽{obstacle_width:.2f}m)，后退转向绕行")
                    return

            # 在减速区域中正常跟踪但限速
            speed_limit = None
            if front_min_dist < SLOW_DOWN_DIST:
                # 线性减速：距离越近速度越低
                speed_ratio = (front_min_dist - EMERGENCY_STOP_DIST) / (SLOW_DOWN_DIST - EMERGENCY_STOP_DIST)
                speed_ratio = max(0.0, min(1.0, speed_ratio))
                speed_limit = VBOT_LINEAR_X_FWD_MIN + speed_ratio * (0.8 - VBOT_LINEAR_X_FWD_MIN)

            linear_x, angular_z = self.pure_pursuit(robot_x, robot_y, robot_yaw, path_poses, speed_limit)
            lx, az = self.publish_vel(linear_x, angular_z)
            self.get_logger().info(
                f"[正常] v={lx:.2f}m/s ω={az:.2f}rad/s 前方障碍={front_min_dist:.2f}m",
                throttle_duration_sec=0.5)

        elif self.state == STATE_EMERGENCY_STOP:
            self.publish_stop()
            # 检查前方是否已清空
            if front_min_dist > SLOW_DOWN_DIST:
                self.get_logger().info("前方障碍已清除，恢复正常跟踪")
                self.transition_to(STATE_NORMAL)
            elif state_elapsed > 1.0:
                # 停了1秒还没清除：尝试后退
                self.transition_to(STATE_BACK_UP)
                self.backup_start_time = now
            self.get_logger().info(
                f"[紧急停车] 等待{state_elapsed:.1f}s 前方障碍={front_min_dist:.2f}m",
                throttle_duration_sec=0.5)

        elif self.state == STATE_BACK_UP:
            backup_elapsed = now - self.backup_start_time
            if backup_elapsed < 1.0:
                # 后退1秒（约0.3m）
                lx, az = self.publish_vel(-0.3, 0.0)  # 最小后退速度
                self.get_logger().info(
                    f"[后退] v={lx:.2f}m/s 后退{backup_elapsed:.1f}s",
                    throttle_duration_sec=0.3)
            else:
                # 后退完成，转向原地转向
                self.turn_target_yaw = self.choose_turn_direction(
                    left_count, right_count, robot_yaw, path_poses)
                self.transition_to(STATE_TURN_IN_PLACE)

        elif self.state == STATE_TURN_IN_PLACE:
            yaw_error = normalize_angle(self.turn_target_yaw - robot_yaw)
            if abs(yaw_error) < 0.15:  # ~8.6度以内认为转到位
                self.get_logger().info("原地转向完成，恢复正常跟踪")
                self.transition_to(STATE_NORMAL)
            elif state_elapsed > 5.0:
                # 转向超时：强制恢复
                self.get_logger().warn("原地转向超时5s，强制恢复正常跟踪")
                self.transition_to(STATE_NORMAL)
            else:
                angular_z = 1.5 * yaw_error  # P控制
                angular_z = max(-VBOT_ANGULAR_Z_MAX_ABS, min(VBOT_ANGULAR_Z_MAX_ABS, angular_z))
                lx, az = self.publish_vel(0.0, angular_z)  # 线速度0，原地转
                self.get_logger().info(
                    f"[原地转向] yaw_err={math.degrees(yaw_error):.1f}° ω={az:.2f}rad/s",
                    throttle_duration_sec=0.3)

        elif self.state == STATE_WAIT_FOR_PASS:
            self.publish_stop()
            wait_elapsed = now - self.wait_start_time
            if front_min_dist > SLOW_DOWN_DIST:
                self.get_logger().info("大型障碍已通过，恢复正常跟踪")
                self.transition_to(STATE_NORMAL)
            elif wait_elapsed > 15.0:
                # 等待15秒还没通过：可能不是移动障碍，尝试绕行
                self.get_logger().warn("等待大型障碍超过15s，尝试后退绕行")
                self.transition_to(STATE_BACK_UP)
                self.backup_start_time = now
            else:
                self.get_logger().info(
                    f"[等待通过] 已等待{wait_elapsed:.1f}s 前方障碍={front_min_dist:.2f}m "
                    f"宽度={obstacle_width:.2f}m",
                    throttle_duration_sec=1.0)


def main(args=None):
    rclpy.init(args=args)
    node = VbotPathFollower()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("接收到退出信号，节点停止...")
        # 停止机器狗：发布零速度
        stop_msg = Twist()
        node.vel_pub.publish(stop_msg)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

