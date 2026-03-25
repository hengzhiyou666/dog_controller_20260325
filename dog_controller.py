#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Vbot机器狗路径跟踪节点
ROS2 Humble | Ubuntu22.04
订阅：/odometry(nav_msgs/msg/Odometry)、/dog_output_local_path(nav_msgs/msg/Path)
发布：/vel_cmd(geometry_msgs/msg/Twist) 适配Vbot速度控制
功能：纯追踪算法跟踪局部路径 + 话题等待 + 计算计时 + 过程打印
"""
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import Twist, PoseStamped
import time
import math

# Vbot速度限制（严格遵循文档要求）
VBOT_LINEAR_X_MIN = 0.3    # 最小前进线速度
VBOT_LINEAR_X_MAX = 1.5    # 最大前进线速度
VBOT_ANGULAR_Z_MIN = -3.0  # 最大右转角速度
VBOT_ANGULAR_Z_MAX = 3.0   # 最大左转角速度
# 纯追踪算法参数
LOOKAHEAD_DISTANCE = 0.8   # 预瞄距离（可根据实际调试）
CONTROL_FREQ = 10.0        # 控制发布频率（匹配路径话题10Hz）

def quaternion_to_euler(x, y, z, w):
    """
    替代tf_transformations的四元数转欧拉角函数（yaw为偏航角）
    :param x,y,z,w: 四元数分量
    :return: roll, pitch, yaw
    """
    # 四元数归一化
    norm = math.sqrt(x**2 + y**2 + z**2 + w**2)
    x /= norm
    y /= norm
    z /= norm
    w /= norm

    # 计算欧拉角
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2 * (w * y - z * x)
    pitch = math.asin(sinp)

    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw

class VbotPathFollower(Node):
    def __init__(self):
        super().__init__('vbot_path_follower_node')
        # QoS配置（适配ROS2 Humble，兼容话题发布频率）
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        # 初始化数据缓存
        self.current_odom = None  # 当前位姿缓存
        self.local_path = None    # 局部路径缓存
        self.odom_received = False
        self.path_received = False

        # 订阅器
        self.odom_sub = self.create_subscription(
            Odometry,
            '/odometry',
            self.odom_callback,
            qos_profile
        )
        self.path_sub = self.create_subscription(
            Path,
            '/dog_output_local_path',
            self.path_callback,
            qos_profile
        )

        # 发布器（Vbot速度控制话题/vel_cmd）
        self.vel_pub = self.create_publisher(
            Twist,
            '/vel_cmd',
            qos_profile
        )

        # 控制定时器（按CONTROL_FREQ执行路径跟踪计算）
        self.control_timer = self.create_timer(
            1.0 / CONTROL_FREQ,
            self.control_callback
        )

        self.get_logger().info("Vbot路径跟踪节点已启动，等待话题数据...")

    def odom_callback(self, msg: Odometry):
        """位姿话题回调：更新当前位姿"""
        self.current_odom = msg
        self.odom_received = True

    def path_callback(self, msg: Path):
        """局部路径话题回调：更新局部路径（过滤空路径）"""
        if len(msg.poses) > 1:  # 路径点至少2个才有效
            self.local_path = msg
            self.path_received = True

    def get_robot_pose(self):
        """从Odometry解析机器人当前x,y,yaw（欧拉角）"""
        if not self.current_odom:
            return None, None, None
        # 位置
        x = self.current_odom.pose.pose.position.x
        y = self.current_odom.pose.pose.position.y
        # 四元数转欧拉角（使用自定义函数，替代tf_transformations）
        q = self.current_odom.pose.pose.orientation
        roll, pitch, yaw = quaternion_to_euler(q.x, q.y, q.z, q.w)
        return x, y, yaw

    def pure_pursuit(self, robot_x, robot_y, robot_yaw, path_poses):
        """
        纯追踪算法计算速度指令
        :param robot_x/robot_y/robot_yaw: 机器人当前位姿
        :param path_poses: 局部路径点列表[PoseStamped, ...]
        :return: linear_x, angular_z 计算后的速度值
        """
        # 1. 遍历路径点，找到预瞄点
        target_x, target_y = None, None
        for pose in path_poses:
            px = pose.pose.position.x
            py = pose.pose.position.y
            # 计算机器人到路径点的距离
            dist = math.hypot(px - robot_x, py - robot_y)
            # 找到第一个距离≥预瞄距离的点作为预瞄点
            if dist >= LOOKAHEAD_DISTANCE:
                target_x = px
                target_y = py
                break
        # 无有效预瞄点：停止移动
        if target_x is None or target_y is None:
            return 0.0, 0.0

        # 2. 计算预瞄点相对机器人的坐标（机器人局部坐标系）
        dx = target_x - robot_x
        dy = target_y - robot_y
        # 转换到机器人局部坐标系（绕yaw旋转）
        local_dx = dx * math.cos(robot_yaw) + dy * math.sin(robot_yaw)
        local_dy = -dx * math.sin(robot_yaw) + dy * math.cos(robot_yaw)

        # 3. 计算需要的角速度（纯追踪核心公式）
        if abs(local_dx) < 1e-6:  # 避免除零
            angular_z = 0.0
        else:
            angular_z = (2 * local_dy * VBOT_LINEAR_X_MIN) / (LOOKAHEAD_DISTANCE ** 2)

        # 4. 速度限幅（严格适配Vbot文档要求）
        linear_x = VBOT_LINEAR_X_MIN  # 固定基础前进速度（可按需调）
        angular_z = max(VBOT_ANGULAR_Z_MIN, min(VBOT_ANGULAR_Z_MAX, angular_z))

        return linear_x, angular_z

    def control_callback(self):
        """控制主回调：核心逻辑执行"""
        # 检查话题是否都接收到数据
        if not self.odom_received and not self.path_received:
            self.get_logger().info("等待。。。/odometry和/dog_output_local_path话题")
            return
        elif not self.odom_received:
            self.get_logger().info("等待。。。/odometry话题")
            return
        elif not self.path_received:
            self.get_logger().info("等待。。。/dog_output_local_path话题")
            return

        # 开始计算：打印标记+计时
        self.get_logger().info("="*20 + "开始路径跟踪计算" + "="*20)
        start_time = time.time()

        # 1. 获取机器人当前位姿
        robot_x, robot_y, robot_yaw = self.get_robot_pose()
        if robot_x is None:
            self.get_logger().error("解析机器人位姿失败，跳过本次计算")
            return

        # 2. 获取局部路径点
        path_poses = self.local_path.poses
        if len(path_poses) < 2:
            self.get_logger().warn("局部路径点数量不足，发布零速度")
            vel_msg = Twist()
            self.vel_pub.publish(vel_msg)
            end_time = time.time()
            calc_time = (end_time - start_time) * 1000  # 转毫秒
            self.get_logger().info(f"="*20 + "结束路径跟踪计算" + "="*20)
            self.get_logger().info(f"本次计算耗时：{calc_time:.2f} 毫秒\n")
            return

        # 3. 纯追踪算法计算速度
        linear_x, angular_z = self.pure_pursuit(robot_x, robot_y, robot_yaw, path_poses)

        # 4. 构造速度消息并发布（适配Vbot/vel_cmd话题）
        vel_msg = Twist()
        vel_msg.linear.x = linear_x
        vel_msg.linear.y = 0.0
        vel_msg.linear.z = 0.0
        vel_msg.angular.x = 0.0
        vel_msg.angular.y = 0.0
        vel_msg.angular.z = angular_z
        self.vel_pub.publish(vel_msg)

        # 结束计算：打印标记+计时结果
        end_time = time.time()
        calc_time = (end_time - start_time) * 1000  # 计算耗时转毫秒
        self.get_logger().info(f"发布速度指令：线速度x={linear_x:.2f}m/s，角速度z={angular_z:.2f}rad/s")
        self.get_logger().info("="*20 + "结束路径跟踪计算" + "="*20)
        self.get_logger().info(f"本次计算耗时：{calc_time:.2f} 毫秒\n")

def main(args=None):
    # 初始化ROS2
    rclpy.init(args=args)
    # 创建节点
    node = VbotPathFollower()
    # 自旋节点
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("接收到退出信号，节点停止...")
        # 停止机器狗：发布零速度
        stop_msg = Twist()
        node.vel_pub.publish(stop_msg)
    finally:
        # 销毁节点
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

