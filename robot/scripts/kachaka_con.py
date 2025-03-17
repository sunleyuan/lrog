#!/usr/bin/env python3


import kachaka_api

client = kachaka_api.KachakaApiClient(target="192.168.241.79:26400")
# image = client.get_tof_camera_ros_image()
# 状態の取得
# 状態の取得
current_pose = client.get_robot_pose()
print(f"current pose: {current_pose}")


# client.set_robot_velocity(0.3, 0.0)

# client.dock_shelf()
# client.undock_shelf()



#!/usr/bin/env python3

import rospy
from std_msgs.msg import Int32  # 使用 Int32 类型接收 action 消息

# 回调函数，处理接收到的 action 消息
def action_callback(msg):
    rospy.loginfo(f"Received action: {msg.data}")
    # 根据接收到的 action 数据进行处理
    if msg.data == 1:
        rospy.loginfo("Action 1: Move forward")
        client.set_robot_velocity(0.1, 0.0)

    elif msg.data == 2:
        rospy.loginfo("Action 2: Turn left")
        client.set_robot_velocity(0.0, 0.05)

    elif msg.data == 3:
        rospy.loginfo("Action 3: Turn right")
        client.set_robot_velocity(0.0, -0.05)
    else:
        rospy.loginfo("Unknown action")

def main():
    rospy.init_node('action_listener_node')  # 初始化 ROS 节点
    rospy.Subscriber('/robot_action', Int32, action_callback)  # 订阅 action 主题
    rospy.loginfo("Listening to /fsp/action topic...")
    rospy.spin()  # 保持节点运行

if __name__ == '__main__':
    main()
