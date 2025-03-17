import sys
sys.path.append(".")


# ros package
import rospy
import actionlib
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point, Pose, Quaternion, Twist, Vector3
import tf
from ros_numpy import numpify
from std_msgs.msg import Int32  # Import Int32 message

# habitat
from arguments import get_args
import numpy as np
from sem_policy import SLAMTrainer


# global variables
global obs
obs = {}


def odom_callback(msg):
    global obs
    obs['gps'] = [msg.pose.pose.position.x, msg.pose.pose.position.y]
    (roll, pitch, yaw) = tf.transformations.euler_from_quaternion([msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w])
    obs['compass'] = [yaw]
    obs['objectgoal'] = [4]


def rgb_callback(msg):
    global obs
    h = msg.height
    w = msg.width
    rs_rgb = numpify(msg)
    rs_rgb = np.reshape(rs_rgb, (h, w, 3))
    obs['rgb'] = rs_rgb


def depth_callback(msg):
    global obs
    h = msg.height
    w = msg.width
    rs_depth = numpify(msg)
    rs_depth = np.reshape(rs_depth, (h, w, 1))
    obs['depth'] = rs_depth


def main():
    rospy.init_node('lrognav_node')

    rospy.Subscriber("/camera/color/image_raw", Image, rgb_callback)
    rospy.Subscriber("/camera/depth/image_raw", Image, depth_callback)
    rospy.Subscriber("/jackal_velocity_controller/odom", Odometry, odom_callback)

    # Create a publisher to publish action values
    action_publisher = rospy.Publisher('/robot_action', Int32, queue_size=10)  # Change to an appropriate topic type (Int32)

    # Create the listener
    listener = tf.TransformListener()

    time_count = 0
    action = 0

    args = get_args()
    agent = SLAMTrainer(args)

    rate = rospy.Rate(10)  # 10hz
    global obs

    while not rospy.is_shutdown():
        if len(obs) > 0:

            if time_count % 25 == 0:  # Execute action every 25 cycles
                try:
                    (trans, rot) = listener.lookupTransform("/map", "/base_link", rospy.Time(0))

                    obs['gps'] = [trans[0], trans[1]]
                    (roll, pitch, yaw) = tf.transformations.euler_from_quaternion(rot)
                    obs['compass'] = [yaw]
                    obs['objectgoal'] = [0]
                    action = agent.act(obs)  # Get the action from agent
                    rospy.loginfo(f"@@@@@@@@@@@@@@ act = {action}")

                    time_count = 1
                except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                    rospy.logerr("TF lookup failed")

            # Directly publish the action value (which is a number)
            action_publisher.publish(action)

            time_count += 1

        rate.sleep()


if __name__ == "__main__":
    main()
