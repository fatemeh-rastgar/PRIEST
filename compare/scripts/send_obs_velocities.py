#!/usr/bin/env python3
import rospy
import actionlib
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
import numpy as np
import time
import matplotlib.pyplot as plt


class obstacles():

    def __init__(self):

        rospy.init_node('obstacle_dynamic')
        self.cmd_vel_obs_1_pub = rospy.Publisher("obs_1/cmd_vel", Twist, queue_size=20)
        self.cmd_vel_obs_2_pub = rospy.Publisher("obs_2/cmd_vel", Twist, queue_size=20)
        self.cmd_vel_obs_3_pub = rospy.Publisher("obs_3/cmd_vel", Twist, queue_size=20)
        self.cmd_vel_obs_4_pub = rospy.Publisher("obs_4/cmd_vel", Twist, queue_size=20)
        self.cmd_vel_obs_5_pub = rospy.Publisher("obs_5/cmd_vel", Twist, queue_size=20)
        self.cmd_vel_obs_6_pub = rospy.Publisher("obs_6/cmd_vel", Twist, queue_size=10)
        self.cmd_vel_obs_7_pub = rospy.Publisher("obs_7/cmd_vel", Twist, queue_size=10)
        self.cmd_vel_obs_8_pub = rospy.Publisher("obs_8/cmd_vel", Twist, queue_size=10)
        self.cmd_vel_obs_9_pub = rospy.Publisher("obs_9/cmd_vel", Twist, queue_size=10)
        self.cmd_vel_obs_10_pub = rospy.Publisher("obs_10/cmd_vel", Twist, queue_size=20)

        self.vx_obs_dy = 0.0*np.ones(10)
        self.vy_obs_dy = -0.1*np.ones(10)


        

    def velocity_publisher(self, ):



        # thread_1.start()
        # thread_2.start()

        vel_cmd_1 = Twist()
        vel_cmd_2 = Twist()
        vel_cmd_3 = Twist()
        vel_cmd_4 = Twist()
        vel_cmd_5 = Twist()
        vel_cmd_6 = Twist()
        vel_cmd_7 = Twist()
        vel_cmd_8 = Twist()
        vel_cmd_9 = Twist()
        vel_cmd_10 = Twist()

        vel_cmd_1.linear.x = self.vx_obs_dy[0]
        vel_cmd_2.linear.x = self.vx_obs_dy[1]
        vel_cmd_3.linear.x = self.vx_obs_dy[2]
        vel_cmd_4.linear.x = self.vx_obs_dy[3]
        vel_cmd_5.linear.x = self.vx_obs_dy[4]
        vel_cmd_6.linear.x = self.vx_obs_dy[5]
        vel_cmd_7.linear.x = self.vx_obs_dy[6]
        vel_cmd_8.linear.x = self.vx_obs_dy[7]
        vel_cmd_9.linear.x = self.vx_obs_dy[8]
        vel_cmd_10.linear.x = self.vx_obs_dy[9]

        vel_cmd_1.linear.y = self.vy_obs_dy[0]
        vel_cmd_2.linear.y = self.vy_obs_dy[1]
        vel_cmd_3.linear.y = self.vy_obs_dy[2]
        vel_cmd_4.linear.y = self.vy_obs_dy[3]
        vel_cmd_5.linear.y = self.vy_obs_dy[4]
        vel_cmd_6.linear.y = self.vy_obs_dy[5]
        vel_cmd_7.linear.y = self.vy_obs_dy[6]
        vel_cmd_8.linear.y = self.vy_obs_dy[7]
        vel_cmd_9.linear.y = self.vy_obs_dy[8]
        vel_cmd_10.linear.y = self.vy_obs_dy[9]



        for i in range(1000):


            self.cmd_vel_obs_1_pub.publish(vel_cmd_1) 
            time.sleep(0.02)
            self.cmd_vel_obs_2_pub.publish(vel_cmd_2)
            time.sleep(0.02)
            self.cmd_vel_obs_3_pub.publish(vel_cmd_3)
            time.sleep(0.02)
            self.cmd_vel_obs_4_pub.publish(vel_cmd_4)
            time.sleep(0.02)
            self.cmd_vel_obs_5_pub.publish(vel_cmd_5)
            time.sleep(0.02)
            self.cmd_vel_obs_6_pub.publish(vel_cmd_6)
            time.sleep(0.02)
            self.cmd_vel_obs_7_pub.publish(vel_cmd_7)
            time.sleep(0.02)
            self.cmd_vel_obs_8_pub.publish(vel_cmd_8)
            time.sleep(0.02)
            self.cmd_vel_obs_9_pub.publish(vel_cmd_9)
            time.sleep(0.02)
            self.cmd_vel_obs_10_pub.publish(vel_cmd_10)
            time.sleep(0.02)



        
            



if __name__ == '__main__':
   

    obstacles_movement = obstacles()
    obstacles_movement.velocity_publisher()
