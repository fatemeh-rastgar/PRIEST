#!/usr/bin/env python3
import rospy
import actionlib
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
import numpy as np
import time
import matplotlib.pyplot as plt


class navigation_movebase():

    def __init__(self):

        rospy.init_node('movebase_client_py')

        self.x_robot = -0.6
        self.y_robot = 1.0

        self.v_x_robot = 0.0
        self.v_y_robot = 0.0

        self.x_fin = -2.2
        self.y_fin = 10.5

        self.x_robot_list = []
        self.y_robot_list = [] 

        self.v_x_robot_list = []
        self.v_y_robot_list = [] 



    def Callback(self, msg):

        self.x_robot = msg.pose.pose.position.x
        self.y_robot = msg.pose.pose.position.y

        self.v_x_robot = msg.twist.twist.linear.x
        self.v_y_robot = msg.twist.twist.linear.y


        self.x_robot_list.append(self.x_robot)
        self.y_robot_list.append(self.y_robot)

        self.v_x_robot_list.append(self.v_x_robot)
        self.v_x_robot_list.append(self.v_y_robot)

        


    def movebase_client(self, ):


        # rospy.Subscriber("/ground_truth/odom", Odometry, self.Callback)
        rospy.Subscriber("/odom", Odometry, self.Callback)
        client = actionlib.SimpleActionClient('move_base',MoveBaseAction)
        client.wait_for_server()

        goal = MoveBaseGoal()
        #goal.target_pose.header.frame_id = "odom" for TEB, DWA
        #goal.target_pose.header.frame_id = "map" #for Mppi
        goal.target_pose.header.frame_id = "odom"
        goal.target_pose.header.stamp = rospy.Time.now()
        goal.target_pose.pose.position.x = self.x_fin
        goal.target_pose.pose.position.y = self.y_fin
        goal.target_pose.pose.orientation.w = 1.0

        client.send_goal(goal)
        # wait = client.wait_for_result()
       
        
        # if not wait:
        #     rospy.logerr("Action server not available!")
        #     rospy.signal_shutdown("Action server not available!")
        # else:
        return self.x_robot_list, self.y_robot_list, self.v_x_robot_list, self.v_y_robot_list, self.x_fin , self.y_fin



if __name__ == '__main__':
    start = time.time()

    move_base_navigation = navigation_movebase()
    x_robot, y_robot, v_x_robot, v_y_robot, x_fin , y_fin = move_base_navigation.movebase_client()

    dist = np.sqrt((x_robot[-1] - (x_fin))**2 + (y_robot[-1] - (y_fin))**2)
    while dist >0.5:
        dist = np.sqrt((x_robot[-1] - (x_fin))**2 + (y_robot[-1] - (y_fin))**2)
    print ("In here")
    x_robot = np.array(x_robot)
    y_robot = np.array(y_robot)

    v_x_robot = np.array(v_x_robot)
    v_y_robot = np.array(v_y_robot)
    

    world_num_MPPI = 240
    np.save("x_robot"+str(world_num_MPPI)+".npy", x_robot)
    np.save("y_robot"+str(world_num_MPPI)+".npy", y_robot)

    np.save("v_x_robot"+str(world_num_MPPI)+".npy", v_x_robot)
    np.save("v_y_robot"+str(world_num_MPPI)+".npy", v_y_robot)
    
    print("Save the data")

    time_total = time.time() - start
    print(time_total)
    np.save("time"+str(world_num_MPPI)+".npy", time_total)
    rospy.loginfo("Goal execution done!")
