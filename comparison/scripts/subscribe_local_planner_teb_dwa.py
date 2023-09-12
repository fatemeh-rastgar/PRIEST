#!/usr/bin/env python3

import rospy
import actionlib
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from nav_msgs.msg import Odometry
from nav_msgs.msg import Path
from geometry_msgs.msg import Twist
import numpy as np
import time
import matplotlib.pyplot as plt
from sensor_msgs.msg import PointCloud
from geometry_msgs.msg import Twist
import tf
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import PoseWithCovarianceStamped 
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray  
from geometry_msgs.msg import Point
import threading


class local_traj():
    
    counter_sub = 0
    def __init__(self):

        rospy.init_node('movebase_teb_dwa')

        self.x_local = np.ones(70)
        self.y_local = np.ones(70) 

        self.x_robot = 0
        self.y_robot = 0

        self.x_homo = []
        self.y_homo = []



    def Callback(self, msg):

        increment_value = 1
        counter = 0
        
    
        print( len(msg.poses), "len")
        for i in range (0, len(msg.poses), increment_value):

            self.x_local[counter] = msg.poses[i].pose.position.x
            self.y_local[counter] = msg.poses[i].pose.position.y
            counter += 1

        
        
        
        


    def local_planner(self, ):

        rospy.Subscriber("/move_base/TebLocalPlannerROS/local_plan", Path, self.Callback)
        x_homotopy = []
        y_homotopy = []

        for i in range(30):

            x_local = self.x_local
            y_local = self.y_local

            x_local = np.array(x_local)
            y_local = np.array(y_local)

            x_homotopy.append(x_local)
            y_homotopy.append(y_local)

            x_homo = x_homotopy
            y_homo = y_homotopy

            x_homo = np.array(x_homo)
            y_homo = np.array(y_homo)
            
            print(i, "=counter" )
            # print(x_homo[:,0])


            rospy.sleep(1)


        x_local = np.array(x_local)
        y_local = np.array(y_local)

        x_homo = np.array(x_homo)
        y_homo = np.array(y_homo)
        
        world_num_MPPI = 278_3

        # # print(x_local)
        # print(np.shape(x_homo))
        # print(x_homo)
        # print(y_homo)
        print("saving")
        
        np.save("x_homo"+str(world_num_MPPI)+".npy", x_homo)
        np.save("y_homo"+str(world_num_MPPI)+".npy", y_homo)

        np.save("x_local"+str(world_num_MPPI)+".npy", x_local)
        np.save("y_local"+str(world_num_MPPI)+".npy", y_local)
        # rospy.Subscriber("/move_base/NavfnROS/plan", Path, self.Callback)
        #rospy.Subscriber("/move_base/DWAPlannerROS/local_plan", Path, self.Callback)

     




   

if __name__ == '__main__':

    motion_planning_1 = local_traj()
    motion_planning_1.local_planner()
    rospy.spin()

    rospy.loginfo("Goal execution done!")
