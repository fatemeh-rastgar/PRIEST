#!/usr/bin/env python3

import rospy
import actionlib
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from nav_msgs.msg import Odometry
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

    def __init__(self):

        rospy.init_node('movebase_client_py')

        self.x_local = np.ones(500)
        self.y_local = np.ones(500) 

        self.x_robot = 0
        self.y_robot = 0



    def Callback(self, msg):

        increment_value = 1
        counter = 0

        for i in range (0, len(msg.points), increment_value):

            self.x_local[counter] = msg.points[i].x
            self.y_local[counter] = msg.points[i].y
            counter += 1

            

        x_local = self.x_local
        y_local = self.y_local

        x_local = np.array(x_local)
        y_local = np.array(y_local)

        world_num_MPPI = 149_3

        print(x_local)
        np.save("x_local"+str(world_num_MPPI)+".npy", x_local)
        np.save("y_local"+str(world_num_MPPI)+".npy", y_local)


            


    def local_planner(self, ):

        rospy.Subscriber("/visualization_marker", Marker, self.Callback)



   

if __name__ == '__main__':

    motion_planning_1 = local_traj()
    motion_planning_1.local_planner()
    rospy.spin()

    rospy.loginfo("Goal execution done!")
