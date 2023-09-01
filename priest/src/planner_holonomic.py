#!/usr/bin/env python3

import rospy 
from laser_assembler.srv import *
from sensor_msgs.msg import PointCloud
from geometry_msgs.msg import Twist
import tf
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import PoseWithCovarianceStamped 
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray  
from geometry_msgs.msg import Point
from geometry_msgs.msg import Pose
from nav_msgs.msg import Odometry
from tf.transformations import quaternion_from_euler
import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
import jax
import bernstein_coeff_order10_arbitinterval
import mpc_expert
import matplotlib.pyplot as plt
from scipy.io import loadmat
import scipy
import time
from jax import vmap, random
from scipy.interpolate import UnivariateSpline
from sensor_msgs.msg import LaserScan
from threading import Lock
from tf2_msgs.msg import TFMessage
import threading
import random as rnd
import open3d
from std_srvs.srv import Empty
from tempfile import TemporaryFile
import message_filters
import bernstein_coeff_order10_arbitinterval

# unpause_physics_service = None
# pause_physics_service = None


class planning_traj():

    def __init__(self):

        rospy.init_node("MPC_holonomic_planning")
        rospy.on_shutdown(self.shutdown)

        self.cmd_vel_pub = rospy.Publisher("robot1/cmd_vel", Twist, queue_size=10)
        self.feas_path_publisher = rospy.Publisher("feas_path", Marker, queue_size=1)
        self.visible_obs_publisher = rospy.Publisher("visible_obs", MarkerArray, queue_size=10)
        self.homotopy_path_publisher = rospy.Publisher("homotopy", MarkerArray, queue_size=1)
        
        ############# Initial conditions###################

        self.vx_init = 0.05
        self.vy_init = 0.1
        self.ax_init = 0.0
        self.ay_init = 0.0
        self.vx_fin = 0.0
        self.vy_fin = 0.0
        self.ax_fin = 0.0
        self.ay_fin = 0.0
        self.x_init = -0.6
        self.y_init = 1
        self.x_fin = -3
        self.y_fin = 15
        self.x_end = -3
        self.y_end = 15

        self.x_best_global = jnp.linspace(self.x_init, self.x_init, 100)
        self.y_best_global = jnp.linspace(self.y_init, self.y_init, 100)

        ################ Boundaries ######################### 

        self.v_max = 0.65
        self.v_min = 0.02
        self.a_max = 1.1
        self.v_des = 0.5

        #####################Setting#############################

        self.maxiter = 1
        self.maxiter_cem = 13
        self.weight_track = 0.001
        self.weight_smoothness = 1

        self.a_obs = 0.58
        self.b_obs = 0.58

        self.t_fin = 10 
        self.num = 100 
        self.num_batch = 110
        self.tot_time = np.linspace(0, self.t_fin, self.num)
        
        self.theta_des = np.arctan2(self.y_fin-self.y_init, self.x_fin-self.x_init) 
        
        self.maxiter_mpc = 1000
        self.mutex = Lock()
        self.updated = False
        self.updated_obs = False
        self.updated_homotopy = False
        self.num_update_state = 1
        self.num_up = 100
        #self.t_update = 0.04#### simulation


        self.x_waypoint = jnp.linspace(self.x_init, self.x_fin + 0.0 * jnp.cos(self.theta_des) , 1000)
        self.y_waypoint = jnp.linspace(self.y_init, self.y_fin + 0.0 * jnp.sin(self.theta_des),  1000)
        self.way_point_shape = 1000

        self.x_obs_init_raw = np.ones(780) * 1000
        self.y_obs_init_raw = np.ones(780) * 1000  

        self.x_obs_down = np.ones(420) * 100
        self.y_obs_down = np.ones(420) * 100

        self.x_obs_init = np.ones(420) * 100
        self.y_obs_init = np.ones(420) * 100  

        self.vx_obs = np.zeros(420) 
        self.vy_obs = np.zeros(420) 

        self.x_obs_visible = 10 * jnp.ones(60)
        self.y_obs_visible = 10 * jnp.ones(60)

        self.xyz = np.random.rand(780, 3)
        self.pcd =open3d.geometry.PointCloud()

    ############ calling robot position and obstacles #######################

    def Callback(self, msg_obs, msg_rob):

        ####calling robot postion
        self.x_init = msg_rob.pose.pose.position.x 
        self.y_init = msg_rob.pose.pose.position.y 

        #### calling obstacles through lasersacn
        counter = 0
        increment_value = 1
        for i in range (0, len(msg_obs.points), increment_value):

            self.x_obs_init_raw[counter] = msg_obs.points[i].x 
            self.y_obs_init_raw[counter] = msg_obs.points[i].y 
            counter += 1

        idxes = np.argwhere((self.x_obs_init_raw[:]>=15) | (self.y_obs_init_raw[:]>=15))
        self.x_obs_init_raw[idxes] = self.x_obs_init_raw[0]
        self.y_obs_init_raw[idxes] = self.y_obs_init_raw[0]

        self.xyz[:,0] = self.x_obs_init_raw
        self.xyz[:,1] = self.y_obs_init_raw
        self.xyz[:,2] = 0.0

        self.pcd.points = open3d.utility.Vector3dVector(self.xyz)
        downpcd = self.pcd.voxel_down_sample(voxel_size = 0.16) 
        downpcd_array = np.asarray(downpcd.points)
        num_down_samples = downpcd_array[:,0].shape[0]

        self.x_obs_down[0:num_down_samples] = downpcd_array[:,0]
        self.y_obs_down[0:num_down_samples] = downpcd_array[:,1]

        self.x_obs_down[num_down_samples-1:-1] = 100
        self.y_obs_down[num_down_samples-1:-1] = 100

        self.x_obs_init = self.x_obs_down
        self.y_obs_init = self.y_obs_down


    #############Plot the closest obstacles: Plotting visible obstacles increase computation time, so this option is commented in our code and only used for sanity check of closest obstacles. #########################

    def drawVisibleObstacles(self, obs_pose, index,marker_array_points):

        marker= Marker()
        marker.header.frame_id = "odom"
        marker.ns = ''
        marker.id = index
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.ns = ""
        marker.action = Marker.ADD
        marker.scale.x = 0.1
        marker.scale.y = 0.1
        marker.scale.z = 0.001
        marker.pose.orientation.w = 1.0
        marker.color.a = 1.0
        marker.color.r = 0.1
        marker.color.g = 0.6
        marker.color.b = 0.3
        marker.pose = obs_pose
        marker_array_points.markers.append(marker)

        return marker_array_points

    
    def PlotObstacles(self):

        marker_array_points = MarkerArray()
        anti_direction = False 
        while(True):  
            if(self.updated_obs==True):

                self.mutex.acquire()
                temp_obs_x = self.x_obs_visible
                temp_obs_y = self.y_obs_visible
                self.mutex.release()

                for j in range(jnp.shape(temp_obs_x)[0]):
                    obs_pose = Pose()
                    obs_pose.position.x = temp_obs_x[j]
                    obs_pose.position.y = temp_obs_y[j]
                    obs_pose.position.z = 0

                    q = quaternion_from_euler(0, 0, 1.57)
                    obs_pose.orientation.x = q[0]
                    obs_pose.orientation.y = q[1]
                    obs_pose.orientation.z = q[2]
                    obs_pose.orientation.w = q[3]
                    marker_array_points = self.drawVisibleObstacles(obs_pose,j,marker_array_points)

                self.visible_obs_publisher.publish(marker_array_points)
                self.updated_obs = False

    ################# Publishing the best path at each iteration#######################

    def pathPublisher(self):

        while (True):
            if(self.updated==True):

                self.mutex.acquire()
                temp_traj_x = self.x_best_global
                temp_traj_y = self.y_best_global

                self.mutex.release()

                marker_feas = Marker()
                marker_feas.id = 0
                marker_feas.header.frame_id = 'odom'
                marker_feas.header.stamp = rospy.Time.now()
                marker_feas.type = Marker.LINE_STRIP
                marker_feas.ns = "feas_path_"+str(1)
                marker_feas.action = Marker.ADD
                marker_feas.scale.x = 0.1
                marker_feas.pose.orientation.w = 1.0
                marker_feas.color.a = 1.0
                marker_feas.color.r = 0.0
                marker_feas.color.g = 0.0
                marker_feas.color.b = 1.0

                for j in range(jnp.shape(temp_traj_x)[0]):

                    feas_point = Point()
                    feas_point.x = temp_traj_x[j]
                    feas_point.y = temp_traj_y[j]
                    marker_feas.points.append(feas_point)
 
                self.feas_path_publisher.publish(marker_feas)
                self.updated = False 

    ######################################

    def planner(self, ):

        vel_cmd = Twist()

        ### subscribe to robot postion and observed obstacles
        sub_PointCloud = message_filters.Subscriber("/pointcloud", PointCloud)
        sub_Odom = message_filters.Subscriber("/robot1/odom", Odometry)
        ts = message_filters.ApproximateTimeSynchronizer([sub_PointCloud, sub_Odom], 1,1, allow_headerless=True )
        ts.registerCallback(self.Callback)

        ########## Best path and obstacles are shown in separate threads 
        # thread_1 = threading.Thread(target=self.pathPublisher)
        #thread_2 = threading.Thread(target=self.PlotObstacles)

        # thread_1.start()
        #thread_2.start()

        #### number of obstacles
        num_obs = 60

        prob = mpc_expert.batch_crowd_nav(self.a_obs, self.b_obs, self.v_max, self.v_min, self.a_max, num_obs, self.t_fin, self.num, self.num_batch, self.maxiter, self.maxiter_cem, self.weight_smoothness, self.weight_track, self.way_point_shape, self.v_des)
        key = random.PRNGKey(0)

        arc_length, arc_vec, x_diff, y_diff =prob.path_spline(self.x_waypoint, self.y_waypoint)
    
        x_best = jnp.linspace(self.x_init, self.x_init + self.v_des * self.t_fin, 100)
        y_best = jnp.linspace(self.y_init, self.y_init + self.v_des * self.t_fin, 100)

        ############### use the following line to save the trajectories
        # c_x_data = []
        # c_y_data = []

        # x_data = []
        # y_data = []

        # vx_data = []
        # vy_data = []

        # ax_data = []
        # ay_data = []

        # x_ellite_data = []
        # y_ellite_data = []

        # total_time = []
        mpc_count = 0

        initial_state = jnp.hstack(( self.x_init, self.y_init, self.vx_init, self.vy_init, self.ax_init, self.ay_init ))

        ################# Providing warming samples
        x_guess_per, y_guess_per = prob.compute_warm_traj(initial_state, self.v_des, self.x_waypoint, self.y_waypoint, arc_vec, x_diff, y_diff)

        for i in range(0, self.maxiter_mpc):

            initial_state = jnp.hstack(( self.x_init, self.y_init, self.vx_init, self.vy_init, self.ax_init, self.ay_init ))

            lamda_x = jnp.zeros((self.num_batch, prob.nvar))
            lamda_y = jnp.zeros((self.num_batch, prob.nvar))

            start = time.time()

            x_obs_init = self.x_obs_init
            y_obs_init = self.y_obs_init

            vx_obs = self.vx_obs
            vy_obs = self.vy_obs

            x_obs_trajectory, y_obs_trajectory, x_obs_trajectory_proj, y_obs_trajectory_proj = prob.compute_obs_traj_prediction(jnp.asarray(x_obs_init).flatten(), jnp.asarray(y_obs_init).flatten(), vx_obs, vy_obs, initial_state[0], initial_state[1] ) ####### obstacle trajectory prediction

            sol_x_bar, sol_y_bar, x_guess, y_guess,  xdot_guess, ydot_guess, xddot_guess, yddot_guess,c_mean, c_cov, x_fin, y_fin = prob.compute_traj_guess( initial_state, x_obs_trajectory, y_obs_trajectory, self.v_des, self.x_waypoint, self.y_waypoint, arc_vec, x_guess_per, y_guess_per, x_diff, y_diff)
            
            self.x_fin = x_fin
            self.y_fin = y_fin  

            # pause_physics_service()

            x_ellite, y_ellite, x, y, c_x_best, c_y_best, x_best, y_best, xdot_best, ydot_best, x_guess_per , y_guess_per = prob.compute_cem(key, initial_state, self.x_fin, self.y_fin, lamda_x, lamda_y, x_obs_trajectory, y_obs_trajectory, x_obs_trajectory_proj, y_obs_trajectory_proj, sol_x_bar, sol_y_bar, x_guess, y_guess,  xdot_guess, ydot_guess, xddot_guess, yddot_guess, self.x_waypoint,  self.y_waypoint, arc_vec, c_mean, c_cov )
            
            delta_t = jnp.array(time.time() - start)
    
            vx_control, vy_control, ax_control, ay_control = prob.compute_controls(c_x_best, c_y_best, delta_t)

            self.x_best_global = x_best
            self.y_best_global = y_best

            self.x_obs_visible = x_obs_trajectory[:,0]
            self.y_obs_visible = y_obs_trajectory[:,0]

            self.updated_obs = True
            self.updated = True

            ###### use the following line if you are not updating the robot position through /odom  
            #self.x_init = self.x_init + vx_init*self.t_update #### ROS odom
            #self.y_init = self.y_init + vy_init*self.t_update #### ROS odom

            ###########################
            self.vx_init = vx_control
            self.vy_init = vy_control

            self.ax_init = ax_control
            self.ay_init = ay_control

            vel_cmd.linear.x = self.vx_init
            vel_cmd.linear.y = self.vy_init


            if i>3:
                # unpause_physics_service()
                self.cmd_vel_pub.publish(vel_cmd)
                rospy.sleep(0.012)
                # total_time.append(time.time()- start)

            print (time.time() - start, "Comp. time")


            # c_x_data.append(c_x_best)
            # c_y_data.append(c_y_best)

            # x_data.append(self.x_init)
            # y_data.append(self.y_init)

            # vx_data.append(self.vx_init)
            # vy_data.append(self.vy_init)

            # ax_data.append(self.ax_init)
            # ay_data.append(self.ay_init)

            # x_ellite_data.append(x_ellite)
            # y_ellite_data.append(y_ellite)


            dist_final_point = np.sqrt((self.x_init - self.x_end)**2 + (self.y_init - self.y_end)**2)
            mpc_count = mpc_count +1

            if (dist_final_point <= 0.5):
                vel_cmd.linear.x = 0.0
                vel_cmd.linear.y = 0.0
                self.cmd_vel_pub.publish(vel_cmd)
                print("Distance is satisfied")
                break
                
        print("Stop mpc")

        # x_data = np.array(x_data)
        # y_data = np.array(y_data)

        # vx_data = np.array(vx_data)
        # vy_data = np.array(vy_data)

        # ax_data = np.array(ax_data)
        # ay_data = np.array(ay_data)

        # c_x_data = np.array(c_x_data)
        # c_y_data = np.array(c_y_data)

        # x_ellite_data = np.array(x_ellite_data)
        # y_ellite_data = np.array(y_ellite_data)

        # total_time = np.array(total_time)

        # cx_data =  c_x_data.reshape(mpc_count , 11)
        # cy_data =  c_y_data.reshape(mpc_count , 11)

        # x_data =  x_data
        # y_data =  y_data

        # world_num = 115
        # np.save("cx_data_"+str(world_num)+".npy", cx_data)
        # np.save("cy_data_"+str(world_num)+".npy", cy_data)

        # np.save("vx_data_"+str(world_num)+".npy", vx_data)
        # np.save("vy_data_"+str(world_num)+".npy", vy_data)

        # np.save("ax_data_"+str(world_num)+".npy", ax_data)
        # np.save("ay_data_"+str(world_num)+".npy", ay_data)

        # np.save("cost_smoothness_"+str(world_num)+".npy", cost_smoothness)
        # np.save("arc_length_"+str(world_num)+".npy", arc_length_best)
        # np.save("trackin_error_cost_"+str(world_num)+".npy", tracking_error_cost)
        # np.save("total_time_"+str(world_num)+".npy", total_time)

        # np.save("x_ellite"+str(world_num)+".npy", x_ellite_data)
        # np.save("y_ellite"+str(world_num)+".npy", y_ellite_data)
        
        print("Done")
        rospy.signal_shutdown('Finished running node')
            
    def shutdown(self):

        rospy.loginfo("Stop")
        rospy.sleep(2)

    
if __name__ == "__main__":

    # pause_physics_service = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
    # unpause_physics_service = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)

    motion_planning = planning_traj()
    motion_planning.planner()
    motion_planning.pathPublisher()
    motion_planning.PlotObstacles()

    




    


    
    




    


    

















	







