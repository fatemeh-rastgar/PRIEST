#!/usr/bin/env python3

import rospy 
from laser_assembler.srv import *
from sensor_msgs.msg import PointCloud
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
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
import mpc_cem_dynamic
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
from tf.transformations import euler_from_quaternion, quaternion_from_euler

unpause_physics_service = None
pause_physics_service = None


class planning_traj():

    def __init__(self):

        rospy.init_node("MPC_planning")
        rospy.on_shutdown(self.shutdown)


        self.cmd_vel_pub = rospy.Publisher("/mppi/cmd_vel", Twist, queue_size=10)
        self.feas_path_publisher = rospy.Publisher("feas_path", Marker, queue_size=1)
        self.visible_obs_publisher = rospy.Publisher("visible_obs", MarkerArray, queue_size=10)
        self.homotopy_path_publisher = rospy.Publisher("homotopy", MarkerArray, queue_size=1)
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
        
        ################################

        self.vx_init = 0.05
        self.vy_init = 0.1
        self.ax_init = 0.0
        self.ay_init = 0.0
        self.vx_fin = 0.0
        self.vy_fin = 0.0
        self.ax_fin = 0.0
        self.ay_fin = 0.0

        ######################################### MPC parameters

        self.v_max = 1.0
        self.v_min = 0.02
        self.a_max = 1.0
        ##################################################

        self.maxiter = 1
        self.maxiter_cem = 13
        self.weight_track = 0.001
        self.weight_smoothness = 1

        self.a_obs_1 = 0.65
        self.b_obs_1 = 0.65

        self.a_obs_2 = 0.65
        self.b_obs_2 = 0.65

        #####################

        self.t_fin = 10 #### time horizon
        self.num = 100 #### number of steps in prediction.
        self.num_batch = 110
        self.tot_time = np.linspace(0, self.t_fin, self.num)
    
        self.x_init = 1.0
        self.y_init = 2.0
    
        self.x_fin = 0.0
        self.y_fin = 15.0

        self.x_fin_t = 0.0
        self.y_fin_t = 15

        self.i = 26
        ###########Dyanmic obstacles position
        data_obs_mat = loadmat('/home/fatemeh/test_ws/src/planner/src/obstacles_position/obstacles_dy_26.mat')
        data_obs = data_obs_mat["obs"]

        print(data_obs, "obs")

        self.x_obs_init_dy = data_obs[:,0] 
        self.y_obs_init_dy = data_obs[:,1]

        self.x_obs_1 = self.x_obs_init_dy[0]
        self.y_obs_1 = self.y_obs_init_dy[0]

        self.x_obs_2 = self.x_obs_init_dy[1]
        self.y_obs_2 = self.y_obs_init_dy[1]

        self.x_obs_3 = self.x_obs_init_dy[2]
        self.y_obs_3 = self.y_obs_init_dy[2]

        self.x_obs_4 = self.x_obs_init_dy[3]
        self.y_obs_4 = self.y_obs_init_dy[3]

        self.x_obs_5 = self.x_obs_init_dy[4]
        self.y_obs_5 = self.y_obs_init_dy[4]

        self.x_obs_6 = self.x_obs_init_dy[5]
        self.y_obs_6 = self.y_obs_init_dy[5]

        self.x_obs_7 = self.x_obs_init_dy[6]
        self.y_obs_7 = self.y_obs_init_dy[6]

        self.x_obs_8 = self.x_obs_init_dy[7]
        self.y_obs_8 = self.y_obs_init_dy[7]        
        
        self.x_obs_9 = self.x_obs_init_dy[8]
        self.y_obs_9 = self.y_obs_init_dy[8]

        self.x_obs_10 = self.x_obs_init_dy[9]
        self.y_obs_10 = self.y_obs_init_dy[9]


        self.x_best_global = jnp.linspace(self.x_init, self.x_init , 100)
        self.y_best_global = jnp.linspace(self.y_init, self.y_init, 100)
            
        self.theta_des = np.arctan2(self.y_fin-self.y_init, self.x_fin-self.x_init) 
        self.v_des = 1.0

        self.maxiter_mpc = 350
        self.mutex = Lock()
        self.updated = False
        self.updated_obs = False
        self.updated_homotopy = False
        self.num_update_state = 1
        self.num_up = 100

        self.x_waypoint = jnp.linspace(self.x_init, self.x_fin + 10.0 * jnp.cos(self.theta_des) , 1000)
        self.y_waypoint = jnp.linspace(self.y_init, self.y_fin + 10.0 * jnp.sin(self.theta_des),  1000)
        self.way_point_shape = 1000

        self.x_obs_init_1 = np.ones(800) * 100
        self.y_obs_init_1 = np.ones(800) * 100 

        self.x_obs_down = np.ones(500) * 100
        self.y_obs_down = np.ones(500) * 100

        self.x_obs_init = np.ones(500) * 100
        self.y_obs_init = np.ones(500) * 100 
      
        self.vx_obs = 0.0*jnp.ones(500)
        self.vy_obs = 0.0*jnp.ones(500)

        self.vx_obs_dy = 0.0*jnp.ones(10)
        self.vy_obs_dy = -0.1*jnp.ones(10)

        self.x_obs_visible = 10 * jnp.ones(60)
        self.y_obs_visible = 10 * jnp.ones(60)

        self.xyz = np.random.rand(800, 3)
        self.pcd =open3d.geometry.PointCloud()


    ###################################

    def convert_angle(self, angle):
        if angle >= 0 and angle <= 2*jnp.pi:
            return angle  # Angle is already in the desired range

        if angle < 0:
            angle += 2*jnp.pi  # Add 2π to negative angle

        
        return angle
    
    def convert_angle_2(self, angle):
        if angle <= 0 and angle >= -2*jnp.pi:
            return angle  # Angle is already in the desired range

        if angle < -2*jnp.pi:
            angle = angle % 2*jnp.pi  # Add 2π to negative angle
        
        return angle

    ####################################
        
    def Callback(self, msg_1, msg_2):

        self.x_init = msg_2.pose.pose.position.x 
        self.y_init = msg_2.pose.pose.position.y 

        orientation_q = msg_2.pose.pose.orientation
        orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]

        (roll, pitch, yaw) = euler_from_quaternion (orientation_list)
        self.theta_init = yaw

        counter = 0
        increment_value = 1
        for i in range (0, len(msg_1.points), increment_value):

            self.x_obs_init_1[counter] = msg_1.points[i].x 
            self.y_obs_init_1[counter] = msg_1.points[i].y 
            counter += 1

        idxes = np.argwhere((self.x_obs_init_1[:]>=30) | (self.y_obs_init_1[:]>=30))
        self.x_obs_init_1[idxes] = self.x_obs_init_1[0]
        self.y_obs_init_1[idxes] = self.y_obs_init_1[0]

        self.xyz[:,0] = self.x_obs_init_1
        self.xyz[:,1] = self.y_obs_init_1
        self.xyz[:,2] = 0.0

        self.pcd.points = open3d.utility.Vector3dVector(self.xyz)
        downpcd = self.pcd.voxel_down_sample(voxel_size = 0.16) 
        downpcd_array = np.asarray(downpcd.points)
        num_down_samples = downpcd_array[:,0].shape[0]

        self.x_obs_down[0:num_down_samples] = downpcd_array[:,0]
        self.y_obs_down[0:num_down_samples] = downpcd_array[:,1]

        self.x_obs_down[num_down_samples-1:-1] = 100
        self.y_obs_down[num_down_samples-1:-1] = 100

        self.x_obs_init = self.x_obs_down#[idx_dis[0:60]]
        self.y_obs_init = self.y_obs_down#[idx_dis[0:60]]


    def OdomCallback_1(self, msg_2):

        self.x_obs_1 = msg_2.pose.pose.position.x 
        self.y_obs_1 = msg_2.pose.pose.position.y 

    ###########################################
    def OdomCallback_2(self, msg_2):

        self.x_obs_2 = msg_2.pose.pose.position.x 
        self.y_obs_2 = msg_2.pose.pose.position.y 
    
    ####################################
    def OdomCallback_3(self, msg_2):

        self.x_obs_3 = msg_2.pose.pose.position.x 
        self.y_obs_3 = msg_2.pose.pose.position.y 
    
    ######################################
    def OdomCallback_4(self, msg_2):

        self.x_obs_4 = msg_2.pose.pose.position.x 
        self.y_obs_4 = msg_2.pose.pose.position.y 

    ###############################
    def OdomCallback_5(self, msg_2):

        self.x_obs_5 = msg_2.pose.pose.position.x 
        self.y_obs_5 = msg_2.pose.pose.position.y 
    
    ##############################
    def OdomCallback_6(self, msg_2):

        self.x_obs_6 = msg_2.pose.pose.position.x 
        self.y_obs_6 = msg_2.pose.pose.position.y 
    
    ###########################
    def OdomCallback_7(self, msg_2):

        self.x_obs_7 = msg_2.pose.pose.position.x 
        self.y_obs_7 = msg_2.pose.pose.position.y

    #################################
    def OdomCallback_8(self, msg_2):

        self.x_obs_8 = msg_2.pose.pose.position.x 
        self.y_obs_8 = msg_2.pose.pose.position.y  
    
    ##############################
    def OdomCallback_9(self, msg_2):

        self.x_obs_9 = msg_2.pose.pose.position.x 
        self.y_obs_9 = msg_2.pose.pose.position.y 

    ##############################
    def OdomCallback_10(self, msg_2):

        self.x_obs_10 = msg_2.pose.pose.position.x 
        self.y_obs_10 = msg_2.pose.pose.position.y 
        
    ######################################

    def drawVisibleObstacles(self, obs_pose, index,marker_array_points):

        marker= Marker()
        marker.header.frame_id = "map"
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

    ###############################
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

    #############################################
    
    def deleteMarkers(self,list_of_points, maker_to_be_deleted):

        marker= Marker()
        marker.header.frame_id = "world"
        marker.ns = ''
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.DELETEALL
        marker.points = list_of_points
        marker.color.r, marker.color.g, marker.color.b = (0, 1, 0)
        marker.color.a = 0.7
        marker.scale.x, marker.scale.y, marker.scale.z = (0.05, 0.05, 0.05)
        maker_to_be_deleted.markers.append(marker)
        return maker_to_be_deleted

    ########################################

    def pathPublisher(self):

        temp_traj_x = []
        temp_traj_y = []

        while (True):
            if(self.updated==True):

                self.mutex.acquire()
                

                marker_feas = Marker()
                marker_feas.id = 0
                marker_feas.header.frame_id = 'world'
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
     
        sub_PointCloud = message_filters.Subscriber("/pointcloud", PointCloud)
        sub_Odom = message_filters.Subscriber("/ground_truth/odom", Odometry)
        ts = message_filters.ApproximateTimeSynchronizer([sub_PointCloud, sub_Odom], 1,1, allow_headerless=True )
        ts.registerCallback(self.Callback)
        
        # thread_1 = threading.Thread(target=self.pathPublisher)
        # thread_2 = threading.Thread(target=self.PlotObstacles)


        rospy.Subscriber("obs_1/odom", Odometry, self.OdomCallback_1)
        rospy.Subscriber("obs_2/odom", Odometry, self.OdomCallback_2)
        rospy.Subscriber("obs_3/odom", Odometry, self.OdomCallback_3)
        rospy.Subscriber("obs_4/odom", Odometry, self.OdomCallback_4)
        rospy.Subscriber("obs_5/odom", Odometry, self.OdomCallback_5)
        rospy.Subscriber("obs_6/odom", Odometry, self.OdomCallback_6)
        rospy.Subscriber("obs_7/odom", Odometry, self.OdomCallback_7)
        rospy.Subscriber("obs_8/odom", Odometry, self.OdomCallback_8)
        rospy.Subscriber("obs_9/odom", Odometry, self.OdomCallback_9)
        rospy.Subscriber("obs_10/odom", Odometry, self.OdomCallback_10)

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


        num_obs_1 = 50
        num_obs_2 = 10

        prob = mpc_cem_dynamic.batch_crowd_nav(self.a_obs_1, self.b_obs_1, self.a_obs_2, self.b_obs_2, self.v_max, self.v_min, self.a_max, num_obs_1, num_obs_2, self.t_fin, self.num, self.num_batch, self.maxiter, self.maxiter_cem, self.weight_smoothness, self.weight_track, self.way_point_shape, self.v_des)


        key = random.PRNGKey(0)

        arc_length, arc_vec, x_diff, y_diff =prob.path_spline(self.x_waypoint, self.y_waypoint)
    
        x_best = jnp.linspace(self.x_init, self.x_init + self.v_des * self.t_fin, 100)
        y_best = jnp.linspace(self.y_init, self.y_init + self.v_des * self.t_fin, 100)

        c_x_data = []
        c_y_data = []

        x_obs_save = []
        y_obs_save = []

        total_time = []
        mpc_count = 0

        initial_state = jnp.hstack(( self.x_init, self.y_init, self.vx_init, self.vy_init, self.ax_init, self.ay_init ))

        x_guess_per, y_guess_per = prob.compute_warm_traj(initial_state, self.v_des, self.x_waypoint, self.y_waypoint, arc_vec, x_diff, y_diff)

        for i in range(0, self.maxiter_mpc):

            initial_state = jnp.hstack(( self.x_init, self.y_init, self.vx_init, self.vy_init, self.ax_init, self.ay_init ))

            lamda_x = jnp.zeros((self.num_batch, prob.nvar))
            lamda_y = jnp.zeros((self.num_batch, prob.nvar))

            start = time.time()

            x_obs_init_dy = self.x_obs_init_dy
            y_obs_init_dy = self.y_obs_init_dy

            x_obs_init = self.x_obs_init
            y_obs_init = self.y_obs_init

            vx_obs = self.vx_obs
            vy_obs = self.vy_obs

            vx_obs_dy = self.vx_obs_dy
            vy_obs_dy = self.vy_obs_dy

            x_obs_trajectory, y_obs_trajectory, x_obs_trajectory_proj, y_obs_trajectory_proj, x_obs_trajectory_dy, y_obs_trajectory_dy = prob.compute_obs_traj_prediction( jnp.asarray(x_obs_init_dy).flatten(), jnp.asarray(y_obs_init_dy).flatten(), vx_obs_dy, vy_obs_dy, jnp.asarray(x_obs_init).flatten(), jnp.asarray(y_obs_init).flatten(), vx_obs, vy_obs, initial_state[0], initial_state[1] ) ####### obstacle trajectory prediction

            sol_x_bar, sol_y_bar, x_guess, y_guess,  xdot_guess, ydot_guess, xddot_guess, yddot_guess,c_mean, c_cov, x_fin, y_fin = prob.compute_traj_guess( initial_state, x_obs_trajectory, y_obs_trajectory, x_obs_trajectory_dy, y_obs_trajectory_dy , self.v_des, self.x_waypoint, self.y_waypoint, arc_vec, x_guess_per, y_guess_per, x_diff, y_diff)
            
            self.x_fin = x_fin
            self.y_fin = y_fin  

            # pause_physics_service()

            x_ellite, y_ellite, x, y, c_x_best, c_y_best, x_best, y_best, xdot_best, ydot_best, x_guess_per , y_guess_per, xddot_best, yddot_best = prob.compute_cem(key, initial_state, self.x_fin, self.y_fin, lamda_x, lamda_y, x_obs_trajectory, y_obs_trajectory, x_obs_trajectory_proj, y_obs_trajectory_proj, x_obs_trajectory_dy, y_obs_trajectory_dy , sol_x_bar, sol_y_bar, x_guess, y_guess,  xdot_guess, ydot_guess, xddot_guess, yddot_guess, self.x_waypoint,  self.y_waypoint, arc_vec, c_mean, c_cov )
            
            delta_t = jnp.array(time.time() - start)
    
            vx_control, vy_control, ax_control, ay_control, norm_v_t, angle_v_t, v_t_control, omega_control = prob.compute_controls(c_x_best, c_y_best, delta_t, self.convert_angle(self.theta_init))

            zeta = self.convert_angle(self.theta_init) - self.convert_angle(angle_v_t)
            v_t_control = norm_v_t * jnp.cos(zeta)
            omega_control = - zeta / (6*self.t_fin*0.01)

            self.x_best_global = x_best
            self.y_best_global = y_best

            self.x_obs_visible = x_obs_trajectory[:,0]
            self.y_obs_visible = y_obs_trajectory[:,0]

            self.updated_obs = True
            self.updated = True
            self.updated_homotopy = True
            
            #self.mutex.release()

            ###########################
            self.vx_init = vx_control#[j]
            self.vy_init = vy_control#[j]

            self.ax_init = ax_control#[j]
            self.ay_init = ay_control#[j]

            vel_cmd.linear.x = 1*v_t_control#v_t
            #vel_cmd.linear.y = self.vy_init
            vel_cmd.angular.z = omega_control#, 0.0, 2*np.pi)#1*omega_control#omega

            if i <=3 :
                vel_cmd.linear.x = 0.0*v_t_control
                vel_cmd.angular.z = omega_control#, 0.0, 2*np.pi)
                self.cmd_vel_pub.publish(vel_cmd)
                


            if i>3:
                # unpause_physics_service()


                vel_cmd_1.linear.x = self.vx_obs[0]
                vel_cmd_2.linear.x = self.vx_obs[1]
                vel_cmd_3.linear.x = self.vx_obs[2]
                vel_cmd_4.linear.x = self.vx_obs[3]
                vel_cmd_5.linear.x = self.vx_obs[4]
                vel_cmd_6.linear.x = self.vx_obs[5]
                vel_cmd_7.linear.x = self.vx_obs[6]
                vel_cmd_8.linear.x = self.vx_obs[7]
                vel_cmd_9.linear.x = self.vx_obs[8]
                vel_cmd_10.linear.x = self.vx_obs[9]

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
               
                self.cmd_vel_obs_1_pub.publish(vel_cmd_1) 
                self.cmd_vel_obs_2_pub.publish(vel_cmd_2)
                self.cmd_vel_obs_3_pub.publish(vel_cmd_3)
                self.cmd_vel_obs_4_pub.publish(vel_cmd_4)
                self.cmd_vel_obs_5_pub.publish(vel_cmd_5)
                self.cmd_vel_obs_6_pub.publish(vel_cmd_6)
                self.cmd_vel_obs_7_pub.publish(vel_cmd_7)
                self.cmd_vel_obs_8_pub.publish(vel_cmd_8)
                self.cmd_vel_obs_9_pub.publish(vel_cmd_9)
                self.cmd_vel_obs_10_pub.publish(vel_cmd_10)
                
                self.cmd_vel_pub.publish(vel_cmd)
                rospy.sleep(0.012)
                total_time.append(time.time()- start)

                self.x_obs_init_dy = jnp.hstack(( self.x_obs_1, self.x_obs_2, self.x_obs_3, self.x_obs_4, self.x_obs_5, self.x_obs_6, self.x_obs_7, self.x_obs_8, self.x_obs_9, self.x_obs_10 ))
                self.y_obs_init_dy = jnp.hstack(( self.y_obs_1, self.y_obs_2, self.y_obs_3, self.y_obs_4, self.y_obs_5, self.y_obs_6, self.y_obs_7, self.y_obs_8, self.y_obs_9, self.y_obs_10 ))

            print (time.time() - start, "time_2")

            x_obs_save.append(x_obs_trajectory[:,0])
            y_obs_save.append(y_obs_trajectory[:,0])

            c_x_data.append(c_x_best)
            c_y_data.append(c_y_best)
     
            dist_final_point = np.sqrt((self.x_init - self.x_fin_t)**2 + (self.y_init - self.y_fin_t)**2)
            mpc_count = mpc_count +1

            if (dist_final_point <= 1):
                vel_cmd.linear.x = 0.0
                vel_cmd.linear.y = 0.0
                self.cmd_vel_pub.publish(vel_cmd)
                print("Distance is satisfied")
                break
                

        print("Stop mpc")


        c_x_data = np.array(c_x_data)
        c_y_data = np.array(c_y_data)

        x_obs_save = np.array(x_obs_save)
        y_obs_save = np.array(y_obs_save)

        total_time = np.array(total_time)

        cx_data =  c_x_data.reshape(mpc_count , 11)
        cy_data =  c_y_data.reshape(mpc_count , 11)

        world_num = self.i
        np.save("cx_data_dy_"+str(world_num)+".npy", cx_data)
        np.save("cy_data_dy_"+str(world_num)+".npy", cy_data)

        np.save("total_time_dy_"+str(world_num)+".npy", total_time)

        np.save("x_obs_save_dy_"+str(world_num)+".npy", x_obs_save)
        np.save("y_obs_save_dy_"+str(world_num)+".npy", y_obs_save)

        print(np.sum(total_time, axis = 0), "total_time")
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
    # motion_planning.pathPublisher()
    # motion_planning.PlotObstacles()

    




    


    
    




    


    

















	







