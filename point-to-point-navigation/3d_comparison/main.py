
#!/usr/bin/env python3




import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
import jax
import bernstein_coeff_order10_arbitinterval

import CEM
import matplotlib.pyplot as plt
from scipy.io import loadmat
import scipy
import time
from jax import vmap, random
from scipy.interpolate import UnivariateSpline
import bernstein_coeff_order10_arbitinterval
import Optim_part

import rockit
from rockit import *
from casadi import vertcat, sumsqr, horzcat
from scipy.io import loadmat
import scipy
# import random as rnd
# from pylab import *
import casadi as cs
import string
import casadi_solver_2

import expert_2
import Optim_part
from mayavi import mlab


class planning_traj():

    def __init__(self):

        #### General Parameters
        self.vx_init = 0.05
        self.vy_init = 0.1
        self.vz_init = 0.001

        self.ax_init = 0.0
        self.ay_init = 0.0
        self.az_init = 0.0

        self.vx_fin = 0.0
        self.vy_fin = 0.0
        self.vz_fin = 0.0

        self.ax_fin = 0.0
        self.ay_fin = 0.0
        self.az_fin = 0.0

        self.v_max = 2.0
        self.v_min = 0.02
        self.a_max = 2.5
        
        self.maxiter = 1
        self.maxiter_cem = 13
        self.weight_track = 0.001
        self.weight_smoothness = 1
        self.max_proj = 15

        self.a_obs_1 = 0.68
        self.b_obs_1 = 0.68
        self.c_obs_1 = 0.68

        self.a_obs_2 = 4.4
        self.b_obs_2 = 4.4
        self.c_obs_2 = 0.1

        self.a_obs_3 = 0.6
        self.b_obs_3 = 0.6
        self.c_obs_3 = 3.0


        self.x_max = 6.2 
        self.y_max = 6.2
        self.z_max = 2.8 

        self.x_min = 0.0
        self.y_min = 0.0
        self.z_min = 0.0


        #####################

        self.t_fin = 10 #### time horizon
        self.num = 100 #### number of steps in prediction.
        self.num_batch = 110
        self.tot_time = np.linspace(0, self.t_fin, self.num)


        self.x_init_array = np.array([ 1.2, 1.2, 1.2, 0.6, 0.6, 0.6, 0.4, 0.4, 0.4, 5.5, 5.5, 5.5, 
                                      0.8, 0.8, 0.8, 1, 1.0, 1.0, 1.3, 1.3, 1.3, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,  
                                       1.3, 1.3, 1.3, 0.6, 0.6, 0.6, 5.8,  5.8, 5.8, 4.0, 4.0, 4.0,
                                        4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 
                                         1.4, 1.4, 1.4, 1.4, 1.4, 1.4, 0.6, 0.6, 0.6, 1.0, 1.0, 1.0,
                                        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.5, 1.5, 1.5, 0.8, 0.8, 0.8, 2.0, 2.0, 2.0, 
                                         0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 
                                           1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7,   ])
        
        self.y_init_array = np.array([ 1.7, 1.7, 1.7, 4.7, 4.7, 4.7, 3.4, 3.4, 3.4, 0.0, 0.0, 0.0, 
                                       0.8, 0.8, 0.8, 0.4, 0.4, 0.4, 1.2, 1.2, 1.2, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
                                        1.3, 1.3, 1.3, 0.6, 0.6, 0.6, 1.0 , 1.0, 1.0, 0.95, 0.95, 0.95, 
                                        0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 
                                        0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1.6, 1.6, 1.6, 3.0, 3.0, 3.0, 
                                        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 3.0, 3.0, 3.0, 2.0, 2.0, 2.0, 0.6, 0.6, 0.6, 
                                        2.0, 2.0, 2.0, 3.2, 3.2, 3.2, 3.2, 3.2, 3.2, 3.2, 3.2, 3.2, 
                                        1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5,  3.3, 3.3, 3.3, 
                                     ])
        
        self.z_init_array = np.array([ 0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                      1.2,  1.2, 1.2, 0.3, 0.3, 0.3, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 
                                      1.2, 1.2, 1.2,  0.6, 0.6, 0.6, 1.5, 1.5, 1.5, 1.0, 1.0, 1.0,  
                                       1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 
                                        0.5, 0.5, 0.5, 0.5,  0.5, 0.5, 0.8, 0.8, 0.8, 1.5, 1.5, 1.5, 
                                          0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.9, 0.9, 0.9,  1.6, 1.6, 1.6, 0.7, 0.7, 0.7, 
                                        1.0, 1.0, 1.0, 0.7, 0.7, 0.7, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 
                                         1.3, 1.3, 1.3, 1.3, 1.3, 1.3, 1.3, 1.3, 1.3, 1.3, 1.3, 1.3, 1.3, 1.3, 1.3, 1.0, 1.0, 1.0, 
                                           ])
        
        self.x_fin_array = np.array([ 5.7, 5.7, 5.7, 5.7, 5.7, 5.7, 5.5, 5.5, 5.5, 2.0, 2.0, 2.0, 
                                     5.7, 5.7, 5.7, 5.9, 5.9, 5.9, 5.8, 5.8, 5.8, 1.0, 1.0, 1.0, 5.0 , 5.0, 5.0, 
                                      5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 0.5, 0.5, 0.5, 4.0, 4.0, 4.0, 
                                       3.2, 3.2, 3.2, 4.7, 4.3, 4.3, 0.8, 0.8, 0.8, 
                                        6.0, 6.0, 6.0, 2.8, 2.8, 2.8, 6.1, 6.1, 6.1, 6.1, 6.1, 6.1, 
                                          6.0, 6.0, 6.0, 6.2, 6.2, 6.2, 6.2, 6.2, 6.2, 6.2,  6.2, 6.2, 5.0, 5.0, 5.0, 
                                           6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0,  
                                            3.7, 3.7, 3.7, 4.3, 4.3, 4.3, 5.7, 5.7, 5.7, 5.7, 5.7, 5.7, 6.1, 6.1, 6.1, 6.1, 6.1, 6.1,  ])
        
        self.y_fin_array = np.array([ 5.7, 5.7, 5.7, 0.3, 0.3, 0.3, 3.8, 3.8, 3.8, 5.5, 5.5, 5.5, 
                                     5.0, 5.0, 5.0, 5.6 , 5.6, 5.6, 3.1, 3.1, 3.1, 5.5, 5.5, 5.5, 2.5,  2.5, 2.5,
                                       5.0, 5.0, 5.0,  5.0, 5.0, 5.0, 3.5, 3.5, 3.5, 5.2, 5.2, 5.2,  
                                       5.3, 5.3, 5.3, 5.3, 5.3, 5.3, 5.5,  5.5, 5.5,
                                         4.3, 4.3, 4.3, 4.5,  4.5, 4.5, 2.9, 2.9, 2.9, 2.9, 2.9, 2.9, 
                                           6.0, 6.0, 6.0, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 5.0, 5.0, 5.0, 
                                             5.0, 5.0, 5.0, 4.0, 4.0, 4.0, 1.2, 1.2, 1.2, 2.2, 2.2, 2.2, 
                                              5.5, 5.5, 5.5, 5.0, 5.0, 5.0, 3.8, 3.8, 3.8, 3.8, 3.8, 3.8, 1.8, 1.8, 1.8, 1.8, 1.8, 1.8,   ])
        
        self.z_fin_array = np.array([ 0.9, 0.9, 0.9, 0.5, 0.5, 0.5 , 0.5, 0.5, 0.5,  1.5 , 1.5, 1.5,
                                     0.3, 0.3, 0.3, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 1.5, 1.5, 1.5,
                                      1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 , 1.0,  0.85, 0.85, 0.85,
                                       0.8, 0.8, 0.8, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0,
                                         1.7, 1.7, 1.7, 1.25, 1.25, 1.25, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2,
                                           2.0,2.0, 2.0, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 2.0, 2.0, 2.0, 
                                              1.0, 1.0, 1.0, 0.7, 0.7, 0.7, 1.5, 1.5, 1.5, 1.6, 1.6, 1.6, 
                                                1.5, 1.5, 1.5, 1.5, 1.5, 1.5,  1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,   ])


        # ############### k = 0 to 11 obstacles, k=12 to 24 obstacles_1, k = 25 to 35 obstacles_2, k = 36 to 47 obstacle_3, k = 48 to 60 obstacles_4, 60 to 72 obstacles_5, 72to 74_obstacles_6, 75 obstacles_7
        ## 53->54 
        k = 104



        self.num_test = k//3
        self.num_repeat =k%3

        self.x_init = self.x_init_array[k]
        self.y_init = self.y_init_array[k]
        self.z_init = self.z_init_array[k]

        self.x_fin = self.x_fin_array[k]
        self.y_fin = self.y_fin_array[k]
        self.z_fin = self.z_fin_array[k]

        print( self.x_init,  self.y_init,  self.z_init )

        data_obs_mat = loadmat('/home/fatemeh//Priest/PRIEST/point-to-point-navigation/3d_comparison/obs_positions/obstacles_19.mat')
        data_obs = data_obs_mat["obs"]

        self.x_obs_init_1 = data_obs[:,0] 
        self.y_obs_init_1 = data_obs[:,1]
        self.z_obs_init_1 = data_obs[:,2]

        self.x_obs_init_2 = jnp.hstack(( 3.5, 3.5 ))
        self.y_obs_init_2 = jnp.hstack(( 3.5, 3.5 ))
        self.z_obs_init_2 = jnp.hstack(( 0.0, 3.0 ))

        self.x_obs_init_3 = jnp.hstack(( jnp.linspace(0.0, 7.0, 8), jnp.linspace(0.0, 7.0, 8), 0.0*jnp.ones(8), 7.0*jnp.ones(8 ) ))
        self.y_obs_init_3 = jnp.hstack(( 0.0*jnp.ones(8 ),  7.0*jnp.ones(8 ), jnp.linspace(0.0, 7.0, 8), jnp.linspace(0.0, 7.0, 8) ))
        self.z_obs_init_3 =  1.7*jnp.ones(32 )

        self.x_best_global = jnp.linspace(self.x_init, self.x_init , 100)
        self.y_best_global = jnp.linspace(self.y_init, self.y_init, 100)
        self.z_best_global = jnp.linspace(self.z_init, self.z_init, 100)
            
        self.theta_des = np.arctan2(self.y_fin-self.y_init, self.x_fin-self.x_init) 

        self.theta_des = np.arctan2(self.y_fin-self.y_init, self.x_fin-self.x_init)
        self.phi_des = np.pi/4 
        self.v_des = 2.0

        self.maxiter_mpc = 1
        self.num_batch = 110
        self.num_goal = int(self.num_batch*0.6)
        self.num_up = 100
        self.nvar = 11
        self.way_point_shape = 1000

        self.vx_obs_1 = np.zeros(25)
        self.vy_obs_1 = np.zeros(25)
        self.vz_obs_1 = np.zeros(25)

        self.vx_obs_2 = np.zeros(2)
        self.vy_obs_2 = np.zeros(2)
        self.vz_obs_2 = np.zeros(2)

        self.vx_obs_3 = np.zeros(32)
        self.vy_obs_3 = np.zeros(32)
        self.vz_obs_3 = np.zeros(32)

        self.vx_des = self.v_des*jnp.cos(self.theta_des) * jnp.sin(self.phi_des)
        self.vy_des = self.v_des*jnp.sin(self.theta_des) * jnp.sin(self.phi_des)
        self.vz_des = self.v_des** jnp.cos(self.phi_des)

        self.x_des_traj = self.x_init + self.vx_des*self.tot_time
        self.y_des_traj = self.y_init + self.vy_des*self.tot_time
        self.z_des_traj = self.z_init + self.vz_des*self.tot_time

        self.x_des_init = self.x_des_traj[0]
        self.y_des_init = self.y_des_traj[0]
        self.z_des_init = self.y_des_traj[0]

        self.x_waypoint = jnp.linspace(self.x_init, self.x_fin + 0.0 * jnp.cos(self.theta_des) , 100)
        self.y_waypoint = jnp.linspace(self.y_init, self.y_fin + 0.0 * jnp.sin(self.theta_des),  100)
        self.z_waypoint = jnp.linspace(self.z_init, self.z_fin + 0.0 ,  100)
        


       
    def compute_min_dist(self, x, y, z, x_obs_1, y_obs_1, z_obs_1, x_obs_2, y_obs_2, z_obs_2, x_obs_3, y_obs_3, z_obs_3):

		##################################### computing minimum distance collision check
        dist_robot = np.zeros(np.shape(x)[0])

        for i in range(0, np.shape(x)[0]):

            dist_x_1 = (x[i]-x_obs_1[:])
            dist_y_1 = (y[i]-y_obs_1[:])
            dist_z_1 = (z[i]-z_obs_1[:])

            dist_1 =  np.max(-(dist_x_1**2/((self.a_obs_1-0.03)**2))-(dist_y_1**2/((self.b_obs_1-0.03)**2))-((dist_z_1**2)/((self.c_obs_1-0.03)**2))+1)
            dist = dist_1
            dist_robot[i] = dist
        
        return dist_robot


    def boundaries(self, x, y, z):

        x_b_up = np.max(x[:] - self.x_max)
        y_b_up = np.max(y[:] - self.y_max)
        z_b_up = np.max(z[:] - self.z_max)
        
        b_up = np.max(np.vstack((x_b_up, y_b_up, z_b_up)) )

        x_b_l = np.min(x - self.x_min)
        y_b_l = np.min(y - self.y_min)
        z_b_l = np.min(z - self.z_min)
        b_l = np.min(np.vstack(( x_b_l, y_b_l, z_b_l)) )

        return b_up, b_l

    def planner(self, ):

        num_obs_1 = 25
        num_obs_2 = 2
        num_obs_3 = 32


        x_guess_naive = np.linspace(self.x_init, self.x_fin, self.num)
        y_guess_naive = np.linspace(self.y_init, self.y_fin, self.num)
        z_guess_naive = np.linspace(self.z_init, self.z_fin, self.num)

        prob_solver = casadi_solver_2.Solvers_func(self.t_fin, self.a_max, self.v_max, self.a_obs_1, self.b_obs_1, self.c_obs_1, self.a_obs_2, self.b_obs_2, self.c_obs_2, self.a_obs_3, self.b_obs_3, self.c_obs_3, self.num, num_obs_1, num_obs_2, num_obs_3)
        Prob_our = expert_2.batch_crowd_nav(self.a_obs_1, self.b_obs_1, self.c_obs_1, self.a_obs_2, self.b_obs_2, self.c_obs_2, self.a_obs_3, self.b_obs_3, self.c_obs_3, self.v_max, self.v_min, self.a_max, num_obs_1, num_obs_2, num_obs_3, self.t_fin, self.num, self.num_batch, self.maxiter, self.maxiter_cem, self.weight_smoothness, self.weight_track, self.way_point_shape, self.v_des)
        prob_CEM = CEM.CEM_optimizer( self.a_obs_1, self.b_obs_1, self.c_obs_1, self.a_obs_2, self.b_obs_2, self.c_obs_2, self.a_obs_3, self.b_obs_3, self.c_obs_3, self.t_fin, self.nvar, self.num, self.num_batch, self.maxiter_cem, self.weight_smoothness, num_obs_1, num_obs_2, num_obs_3, self.way_point_shape, self.v_max, self.a_max)
        Prob_proj_cem = Optim_part.Optim(self.way_point_shape, self.max_proj, self.num_goal, self.v_max, self.a_max, self.weight_smoothness,  self.nvar, self.num, num_obs_1, self.num_batch, self.a_obs_1, self.b_obs_1, self.c_obs_1, self.t_fin, self.vx_obs_1, self.vy_obs_1, self.vz_obs_1, self.vx_des, self.vy_des, self.vz_des )       
        ########################################################    
        key = random.PRNGKey(0)

        arc_length, arc_vec, x_diff, y_diff, z_diff =  Prob_our.path_spline(self.x_waypoint, self.y_waypoint, self.z_waypoint)
    
        # ###########
        lamda_x_our_method = jnp.zeros((self.num_batch, Prob_our.nvar))
        lamda_y_our_method = jnp.zeros((self.num_batch, Prob_our.nvar))
        lamda_z_our_method = jnp.zeros((self.num_batch, Prob_our.nvar))

        lamda_x_cem = jnp.zeros((self.num_batch, prob_CEM.nvar))
        lamda_y_cem = jnp.zeros((self.num_batch, prob_CEM.nvar))
        lamda_z_cem = jnp.zeros((self.num_batch, prob_CEM.nvar))


        initial_state = jnp.hstack(( self.x_init, self.y_init, self.z_init, self.vx_init, self.vy_init, self.vz_init, self.ax_init, self.ay_init, self.az_init ))

        x_obs_trajectory_1, y_obs_trajectory_1, z_obs_trajectory_1, x_obs_trajectory_2, y_obs_trajectory_2, z_obs_trajectory_2, x_obs_trajectory_3, y_obs_trajectory_3, z_obs_trajectory_3 = Prob_our.compute_obs_traj_prediction( jnp.asarray(self.x_obs_init_1).flatten(), jnp.asarray(self.y_obs_init_1).flatten(), jnp.asarray(self.z_obs_init_1).flatten(), jnp.asarray(self.x_obs_init_2).flatten(), jnp.asarray(self.y_obs_init_2).flatten(), jnp.asarray(self.z_obs_init_2).flatten(), jnp.asarray(self.x_obs_init_3).flatten(), jnp.asarray(self.y_obs_init_3).flatten(), jnp.asarray(self.z_obs_init_3).flatten(), self.vx_obs_1, self.vy_obs_1, self.vz_obs_1, self.vx_obs_2, self.vy_obs_2, self.vz_obs_2, self.vx_obs_3, self.vy_obs_3, self.vz_obs_3 )
      
        sol_x_bar, sol_y_bar, sol_z_bar, x_guess, y_guess, z_guess, xdot_guess, ydot_guess, zdot_guess, xddot_guess, yddot_guess, zddot_guess, c_mean, c_cov = Prob_our.compute_traj_guess(key,  initial_state, x_obs_trajectory_1, y_obs_trajectory_1, z_obs_trajectory_1, x_obs_trajectory_2, y_obs_trajectory_2, z_obs_trajectory_2, x_obs_trajectory_3, y_obs_trajectory_3, z_obs_trajectory_3, self.x_fin , self.y_fin, self.z_fin)
        for j in range(4):

            lamda_x_our_method = jnp.zeros((self.num_batch, Prob_our.nvar))
            lamda_y_our_method = jnp.zeros((self.num_batch, Prob_our.nvar))
            lamda_z_our_method = jnp.zeros((self.num_batch, Prob_our.nvar))

            lamda_x = jnp.zeros((self.num_goal, Prob_our.nvar))
            lamda_y = jnp.zeros((self.num_goal, Prob_our.nvar))
            lamda_z = jnp.zeros((self.num_goal, Prob_our.nvar))


            ################### Our method
            start = time.time()
            x_best_our_method, y_best_our_method, z_best_our_method = Prob_our.compute_cem(key, initial_state, self.x_fin, self.y_fin, self.z_fin, lamda_x_our_method, lamda_y_our_method,lamda_z_our_method, x_obs_trajectory_1, y_obs_trajectory_1, z_obs_trajectory_1, x_obs_trajectory_2, y_obs_trajectory_2, z_obs_trajectory_2, x_obs_trajectory_3, y_obs_trajectory_3, z_obs_trajectory_3, sol_x_bar, sol_y_bar, sol_z_bar, x_guess, y_guess, z_guess,  xdot_guess, ydot_guess, zdot_guess, xddot_guess, yddot_guess, zddot_guess, self.x_waypoint,  self.y_waypoint, self.z_waypoint, arc_vec, c_mean, c_cov )
            
            Time_our = time.time()-start
            
            ################## Previous method
            #######CEM_Base

            start = time.time()
            x_initial_compare, y_initial_compare, z_initial_compare, x_samples_cem, y_samples_cem, z_samples_cem, xdot_samples_cem, ydot_samples_cem, zdot_samples_cem, xddot_samples_cem, yddot_samples_cem, zddot_samples_cem = prob_CEM.compute_cem_part(   initial_state,   x_obs_trajectory_1, y_obs_trajectory_1, z_obs_trajectory_1, x_obs_trajectory_2, y_obs_trajectory_2, z_obs_trajectory_2, x_obs_trajectory_3, y_obs_trajectory_3, z_obs_trajectory_3, self.x_fin, self.y_fin, self.z_fin, sol_x_bar, sol_y_bar, sol_z_bar, c_mean, c_cov, x_guess, y_guess, z_guess, arc_vec )

            Time_CEM = time.time()-start

            #################3CEM +optimzer
            start = time.time()
            sol_x_cem_proj, sol_y_cem_proj, sol_z_cem_proj, x_cem_proj, y_cem_proj, z_cem_proj, x, y, z = Prob_proj_cem.SolveOpt1(self.x_waypoint, self.y_waypoint, self.z_waypoint, arc_vec, x_samples_cem, y_samples_cem, z_samples_cem, xdot_samples_cem, ydot_samples_cem, zdot_samples_cem, xddot_samples_cem, yddot_samples_cem, zddot_samples_cem, x_obs_trajectory_1, y_obs_trajectory_1, z_obs_trajectory_1, self.x_des_init, self.y_des_init, self.z_des_init,initial_state, self.x_fin, self.y_fin, self.z_fin, lamda_x, lamda_y, lamda_z)
            
            Time_per = time.time()-start 
                                

        print("rockit_naive")
        sol_rockit, x_rockit_naive, y_rockit_naive, z_rockit_naive = prob_solver.SolveRockit(initial_state, self.x_fin, self.y_fin, self.z_fin, x_guess_naive, y_guess_naive, z_guess_naive, self.x_obs_init_1, self.y_obs_init_1, self.z_obs_init_1, self.x_obs_init_2, self.y_obs_init_2, self.z_obs_init_2, self.x_obs_init_3, self.y_obs_init_3, self.z_obs_init_3)
                                                                            
        print("fatrop_naive")
        sol_fatrop, x_fatrop_naive, y_fatrop_naive, z_fatrop_naive = prob_solver.SolveFatrop(initial_state, self.x_fin, self.y_fin, self.z_fin, x_guess_naive, y_guess_naive,  z_guess_naive, self.x_obs_init_1, self.y_obs_init_1, self.z_obs_init_1, self.x_obs_init_2, self.y_obs_init_2, self.z_obs_init_2,  self.x_obs_init_3, self.y_obs_init_3, self.z_obs_init_3)

        print("rockit")
        sol_rockit, x_rockit, y_rockit, z_rockit = prob_solver.SolveRockit(initial_state, self.x_fin, self.y_fin, self.z_fin, x_initial_compare, y_initial_compare, z_initial_compare, self.x_obs_init_1, self.y_obs_init_1, self.z_obs_init_1, self.x_obs_init_2, self.y_obs_init_2, self.z_obs_init_2,  self.x_obs_init_3, self.y_obs_init_3, self.z_obs_init_3)
        
        print("fatrop")

        sol_fatrop, x_fatrop, y_fatrop, z_fatrop = prob_solver.SolveFatrop(initial_state, self.x_fin, self.y_fin, self.z_fin, x_initial_compare, y_initial_compare, z_initial_compare, self.x_obs_init_1, self.y_obs_init_1, self.z_obs_init_1, self.x_obs_init_2, self.y_obs_init_2, self.z_obs_init_2,  self.x_obs_init_3, self.y_obs_init_3, self.z_obs_init_3 )
        
        print(Time_CEM, "CEM_time")
        print(Time_our, "our")
        print(Time_per, "Proj")
        x_rockit_naive = np.array(x_rockit_naive)
        y_rockit_naive = np.array(y_rockit_naive)
        z_rockit_naive = np.array(z_rockit_naive)

        x_rockit = np.array(x_rockit)
        y_rockit = np.array(y_rockit)
        z_rockit = np.array(z_rockit)

        x_fatrop_naive = np.array(x_fatrop_naive)
        y_fatrop_naive = np.array(y_fatrop_naive)
        z_fatrop_naive = np.array(z_fatrop_naive)

        x_fatrop = np.array(x_fatrop)
        y_fatrop = np.array(y_fatrop)
        z_fatrop = np.array(z_fatrop)

        x_our = np.array(x_best_our_method)
        y_our = np.array(y_best_our_method)
        z_our = np.array(z_best_our_method)

        x_cem = np.array(x_initial_compare)
        y_cem = np.array(y_initial_compare)
        z_cem = np.array(z_initial_compare)

        x_cem_proj = np.array(x_cem_proj)
        y_cem_proj = np.array(y_cem_proj)
        z_cem_proj = np.array(z_cem_proj)

        dist_rockit_naive = self.compute_min_dist( x_rockit_naive, y_rockit_naive, z_rockit_naive, self.x_obs_init_1, self.y_obs_init_1, self.z_obs_init_1, self.x_obs_init_2, self.y_obs_init_2, self.z_obs_init_2,  self.x_obs_init_3, self.y_obs_init_3, self.z_obs_init_3)
        dist_rockit = self.compute_min_dist( x_rockit, y_rockit, z_rockit, self.x_obs_init_1, self.y_obs_init_1, self.z_obs_init_1, self.x_obs_init_2, self.y_obs_init_2, self.z_obs_init_2,  self.x_obs_init_3, self.y_obs_init_3, self.z_obs_init_3)
        
        dist_fatrop_naive = self.compute_min_dist( x_fatrop_naive, y_fatrop_naive, z_fatrop_naive, self.x_obs_init_1, self.y_obs_init_1, self.z_obs_init_1, self.x_obs_init_2, self.y_obs_init_2, self.z_obs_init_2,  self.x_obs_init_3, self.y_obs_init_3, self.z_obs_init_3)
        dist_fatrop = self.compute_min_dist( x_fatrop, y_fatrop, z_fatrop_naive, self.x_obs_init_1, self.y_obs_init_1, self.z_obs_init_1, self.x_obs_init_2, self.y_obs_init_2, self.z_obs_init_2,  self.x_obs_init_3, self.y_obs_init_3, self.z_obs_init_3)

        dist_cem = self.compute_min_dist( x_cem, y_cem, z_cem, self.x_obs_init_1, self.y_obs_init_1, self.z_obs_init_1, self.x_obs_init_2, self.y_obs_init_2, self.z_obs_init_2,  self.x_obs_init_3, self.y_obs_init_3, self.z_obs_init_3)
        dist_our = self.compute_min_dist( x_our, y_our, z_our, self.x_obs_init_1, self.y_obs_init_1, self.z_obs_init_1, self.x_obs_init_2, self.y_obs_init_2, self.z_obs_init_2,  self.x_obs_init_3, self.y_obs_init_3, self.z_obs_init_3)
        dist_cem_proj = self.compute_min_dist( x_cem_proj, y_cem_proj, z_cem_proj, self.x_obs_init_1, self.y_obs_init_1, self.z_obs_init_1, self.x_obs_init_2, self.y_obs_init_2, self.z_obs_init_2,  self.x_obs_init_3, self.y_obs_init_3, self.z_obs_init_3)


        rockit_naive_up , rockit_naive_l = self.boundaries( x_rockit_naive, y_rockit_naive, z_rockit_naive,) 
        fatrop_naive_up , fatrop_naive_l = self.boundaries( x_fatrop_naive, y_fatrop_naive, z_fatrop_naive,) 
        rockit_up , rockit_l = self.boundaries( x_rockit, y_rockit, z_rockit,) 
        fatrop_up , fatrop_l = self.boundaries( x_fatrop, y_fatrop, z_fatrop,) 
        our_up , our_l = self.boundaries( x_our, y_our, z_our,) 
        cem_up , cem_l = self.boundaries( x_cem, y_cem, z_cem,) 
        cem_proj_up , cem_proj_l = self.boundaries( x_cem_proj, y_cem_proj, z_cem_proj,) 
        # #################################################
        Min_allowed_dist = 0.0
        

        if np.max(dist_rockit_naive) > Min_allowed_dist :
            print(np.max(dist_rockit_naive), "Rockit naive is unsucessful", )
        else: 
            print(np.max(dist_rockit_naive), "Rockit naive is ok", rockit_naive_up, rockit_naive_l )
        
        if np.max(dist_rockit) > Min_allowed_dist:
            print(np.max(dist_rockit), "Rockit is unsucessful")
        else: 
            print(np.max(dist_rockit), "Rockit is ok", rockit_up, rockit_l)
        
        if np.max(dist_fatrop_naive) > Min_allowed_dist:
            print(np.max(dist_fatrop_naive), "fatrop naive is unsucessful")
        else: 
            print(np.max(dist_fatrop_naive), "fatrop naive is ok", fatrop_naive_up, fatrop_naive_l)
        
        if np.max(dist_fatrop) > Min_allowed_dist:
            print(np.max(dist_fatrop), "fatrop is unsucessful")
        else: 
            print(np.max(dist_fatrop), "fatrop is ok", fatrop_up, fatrop_l)

        if np.max(dist_cem) > Min_allowed_dist:
            print(np.max(dist_cem), "cem is unsucessful")
        else: 
            print(np.max(dist_cem), "cem is ok", cem_up, cem_l)

        if np.max(dist_our) > Min_allowed_dist:
            print(np.max(dist_our), "our method is unsucessful")
        else: 
            print(np.max(dist_our), "our method is ok", our_up, our_l)

        if np.max(dist_cem_proj) > Min_allowed_dist:
            print(np.max(dist_cem_proj), "CEM Proj is unsucessful")
        else: 
            print(np.max(dist_cem_proj), "CEM Proj is ok", cem_proj_up, cem_proj_l)


        ts = np.linspace(0,2*np.pi, self.num)
        ts2 = np.linspace(0,1*np.pi, self.num)
        # print(Time_our, Time_CEM, Time_per+ Time_CEM, "time_our", "time_cem", "time_perv")
        print(self.x_init, self.y_init, self.z_init, self.x_fin, self.y_fin, self.z_fin)
        

        # np.save("x_rockit_naive_n"+str(self.num_test)+"n_"+str(self.num_repeat)+".npy", x_rockit_naive)
        # np.save("y_rockit_naive_n"+str(self.num_test)+"n_"+str(self.num_repeat)+".npy", y_rockit_naive)
        # np.save("z_rockit_naive_n"+str(self.num_test)+"n_"+str(self.num_repeat)+".npy", z_rockit_naive)
        
        # np.save("x_rockit_n"+str(self.num_test)+"n_"+str(self.num_repeat)+".npy", x_rockit)
        # np.save("y_rockit_n"+str(self.num_test)+"n_"+str(self.num_repeat)+".npy", y_rockit)
        # np.save("z_rockit_n"+str(self.num_test)+"n_"+str(self.num_repeat)+".npy", z_rockit)

        # np.save("x_fatrop_naive_n"+str(self.num_test)+"n_"+str(self.num_repeat)+".npy", x_fatrop_naive)
        # np.save("y_fatrop_naive_n"+str(self.num_test)+"n_"+str(self.num_repeat)+".npy", y_fatrop_naive)
        # np.save("z_fatrop_naive_n"+str(self.num_test)+"n_"+str(self.num_repeat)+".npy", z_fatrop_naive)

        # np.save("x_fatrop_n"+str(self.num_test)+"n_"+str(self.num_repeat)+".npy", x_fatrop)
        # np.save("y_fatrop_n"+str(self.num_test)+"n_"+str(self.num_repeat)+".npy", y_fatrop)
        # np.save("z_fatrop_n"+str(self.num_test)+"n_"+str(self.num_repeat)+".npy", z_fatrop)

        # np.save("x_n_cem"+str(self.num_test)+"n_"+str(self.num_repeat)+".npy", x_cem)
        # np.save("y_n_cem"+str(self.num_test)+"n_"+str(self.num_repeat)+".npy", y_cem)
        # np.save("z_n_cem"+str(self.num_test)+"n_"+str(self.num_repeat)+".npy", z_cem)

        # np.save("x_our"+str(self.num_test)+"n_"+str(self.num_repeat)+".npy", x_our)
        # np.save("y_our"+str(self.num_test)+"n_"+str(self.num_repeat)+".npy", y_our)
        # np.save("z_our"+str(self.num_test)+"n_"+str(self.num_repeat)+".npy", z_our)

        # np.save("x_cem_proj"+str(self.num_test)+"n_"+str(self.num_repeat)+".npy", x_cem_proj)
        # np.save("y_cem_proj"+str(self.num_test)+"n_"+str(self.num_repeat)+".npy", y_cem_proj)
        # np.save("z_cem_proj"+str(self.num_test)+"n_"+str(self.num_repeat)+".npy", z_cem_proj)


        u = jnp.linspace(0, 2 * jnp.pi, 100)
        v = jnp.linspace(0, jnp.pi, 100)

        figure = mlab.figure(bgcolor=(1, 1, 1), size=(800, 600))
        for i in range(25):
            x = self.x_obs_init_1.flatten()[i] + (self.a_obs_1-0.03) * jnp.outer(jnp.cos(u), jnp.sin(v))
            y = self.y_obs_init_1.flatten()[i] + (self.b_obs_1-0.03) * jnp.outer(jnp.sin(u), jnp.sin(v))
            z = self.z_obs_init_1.flatten()[i] + (self.c_obs_1-0.03) * jnp.outer(jnp.ones(jnp.size(u)), jnp.cos(v))
            mlab.mesh(x, y, z, color=(0.5, 0.5, 0.5))

        for i in range(2):
            x = self.x_obs_init_2.flatten()[i] + (self.a_obs_2-0.03) * jnp.outer(jnp.cos(u), jnp.sin(v))
            y = self.y_obs_init_2.flatten()[i] + (self.b_obs_2-0.03) * jnp.outer(jnp.sin(u), jnp.sin(v))
            z = self.z_obs_init_2.flatten()[i] + (self.c_obs_2-0.03) * jnp.outer(jnp.ones(jnp.size(u)), jnp.cos(v))
            # mlab.mesh(x, y, z, color=(0.9, 0.9, 0.9))

        for i in range(32):
            x = self.x_obs_init_3.flatten()[i] + (self.a_obs_3-0.03) * jnp.outer(jnp.cos(u), jnp.sin(v))
            y = self.y_obs_init_3.flatten()[i] + (self.b_obs_3-0.03) * jnp.outer(jnp.sin(u), jnp.sin(v))
            z = self.z_obs_init_3.flatten()[i] + (self.c_obs_3-0.03) * jnp.outer(jnp.ones(jnp.size(u)), jnp.cos(v))
            # mlab.mesh(x, y, z, color=(0.1, 0.1, 0.1))
            # mlab.mesh(x, y, z, color=(0.5, 0.5, 0.5))
        mlab.plot3d(x_best_our_method, y_best_our_method, z_best_our_method, color=(0,0,0))
        mlab.plot3d(x_initial_compare, y_initial_compare, z_initial_compare, color=(0,0,1))
        mlab.plot3d(x_rockit_naive, y_rockit_naive, z_rockit_naive,  color=(1,0, 0)) 
        mlab.plot3d(x_fatrop_naive, y_fatrop_naive, z_fatrop_naive,  color=(0,1,0) ) 
        # mlab.plot3d(x_rockit, y_rockit, z_rockit, color="(0.6, 0.0, 0.0)") 
        mlab.plot3d(x_fatrop, y_fatrop, z_fatrop,   color=(0.0, 1.0, 0.5) )
        mlab.plot3d(x_cem_proj, y_cem_proj, z_cem_proj,   color=(0.0, 0.7, 0.8) )



        mlab.view(azimuth=45, elevation=60, distance=6, focalpoint=(0.5, 0.5, 0.5))
        mlab.show()

       
        

      




    
if __name__ == "__main__":


    motion_planning = planning_traj()
    motion_planning.planner()


    