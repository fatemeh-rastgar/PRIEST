
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
import casadi_solver
import expert


class planning_traj():

    def __init__(self):

        #### General Parameters
        self.vx_init = 0.05
        self.vy_init = 0.1
        self.ax_init = 0.0
        self.ay_init = 0.0
        self.vx_fin = 0.0
        self.vy_fin = 0.0
        self.ax_fin = 0.0
        self.ay_fin = 0.0
        self.v_max = 2.5
        self.v_min = 0.02
        self.a_max = 3.0
        
        self.maxiter = 1
        self.maxiter_cem = 13
        self.weight_track = 0.001
        self.weight_smoothness = 1
        self.max_proj = 13

        self.a_obs = 0.4
        self.b_obs = 0.4

        #####################

        self.t_fin = 10 #### time horizon
        self.num = 100 #### number of steps in prediction.
        self.num_batch = 110
        self.tot_time = np.linspace(0, self.t_fin, self.num)

        x_init_arr = np.array([-2, -2, -2, -1, -1, -1, 5, 5, 5, -2, -2, -2, -2, -2, -2, -2, -2, -2, 4, 4, 4, 4, 4, 4, 2.8, 2.8, 2.8, -1, 
                               -1, -1, -2, -2, -2, 5, 5, 5, 11, 11, 11, 2, 2, 2, -1, -1, -1, 8, 8, 8, -2, -2, -2, -2, 
                               -2, -2, -2, -2, -2, 10, 10, 10, 0, 0, 0, 2, 2, 2, 8.5, 8.5, 8.5, -1, -1, -1, 2.5, 2.5, 2.5, -1, -1, 
                               -1, 8, 8, 8, 2, 2, 2, 0, 0, 0, 0, 0, 0, -1.2, -1.2, -1.2, 7, 7, 7, 10, 10, 10, 8.7, 
                               8.7, 8.7, 10, 10, 10, 6.5, 6.5, 6.5, 1, 1, 1, -1, -1, -1, 1, 1, 1])
        
        y_init_arr = np.array([8, 8, 8, 5, 5, 5, -0.8, -0.8, -0.8, -2, -2, -2, 2, 2, 2, 1, 1, 1, 1, 1
                               , 1, 0, 0, 0, 2.5, 2.5, 2.5, 4, 4, 4, 6, 6, 6, 1, 1, 1, 5, 5, 5, 2, 2, 2, 
                               9, 9, 9, 11, 11, 11, 3, 3, 3, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2, 2, 2, 8, 8, 
                               8, 0, 0, 0, 6.2, 6.2, 6.2, 7, 7, 7, 11, 11, 11, 5, 5, 5, 4, 4, 4, 11, 11, 11,
                                 11, 11, 11, 0, 0, 0, 1.7, 1.7, 1.7, 9, 9, 9, 5, 5, 5, 0.5, 0.5, 0.5, 9, 9, 9, 6, 6, 6, 2, 2, 2, 5, 5, 5, 7, 7, 7])


        x_fin_arr = np.array([12, 12, 12, 6.5, 6.5, 6.5, 2, 2, 2, 6, 6, 6, 11, 11, 11, 11, 11, 11, 8, 8, 
                              8, 7.5, 7.5, 7.5, 7.5, 7.5, 7.5,  9, 9, 9, 11, 11, 11, 9, 9, 9, -1, -1, -1, 12, 12, 12, 10, 
                              10, 10, 8, 8, 8, 11, 11, 11, 10, 10, 10, 10, 10, 10, -1, -1, -1, 11, 11, 11, 7, 7, 7, 0, 
                              0, 0, 11, 11, 11, 8, 8, 8, 10, 10, 10, 0, 0, 0, 8, 8, 8, 6, 6, 6, 7, 7, 7, 10, 
                              10, 10, 4, 4, 4, 2, 2, 2, 4, 4, 4, 2, 2, 2, 0, 0, 0, 10, 10, 10, 11, 11, 11, 7, 7, 7])


        y_fin_arr = np.array([3, 3, 3, 5, 5, 5, 7, 7, 7, 10, 10, 10, 9, 9, 9, 5, 5, 5, 5,
                               5, 5, 11, 11, 11, 12, 12, 12, 11, 11, 11, 6, 6, 6, 12, 12, 
                               12, 9, 9, 9, 12, 12, 12, 5, 5, 5, -1.5, -1.5, -1.5, 9, 9, 9, 
                               2, 2, 2, 2, 2, 2, 7, 7, 7, 0, 0, 0, 11, 11, 11, 7, 7, 7, 2, 2,
                                 2, -0.5, -0.5, -0.5, 5, 5, 5, 7, 7, 7, 2, 2, 2, -2, -2, -2, 9, 9, 
                                 9, 2, 2, 2, -1, -1, -1, 5, 5, 5, 10, 10, 10, 5, 5, 5, 9, 9, 9, 8, 8, 8, 5, 5, 5, 2, 2, 2])

        i = 116

       

        self.x_init = x_init_arr[i]
        self.y_init = y_init_arr[i]

        self.x_fin = x_fin_arr[i]
        self.y_fin = y_fin_arr[i]

        self.x_best_global = jnp.linspace(self.x_init, self.x_init , 100)
        self.y_best_global = jnp.linspace(self.y_init, self.y_init, 100)
            
        self.theta_des = np.arctan2(self.y_fin-self.y_init, self.x_fin-self.x_init) 
        self.v_des = 0.5

        self.theta_des = np.arctan2(self.y_fin-self.y_init, self.x_fin-self.x_init) 
        self.v_des = 2.0

        self.maxiter_mpc = 1
        self.num_batch = 500
        self.num_goal = int(self.num_batch*0.6)
        self.num_up = 100
        self.nvar = 11

        self.x_waypoint = jnp.linspace(self.x_init, self.x_fin + 0.0 * jnp.cos(self.theta_des) , 1000)
        self.y_waypoint = jnp.linspace(self.y_init, self.y_fin + 0.0 * jnp.sin(self.theta_des),  1000)
        self.way_point_shape = 1000

        data_x_obs = loadmat('/home/fatemeh/Priest/PRIEST/point-to-point-navigation/2d_comparison/obs_postions/x_obs_10.mat')
        data_y_obs = loadmat('/home/fatemeh/Priest/PRIEST/point-to-point-navigation/2d_comparison/obs_postions/y_obs_10.mat')

        self.x_obs_init = (data_x_obs['x_obs'][0])[0:50] 
        self.y_obs_init = (data_y_obs['y_obs'][0])[0:50]

        ts = np.linspace(0,2*np.pi, self.num)

        self.vx_obs = np.zeros(50)
        self.vy_obs = np.zeros(50)

        self.vx_des = self.v_des*jnp.cos(self.theta_des) 
        self.vy_des = self.v_des*jnp.sin(self.theta_des) 

        self.x_des_traj = self.x_init + self.vx_des*self.tot_time
        self.y_des_traj = self.y_init + self.vy_des*self.tot_time

        self.x_des_init = self.x_des_traj[0]
        self.y_des_init = self.y_des_traj[0]


        self.num_test = i//3
        self.num_repeat = i%3


    def planner(self, ):


        num_obs = 50

        x_guess_naive = np.linspace(self.x_init, self.x_fin, self.num)
        y_guess_naive = np.linspace(self.y_init, self.y_fin, self.num)

        prob_solver = casadi_solver.Solvers_func(self.t_fin, self.a_max, self.v_max, self.a_obs, self.num, num_obs)
        Prob_our = expert.batch_crowd_nav(self.a_obs, self.b_obs, self.v_max, self.v_min, self.a_max, num_obs, self.t_fin, self.num, self.num_batch, self.maxiter, self.maxiter_cem, self.weight_smoothness, self.weight_track, self.way_point_shape, self.v_des)
        prob_CEM = CEM.CEM_optimizer( self.a_obs, self.b_obs, self.t_fin, self.nvar, self.num, self.num_batch, self.maxiter_cem, self.weight_smoothness, num_obs,self.way_point_shape, self.v_max)
        prob_opt = Optim_part.Optim(self.way_point_shape, self.max_proj, self.num_goal, self.v_max, self.a_max, self.weight_smoothness, self.nvar, self.num, num_obs, self.num_batch, self.a_obs, self.b_obs, self.t_fin, self.vx_obs, self.vy_obs, self.vx_des, self.vy_des)
        

        ########################################################    
        key = random.PRNGKey(0)

        #t_update = 0.04#### simulation

        arc_length, arc_vec, x_diff, y_diff =  Prob_our.path_spline(self.x_waypoint, self.y_waypoint)
    
        # ###########
        lamda_x_our_method = jnp.zeros((self.num_batch, Prob_our.nvar))
        lamda_y_our_method = jnp.zeros((self.num_batch, Prob_our.nvar))

        lamda_x_cem = jnp.zeros((self.num_batch, prob_CEM.nvar))
        lamda_y_cem = jnp.zeros((self.num_batch, prob_CEM.nvar))

        lamda_x_per = jnp.zeros((self.num_goal, prob_CEM.nvar))
        lamda_y_per = jnp.zeros((self.num_goal, prob_CEM.nvar))

        ##########warm_up
        for j in range(2):
            initial_state = jnp.hstack(( self.x_init, self.y_init, self.vx_init, self.vy_init, self.ax_init, self.ay_init ))

            x_obs_trajectory_w, y_obs_trajectory_w, x_obs_trajectory_proj_w, y_obs_trajectory_proj_w = Prob_our.compute_obs_traj_prediction(jnp.asarray(self.x_obs_init).flatten(), jnp.asarray(self.y_obs_init).flatten(), self.vx_obs, self.vy_obs, initial_state[0], initial_state[1] ) ####### obstacle trajectory prediction

            sol_x_bar_w, sol_y_bar_w, x_guess_w, y_guess_w, xdot_guess_w, ydot_guess_w, xddot_guess_w, yddot_guess_w,c_mean_w, c_cov_w = Prob_our.compute_traj_guess( initial_state, x_obs_trajectory_w, y_obs_trajectory_w, self.v_des, self.x_waypoint, self.y_waypoint, arc_vec, x_diff, y_diff, self.x_fin, self.y_fin)

            x_best_w, y_best_w, com = Prob_our.compute_cem(key, initial_state, self.x_fin, self.y_fin, lamda_x_our_method, lamda_y_our_method, x_obs_trajectory_w, y_obs_trajectory_w, x_obs_trajectory_proj_w, y_obs_trajectory_proj_w, sol_x_bar_w, sol_y_bar_w, x_guess_w, y_guess_w,  xdot_guess_w, ydot_guess_w, xddot_guess_w, yddot_guess_w, self.x_waypoint,  self.y_waypoint, arc_vec, c_mean_w, c_cov_w )
            #########################################################


        for i in range(0, self.maxiter_mpc):

            initial_state = jnp.hstack(( self.x_init, self.y_init, self.vx_init, self.vy_init, self.ax_init, self.ay_init ))

            x_obs_trajectory, y_obs_trajectory, x_obs_trajectory_proj, y_obs_trajectory_proj = Prob_our.compute_obs_traj_prediction(jnp.asarray(self.x_obs_init).flatten(), jnp.asarray(self.y_obs_init).flatten(), self.vx_obs, self.vy_obs, initial_state[0], initial_state[1] ) ####### obstacle trajectory prediction

            sol_x_bar, sol_y_bar, x_guess, y_guess, xdot_guess, ydot_guess, xddot_guess, yddot_guess,c_mean, c_cov = Prob_our.compute_traj_guess( initial_state, x_obs_trajectory, y_obs_trajectory, self.v_des, self.x_waypoint, self.y_waypoint, arc_vec, x_diff, y_diff, self.x_fin , self.y_fin)

            for j in range(3):

                lamda_x_our_method = jnp.zeros((self.num_batch, Prob_our.nvar))
                lamda_y_our_method = jnp.zeros((self.num_batch, Prob_our.nvar))

                ################### Our method
                start = time.time()
                x_best_our_method, y_best_our_method, comp_iter = Prob_our.compute_cem(key, initial_state, self.x_fin, self.y_fin, lamda_x_our_method, lamda_y_our_method, x_obs_trajectory, y_obs_trajectory, x_obs_trajectory_proj, y_obs_trajectory_proj, sol_x_bar, sol_y_bar, x_guess, y_guess,  xdot_guess, ydot_guess, xddot_guess, yddot_guess, self.x_waypoint,  self.y_waypoint, arc_vec, c_mean, c_cov )
                
                # if j ==2:
                Time_our = time.time() -start
                print(comp_iter)
                
                # print("our method is working")



        #         ################## Previous method
        #         #######CEM_Base

        #         start = time.time()
        #         x_initial_compare, y_initial_compare, x_samples_cem, y_samples_cem, xdot_samples_cem, ydot_samples_cem, xddot_samples_cem, yddot_samples_cem = prob_CEM.compute_cem_part(initial_state, x_obs_trajectory, y_obs_trajectory, self.vx_des, self.vy_des,  self.v_des, self.x_waypoint, self.y_waypoint,  x_obs_trajectory, y_obs_trajectory, lamda_x_cem, lamda_y_cem, self.x_fin, self.y_fin, sol_x_bar, sol_y_bar, c_mean, c_cov, x_guess, y_guess, arc_vec )
                
        #         # if j ==2:
        #         Time_CEM = time.time()-start
        #         # print("CEM base is working")

        #         # plt.figure(3)
        #         # plt.plot(x_samples_cem.T, y_samples_cem.T)
        #         # plt.show()
        #         ############## CEM + Projection

        #         start = time.time()

        #         sol_x_best, sol_y_best, x_best, y_best, x, y = prob_opt.SolveOpt1(self.x_waypoint, self.y_waypoint, arc_vec, x_samples_cem, y_samples_cem, xdot_samples_cem, ydot_samples_cem, xddot_samples_cem, yddot_samples_cem, x_obs_trajectory, y_obs_trajectory,  self.x_des_init, self.y_des_init, initial_state, self.x_fin, self.y_fin, lamda_x_per, lamda_y_per)
                
        #         # if j ==2:
        #         Time_per = time.time()-start 
                
        #         # plt.figure(3)
        #         # plt.plot(x_initial_compare, y_initial_compare)
        #         # plt.plot(x_best, y_best)
        #         # plt.show()

        #         ######################################
            
        #     print("rockit_naive")
        #     sol_rockit, x_rockit_naive, y_rockit_naive = prob_solver.SolveRockit(initial_state, self.x_fin, self.y_fin, x_guess_naive, y_guess_naive, self.x_obs_init, self.y_obs_init)
            
        #     print("fatrop_naive")
        #     sol_fatrop, x_fatrop_naive, y_fatrop_naive = prob_solver.SolveFatrop(initial_state, self.x_fin, self.y_fin, x_guess_naive, y_guess_naive, self.x_obs_init, self.y_obs_init )

        #     print("rockit")
        #     sol_rockit, x_rockit, y_rockit = prob_solver.SolveRockit(initial_state, self.x_fin, self.y_fin, x_initial_compare, y_initial_compare, self.x_obs_init, self.y_obs_init)
            
        #     print("fatrop")
        #     sol_fatrop, x_fatrop, y_fatrop = prob_solver.SolveFatrop(initial_state, self.x_fin, self.y_fin, x_initial_compare, y_initial_compare, self.x_obs_init, self.y_obs_init )


        # x_rockit_naive = np.array(x_rockit_naive)
        # y_rockit_naive = np.array(y_rockit_naive)

        # x_rockit = np.array(x_rockit)
        # y_rockit = np.array(y_rockit)

        # x_fatrop_naive = np.array(x_fatrop_naive)
        # y_fatrop_naive = np.array(y_fatrop_naive)

        # x_fatrop = np.array(x_fatrop)
        # y_fatrop = np.array(y_fatrop)

        # x_opt = np.array(x_best)
        # y_opt = np.array(y_best)

        x_our = np.array(x_best_our_method)
        y_our = np.array(y_best_our_method)

        # x_cem = np.array(x_initial_compare)
        # y_cem = np.array(y_initial_compare)

        ts = np.linspace(0,2*np.pi, self.num)
        print(Time_our, )
        print(self.x_init, self.y_init, self.x_fin, self.y_fin)

        # np.save("x_rockit_naive_n"+str(self.num_test)+"n_"+str(self.num_repeat)+".npy", x_rockit_naive)
        # np.save("y_rockit_naive_n"+str(self.num_test)+"n_"+str(self.num_repeat)+".npy", y_rockit_naive)
        
        # np.save("x_rockit_n"+str(self.num_test)+"n_"+str(self.num_repeat)+".npy", x_rockit)
        # np.save("y_rockit_n"+str(self.num_test)+"n_"+str(self.num_repeat)+".npy", y_rockit)

        # np.save("x_fatrop_naive_n"+str(self.num_test)+"n_"+str(self.num_repeat)+".npy", x_fatrop_naive)
        # np.save("y_fatrop_naive_n"+str(self.num_test)+"n_"+str(self.num_repeat)+".npy", y_fatrop_naive)

        # np.save("x_fatrop_n"+str(self.num_test)+"n_"+str(self.num_repeat)+".npy", x_fatrop)
        # np.save("y_fatrop_n"+str(self.num_test)+"n_"+str(self.num_repeat)+".npy", y_fatrop)

        # np.save("x_n"+str(self.num_test)+"n_"+str(self.num_repeat)+".npy", x_opt)
        # np.save("y_n"+str(self.num_test)+"n_"+str(self.num_repeat)+".npy", y_opt)

        # np.save("x_n_cem"+str(self.num_test)+"n_"+str(self.num_repeat)+".npy", x_cem)
        # np.save("y_n_cem"+str(self.num_test)+"n_"+str(self.num_repeat)+".npy", y_cem)

        np.save("x_our"+str(self.num_test)+"n_"+str(self.num_repeat)+".npy", x_our)
        np.save("y_our"+str(self.num_test)+"n_"+str(self.num_repeat)+".npy", y_our)

        plt.figure()
        # plt.plot(x_rockit_naive, y_rockit_naive, color = "red", label = "rockit_naive" )
        # plt.plot(x_fatrop_naive, y_fatrop_naive, color = "brown",  label = "fatrop_naive")
        # plt.plot(x_rockit, y_rockit, color = "blue",  label = "rockit")
        # plt.plot(x_fatrop, y_fatrop, color = "green",  label = "Fatrop")
        # plt.plot(x_best, y_best, color = "cyan",  label = "per")
        # plt.plot(x_initial_compare, y_initial_compare, color = "pink", label = "cem")
        #plot(x.T, y.T)
        plt.plot( x_best_our_method, y_best_our_method, color = "black", label = "proposed")
        for i in range(self.x_obs_init.shape[0]):
            plt.plot(self.x_obs_init[i]+1*(self.a_obs-0.04)*np.cos(ts),self.y_obs_init[i]+1*(self.a_obs-0.04)*np.sin(ts),'r-')

        plt.axis('equal')
        plt.legend()
        plt.show(block=True)

      




    
if __name__ == "__main__":


    motion_planning = planning_traj()
    motion_planning.planner()


    