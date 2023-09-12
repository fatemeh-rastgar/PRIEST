import numpy as np
import jax.numpy as jnp
from functools import partial
from jax import jit, random
import bernstein_coeff_order10_arbitinterval
import time
import matplotlib.pyplot as plt 
import jax
from jax.ops import index_update, index
from scipy.interpolate import CubicSpline

import matplotlib.pyplot as plt 
from scipy.interpolate import BarycentricInterpolator

from scipy.interpolate import UnivariateSpline
from jax import lax
import jax.lax as lax



class batch_crowd_nav():

    def __init__(self, a_obs_1, b_obs_1, a_obs_2, b_obs_2, v_max, v_min, a_max, num_obs_1, num_obs_2, t_fin, num, num_batch, maxiter, maxiter_cem, weight_smoothness, weight_track, way_point_shape, v_des):

        
        self.maxiter = maxiter
        self.maxiter_cem = maxiter_cem
        self.weight_smoothness = weight_smoothness
        self.weight_track = weight_track
        self.v_des = v_des

        self.v_max = v_max
        self.v_min = v_min
        self.a_max = a_max

        self.a_obs_1 = a_obs_1
        self.b_obs_1 = b_obs_1
        self.a_obs_2 = a_obs_2
        self.b_obs_2 = b_obs_2

        self.t_fin = t_fin
        self.num = num
        self.t = self.t_fin/self.num
        self.num_batch = num_batch
        self.ellite_num = int(self.num_batch*0.6)
        self.ellite_num_const = self.num_batch - self.ellite_num 
        self.ellite_num = 20

        self.num_obs_1 = num_obs_1
        self.num_obs_2 = num_obs_2
        self.num_obs_proj = 20

        tot_time = np.linspace(0, t_fin, num)
        tot_time_copy = tot_time.reshape(num, 1)

        self.P, self.Pdot, self.Pddot = bernstein_coeff_order10_arbitinterval.bernstein_coeff_order10_new(10, tot_time_copy[0], tot_time_copy[-1], tot_time_copy)
        
        self.tot_time = tot_time
        
        self.P_jax, self.Pdot_jax, self.Pddot_jax = jnp.asarray(self.P), jnp.asarray(self.Pdot), jnp.asarray(self.Pddot)
        
        self.nvar = jnp.shape(self.P_jax)[1]

        self.cost_smoothness = self.weight_smoothness*jnp.dot(self.Pddot_jax.T, self.Pddot_jax)	
        self.A_projection = jnp.identity( self.nvar)			
        self.A_eq = jnp.vstack(( self.P_jax[0], self.Pdot_jax[0], self.Pddot_jax[0] , self.P_jax[-1] ))
                    
        self.A_vel = self.Pdot_jax 
        self.A_acc = self.Pddot_jax
        self.A_obs = jnp.tile(self.P_jax, (self.num_obs_proj+self.num_obs_2, 1))

        ########## for up sampling

        num_up = 100
        dt_up = t_fin/num_up
        tot_time_up = np.linspace(0, t_fin, num_up)
        self.tot_time_up = tot_time_up
        tot_time_copy_up = tot_time_up.reshape(num_up, 1)

        P_up, Pdot_up, Pddot_up = bernstein_coeff_order10_arbitinterval.bernstein_coeff_order10_new(10, tot_time_copy_up[0], tot_time_copy_up[-1], tot_time_copy_up)

        self.P_up_jax = jnp.asarray(P_up)
        self.Pdot_up_jax = jnp.asarray(Pdot_up)
        self.Pddot_up_jax = jnp.asarray(Pddot_up)

        self.maxitet_proj = 15

        ##########################3 computing initial covariance for trajectories
        
        A = np.diff(np.diff(np.identity(self.num), axis = 0), axis = 0)

        temp_1 = np.zeros(self.num)
        temp_2 = np.zeros(self.num)
        temp_3 = np.zeros(self.num)
        temp_4 = np.zeros(self.num)

        temp_1[0] = 1.0
        temp_2[0] = -2
        temp_2[1] = 1
        temp_3[-1] = -2
        temp_3[-2] = 1

        temp_4[-1] = 1.0

        A_mat = -np.vstack(( temp_1, temp_2, A, temp_3, temp_4   ))
        R = np.dot(A_mat.T, A_mat)
        mu = np.zeros(self.num)
        cov = np.linalg.pinv(R)
        ################# Gaussian Trajectory Sampling
        self.initial_up_sampling = 30

        eps_k = np.random.multivariate_normal(mu, 0.1*cov, (int(self.num_batch/1), ))
        eps_k_up_sampling = np.random.multivariate_normal(mu, 0.1*cov, (int((self.num_batch * self.initial_up_sampling)), ))

        self.eps_k = jnp.asarray(eps_k)
        self.num_sample = 6
        self.num_sample_warm =  self.num_batch //2

        self.beta = 4
        self.lamda = 0.9
        self.alpha = 0.7
        self.vec_product = jit(jax.vmap(self.comp_prod, 0, out_axes=(0)))
        self.vectorized_projection = jit(jax.vmap(self.compute_contouring_error, in_axes=(None, None, 0, 0, None, ) ))
        self.way_point_shape = way_point_shape
        ###############

        self.cost_x =  jnp.dot(self.P_jax.T, self.P_jax) + 0.0001 * jnp.identity(self.nvar) + 1*jnp.dot(self.Pddot_jax.T, self.Pddot_jax)
        self.cost_y = self.cost_x
        self.cost_mat_x = self.cost_x
        self.cost_mat_inv_x = jnp.linalg.inv(self.cost_mat_x)
        self.cost_mat_y = self.cost_y
        self.cost_mat_inv_y = jnp.linalg.inv(self.cost_mat_y)
        ######################################

        self.rho_proj = 1
        self.rho_obs = 1
        self.rho_ineq = 1

        self.cost_x_projection = self.rho_proj * jnp.dot( self.A_projection.T, self.A_projection) + self.rho_obs*jnp.dot(self.A_obs.T, self.A_obs)+self.rho_ineq*jnp.dot(self.A_acc.T, self.A_acc)+self.rho_ineq*jnp.dot(self.A_vel.T, self.A_vel) 
        #self.cost_y_projection = self.cost_x_projection

        self.cost_mat_x_projection = jnp.vstack((  jnp.hstack(( self.cost_x_projection, self.A_eq.T )), jnp.hstack(( self.A_eq, jnp.zeros(( jnp.shape(self.A_eq)[0], jnp.shape(self.A_eq)[0] )) )) ))
        self.cost_mat_inv_x_projection = jnp.linalg.inv(self.cost_mat_x_projection)
        self.cost_mat_inv_y_projection = self.cost_mat_inv_x_projection


        self.scale_factor = np.random.normal( 0, 0.8, size=self.num_batch * self.initial_up_sampling)#np.random.uniform(low=-2, high=2.0, size=self.num_batch * self.initial_up_sampling)# np.random.normal(size=self.num_batch * self.initial_up_sampling)
        self.scale_factor_warm = np.random.normal( 0, 0.8, size=self.num_sample_warm)
        self.scale_factor_1 = np.random.normal( 0, 0.8, size=self.num_batch * self.initial_up_sampling)
        self.scale_factor_2 = np.random.normal( 0, 0.8, size=self.num_batch * self.initial_up_sampling)
        self.scale_factor_3 = np.random.normal( 0, 0.8, size=self.num_batch * self.initial_up_sampling)
        self.scale_factor_4 = np.random.normal( 0, 0.8, size=self.num_batch * self.initial_up_sampling)
        self.k_max = 2

        self.cost = jnp.identity(self.nvar)
        



    @partial(jit, static_argnums=(0,),backend="gpu")
    def path_spline(self, x_waypoint, y_waypoint):

        x_diff = jnp.diff(x_waypoint)
        y_diff = jnp.diff(y_waypoint)

        arc = jnp.cumsum(jnp.sqrt( x_diff**2 + y_diff**2))
        arc_length = arc[-1]

        arc_vec = jnp.linspace(0, arc_length, self.way_point_shape)

        return arc_length, arc_vec, x_diff, y_diff


    @partial(jit, static_argnums=(0,),backend="gpu")
    def compute_contouring_error(self, x_waypoint, y_waypoint, x_target_point, y_target_point, arc_vec):

        dist = jnp.sqrt( (x_waypoint - x_target_point)**2 + (y_waypoint - y_target_point)**2 )
        index = jnp.argmin(dist)

        arc_point = arc_vec[index]

        x_project = x_waypoint[index]
        y_project = y_waypoint[index]

        return arc_point, x_project, y_project

    #######################################################
    @partial(jit, static_argnums=(0,),backend="gpu")
    def compute_obs_traj_prediction(self, x_obs_init_dy, y_obs_init_dy, vx_obs_dy, vy_obs_dy, x_obs_init, y_obs_init, vx_obs, vy_obs, x_init, y_init):

        x_temp = x_obs_init + vx_obs * self.tot_time[:,jnp.newaxis]
        x_obs_trajectory = x_temp.T 

        y_temp = y_obs_init+vy_obs*self.tot_time[:,jnp.newaxis]
        y_obs_trajectory = y_temp.T

        x_temp_dy = x_obs_init_dy + vx_obs_dy * self.tot_time[:,jnp.newaxis]
        x_obs_trajectory_dy = x_temp_dy.T 

        y_temp_dy = y_obs_init_dy + vy_obs_dy *self.tot_time[:,jnp.newaxis]
        y_obs_trajectory_dy = y_temp_dy.T

        Dist = jnp.sqrt( (x_init - x_obs_trajectory[:,0])**2 +(y_init - y_obs_trajectory[:,0])**2) 
        idx_dis = jnp.argsort(Dist)

        x_obs_trajectory = x_obs_trajectory[idx_dis[0:(self.num_obs_1 )],:]
        y_obs_trajectory = y_obs_trajectory[idx_dis[0:(self.num_obs_1) ],:]

        x_obs_trajectory_proj = x_obs_trajectory[idx_dis[0:(self.num_obs_proj)],:]
        y_obs_trajectory_proj = y_obs_trajectory[idx_dis[0:(self.num_obs_proj)],:]

        return x_obs_trajectory, y_obs_trajectory, x_obs_trajectory_proj, y_obs_trajectory_proj, x_obs_trajectory_dy, y_obs_trajectory_dy

    ###############################################################

    @partial(jit, static_argnums=(0,),backend="gpu")	
    def compute_boundary_vec(self, initial_state, x_fin, y_fin):		

        x_init, y_init, vx_init, vy_init, ax_init, ay_init = initial_state

        x_init_vec = x_init*jnp.ones((self.num_batch, 1))
        y_init_vec = y_init*jnp.ones((self.num_batch, 1)) 

        vx_init_vec = vx_init*jnp.ones((self.num_batch, 1))
        vy_init_vec = vy_init*jnp.ones((self.num_batch, 1))

        ax_init_vec = ax_init*jnp.ones((self.num_batch, 1))
        ay_init_vec = ay_init*jnp.ones((self.num_batch, 1))

        x_fin_vec = x_fin*jnp.ones((self.num_batch, 1))
        y_fin_vec = y_fin*jnp.ones((self.num_batch, 1)) 

        b_eq_x = jnp.hstack(( x_init_vec, vx_init_vec, ax_init_vec, x_fin_vec   ))
        b_eq_y = jnp.hstack(( y_init_vec, vy_init_vec, ay_init_vec, y_fin_vec  ))
		
        return b_eq_x, b_eq_y
    ###########################################################

    @partial(jit, static_argnums=(0,),backend="gpu")
    def compute_warm_traj(self, initial_state, v_des, x_waypoint, y_waypoint, arc_vec, x_diff, y_diff):

        x_init, y_init, vx_init, vy_init, ax_init, ay_init = initial_state

        dist = jnp.sqrt( (x_waypoint - x_init)**2 + (y_waypoint - y_init)**2 )
        index = jnp.argmin(dist)
        arc_point = arc_vec[index]

        look_ahead_point_path = arc_point+v_des*self.t_fin
        look_ahead_point_path = jnp.clip(look_ahead_point_path, arc_vec[0], arc_vec[-1])
        index_final_path = jnp.argmin(jnp.abs(look_ahead_point_path-arc_vec)  )

        x_fin_path = x_waypoint[index_final_path]
        y_fin_path = y_waypoint[index_final_path]
        #########################################

        look_ahead__path = arc_point+v_des*self.t_fin*0.5
        look_ahead__path = jnp.clip(look_ahead__path, arc_vec[0], arc_vec[-1])
        index_mid_path = jnp.argmin(jnp.abs(look_ahead__path-arc_vec)  )

        x_m = x_waypoint[index_mid_path]
        y_m = y_waypoint[index_mid_path]

        xdot_m = x_diff[index_mid_path]
        ydot_m = y_diff[index_mid_path]

        normal_x =  -1*(ydot_m)/jnp.sqrt(xdot_m**2 + ydot_m**2  )
        normal_y =   1*(xdot_m)/jnp.sqrt(xdot_m**2 + ydot_m**2  )

        x_guess_mid = x_m + self.scale_factor_warm * normal_x
        y_guess_mid = y_m + self.scale_factor_warm * normal_y
        ######################################################33333

        key = random.PRNGKey(0)

        x_init_vec = x_init*jnp.ones((self.num_sample_warm, 1))
        y_init_vec = y_init*jnp.ones((self.num_sample_warm, 1)) 

        vx_init_vec = vx_init*jnp.ones((self.num_sample_warm, 1))
        vy_init_vec = vy_init*jnp.ones((self.num_sample_warm, 1))
    
        ax_init_vec = ax_init*jnp.ones((self.num_sample_warm, 1))
        ay_init_vec = ay_init*jnp.ones((self.num_sample_warm, 1))

        x_mid_vec = x_guess_mid.reshape(self.num_sample_warm, 1)
        y_mid_vec = y_guess_mid.reshape(self.num_sample_warm, 1)

        x_fin_vec = x_fin_path*jnp.ones((self.num_sample_warm, 1))
        y_fin_vec = y_fin_path*jnp.ones((self.num_sample_warm, 1)) 
    
        b_eq_x = jnp.hstack(( x_init_vec, vx_init_vec, ax_init_vec, x_mid_vec, x_fin_vec   ))
        b_eq_y = jnp.hstack(( y_init_vec, vy_init_vec, ay_init_vec, y_mid_vec, y_fin_vec ))

        A_eq = jnp.vstack(( self.P_jax[0], self.Pdot_jax[0], self.Pddot_jax[0], self.P_jax[50], self.P_jax[-1]  ))

        cost_mat = jnp.vstack((  jnp.hstack(( jnp.dot(self.Pddot_jax.T, self.Pddot_jax), A_eq.T )), jnp.hstack(( A_eq, jnp.zeros(( jnp.shape(A_eq)[0], jnp.shape(A_eq)[0] )) )) ))
        cost_mat_inv = jnp.linalg.inv(cost_mat)

        sol_x = jnp.dot(cost_mat_inv, jnp.hstack(( -jnp.zeros((self.num_sample_warm, self.nvar)), b_eq_x )).T).T
        sol_y = jnp.dot(cost_mat_inv, jnp.hstack(( -jnp.zeros((self.num_sample_warm, self.nvar)), b_eq_y )).T).T
    
        x_guess_per = jnp.dot(self.P_jax, sol_x[:,0:self.nvar].T).T
        y_guess_per = jnp.dot(self.P_jax, sol_y[:,0:self.nvar].T).T

        return x_guess_per, y_guess_per 

    #################################################

    @partial(jit, static_argnums=(0,),backend="gpu")
    def compute_traj_guess(self, initial_state, x_obs_trajectory, y_obs_trajectory, x_obs_trajectory_dy, y_obs_trajectory_dy, v_des, x_waypoint, y_waypoint, arc_vec, x_guess_per , y_guess_per, x_diff, y_diff):

        x_init, y_init, vx_init, vy_init, ax_init, ay_init = initial_state
        
        dist = jnp.sqrt( (x_waypoint - x_init)**2 + (y_waypoint - y_init)**2 )
        index = jnp.argmin(dist)
        arc_point = arc_vec[index]

        look_ahead_point_path = arc_point+v_des*self.t_fin
        look_ahead_point_path = jnp.clip(look_ahead_point_path, arc_vec[0], arc_vec[-1])
        index_final_path = jnp.argmin(jnp.abs(look_ahead_point_path-arc_vec)  )

        x_fin_path = x_waypoint[index_final_path]
        y_fin_path = y_waypoint[index_final_path]
        #########################################

        look_ahead__path = arc_point+v_des*self.t_fin*0.5
        look_ahead__path = jnp.clip(look_ahead__path, arc_vec[0], arc_vec[-1])
        index_mid_path = jnp.argmin(jnp.abs(look_ahead__path-arc_vec)  )

        x_m = x_waypoint[index_mid_path]
        y_m = y_waypoint[index_mid_path]

        xdot_m = x_diff[index_mid_path]
        ydot_m = y_diff[index_mid_path]

        normal_x =  -1*(ydot_m)/jnp.sqrt(xdot_m**2 + ydot_m**2  )
        normal_y =   1*(xdot_m)/jnp.sqrt(xdot_m**2 + ydot_m**2  )

        x_guess_mid = x_m + self.scale_factor * normal_x
        y_guess_mid = y_m + self.scale_factor * normal_y

        #########################################

        look_ahead__path1 = arc_point+v_des*self.t_fin*0.75
        look_ahead__path1 = jnp.clip(look_ahead__path1, arc_vec[0], arc_vec[-1])
        index_mid_path1 = jnp.argmin(jnp.abs(look_ahead__path1-arc_vec)  )

        x_m1 = x_waypoint[index_mid_path1]
        y_m1 = y_waypoint[index_mid_path1]

        xdot_m1 = x_diff[index_mid_path1]
        ydot_m1 = y_diff[index_mid_path1]

        normal_x1 =  -1*(ydot_m1)/jnp.sqrt(xdot_m1**2 + ydot_m1**2  )
        normal_y1 =   1*(xdot_m1)/jnp.sqrt(xdot_m1**2 + ydot_m1**2  )

        x_guess_mid1 = x_m1 + self.scale_factor_1 * normal_x1
        y_guess_mid1 = y_m1 + self.scale_factor_1 * normal_y1
        #########################################################

        look_ahead__path2 = arc_point+v_des*self.t_fin*0.25
        look_ahead__path2 = jnp.clip(look_ahead__path2, arc_vec[0], arc_vec[-1])
        index_mid_path2 = jnp.argmin(jnp.abs(look_ahead__path2-arc_vec)  )

        x_m2 = x_waypoint[index_mid_path2]
        y_m2 = y_waypoint[index_mid_path2]

        xdot_m2 = x_diff[index_mid_path2]
        ydot_m2 = y_diff[index_mid_path2]

        normal_x2 =  -1*(ydot_m2)/jnp.sqrt(xdot_m2**2 + ydot_m2**2  )
        normal_y2 =   1*(xdot_m2)/jnp.sqrt(xdot_m2**2 + ydot_m2**2  )

        x_guess_mid2 = x_m2 + self.scale_factor_2 * normal_x2
        y_guess_mid2 = y_m2 + self.scale_factor_2 * normal_y2

        ##################
        key = random.PRNGKey(0)

        x_init_vec = x_init*jnp.ones((self.num_batch * self.initial_up_sampling, 1))
        y_init_vec = y_init*jnp.ones((self.num_batch * self.initial_up_sampling, 1)) 

        vx_init_vec = vx_init*jnp.ones((self.num_batch * self.initial_up_sampling, 1))
        vy_init_vec = vy_init*jnp.ones((self.num_batch * self.initial_up_sampling, 1))
    
        ax_init_vec = ax_init*jnp.ones((self.num_batch * self.initial_up_sampling, 1))
        ay_init_vec = ay_init*jnp.ones((self.num_batch * self.initial_up_sampling, 1))

        x_mid_vec = x_guess_mid.reshape(self.num_batch * self.initial_up_sampling, 1)
        y_mid_vec = y_guess_mid.reshape(self.num_batch * self.initial_up_sampling, 1)

        x_75_vec = x_guess_mid1.reshape(self.num_batch * self.initial_up_sampling, 1)
        y_75_vec = y_guess_mid1.reshape(self.num_batch * self.initial_up_sampling, 1)

        x_25_vec = x_guess_mid2.reshape(self.num_batch * self.initial_up_sampling, 1)
        y_25_vec = y_guess_mid2.reshape(self.num_batch * self.initial_up_sampling, 1)

        x_fin_vec = x_fin_path*jnp.ones((self.num_batch* self.initial_up_sampling, 1))
        y_fin_vec = y_fin_path*jnp.ones((self.num_batch* self.initial_up_sampling, 1)) 
    
        b_eq_x = jnp.hstack(( x_init_vec, vx_init_vec, ax_init_vec,  x_25_vec, x_75_vec, x_mid_vec,  x_fin_vec   ))
        b_eq_y = jnp.hstack(( y_init_vec, vy_init_vec, ay_init_vec,  y_25_vec, y_75_vec, y_mid_vec,  y_fin_vec ))

        A_eq = jnp.vstack(( self.P_jax[0], self.Pdot_jax[0], self.Pddot_jax[0],  self.P_jax[25],  self.P_jax[75], self.P_jax[50], self.P_jax[-1]  ))
        cost_mat = jnp.vstack((  jnp.hstack(( jnp.dot(self.Pddot_jax.T, self.Pddot_jax), A_eq.T )), jnp.hstack(( A_eq, jnp.zeros(( jnp.shape(A_eq)[0], jnp.shape(A_eq)[0] )) )) ))
        cost_mat_inv = jnp.linalg.inv(cost_mat)

        sol_x = jnp.dot(cost_mat_inv, jnp.hstack(( -jnp.zeros((self.num_batch* self.initial_up_sampling, self.nvar)), b_eq_x )).T).T
        sol_y = jnp.dot(cost_mat_inv, jnp.hstack(( -jnp.zeros((self.num_batch* self.initial_up_sampling, self.nvar)), b_eq_y )).T).T
    
        x_guess_temp = jnp.dot(self.P_jax, sol_x[:,0:self.nvar].T).T
        y_guess_temp = jnp.dot(self.P_jax, sol_y[:,0:self.nvar].T).T
        
        #x_fin_path = x_waypoint[index+100]#(x_init_project +  vx_des * self.tot_time)[-1]
        #y_fin_path = y_waypoint[index+100]#(y_init_project +  vy_des * self.tot_time)[-1]

        # x_interp = jnp.linspace(x_init, x_fin_path, self.num)
        # y_interp = jnp.linspace(y_init, y_fin_path, self.num)

        # x_guess_temp = x_interp+ 0.0*self.eps_k_up_sampling 
        # y_guess_temp = y_interp+ self.eps_k_up_sampling

        x_guess_sampling = x_guess_temp #x_guess_temp#x_guess_temp * jnp.cos(goal_rot) + y_guess_temp * jnp.sin(goal_rot)
        y_guess_sampling = y_guess_temp #y_guess_temp#- x_guess_temp * jnp.sin(goal_rot) + y_guess_temp * jnp.cos(goal_rot)

        wc_alpha_temp = (x_guess_sampling-x_obs_trajectory[:,jnp.newaxis])
        ws_alpha_temp = (y_guess_sampling-y_obs_trajectory[:,jnp.newaxis])

        wc_alpha_temp_dy = (x_guess_sampling-x_obs_trajectory_dy[:,jnp.newaxis])
        ws_alpha_temp_dy = (y_guess_sampling-y_obs_trajectory_dy[:,jnp.newaxis])

        wc_alpha = wc_alpha_temp.transpose(1, 0, 2)
        ws_alpha = ws_alpha_temp.transpose(1, 0, 2)

        wc_alpha_dy = wc_alpha_temp_dy.transpose(1, 0, 2)
        ws_alpha_dy = ws_alpha_temp_dy.transpose(1, 0, 2)

        wc_alpha = wc_alpha.reshape(self.num_batch * self.initial_up_sampling, self.num*self.num_obs_1)
        ws_alpha = ws_alpha.reshape(self.num_batch * self.initial_up_sampling, self.num*self.num_obs_1)

        wc_alpha_dy = wc_alpha_dy.reshape(self.num_batch * self.initial_up_sampling, self.num*self.num_obs_2)
        ws_alpha_dy = ws_alpha_dy.reshape(self.num_batch * self.initial_up_sampling, self.num*self.num_obs_2)

        dist_obs_st = -wc_alpha**2/(self.a_obs_1**2)-ws_alpha**2/(self.b_obs_1**2)+1
        dist_obs_dy = -wc_alpha_dy**2/(self.a_obs_2**2)-ws_alpha_dy**2/(self.b_obs_2**2)+1

        dist_obs = jnp.hstack(( dist_obs_st, dist_obs_dy ))

        cost_obs_penalty = jnp.linalg.norm(jnp.maximum(jnp.zeros(( self.initial_up_sampling*self.num_batch, self.num*(self.num_obs_1 + self.num_obs_2)  )), dist_obs), axis = 1)# jnp.linalg.norm(1.6*jnp.minimum(jnp.zeros(( self.initial_up_sampling*self.num_batch, self.num*self.num_obs  )), -dist_obs), axis = 1)

        idx_ellite = jnp.argsort(cost_obs_penalty)

        x_guess_sample = x_guess_sampling[idx_ellite[0:self.num_batch - self.num_sample_warm]]
        y_guess_sample = y_guess_sampling[idx_ellite[0:self.num_batch - self.num_sample_warm]]

        x_guess = jnp.vstack(( x_guess_sample, x_guess_per))
        y_guess = jnp.vstack(( y_guess_sample, y_guess_per))

        lincost_x = -jnp.dot(self.P_jax.T, x_guess.T).T
        lincost_y = -jnp.dot(self.P_jax.T, y_guess.T).T

        sol_x = jnp.dot(self.cost_mat_inv_x, -lincost_x.T).T
        sol_y = jnp.dot(self.cost_mat_inv_y, -lincost_y.T).T

        sol_x_bar = sol_x[:,0:self.nvar]
        sol_y_bar = sol_y[:,0:self.nvar]

        x_guess = jnp.dot(self.P_jax, sol_x_bar.T).T
        xdot_guess = jnp.dot(self.Pdot_jax, sol_x_bar.T).T
        xddot_guess = jnp.dot(self.Pddot_jax, sol_x_bar.T).T

        y_guess = jnp.dot(self.P_jax, sol_y_bar.T).T
        ydot_guess = jnp.dot(self.Pdot_jax, sol_y_bar.T).T
        yddot_guess = jnp.dot(self.Pddot_jax, sol_y_bar.T).T

        c_mean = jnp.mean( jnp.hstack(( sol_x_bar, sol_y_bar )), axis = 0  )
        c_cov = jnp.cov( jnp.hstack(( sol_x_bar, sol_y_bar )).T  )

        return sol_x_bar, sol_y_bar, x_guess, y_guess,  xdot_guess, ydot_guess, xddot_guess, yddot_guess,c_mean, c_cov, x_fin_path, y_fin_path

    ##################################################
    @partial(jit, static_argnums=(0,),backend="gpu")
    def compute_projection(self, b_eq_x, b_eq_y, sol_x_bar, sol_y_bar ):

        b_projection_x = sol_x_bar
        b_projection_y = sol_y_bar

        cost_mat = jnp.vstack(( jnp.hstack(( self.cost, self.A_eq.T )), jnp.hstack((self.A_eq, jnp.zeros(( 4, 4 )) )) ))
        cost_mat_inv = jnp.linalg.inv(cost_mat)

        lincost_x = - b_projection_x 
        lincost_y = - b_projection_y

        sol_x = jnp.dot(cost_mat_inv, jnp.hstack(( -lincost_x, b_eq_x )).T).T
        sol_y = jnp.dot(cost_mat_inv, jnp.hstack(( -lincost_y, b_eq_y )).T).T

        primal_sol_x = sol_x[:,0:self.nvar]
        primal_sol_y = sol_y[:,0:self.nvar]

        x = jnp.dot(self.P_jax, primal_sol_x.T).T
        xdot = jnp.dot(self.Pdot_jax, primal_sol_x.T).T
        xddot = jnp.dot(self.Pddot_jax, primal_sol_x.T).T

        y = jnp.dot(self.P_jax, primal_sol_y.T).T
        ydot = jnp.dot(self.Pdot_jax, primal_sol_y.T).T
        yddot = jnp.dot(self.Pddot_jax, primal_sol_y.T).T

        return primal_sol_x, primal_sol_y, x, y, xdot, ydot, xddot, yddot

   

   

    @partial(jit, static_argnums=(0,))
    def compute_cost_batch(self, x, y, xdot, ydot, xddot, yddot, x_project, y_project, x_fin, y_fin,  x_obs_trajectory, y_obs_trajectory, x_obs_trajectory_dy, y_obs_trajectory_dy):

        wc_alpha_temp = (x-x_obs_trajectory[:,jnp.newaxis])
        ws_alpha_temp = (y-y_obs_trajectory[:,jnp.newaxis])

        wc_alpha_temp_dy = (x-x_obs_trajectory_dy[:,jnp.newaxis])
        ws_alpha_temp_dy = (y-y_obs_trajectory_dy[:,jnp.newaxis])

        wc_alpha = wc_alpha_temp.transpose(1, 0, 2)
        ws_alpha = ws_alpha_temp.transpose(1, 0, 2)

        wc_alpha_dy = wc_alpha_temp_dy.transpose(1, 0, 2)
        ws_alpha_dy = ws_alpha_temp_dy.transpose(1, 0, 2)

        wc_alpha = wc_alpha.reshape(self.num_batch, self.num*self.num_obs_1)
        ws_alpha = ws_alpha.reshape(self.num_batch, self.num*self.num_obs_1)

        wc_alpha_dy = wc_alpha_dy.reshape(self.num_batch, self.num*self.num_obs_2)
        ws_alpha_dy = ws_alpha_dy.reshape(self.num_batch, self.num*self.num_obs_2)

        dist_obs_st = -wc_alpha**2/(self.a_obs_1**2)-ws_alpha**2/(self.b_obs_1**2)+1
        dist_obs_dy = -wc_alpha_dy**2/(self.a_obs_2**2)-ws_alpha_dy**2/(self.b_obs_2**2)+1

        dist_obs = jnp.hstack(( dist_obs_st, dist_obs_dy))

        clearance_cost = - jnp.min(dist_obs, axis = 1)
        cost_obs = jnp.linalg.norm(jnp.maximum(jnp.zeros(( self.num_batch, self.num*(self.num_obs_1+self.num_obs_2)  )), dist_obs), axis = 1)

        cost_smoothness =  ( (jnp.linalg.norm(xddot, axis = 1)) + (jnp.linalg.norm(yddot, axis = 1)) )
        cost_track = jnp.linalg.norm(x-x_project, axis = 1)+jnp.linalg.norm(y-y_project, axis = 1) #(x-x_project)**2+(y-y_project)**2
        cost_velocity = jnp.linalg.norm(  jnp.sqrt( xdot**2 + ydot**2**2) - self.v_max , axis = 1 )
        cost_acceleration = jnp.linalg.norm(  jnp.sqrt( xddot**2 + yddot**2**2) - self.a_max , axis = 1 )

        cost_batch =  0.15*cost_smoothness + 0.01* cost_track + 3.0*cost_obs + cost_velocity +1.0*cost_acceleration#+ 0.1 * cost_velocity#+ 0.1* cost_fin 

        return cost_batch
    #################################

    @partial(jit, static_argnums=(0,))
    def comp_prod(self, diffs, d ):

        term_1 = jnp.expand_dims(diffs, axis = 1)
        term_2 = jnp.expand_dims(diffs, axis = 0)

        prods = d * jnp.outer(term_1,term_2)

        return prods 

    ###########################################################
    @partial(jit, static_argnums=(0,))	
    def compute_shifted_samples(self, key, cost_batch, c_x_ellite, c_y_ellite, x_obs_trajectory, y_obs_trajectory, x_obs_trajectory_dy, y_obs_trajectory_dy, i, c_mean_prev, c_cov_prev):

        c_ellite = jnp.hstack(( c_x_ellite, c_y_ellite  ))

        beta_param = jnp.min(cost_batch)
        d = cost_batch 
        d = jnp.exp(-(1/self.lamda) * (d-beta_param) )
        sum_d= jnp.sum(d, axis = 0) 
        c_mean = (1- self.alpha) * c_mean_prev + self.alpha * jnp.sum( (c_ellite * d[:,jnp.newaxis]) , axis= 0)/ sum_d

        diffs = (c_ellite - c_mean)#.reshape(self.ellite_num_const * 2* self.nvar)
        prod_result = self.vec_product(diffs, d)
        c_cov =   (1- self.alpha) * c_cov_prev + self.alpha * ( (jnp.sum( prod_result , axis = 0)) / sum_d) #(( sum_d**2 - jnp.sum(d**2, axis = 0))/jnp.sum(d, axis = 0)))+0.01*jnp.identity(2*self.nvar)


        c_ellite_shift_temp = jax.random.multivariate_normal(key, c_mean, c_cov, (self.initial_up_sampling*self.num_batch, ))

        c_x_ellite_shift_temp = c_ellite_shift_temp[:, 0:11]
        c_y_ellite_shift_temp = c_ellite_shift_temp[:, 11:22]

        x_guess_temp = jnp.dot(self.P, c_x_ellite_shift_temp.T).T 
        y_guess_temp = jnp.dot(self.P, c_y_ellite_shift_temp.T).T 

        wc_alpha_temp = (x_guess_temp-x_obs_trajectory[:,jnp.newaxis])
        ws_alpha_temp = (y_guess_temp-y_obs_trajectory[:,jnp.newaxis])

        wc_alpha_temp_dy = (x_guess_temp-x_obs_trajectory_dy[:,jnp.newaxis])
        ws_alpha_temp_dy = (y_guess_temp-y_obs_trajectory_dy[:,jnp.newaxis])

        wc_alpha = wc_alpha_temp.transpose(1, 0, 2)
        ws_alpha = ws_alpha_temp.transpose(1, 0, 2)

        wc_alpha_dy = wc_alpha_temp_dy.transpose(1, 0, 2)
        ws_alpha_dy = ws_alpha_temp_dy.transpose(1, 0, 2)

        wc_alpha = wc_alpha.reshape(self.num_batch* self.initial_up_sampling, self.num*self.num_obs_1)
        ws_alpha = ws_alpha.reshape(self.num_batch* self.initial_up_sampling, self.num*self.num_obs_1)

        wc_alpha_dy = wc_alpha_dy.reshape(self.num_batch* self.initial_up_sampling, self.num*self.num_obs_2)
        ws_alpha_dy = ws_alpha_dy.reshape(self.num_batch* self.initial_up_sampling, self.num*self.num_obs_2)

        dist_obs_st = -wc_alpha**2/(self.a_obs_1**2)-ws_alpha**2/(self.b_obs_1**2)+1
        dist_obs_dy = -wc_alpha_dy**2/(self.a_obs_2**2)-ws_alpha_dy**2/(self.b_obs_2**2)+1

        dist_obs = jnp.hstack(( dist_obs_st, dist_obs_dy))

        cost_obs_penalty = jnp.linalg.norm(jnp.maximum(jnp.zeros(( self.initial_up_sampling*self.num_batch, self.num*(self.num_obs_1 +self.num_obs_2) )), dist_obs), axis = 1)#jnp.linalg.norm(1.6*jnp.minimum(jnp.zeros(( self.initial_up_sampling*self.num_batch, self.num*self.num_obs  )), -dist_obs), axis = 1)
        
        idx_ellite = jnp.argsort(cost_obs_penalty)

        c_x_ellite_shift = c_x_ellite_shift_temp[idx_ellite[0:(self.num_batch - self.ellite_num)]]
        c_y_ellite_shift = c_y_ellite_shift_temp[idx_ellite[0:(self.num_batch - self.ellite_num)]]

        sol_x_bar = jnp.vstack(( c_x_ellite, c_x_ellite_shift ))
        sol_y_bar = jnp.vstack(( c_y_ellite, c_y_ellite_shift ))

        x_guess = jnp.dot(self.P_jax, sol_x_bar.T).T
        xdot_guess = jnp.dot(self.Pdot_jax, sol_x_bar.T).T
        xddot_guess = jnp.dot(self.Pddot_jax, sol_x_bar.T).T

        y_guess = jnp.dot(self.P_jax, sol_y_bar.T).T
        ydot_guess = jnp.dot(self.Pdot_jax, sol_y_bar.T).T
        yddot_guess = jnp.dot(self.Pddot_jax, sol_y_bar.T).T
            
        return sol_x_bar, sol_y_bar, x_guess, y_guess, xdot_guess, ydot_guess, xddot_guess, yddot_guess, c_mean, c_cov
        

    @partial(jit, static_argnums=(0, ))
    def compute_controls(self, c_x_best, c_y_best, delta_t, theta_init):

        xdot_best = jnp.dot(self.Pdot_up_jax, c_x_best)
        ydot_best = jnp.dot(self.Pdot_up_jax, c_y_best)

        xddot_best = jnp.dot(self.Pddot_up_jax, c_x_best)
        yddot_best = jnp.dot(self.Pddot_up_jax, c_y_best)
        
        vx_control = jnp.mean(xdot_best[0:self.num_sample])
        vy_control = jnp.mean(ydot_best[0:self.num_sample])

        ax_control = jnp.mean(xddot_best[0:self.num_sample])#(xdot_best[self.num_sample] - xdot_best[0]) /  ((self.num_sample-1)*0.1) #
        ay_control = jnp.mean(yddot_best[0:self.num_sample])#(ydot_best[self.num_sample] - ydot_best[0]) /  ((self.num_sample-1)*0.1)#

        norm_v_t = jnp.sqrt(vx_control**2 + vy_control**2)
        angle_v_t = jnp.arctan2(vy_control, vx_control)
        angle_robot = theta_init

        zeta = angle_robot - angle_v_t
        v_t_control = norm_v_t * jnp.cos(zeta)
        omega_control = - zeta / (0.6)
       
        return vx_control, vy_control, ax_control, ay_control, norm_v_t, angle_v_t, v_t_control, omega_control

    @partial(jit, static_argnums=(0, ))	
    def compute_cem(self, key, initial_state, x_fin, y_fin, lamda_x, lamda_y, x_obs_trajectory, y_obs_trajectory, x_obs_trajectory_proj, y_obs_trajectory_proj,  x_obs_trajectory_dy, y_obs_trajectory_dy, sol_x_bar, sol_y_bar, x_guess, y_guess,  xdot_guess, ydot_guess, xddot_guess, yddot_guess, x_waypoint,  y_waypoint, arc_vec, c_mean, c_cov):
        
        c_mean_prev = c_mean
        c_cov_prev = c_cov
            
        for i in range(0, self.maxiter_cem):

            b_eq_x, b_eq_y = self.compute_boundary_vec(initial_state, x_fin, y_fin)

            c_x, c_y, x, y, xdot, ydot, xddot, yddot = self.compute_projection( b_eq_x, b_eq_y, sol_x_bar, sol_y_bar )

            x_project = jnp.linspace(initial_state[0], x_fin, self.num)
            y_project = jnp.linspace(initial_state[1], y_fin, self.num)
            

            cost_batch  = self.compute_cost_batch(x, y, xdot, ydot, xddot, yddot, x_project, y_project, x_fin, y_fin, x_obs_trajectory, y_obs_trajectory, x_obs_trajectory_dy, y_obs_trajectory_dy )

            idx_sort = jnp.argsort(cost_batch)

            c_x_ellite = c_x[idx_sort[0:self.ellite_num]]
            c_y_ellite = c_y[idx_sort[0:self.ellite_num]]

            x_ellite = x[idx_sort[0:self.ellite_num]]
            y_ellite = y[idx_sort[0:self.ellite_num]]

            cost_cem_ellite = cost_batch[idx_sort[0:self.ellite_num]]

            sol_x_bar, sol_y_bar, x_guess, y_guess, xdot_guess, ydot_guess, xddot_guess, yddot_guess, c_mean_prev, c_cov_prev = self.compute_shifted_samples(key, cost_cem_ellite, c_x_ellite, c_y_ellite,  x_obs_trajectory, y_obs_trajectory,  x_obs_trajectory_dy, y_obs_trajectory_dy, i, c_mean_prev, c_cov_prev)
            

        c_x_best = c_x_ellite[idx_sort[0]]
        c_y_best = c_y_ellite[idx_sort[0]]
        
        x_best = x[idx_sort[0]]
        y_best = y[idx_sort[0]]

        xdot_best = xdot[idx_sort[0]]
        ydot_best = ydot[idx_sort[0]]

        xddot_best = xddot[idx_sort[0]]
        yddot_best = yddot[idx_sort[0]]

        x_guess_per  = x_guess[idx_sort[0:self.num_sample_warm]]
        y_guess_per  = y_guess[idx_sort[0:self.num_sample_warm]]

        x_guess_per  = x_guess[idx_sort[0:self.num_sample_warm]]
        y_guess_per  = y_guess[idx_sort[0:self.num_sample_warm]]


        return  x_ellite, y_ellite, x, y, c_x_best, c_y_best, x_best, y_best, xdot_best, ydot_best, x_guess_per, y_guess_per, xddot_best, yddot_best