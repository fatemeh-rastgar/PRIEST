
import numpy as np
import jax.numpy as jnp
from functools import partial
import jax
from jax import jit, random
import bernstein_coeff_order10_arbitinterval
from jax.ops import index_update, index
import time
import matplotlib.pyplot as plt 
import bernstein_coeff_order10_arbitinterval




class CEM_optimizer():

    def __init__(self, a_obs_1, b_obs_1, c_obs_1, a_obs_2, b_obs_2, c_obs_2, a_obs_3, b_obs_3, c_obs_3, t_fin, nvar, num, num_batch, maxiter_cem, weight_smoothness, num_obs_1, num_obs_2, num_obs_3, way_point_shape, v_max, a_max ):

        self.t_fin = t_fin
        self.num = num
        self.nvar = nvar
        self.way_point_shape = way_point_shape
        self.num_batch = num_batch
        self.ellite_num = int(self.num_batch*0.6)
        self.ellite_num_const = self.num_batch - self.ellite_num 
        self.maxiter_cem = maxiter_cem
        self.num_obs_1 = num_obs_1
        self.num_obs_2 = num_obs_2
        self.num_obs_3 = num_obs_3

        self.a_obs_1 = a_obs_1
        self.b_obs_1 = b_obs_1
        self.c_obs_1 = c_obs_1
        self.a_obs_2 = a_obs_2
        self.b_obs_2 = b_obs_2
        self.c_obs_2 = c_obs_2
        self.a_obs_3 = a_obs_3
        self.b_obs_3 = b_obs_3
        self.c_obs_3 = c_obs_3

        self.num_up_sample = 1
        self.rho_proj = 1
        self.v_max = v_max
        self.a_max = a_max

        self.tot_time = np.linspace(0, self.t_fin, self.num)
        self.tot_time_copy = self.tot_time.reshape(self.num, 1)

        ########################################################
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
        self.initial_up_sampling = 30

        eps_k = np.random.multivariate_normal(mu, 0.01*cov, (self.num_batch,) )
        self.eps_k = jnp.asarray(eps_k)

        eps_k_up = np.random.multivariate_normal(mu, 0.01*cov, (int((self.num_batch * self.initial_up_sampling)), ))
        self.eps_k_up = jnp.asarray(eps_k_up)


        ###################################################
        self.P, self.Pdot, self.Pddot = bernstein_coeff_order10_arbitinterval.bernstein_coeff_order10_new(10, self.tot_time_copy[0], self.tot_time_copy[-1], self.tot_time_copy)
        self.P_jax, self.Pdot_jax, self.Pddot_jax = jnp.asarray(self.P), jnp.asarray(self.Pdot), jnp.asarray(self.Pddot)

        self.A_eq = jnp.vstack(( self.P_jax[0], self.Pdot_jax[0], self.Pddot_jax[0], self.P_jax[-1] ))
        self.A_projection = jnp.identity( self.nvar)	

        ####################################
        self.cost = jnp.identity(self.nvar)
        self.lamda = 0.9
        self.alpha = 0.7
        self.beta = 4

        ########################################################################
        self.vec_product = jit(jax.vmap(self.comp_prod, 0, out_axes=(0)))
        self.way_point_shape = way_point_shape
        self.vectorized_projection = jit(jax.vmap(self.compute_contouring_error, in_axes=(None, None, 0, 0, None, ) ))

        self.x_max = 5.8
        self.y_max = 5.8
        self.z_max = 2.5

        self.x_min = 0.4
        self.y_min = 0.4
        self.z_min = 0.2

        

    
    @partial(jit, static_argnums=(0,))
    def comp_prod(self, diffs, d ):

        term_1 = jnp.expand_dims(diffs, axis = 1)
        term_2 = jnp.expand_dims(diffs, axis = 0)

        prods = d * jnp.outer(term_1,term_2)

        return prods 
        
    ###########################################

    @partial(jit, static_argnums=(0,),backend="gpu")
    def path_spline(self, x_waypoint, y_waypoint, z_waypoint):

        x_diff = jnp.diff(x_waypoint)
        y_diff = jnp.diff(y_waypoint)
        z_diff = jnp.diff(z_waypoint)

        arc = jnp.cumsum(jnp.sqrt( x_diff**2 + y_diff**2+ z_diff**2))
        arc_length = arc[-1]

        arc_vec = jnp.linspace(0, arc_length, self.way_point_shape)

        return arc_length, arc_vec

    #####################################
    @partial(jit, static_argnums=(0,))
    def compute_contouring_error(self, x_waypoint, y_waypoint, z_waypoint, x_target_point, y_target_point, z_target_point, arc_vec):

        dist = jnp.sqrt( (x_waypoint - x_target_point)**2 + (y_waypoint - y_target_point)**2 + (z_waypoint - z_target_point)**2 )
        index = jnp.argmin(dist)

        arc_point = arc_vec[index]

        x_project = x_waypoint[index]
        y_project = y_waypoint[index]
        z_project = z_waypoint[index]

        return arc_point, x_project, y_project, z_project
        
    #######################################################

    @partial(jit, static_argnums=(0,),backend="gpu")	
    def compute_boundary_vec(self, initial_state, x_fin, y_fin, z_fin):		

        x_init, y_init, z_init, vx_init, vy_init, vz_init, ax_init, ay_init, az_init = initial_state

        x_init_vec = x_init*jnp.ones((self.num_batch, 1))
        y_init_vec = y_init*jnp.ones((self.num_batch, 1))
        z_init_vec = z_init*jnp.ones((self.num_batch, 1))  

        vx_init_vec = vx_init*jnp.ones((self.num_batch, 1))
        vy_init_vec = vy_init*jnp.ones((self.num_batch, 1))
        vz_init_vec = vz_init*jnp.ones((self.num_batch, 1))

        ax_init_vec = ax_init*jnp.ones((self.num_batch, 1))
        ay_init_vec = ay_init*jnp.ones((self.num_batch, 1))
        az_init_vec = az_init*jnp.ones((self.num_batch, 1))

        x_fin_vec = x_fin*jnp.ones((self.num_batch, 1))
        y_fin_vec = y_fin*jnp.ones((self.num_batch, 1)) 
        z_fin_vec = z_fin*jnp.ones((self.num_batch, 1)) 

        b_eq_x = jnp.hstack(( x_init_vec, vx_init_vec, ax_init_vec, x_fin_vec   ))
        b_eq_y = jnp.hstack(( y_init_vec, vy_init_vec, ay_init_vec, y_fin_vec  ))
        b_eq_z = jnp.hstack(( z_init_vec, vz_init_vec, az_init_vec, z_fin_vec  ))
		
        return b_eq_x, b_eq_y, b_eq_z

    #####################################################

    @partial(jit, static_argnums = (0,))
    def compute_projection(self, sol_x_bar, sol_y_bar, sol_z_bar, b_x_eq, b_y_eq, b_z_eq ):

        b_projection_x = sol_x_bar
        b_projection_y = sol_y_bar
        b_projection_z = sol_z_bar

        cost_mat = jnp.vstack(( jnp.hstack(( self.cost, self.A_eq.T )), jnp.hstack((self.A_eq, jnp.zeros(( 4, 4 )) )) ))
        cost_mat_inv = jnp.linalg.inv(cost_mat)
       
        lincost_x = - b_projection_x 
        lincost_y = - b_projection_y
        lincost_z = - b_projection_z

        sol_x = jnp.dot(cost_mat_inv, jnp.hstack(( -lincost_x, b_x_eq )).T ).T
        sol_y = jnp.dot(cost_mat_inv, jnp.hstack(( -lincost_y, b_y_eq )).T ).T
        sol_z = jnp.dot(cost_mat_inv, jnp.hstack(( -lincost_z, b_z_eq )).T ).T

        sol_x_samples = sol_x[:, 0:self.nvar]
        sol_y_samples = sol_y[:, 0:self.nvar]
        sol_z_samples = sol_z[:, 0:self.nvar]

        x_samples = jnp.dot(self.P_jax, sol_x_samples.T).T 
        y_samples = jnp.dot(self.P_jax, sol_y_samples.T).T
        z_samples = jnp.dot(self.P_jax, sol_z_samples.T).T


        xddot_samples = jnp.dot(self.Pddot_jax, sol_x_samples.T).T 
        yddot_samples = jnp.dot(self.Pddot_jax, sol_y_samples.T).T 
        zddot_samples = jnp.dot(self.Pddot_jax, sol_z_samples.T).T 

        xdot_samples = jnp.dot(self.Pdot_jax, sol_x_samples.T).T 
        ydot_samples = jnp.dot(self.Pdot_jax, sol_y_samples.T).T 
        zdot_samples = jnp.dot(self.Pdot_jax, sol_z_samples.T).T 

        return sol_x_samples, sol_y_samples, sol_z_samples, x_samples, y_samples, z_samples, xdot_samples, ydot_samples, zdot_samples, xddot_samples, yddot_samples, zddot_samples

    ######################### 
    @partial(jit, static_argnums=(0,),backend="gpu")
    def compute_contouring_error(self, x_waypoint, y_waypoint, z_waypoint, x_target_point, y_target_point, z_target_point, arc_vec):

        dist = jnp.sqrt( (x_waypoint - x_target_point)**2 + (y_waypoint - y_target_point)**2 + (z_waypoint - z_target_point)**2 )
        index = jnp.argmin(dist)

        arc_point = arc_vec[index]

        x_project = x_waypoint[index]
        y_project = y_waypoint[index]
        z_project = z_waypoint[index]

        return arc_point, x_project, y_project, z_project


    ####################################################################################
    @partial(jit, static_argnums = (0,) )
    def compute_cost_batch(self, x_samples, y_samples, z_samples,  xdot_samples, ydot_samples, zdot_samples, xddot_samples, yddot_samples, zddot_samples,  x_obs_trajectory_1, y_obs_trajectory_1, z_obs_trajectory_1,  x_obs_trajectory_2, y_obs_trajectory_2, z_obs_trajectory_2, x_obs_trajectory_3, y_obs_trajectory_3, z_obs_trajectory_3, x_project, y_project, z_project ):

        dist_x_temp_1 = (x_samples-x_obs_trajectory_1[:,jnp.newaxis])
        dist_y_temp_1 = (y_samples-y_obs_trajectory_1[:,jnp.newaxis])
        dist_z_temp_1 = (z_samples-z_obs_trajectory_1[:,jnp.newaxis])

        # dist_x_temp_2 = (x_samples-x_obs_trajectory_2[:,jnp.newaxis])
        # dist_y_temp_2 = (y_samples-y_obs_trajectory_2[:,jnp.newaxis])
        # dist_z_temp_2 = (z_samples-z_obs_trajectory_2[:,jnp.newaxis])

        # dist_x_temp_3 = (x_samples-x_obs_trajectory_3[:,jnp.newaxis])
        # dist_y_temp_3 = (y_samples-y_obs_trajectory_3[:,jnp.newaxis])
        # dist_z_temp_3 = (z_samples-z_obs_trajectory_3[:,jnp.newaxis])

        dist_x_1 = dist_x_temp_1.transpose(1, 0, 2)
        dist_y_1 = dist_y_temp_1.transpose(1, 0, 2)
        dist_z_1 = dist_z_temp_1.transpose(1, 0, 2)

        # dist_x_2 = dist_x_temp_2.transpose(1, 0, 2)
        # dist_y_2 = dist_y_temp_2.transpose(1, 0, 2)
        # dist_z_2 = dist_z_temp_2.transpose(1, 0, 2)

        # dist_x_3 = dist_x_temp_3.transpose(1, 0, 2)
        # dist_y_3 = dist_y_temp_3.transpose(1, 0, 2)
        # dist_z_3 = dist_z_temp_3.transpose(1, 0, 2)

        dist_x_1 = dist_x_1.reshape(self.num_batch, self.num*self.num_obs_1)
        dist_y_1 = dist_y_1.reshape(self.num_batch, self.num*self.num_obs_1)
        dist_z_1 = dist_z_1.reshape(self.num_batch, self.num*self.num_obs_1)

        # dist_x_2 = dist_x_2.reshape(self.num_batch, self.num*self.num_obs_2)
        # dist_y_2 = dist_y_2.reshape(self.num_batch, self.num*self.num_obs_2)
        # dist_z_2 = dist_z_2.reshape(self.num_batch, self.num*self.num_obs_2)

        # dist_x_3 = dist_x_3.reshape(self.num_batch, self.num*self.num_obs_3)
        # dist_y_3 = dist_y_3.reshape(self.num_batch, self.num*self.num_obs_3)
        # dist_z_3 = dist_z_3.reshape(self.num_batch, self.num*self.num_obs_3)

        dist_obs_1 = -(dist_x_1**2/(self.a_obs_1**2))-(dist_y_1**2/(self.b_obs_1**2))-((dist_z_1**2)/(self.c_obs_1**2))+1
        # dist_obs_2 = -(dist_x_2**2/(self.a_obs_2**2))-(dist_y_2**2/(self.b_obs_2**2))-((dist_z_2**2)/(self.c_obs_2**2))+1
        # dist_obs_3 = -(dist_x_3**2/(self.a_obs_3**2))-(dist_y_3**2/(self.b_obs_3**2))-((dist_z_3**2)/(self.c_obs_3**2))+1
        
        dist_obs =  dist_obs_1

        cost_obs = jnp.linalg.norm(jnp.maximum(jnp.zeros(( self.num_batch, self.num*(self.num_obs_1 )  )), dist_obs), axis = 1)

        cost_max_x = jnp.linalg.norm( jnp.maximum( jnp.zeros((self.num_batch, self.num)),  (x_samples-self.x_max) ), axis = 1)#+jnp.linalg.norm(y-y_project, axis = 1)+2*jnp.linalg.norm(z-z_project, axis = 1) 
        cost_max_y = jnp.linalg.norm( jnp.maximum( jnp.zeros((self.num_batch, self.num)),  (y_samples-self.y_max) ), axis = 1)
        cost_max_z = jnp.linalg.norm( jnp.maximum( jnp.zeros((self.num_batch, self.num)),  (z_samples-self.z_max) ), axis = 1)
        cost_max = cost_max_x + cost_max_y + cost_max_z

        cost_min_x = jnp.linalg.norm( jnp.maximum( jnp.zeros((self.num_batch, self.num)),  (-x_samples+self.x_min) ), axis = 1)#+jnp.linalg.norm(y-y_project, axis = 1)+2*jnp.linalg.norm(z-z_project, axis = 1) 
        cost_min_y = jnp.linalg.norm( jnp.maximum( jnp.zeros((self.num_batch, self.num)),  (-y_samples+self.y_min) ), axis = 1)
        cost_min_z = jnp.linalg.norm( jnp.maximum( jnp.zeros((self.num_batch, self.num)),  (-z_samples+self.z_min) ), axis = 1)
        cost_min = cost_min_x + cost_min_y + cost_min_z


        cost_smoothness =  ( (jnp.linalg.norm(xddot_samples, axis = 1)) + (jnp.linalg.norm(yddot_samples, axis = 1)) + (jnp.linalg.norm(zddot_samples, axis = 1)) )
        cost_track = jnp.linalg.norm(x_samples-x_project, axis = 1)+jnp.linalg.norm(y_samples-y_project, axis = 1)+4*jnp.linalg.norm(z_samples-z_project, axis = 1)
        cost_velocity = jnp.linalg.norm(  jnp.sqrt( xdot_samples**2 + ydot_samples**2+ zdot_samples**2) - self.v_max , axis = 1 )
        cost_acceleration = jnp.linalg.norm(  jnp.sqrt( xddot_samples**2 + yddot_samples**2+ zddot_samples**2) - self.a_max , axis = 1 )
        cost_batch = 1*cost_obs + 0.5* cost_smoothness+ 1.0*cost_velocity+0.0*cost_track +1.0*cost_acceleration + cost_max + cost_min  #+ clearance_cost # +0.2*cost_track

        return cost_batch



    #####################################

    @partial(jit, static_argnums=(0,))
    def compute_mean_cov(self, key, sol_x_ellite, sol_y_ellite, sol_z_ellite, cost_cem, sol_mean_prev, sol_cov_prev,x_obs_trajectory_1, y_obs_trajectory_1, z_obs_trajectory_1,  x_obs_trajectory_2, y_obs_trajectory_2, z_obs_trajectory_2, x_obs_trajectory_3, y_obs_trajectory_3, z_obs_trajectory_3  ):

        
        c_ellite = jnp.hstack(( sol_x_ellite, sol_y_ellite, sol_z_ellite  ))
        beta_param = jnp.min(cost_cem)
        d = cost_cem 
        d = jnp.exp(-(1/self.lamda) * (d-beta_param) )
        sum_d= jnp.sum(d, axis = 0) 
    
        sol_mean = (1- self.alpha) * sol_mean_prev + self.alpha * jnp.sum( (c_ellite * d[:,jnp.newaxis]) , axis= 0)/ sum_d#jnp.mean(c_ellite, axis = 0)#
        diffs = (c_ellite - sol_mean)
        prod_result = self.vec_product(diffs, d)
        sol_cov =   (1- self.alpha) * sol_cov_prev + self.alpha * ( (jnp.sum( prod_result , axis = 0)) / sum_d) 
        
        c_ellite_shift_temp = jax.random.multivariate_normal(key, sol_mean, sol_cov, (self.initial_up_sampling*self.num_batch, ),  method='svd')

        c_x_ellite_shift_temp = c_ellite_shift_temp[:, 0:11]
        c_y_ellite_shift_temp = c_ellite_shift_temp[:, 11:22]
        c_z_ellite_shift_temp = c_ellite_shift_temp[:, 22:33]

        x_guess_temp = jnp.dot(self.P, c_x_ellite_shift_temp.T).T 
        y_guess_temp = jnp.dot(self.P, c_y_ellite_shift_temp.T).T 
        z_guess_temp = jnp.dot(self.P, c_z_ellite_shift_temp.T).T 

        x_guess_sampling = x_guess_temp 
        y_guess_sampling = y_guess_temp 
        z_guess_sampling = z_guess_temp

        x_dist_temp_1 = (x_guess_sampling-x_obs_trajectory_1[:,jnp.newaxis])
        y_dist_temp_1 = (y_guess_sampling-y_obs_trajectory_1[:,jnp.newaxis])
        z_dist_temp_1 = (z_guess_sampling-z_obs_trajectory_1[:,jnp.newaxis])

        x_dist_temp_2 = (x_guess_sampling-x_obs_trajectory_2[:,jnp.newaxis])
        y_dist_temp_2 = (y_guess_sampling-y_obs_trajectory_2[:,jnp.newaxis])
        z_dist_temp_2 = (z_guess_sampling-z_obs_trajectory_2[:,jnp.newaxis])

        x_dist_temp_3 = (x_guess_sampling-x_obs_trajectory_3[:,jnp.newaxis])
        y_dist_temp_3 = (y_guess_sampling-y_obs_trajectory_3[:,jnp.newaxis])
        z_dist_temp_3 = (z_guess_sampling-z_obs_trajectory_3[:,jnp.newaxis])

        x_dist_1 = x_dist_temp_1.transpose(1, 0, 2)
        y_dist_1 = y_dist_temp_1.transpose(1, 0, 2)
        z_dist_1 = z_dist_temp_1.transpose(1, 0, 2)

        x_dist_2 = x_dist_temp_2.transpose(1, 0, 2)
        y_dist_2 = y_dist_temp_2.transpose(1, 0, 2)
        z_dist_2 = z_dist_temp_2.transpose(1, 0, 2)

        x_dist_3 = x_dist_temp_3.transpose(1, 0, 2)
        y_dist_3 = y_dist_temp_3.transpose(1, 0, 2)
        z_dist_3 = z_dist_temp_3.transpose(1, 0, 2)

        x_dist_1 = x_dist_1.reshape(self.num_batch * self.initial_up_sampling, self.num*self.num_obs_1)
        y_dist_1 = y_dist_1.reshape(self.num_batch * self.initial_up_sampling, self.num*self.num_obs_1)
        z_dist_1 = z_dist_1.reshape(self.num_batch * self.initial_up_sampling, self.num*self.num_obs_1)

        x_dist_2 = x_dist_2.reshape(self.num_batch * self.initial_up_sampling, self.num*self.num_obs_2)
        y_dist_2 = y_dist_2.reshape(self.num_batch * self.initial_up_sampling, self.num*self.num_obs_2)
        z_dist_2 = z_dist_2.reshape(self.num_batch * self.initial_up_sampling, self.num*self.num_obs_2)

        x_dist_3 = x_dist_3.reshape(self.num_batch * self.initial_up_sampling, self.num*self.num_obs_3)
        y_dist_3 = y_dist_3.reshape(self.num_batch * self.initial_up_sampling, self.num*self.num_obs_3)
        z_dist_3 = z_dist_3.reshape(self.num_batch * self.initial_up_sampling, self.num*self.num_obs_3)

        dist_obs_1 = -(x_dist_1**2/(self.a_obs_1**2))-(y_dist_1**2/(self.b_obs_1**2))-((z_dist_1**2)/(self.c_obs_1**2))+1
        dist_obs_2 = -(x_dist_2**2/(self.a_obs_2**2))-(y_dist_2**2/(self.b_obs_2**2))-((z_dist_2**2)/(self.c_obs_2**2))+1
        dist_obs_3 = -(x_dist_3**2/(self.a_obs_3**2))-(y_dist_3**2/(self.b_obs_3**2))-((z_dist_3**2)/(self.c_obs_3**2))+1
        
        dist_obs = jnp.hstack(( dist_obs_1, dist_obs_2, dist_obs_3))
        cost_obs_penalty = jnp.linalg.norm(jnp.maximum(jnp.zeros(( self.initial_up_sampling*self.num_batch, self.num*(self.num_obs_1 + self.num_obs_2 + self.num_obs_3)  )), dist_obs), axis = 1)

        idx_ellite = jnp.argsort(cost_obs_penalty)

        c_x_ellite_shift = c_x_ellite_shift_temp[idx_ellite[0:self.ellite_num_const]]
        c_y_ellite_shift = c_y_ellite_shift_temp[idx_ellite[0:self.ellite_num_const]]
        c_z_ellite_shift = c_z_ellite_shift_temp[idx_ellite[0:self.ellite_num_const]]

        sol_x_bar = jnp.vstack(( sol_x_ellite, c_x_ellite_shift ))
        sol_y_bar = jnp.vstack(( sol_y_ellite, c_y_ellite_shift ))
        sol_z_bar = jnp.vstack(( sol_z_ellite, c_z_ellite_shift ))

        x_sample = jnp.dot(self.P_jax, sol_x_bar.T).T
        xdot_sample = jnp.dot(self.Pdot_jax, sol_x_bar.T).T
        xddot_sample = jnp.dot(self.Pddot_jax, sol_x_bar.T).T

        y_sample = jnp.dot(self.P_jax, sol_y_bar.T).T
        ydot_sample = jnp.dot(self.Pdot_jax, sol_y_bar.T).T
        yddot_sample = jnp.dot(self.Pddot_jax, sol_y_bar.T).T

        z_sample = jnp.dot(self.P_jax, sol_z_bar.T).T
        zdot_sample = jnp.dot(self.Pdot_jax, sol_z_bar.T).T
        zddot_sample = jnp.dot(self.Pddot_jax, sol_z_bar.T).T


        return sol_mean, sol_cov, x_sample, y_sample, z_sample, xdot_sample, ydot_sample, zdot_sample, xddot_sample, yddot_sample, zddot_sample, sol_x_bar, sol_y_bar, sol_z_bar


    #################################################################

    @partial(jit, static_argnums = (0,) )
    def compute_cem_part(self, initial_state,  x_obs_trajectory_1, y_obs_trajectory_1, z_obs_trajectory_1, x_obs_trajectory_2, y_obs_trajectory_2, z_obs_trajectory_2, x_obs_trajectory_3, y_obs_trajectory_3, z_obs_trajectory_3, x_fin , y_fin, z_fin, sol_x_bar, sol_y_bar, sol_z_bar, sol_mean, sol_cov, x_guess, y_guess, z_guess, arc_vec):

                                 
        key = random.PRNGKey(0)

        b_x_eq, b_y_eq, b_z_eq = self.compute_boundary_vec(initial_state, x_fin, y_fin, z_fin)

        for i in range(0, self.maxiter_cem):

            key, subkey = random.split(key)

            sol_x_samples, sol_y_samples, sol_z_samples, x_samples, y_samples, z_samples, xdot_samples, ydot_samples, zdot_samples, xddot_samples, yddot_samples, zddot_samples = self.compute_projection(sol_x_bar, sol_y_bar, sol_z_bar, b_x_eq, b_y_eq, b_z_eq)

            x_sample_flat = x_samples.reshape(self.num_batch, self.num)
            y_sample_flat = y_samples.reshape(self.num_batch, self.num)
            z_sample_flat = z_samples.reshape(self.num_batch, self.num)

            x_project = jnp.linspace(initial_state[0], x_fin, self.num)
            y_project = jnp.linspace(initial_state[1], y_fin, self.num)
            z_project = jnp.linspace(initial_state[2], z_fin, self.num)

            cost_cem  = self.compute_cost_batch(x_samples, y_samples, z_samples,  xdot_samples, ydot_samples, zdot_samples, xddot_samples, yddot_samples, zddot_samples,  x_obs_trajectory_1, y_obs_trajectory_1, z_obs_trajectory_1, x_obs_trajectory_2, y_obs_trajectory_2, z_obs_trajectory_2, x_obs_trajectory_3, y_obs_trajectory_3, z_obs_trajectory_3, x_project, y_project, z_project )
            
            idx = jnp.argsort(cost_cem)

            sol_x_ellite = sol_x_samples[idx[0:self.ellite_num]]
            sol_y_ellite = sol_y_samples[idx[0:self.ellite_num]]
            sol_z_ellite = sol_z_samples[idx[0:self.ellite_num]]

            cost_cem_ellite = cost_cem[idx[0:self.ellite_num]]

            sol_mean, sol_cov, x_sample, y_sample, z_sample, xdot_sample, ydot_sample, zdot_sample, xddot_sample, yddot_sample, zddot_sample, sol_x_bar, sol_y_bar, sol_z_bar = self.compute_mean_cov(key, sol_x_ellite, sol_y_ellite, sol_z_ellite, cost_cem_ellite, sol_mean, sol_cov,  x_obs_trajectory_1, y_obs_trajectory_1, z_obs_trajectory_1, x_obs_trajectory_2, y_obs_trajectory_2, z_obs_trajectory_2, x_obs_trajectory_3, y_obs_trajectory_3, z_obs_trajectory_3)		
            

            x_sample_best = x_samples[idx[0]]
            y_sample_best = y_samples[idx[0]]	
            z_sample_best = z_samples[idx[0]]	

            x_samples = x_samples[idx[0:self.ellite_num]]
            xdot_samples = xdot_samples[idx[0:self.ellite_num]]
            xddot_samples = xddot_samples[idx[0:self.ellite_num]]

            y_samples = y_samples[idx[0:self.ellite_num]]
            ydot_samples = ydot_samples[idx[0:self.ellite_num]]
            yddot_samples = yddot_samples[idx[0:self.ellite_num]]

            z_samples = z_samples[idx[0:self.ellite_num]]
            zdot_samples = zdot_samples[idx[0:self.ellite_num]]
            zddot_samples = zddot_samples[idx[0:self.ellite_num]]


            

        return x_sample_best, y_sample_best, z_sample_best, x_samples, y_samples, z_samples, xdot_samples, ydot_samples, zdot_samples, xddot_samples, yddot_samples, zddot_samples
        

