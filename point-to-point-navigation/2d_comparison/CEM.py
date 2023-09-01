
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

    def __init__(self, a_obs, b_obs, t_fin, nvar, num, num_batch, maxiter_cem, weight_smoothness, num_obs,way_point_shape, v_max):

        self.t_fin = t_fin
        self.num = num
        self.nvar = nvar
        self.way_point_shape = way_point_shape
        self.num_batch = num_batch
        self.ellite_num = int(self.num_batch*0.6)
        self.ellite_num_const = self.num_batch - self.ellite_num 
        self.maxiter_cem = maxiter_cem
        self.num_obs = num_obs
        self.a_obs = a_obs
        self.b_obs = b_obs
        self.num_up_sample = 1
        self.rho_proj = 1
        self.v_max = v_max

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
        

    

    @partial(jit, static_argnums=(0,))
    def compute_initial_guess(self, x_obs_trajectory, y_obs_trajectory, x_waypoint, y_waypoint, v_des, initial_state, x_fin, y_fin ):

        x_fin = x_fin#x_waypoint[index_final_path]
        y_fin = y_fin#[index_final_path]

        #x_fin = initial_state[0]+ vx_des * self.t_fin
        #y_fin = initial_state[1]+ vy_des * self.t_fin

        x_interp = jnp.linspace(initial_state[0], x_fin, self.num)
        y_interp = jnp.linspace(initial_state[1], y_fin, self.num)
        
        x_samples_init =  0.0*self.eps_k_up + x_interp
        y_samples_init =  1.0*self.eps_k_up + y_interp 
        
        wc_alpha_temp = (x_samples_init - x_obs_trajectory[:,jnp.newaxis])
        ws_alpha_temp = (y_samples_init - y_obs_trajectory[:,jnp.newaxis])
       
        wc_alpha = wc_alpha_temp.transpose(1, 0, 2)
        ws_alpha = ws_alpha_temp.transpose(1, 0, 2)
       
        wc_alpha = wc_alpha.reshape(self.num_batch * self.initial_up_sampling, self.num*self.num_obs)
        ws_alpha = ws_alpha.reshape(self.num_batch * self.initial_up_sampling, self.num*self.num_obs)
        
        dist_obs = -wc_alpha**2/(self.a_obs**2)-ws_alpha**2/(self.b_obs**2)+1

        cost_obs_penalty = jnp.linalg.norm(jnp.maximum(jnp.zeros(( self.initial_up_sampling*self.num_batch, self.num*self.num_obs  )), dist_obs), axis = 1)

        idx_ellite = jnp.argsort(cost_obs_penalty)

        x_guess = x_samples_init[idx_ellite[0:self.num_batch]]
        y_guess = y_samples_init[idx_ellite[0:self.num_batch]]
       
        cost = jnp.dot(self.P_jax.T, self.P_jax) #+ 0.0001 * jnp.identity(self.nvar) + 1*jnp.dot(self.Pddot_jax.T, self.Pddot_jax)

        lincost_x = -jnp.dot(self.P_jax.T, x_guess.T).T
        lincost_y = -jnp.dot(self.P_jax.T, y_guess.T).T

        cost_mat_inv = jnp.linalg.inv(cost)

        sol_x = jnp.dot(cost_mat_inv, -lincost_x.T).T
        sol_y = jnp.dot(cost_mat_inv, -lincost_y.T).T

        x_guess = jnp.dot(self.P_jax, sol_x.T).T
        y_guess = jnp.dot(self.P_jax, sol_y.T).T

        sol_mean = jnp.mean(jnp.hstack(( sol_x, sol_y)), axis = 0)
        sol_cov = jnp.cov( jnp.hstack(( sol_x, sol_y)).T )
        
        return x_fin, y_fin, sol_x, sol_y, sol_mean, sol_cov, x_guess, y_guess


    	
    @partial(jit, static_argnums=(0,))
    def comp_prod(self, diffs, d ):

        term_1 = jnp.expand_dims(diffs, axis = 1)
        term_2 = jnp.expand_dims(diffs, axis = 0)

        prods = d * jnp.outer(term_1,term_2)

        return prods 
        
    ###########################################

    @partial(jit, static_argnums=(0,))
    def path_spline(self, x_waypoint, y_waypoint):

        x_diff = jnp.diff(x_waypoint)
        y_diff = jnp.diff(y_waypoint)

        arc = jnp.cumsum(jnp.sqrt( x_diff**2 + y_diff**2 ))
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
        
    ############3

    @partial(jit, static_argnums=(0,))
    def compute_obs_traj_prediction(self, x_obs_init, y_obs_init, vx_obs, vy_obs):

        x_temp = x_obs_init+vx_obs*self.tot_time[:,jnp.newaxis]
        x_obs_trajectory = x_temp.T 

        y_temp = y_obs_init+vy_obs*self.tot_time[:,jnp.newaxis]
        y_obs_trajectory = y_temp.T

        #Dist = jnp.sqrt( (x_init - x_obs_trajectory[:,0])**2 +(y_init - y_obs_trajectory[:,0])**2+(z_init - z_obs_trajectory[:,0])**2 ) 
        #idx_dis = jnp.argsort(Dist)

        return x_obs_trajectory, y_obs_trajectory


    #######################################################

    @partial(jit, static_argnums=(0,))	
    def compute_boundary_vec(self, initial_state, x_fin, y_fin ):		

        x_init, y_init, vx_init, vy_init, ax_init, ay_init = initial_state

        x_init_vec = x_init*jnp.ones((self.num_batch, 1))
        y_init_vec = y_init*jnp.ones((self.num_batch, 1)) 

        vx_init_vec = vx_init*jnp.ones((self.num_batch, 1))
        vy_init_vec = vy_init*jnp.ones((self.num_batch, 1))

        ax_init_vec = ax_init*jnp.ones((self.num_batch, 1))
        ay_init_vec = ay_init*jnp.ones((self.num_batch, 1))

        x_fin_vec = x_fin*jnp.ones((self.num_batch, 1))
        y_fin_vec = y_fin*jnp.ones((self.num_batch, 1)) 

        b_eq_x = jnp.hstack(( x_init_vec, vx_init_vec, ax_init_vec, x_fin_vec ))
        b_eq_y = jnp.hstack(( y_init_vec, vy_init_vec, ay_init_vec, y_fin_vec  ))

        return b_eq_x, b_eq_y
    
    ##################################################

    #@partial(jit,static_argnums=(0, ) )
    #def  compute_ellite_samples(self, key, sol_x_mean, sol_y_mean, cov_x, cov_y):

        #sol_x_samples = jax.random.multivariate_normal(key, sol_x_mean, cov_x + 0.001*jnp.identity(11), (self.num_batch, ))
        #sol_y_samples = jax.random.multivariate_normal(key, sol_y_mean, cov_y + 0.001*jnp.identity(11), (self.num_batch, ))

        #return sol_x_samples, sol_y_samples

    #################################################

    #@partial(jit,static_argnums=(0, ) )
    #def  compute_ellite_samples(self, key, sol_mean, sol_cov):

        #sol_samples = jax.random.multivariate_normal(key, sol_mean, sol_cov + 0.01*jnp.identity(2*self.nvar), (self.num_batch, ))

        #sol_x_samples = sol_samples[:, 0:11]
        #sol_y_samples = sol_samples[:, 11:22]

        #return sol_x_samples, sol_y_samples

    #####################################################

    @partial(jit, static_argnums = (0,))
    def compute_projection(self, sol_x_bar, sol_y_bar, b_x_eq, b_y_eq ):

        b_projection_x = sol_x_bar
        b_projection_y = sol_y_bar

        cost_mat = jnp.vstack(( jnp.hstack(( self.cost, self.A_eq.T )), jnp.hstack((self.A_eq, jnp.zeros(( 4, 4 )) )) ))
        cost_mat_inv = jnp.linalg.inv(cost_mat)
       
        lincost_x = - b_projection_x 
        lincost_y = - b_projection_y

        sol_x = jnp.dot(cost_mat_inv, jnp.hstack(( -lincost_x, b_x_eq )).T ).T
        sol_y = jnp.dot(cost_mat_inv, jnp.hstack(( -lincost_y, b_y_eq )).T ).T

        sol_x_samples = sol_x[:, 0:self.nvar]
        sol_y_samples = sol_y[:, 0:self.nvar]

        x_samples = jnp.dot(self.P_jax, sol_x_samples.T).T 
        y_samples = jnp.dot(self.P_jax, sol_y_samples.T).T

        xddot_samples = jnp.dot(self.Pddot_jax, sol_x_samples.T).T 
        yddot_samples = jnp.dot(self.Pddot_jax, sol_y_samples.T).T 

        xdot_samples = jnp.dot(self.Pdot_jax, sol_x_samples.T).T 
        ydot_samples = jnp.dot(self.Pdot_jax, sol_y_samples.T).T 

        return sol_x_samples, sol_y_samples, x_samples, y_samples, xdot_samples, ydot_samples, xddot_samples, yddot_samples

    ######################### 
    @partial(jit, static_argnums=(0,),backend="gpu")
    def compute_contouring_error(self, x_waypoint, y_waypoint, x_target_point, y_target_point, arc_vec):

        dist = jnp.sqrt( (x_waypoint - x_target_point)**2 + (y_waypoint - y_target_point)**2 )
        index = jnp.argmin(dist)

        arc_point = arc_vec[index]

        x_project = x_waypoint[index]
        y_project = y_waypoint[index]

        return arc_point, x_project, y_project
    ####################################################################################
    @partial(jit, static_argnums = (0,) )
    def compute_cost_batch(self, x_samples, y_samples,  xdot_samples, ydot_samples, xddot_samples, yddot_samples, x_obs, y_obs, x_project, y_project ):

        dist_x_temp = (x_samples-x_obs[:,jnp.newaxis])
        dist_y_temp = (y_samples-y_obs[:,jnp.newaxis])

        dist_x = ( dist_x_temp.transpose(1, 0, 2) )#[:, :, 0:30]
        dist_y = ( dist_y_temp.transpose(1, 0, 2) )#[:, :, 0:30]

        dist_x = dist_x.reshape(self.num_batch, self.num*self.num_obs)
        dist_y = dist_y.reshape(self.num_batch, self.num*self.num_obs)

        dist_obs = (-(dist_x**2)/(self.a_obs**2)- (dist_y**2)/(self.b_obs**2)+1) 
        cost_obs = jnp.linalg.norm(jnp.maximum(jnp.zeros(( self.num_batch, self.num*self.num_obs  )), dist_obs), axis = 1)
        cost_smoothness =  ( (jnp.linalg.norm(xddot_samples, axis = 1)) + (jnp.linalg.norm(yddot_samples, axis = 1)) )
        cost_track = jnp.linalg.norm(x_samples-x_project, axis = 1)+jnp.linalg.norm(y_samples-y_project, axis = 1)
        clearance_cost = - jnp.min(dist_obs, axis = 1)
        cost_velocity = jnp.linalg.norm(  jnp.sqrt( xdot_samples**2 + ydot_samples**2) - self.v_max , axis = 1 )
        cost_batch = 1*cost_obs + 1* cost_smoothness+ 0.25*cost_velocity+0.0*cost_track  #+ clearance_cost # +0.2*cost_track

        return cost_batch



    #####################################

    @partial(jit, static_argnums=(0,))
    def compute_mean_cov(self, key, sol_x_ellite, sol_y_ellite, cost_cem, sol_mean_prev, sol_cov_prev,x_obs_trajectory, y_obs_trajectory  ):

        
        c_ellite = jnp.hstack(( sol_x_ellite, sol_y_ellite  ))

        beta_param = jnp.min(cost_cem)
        d = cost_cem 
        d = jnp.exp(-(1/self.lamda) * (d-beta_param) )
        sum_d= jnp.sum(d, axis = 0) 
    
        sol_mean = (1- self.alpha) * sol_mean_prev + self.alpha * jnp.sum( (c_ellite * d[:,jnp.newaxis]) , axis= 0)/ sum_d#jnp.mean(c_ellite, axis = 0)#

        diffs = (c_ellite - sol_mean)#.reshape(self.ellite_num_const * 2* self.nvar)
        prod_result = self.vec_product(diffs, d)
        sol_cov =   (1- self.alpha) * sol_cov_prev + self.alpha * ( (jnp.sum( prod_result , axis = 0)) / sum_d) #jnp.cov(c_ellite.T)##(( sum_d**2 - jnp.sum(d**2, axis = 0))/jnp.sum(d, axis = 0)))+0.01*jnp.identity(2*self.nvar)

        
        c_ellite_shift_temp = jax.random.multivariate_normal(key, sol_mean, sol_cov, (self.initial_up_sampling*self.num_batch, ),  method='svd')

        c_x_ellite_shift_temp = c_ellite_shift_temp[:, 0:11]
        c_y_ellite_shift_temp = c_ellite_shift_temp[:, 11:22]

        x_guess_temp = jnp.dot(self.P, c_x_ellite_shift_temp.T).T 
        y_guess_temp = jnp.dot(self.P, c_y_ellite_shift_temp.T).T 

        wc_alpha_temp = (x_guess_temp-x_obs_trajectory[:,jnp.newaxis])
        ws_alpha_temp = (y_guess_temp-y_obs_trajectory[:,jnp.newaxis])

        wc_alpha = wc_alpha_temp.transpose(1, 0, 2)
        ws_alpha = ws_alpha_temp.transpose(1, 0, 2)

        wc_alpha = wc_alpha.reshape(self.num_batch * self.initial_up_sampling, self.num*self.num_obs)
        ws_alpha = ws_alpha.reshape(self.num_batch * self.initial_up_sampling, self.num*self.num_obs)

        dist_obs = -wc_alpha**2/(self.a_obs**2)-ws_alpha**2/(self.b_obs**2)+1

        cost_obs_penalty = jnp.linalg.norm(jnp.maximum(jnp.zeros(( self.initial_up_sampling*self.num_batch, self.num*self.num_obs  )), dist_obs), axis = 1)

        idx_ellite = jnp.argsort(cost_obs_penalty)

        c_x_ellite_shift = c_x_ellite_shift_temp[idx_ellite[0:self.ellite_num_const]]
        c_y_ellite_shift = c_y_ellite_shift_temp[idx_ellite[0:self.ellite_num_const]]

        sol_x_bar = jnp.vstack(( sol_x_ellite, c_x_ellite_shift ))
        sol_y_bar = jnp.vstack(( sol_y_ellite, c_y_ellite_shift ))

        x_sample = jnp.dot(self.P_jax, sol_x_bar.T).T
        xdot_sample = jnp.dot(self.Pdot_jax, sol_x_bar.T).T
        xddot_sample = jnp.dot(self.Pddot_jax, sol_x_bar.T).T

        y_sample = jnp.dot(self.P_jax, sol_y_bar.T).T
        ydot_sample = jnp.dot(self.Pdot_jax, sol_y_bar.T).T
        yddot_sample = jnp.dot(self.Pddot_jax, sol_y_bar.T).T


        return sol_mean, sol_cov, x_sample, y_sample, xdot_sample, ydot_sample, xddot_sample, yddot_sample, sol_x_bar, sol_y_bar

    


    #################################################################

    @partial(jit, static_argnums = (0,) )
    def compute_cem_part(self, initial_state, x_obs, y_obs, vx_des, vy_des, v_des, x_waypoint, y_waypoint, x_obs_trajectory, y_obs_trajectory, lamda_x, lamda_y, x_fin , y_fin, sol_x_bar, sol_y_bar, sol_mean, sol_cov, x_guess, y_guess, arc_vec):

        key = random.PRNGKey(0)

        # x_fin, y_fin, sol_x_bar, sol_y_bar, sol_mean, sol_cov, x_samples, y_samples  = self.compute_initial_guess(x_obs_trajectory, y_obs_trajectory, x_waypoint, y_waypoint, v_des, initial_state, x_fin, y_fin )

        b_x_eq, b_y_eq = self.compute_boundary_vec(initial_state, x_fin, y_fin)

        for i in range(0, self.maxiter_cem):

            key, subkey = random.split(key)

            sol_x_samples, sol_y_samples, x_samples, y_samples, xdot_samples, ydot_samples, xddot_samples, yddot_samples = self.compute_projection(sol_x_bar, sol_y_bar, b_x_eq, b_y_eq)

            x_sample_flat = x_samples.reshape(self.num_batch, self.num)
            y_sample_flat = y_samples.reshape(self.num_batch, self.num)

            x_project = jnp.linspace(initial_state[0], x_fin, self.num)
            y_project = jnp.linspace(initial_state[1], y_fin, self.num)

            cost_cem  = self.compute_cost_batch(x_samples, y_samples,  xdot_samples, ydot_samples,xddot_samples, yddot_samples, x_obs, y_obs, x_project, y_project )
            
            idx = jnp.argsort(cost_cem)

            sol_x_ellite = sol_x_samples[idx[0:self.ellite_num]]
            sol_y_ellite = sol_y_samples[idx[0:self.ellite_num]]

            cost_cem_ellite = cost_cem[idx[0:self.ellite_num]]

            sol_mean, sol_cov, x_sample, y_sample, xdot_sample, ydot_sample, xddot_sample, yddot_sample, sol_x_bar, sol_y_bar = self.compute_mean_cov(key, sol_x_ellite, sol_y_ellite, cost_cem_ellite, sol_mean, sol_cov, x_obs_trajectory, y_obs_trajectory)		
            
            #sol_x_mean, sol_y_mean, cov_x, cov_y = self.compute_mean_cov( sol_x_ellite, sol_y_ellite )	
            x_sample_best = x_samples[idx[0]]
            y_sample_best = y_samples[idx[0]]	

            x_samples = x_samples[idx[0:self.ellite_num]]
            xdot_samples = xdot_samples[idx[0:self.ellite_num]]
            xddot_samples = xddot_samples[idx[0:self.ellite_num]]

            y_samples = y_samples[idx[0:self.ellite_num]]
            ydot_samples = ydot_samples[idx[0:self.ellite_num]]
            yddot_samples = yddot_samples[idx[0:self.ellite_num]]
            

        return x_sample_best, y_sample_best, x_samples, y_samples, xdot_samples, ydot_samples, xddot_samples, yddot_samples
        

