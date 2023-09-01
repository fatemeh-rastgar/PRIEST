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
from mayavi import mlab




class batch_crowd_nav():

    def __init__(self, a_obs_1, b_obs_1, c_obs_1, a_obs_2, b_obs_2, c_obs_2,  a_obs_3, b_obs_3, c_obs_3, v_max, v_min, a_max, num_obs_1, num_obs_2, num_obs_3, t_fin, num, num_batch, maxiter, maxiter_cem, weight_smoothness, weight_track, way_point_shape, v_des):

        
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
        self.c_obs_1 = c_obs_1
        self.a_obs_2 = a_obs_2
        self.b_obs_2 = b_obs_2
        self.c_obs_2 = c_obs_2
        self.a_obs_3 = a_obs_3
        self.b_obs_3 = b_obs_3
        self.c_obs_3 = c_obs_3

        self.t_fin = t_fin
        self.num = num
        self.t = self.t_fin/self.num
        self.num_batch = num_batch		# number of goals
        self.ellite_num_const = 80
        self.ellite_num = 20

        self.num_obs_1 = num_obs_1
        self.num_obs_2 = num_obs_2
        self.num_obs_3 = num_obs_3

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
        self.A_obs = jnp.tile(self.P_jax, (self.num_obs_1, 1))

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

        ################ proximal weights
        self.rho_obs = 1.0
        self.rho_ineq = 1.0
        self.rho_track = 1.0
        self.maxitet_proj = 13
        self.rho_proj = 1

        self.beta = 4
        self.lamda = 0.9
        self.alpha = 0.7
        self.vec_product = jit(jax.vmap(self.comp_prod, 0, out_axes=(0)))
        self.vectorized_projection = jit(jax.vmap(self.compute_contouring_error, in_axes=(None,None, None, 0, 0, 0, None, ) ))
        self.way_point_shape = way_point_shape
        ###############

        self.cost_x =  jnp.dot(self.P_jax.T, self.P_jax) + 0.0001 * jnp.identity(self.nvar) + 1*jnp.dot(self.Pddot_jax.T, self.Pddot_jax)
        self.cost_y = self.cost_x
        self.cost_mat_x = self.cost_x
        self.cost_mat_inv_x = jnp.linalg.inv(self.cost_mat_x)
        self.cost_mat_y = self.cost_y
        self.cost_mat_inv_y = jnp.linalg.inv(self.cost_mat_y)
        ######################################
        self.A_bound = jnp.vstack(( self.P_jax, -self.P_jax))
        self.rho_b  = 0.00

        self.cost_x_projection = self.rho_proj * jnp.dot( self.A_projection.T, self.A_projection) + self.rho_obs*jnp.dot(self.A_obs.T, self.A_obs)+self.rho_ineq*jnp.dot(self.A_acc.T, self.A_acc)+self.rho_ineq*jnp.dot(self.A_vel.T, self.A_vel) #+ self.rho_b *jnp.dot(self.A_bound.T, self.A_bound)
        self.cost_mat_x_projection = jnp.vstack((  jnp.hstack(( self.cost_x_projection, self.A_eq.T )), jnp.hstack(( self.A_eq, jnp.zeros(( jnp.shape(self.A_eq)[0], jnp.shape(self.A_eq)[0] )) )) ))
        self.cost_mat_inv_x_projection = jnp.linalg.inv(self.cost_mat_x_projection)
        self.cost_mat_inv_y_projection = self.cost_mat_inv_x_projection

        eps_k = np.random.multivariate_normal(mu, 0.1*cov, (self.num_batch,) )
        self.eps_k = jnp.asarray(eps_k)

        eps_k_up = np.random.multivariate_normal(mu, 0.005*cov, (int((self.num_batch * self.initial_up_sampling)), ))
        self.eps_k_up = jnp.asarray(eps_k_up)

        eps_k_up_1 = np.random.multivariate_normal(mu, 0.002*cov, (int(0.6*(self.num_batch * self.initial_up_sampling)), ))
        self.eps_k_up_1 = jnp.asarray(eps_k_up_1)

        eps_k_up_2 = np.random.multivariate_normal(mu, 0.002*cov, (int(0.2*(self.num_batch * self.initial_up_sampling)), ))
        self.eps_k_up_2 = jnp.asarray(eps_k_up_2)

        eps_k_up_3 = np.random.multivariate_normal(mu, 0.002*cov, (int(0.2*(self.num_batch * self.initial_up_sampling)), ))
        self.eps_k_up_3 = jnp.asarray(eps_k_up_3)

        self.x_max = 6.2
        self.y_max = 6.2
        self.z_max = 2.8

        self.x_min = 0.0
        self.y_min = 0.0
        self.z_min = 0.0

    

    ########################################
    @partial(jit, static_argnums=(0,),backend="gpu")
    def path_spline(self, x_waypoint, y_waypoint, z_waypoint):

        x_diff = jnp.diff(x_waypoint)
        y_diff = jnp.diff(y_waypoint)
        z_diff = jnp.diff(z_waypoint)

        arc = jnp.cumsum(jnp.sqrt( x_diff**2 + y_diff**2+ z_diff**2))
        arc_length = arc[-1]
        arc_vec = jnp.linspace(0, arc_length, self.way_point_shape)

        return arc_length, arc_vec, x_diff, y_diff, z_diff
    ###################################3

    @partial(jit, static_argnums=(0,),backend="gpu")
    def compute_contouring_error(self, x_waypoint, y_waypoint, z_waypoint, x_target_point, y_target_point, z_target_point, arc_vec):

        dist = jnp.sqrt( (x_waypoint - x_target_point)**2 + (y_waypoint - y_target_point)**2 + (z_waypoint - z_target_point)**2  )
        index = 99#jnp.argmin(dist)

        arc_point = arc_vec[index]

        x_project = x_waypoint[index]
        y_project = y_waypoint[index]
        z_project = z_waypoint[index]

        return arc_point, x_project, y_project, z_project



    @partial(jit, static_argnums=(0,))
    def compute_obs_traj_prediction(self, x_obs_init_1, y_obs_init_1, z_obs_init_1, x_obs_init_2, y_obs_init_2, z_obs_init_2, x_obs_init_3, y_obs_init_3, z_obs_init_3,  vx_obs_1, vy_obs_1, vz_obs_1,  vx_obs_2, vy_obs_2, vz_obs_2,  vx_obs_3, vy_obs_3, vz_obs_3):

        x_temp_1 = x_obs_init_1+vx_obs_1*self.tot_time[:,jnp.newaxis]
        x_obs_trajectory_1 = x_temp_1.T 

        y_temp_1 = y_obs_init_1+vy_obs_1*self.tot_time[:,jnp.newaxis]
        y_obs_trajectory_1 = y_temp_1.T
        
        z_temp = z_obs_init_1+vz_obs_1*self.tot_time[:,jnp.newaxis]
        z_obs_trajectory_1 = z_temp.T

        x_temp = x_obs_init_2+vx_obs_2*self.tot_time[:,jnp.newaxis]
        x_obs_trajectory_2 = x_temp.T 

        y_temp_2 = y_obs_init_2+vy_obs_2*self.tot_time[:,jnp.newaxis]
        y_obs_trajectory_2 = y_temp_2.T
        
        z_temp = z_obs_init_2+vz_obs_2*self.tot_time[:,jnp.newaxis]
        z_obs_trajectory_2 = z_temp.T

        x_temp_3 = x_obs_init_3+vx_obs_3*self.tot_time[:,jnp.newaxis]
        x_obs_trajectory_3 = x_temp_3.T 

        y_temp_3 = y_obs_init_3+vy_obs_3*self.tot_time[:,jnp.newaxis]
        y_obs_trajectory_3 = y_temp_3.T
        
        z_temp = z_obs_init_3+vz_obs_3*self.tot_time[:,jnp.newaxis]
        z_obs_trajectory_3 = z_temp.T

        return x_obs_trajectory_1, y_obs_trajectory_1, z_obs_trajectory_1, x_obs_trajectory_2, y_obs_trajectory_2, z_obs_trajectory_2, x_obs_trajectory_3, y_obs_trajectory_3, z_obs_trajectory_3


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
    


    @partial(jit, static_argnums=(0,),backend="gpu")
    def compute_traj_guess(self, key, initial_state, x_obs_trajectory_1, y_obs_trajectory_1, z_obs_trajectory_1, x_obs_trajectory_2, y_obs_trajectory_2, z_obs_trajectory_2, x_obs_trajectory_3, y_obs_trajectory_3, z_obs_trajectory_3, x_fin ,y_fin, z_fin):

        x_fin = x_fin
        y_fin = y_fin
        z_fin = z_fin

        x_interp = jnp.linspace(initial_state[0], x_fin, self.num)
        y_interp = jnp.linspace(initial_state[1], y_fin, self.num)
        z_interp = jnp.linspace(initial_state[2], z_fin, self.num)
        # cost_x =  jnp.dot(self.P_jax.T, self.P_jax) + 0.0001 * jnp.identity(self.nvar)# + 1*jnp.dot(self.Pddot_jax.T, self.Pddot_jax)
        # cost_mat_x = cost_x
        # cost_mat_inv_x = jnp.linalg.inv(cost_mat_x)

        # lincost_x_init = -jnp.dot(self.P_jax.T, x_interp.T).T
        # lincost_y_init = -jnp.dot(self.P_jax.T, y_interp.T).T
        # lincost_z_init = -jnp.dot(self.P_jax.T, z_interp.T).T

        # sol_x_init = ( jnp.dot(cost_mat_inv_x, -lincost_x_init.T).T )
        # sol_y_init = ( jnp.dot(cost_mat_inv_x, -lincost_y_init.T).T )
        # sol_z_init = ( jnp.dot(cost_mat_inv_x, -lincost_z_init.T).T )

        # sol_cov_x = 40.0 * jnp.identity(self.nvar-6)
        # sol_cov_y = 60.0 * jnp.identity(self.nvar-6)
        # sol_cov_z = 1.0 * jnp.identity(self.nvar-6)

        # sol_x_initial_point = sol_x_init[:3].flatten()
        # sol_y_initial_point = sol_y_init[:3].flatten()
        # sol_z_initial_point = sol_z_init[:3].flatten()

        # sol_x_final_point = sol_x_init[-3:].flatten()
        # sol_y_final_point = sol_y_init[-3:].flatten()
        # sol_z_final_point = sol_z_init[-3:].flatten()

        # epk_x = jax.random.multivariate_normal(key, sol_x_init[3:8], sol_cov_x, (int((self.num_batch * self.initial_up_sampling)), )) 
        # epk_x = jnp.vstack(( sol_x_initial_point[:,jnp.newaxis]* jnp.ones((3, self.num_batch * self.initial_up_sampling)) , epk_x.T , sol_x_final_point[:,jnp.newaxis]* jnp.ones((3, self.num_batch * self.initial_up_sampling))           )).T

        # epk_y = jax.random.multivariate_normal(key, sol_y_init[3:8], sol_cov_y, (int((self.num_batch * self.initial_up_sampling)), )) 
        # epk_y = jnp.vstack(( sol_y_initial_point[:,jnp.newaxis]* jnp.ones((3, self.num_batch * self.initial_up_sampling)) , epk_y.T , sol_y_final_point[:,jnp.newaxis]* jnp.ones((3, self.num_batch * self.initial_up_sampling))           )).T

        # epk_z = jax.random.multivariate_normal(key, sol_z_init[3:8], sol_cov_z, (int((self.num_batch * self.initial_up_sampling)), )) 
        # epk_z = jnp.vstack(( sol_z_initial_point[:,jnp.newaxis]* jnp.ones((3, self.num_batch * self.initial_up_sampling)) , epk_z.T, sol_z_final_point[:,jnp.newaxis]* jnp.ones((3, self.num_batch * self.initial_up_sampling))            )).T

        # x_guess_temp = jnp.dot(self.P_jax, epk_x.T).T
        # y_guess_temp = jnp.dot(self.P_jax, epk_y.T).T
        # z_guess_temp = jnp.dot(self.P_jax, epk_z.T).T

        # x_guess_sampling = x_guess_temp 
        # y_guess_sampling = y_guess_temp 
        # z_guess_sampling = z_guess_temp

        # x_dist_temp = (x_guess_sampling-x_obs_trajectory_1[:,jnp.newaxis])
        # y_dist_temp = (y_guess_sampling-y_obs_trajectory_1[:,jnp.newaxis])
        # z_dist_temp = (z_guess_sampling-z_obs_trajectory_1[:,jnp.newaxis])

        # x_dist = x_dist_temp.transpose(1, 0, 2)
        # y_dist = y_dist_temp.transpose(1, 0, 2)
        # z_dist = z_dist_temp.transpose(1, 0, 2)

        # x_dist = x_dist.reshape(self.num_batch * self.initial_up_sampling, self.num*self.num_obs_1)
        # y_dist = y_dist.reshape(self.num_batch * self.initial_up_sampling, self.num*self.num_obs_1)
        # z_dist = z_dist.reshape(self.num_batch * self.initial_up_sampling, self.num*self.num_obs_1)

        # dist_obs = -x_dist**2/(self.a_obs_1**2)-y_dist**2/(self.b_obs_1**2)-z_dist**2/(self.c_obs_1**2)+1

        # cost_obs_penalty = jnp.linalg.norm(jnp.maximum(jnp.zeros(( self.initial_up_sampling*self.num_batch, self.num*self.num_obs_1 )), dist_obs), axis = 1)

        # idx_ellite = jnp.argsort(cost_obs_penalty)

        # x_guess_sample = x_guess_sampling[idx_ellite[0:self.num_batch ]]
        # y_guess_sample = y_guess_sampling[idx_ellite[0:self.num_batch ]]
        # z_guess_sample = z_guess_sampling[idx_ellite[0:self.num_batch]]

        # x_guess =  x_guess_sample
        # y_guess =  y_guess_sample
        # z_guess = z_guess_sample

        # lincost_x = -jnp.dot(self.P_jax.T, x_guess.T).T
        # lincost_y = -jnp.dot(self.P_jax.T, y_guess.T).T
        # lincost_z = -jnp.dot(self.P_jax.T, z_guess.T).T

        # sol_x = jnp.dot(self.cost_mat_inv_x, -lincost_x.T).T
        # sol_y = jnp.dot(self.cost_mat_inv_y, -lincost_y.T).T
        # sol_z = jnp.dot(self.cost_mat_inv_y, -lincost_z.T).T

        x_samples_init_1 = 1.0*self.eps_k_up_1 + x_interp #x_guess_temp# 
        y_samples_init_1 = 0.2*self.eps_k_up_1 + y_interp #y_guess_temp#
        z_samples_init_1 = 0.0*self.eps_k_up_1 + z_interp#z_guess_temp#

        x_samples_init_2 = 0.0*self.eps_k_up_2 + x_interp #x_guess_temp# 
        y_samples_init_2 = 1.0*self.eps_k_up_2 + y_interp #y_guess_temp#
        z_samples_init_2 = -0.3*self.eps_k_up_2 + z_interp#z_guess_temp#

        x_samples_init_3 = 0.2*self.eps_k_up_3 + x_interp #x_guess_temp# 
        y_samples_init_3 = 0.2*self.eps_k_up_3 + y_interp #y_guess_temp#
        z_samples_init_3 = 1.0*self.eps_k_up_3 + z_interp#z_guess_temp#

        x_samples_init = jnp.vstack(( x_samples_init_1, x_samples_init_2, x_samples_init_3))
        y_samples_init = jnp.vstack(( y_samples_init_1, y_samples_init_2, y_samples_init_3))
        z_samples_init = jnp.vstack(( z_samples_init_1, z_samples_init_2, z_samples_init_3))

        x_dist_temp_1 = (x_samples_init-x_obs_trajectory_1[:,jnp.newaxis])
        y_dist_temp_1 = (y_samples_init-y_obs_trajectory_1[:,jnp.newaxis])
        z_dist_temp_1 = (z_samples_init-z_obs_trajectory_1[:,jnp.newaxis])

        x_dist_temp_2 = (x_samples_init-x_obs_trajectory_2[:,jnp.newaxis])
        y_dist_temp_2 = (y_samples_init-y_obs_trajectory_2[:,jnp.newaxis])
        z_dist_temp_2 = (z_samples_init-z_obs_trajectory_2[:,jnp.newaxis])

        x_dist_temp_3 = (x_samples_init-x_obs_trajectory_3[:,jnp.newaxis])
        y_dist_temp_3 = (y_samples_init-y_obs_trajectory_3[:,jnp.newaxis])
        z_dist_temp_3 = (z_samples_init-z_obs_trajectory_3[:,jnp.newaxis])

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


        dist_obs_1 = -x_dist_1**2/(self.a_obs_1**2)-y_dist_1**2/(self.b_obs_1**2)-z_dist_1**2/(self.c_obs_1**2)+1
        dist_obs_2 = -x_dist_2**2/(self.a_obs_2**2)-y_dist_2**2/(self.b_obs_2**2)-z_dist_2**2/(self.c_obs_2**2)+1
        dist_obs_3 = -x_dist_3**2/(self.a_obs_3**2)-y_dist_3**2/(self.b_obs_3**2)-z_dist_3**2/(self.c_obs_3**2)+1

        dist_obs = dist_obs_1

        cost_obs_penalty = jnp.linalg.norm(jnp.maximum(jnp.zeros(( self.initial_up_sampling*self.num_batch, self.num*(self.num_obs_1 ) )), dist_obs), axis = 1)

        idx_ellite = jnp.argsort(cost_obs_penalty)

        x_guess = x_samples_init[idx_ellite[0:self.num_batch ]]
        y_guess = y_samples_init[idx_ellite[0:self.num_batch ]]
        z_guess = z_samples_init[idx_ellite[0:self.num_batch ]]

        lincost_x = -jnp.dot(self.P_jax.T, x_guess.T).T
        lincost_y = -jnp.dot(self.P_jax.T, y_guess.T).T
        lincost_z = -jnp.dot(self.P_jax.T, z_guess.T).T

        sol_x = jnp.dot(self.cost_mat_inv_x, -lincost_x.T).T
        sol_y = jnp.dot(self.cost_mat_inv_y, -lincost_y.T).T
        sol_z = jnp.dot(self.cost_mat_inv_y, -lincost_z.T).T

        sol_x_bar = sol_x[:,0:self.nvar]
        sol_y_bar = sol_y[:,0:self.nvar]
        sol_z_bar = sol_z[:,0:self.nvar]

        x_guess = jnp.dot(self.P_jax, sol_x_bar.T).T
        xdot_guess = jnp.dot(self.Pdot_jax, sol_x_bar.T).T
        xddot_guess = jnp.dot(self.Pddot_jax, sol_x_bar.T).T

        y_guess = jnp.dot(self.P_jax, sol_y_bar.T).T
        ydot_guess = jnp.dot(self.Pdot_jax, sol_y_bar.T).T
        yddot_guess = jnp.dot(self.Pddot_jax, sol_y_bar.T).T

        z_guess = jnp.dot(self.P_jax, sol_z_bar.T).T
        zdot_guess = jnp.dot(self.Pdot_jax, sol_z_bar.T).T
        zddot_guess = jnp.dot(self.Pddot_jax, sol_z_bar.T).T

        c_mean = jnp.mean( jnp.hstack(( sol_x_bar, sol_y_bar, sol_z_bar )), axis = 0  )
        c_cov = jnp.cov( jnp.hstack(( sol_x_bar, sol_y_bar, sol_z_bar )).T  )
        

        u = jnp.linspace(0, 2 * jnp.pi, 100)
        v = jnp.linspace(0, jnp.pi, 100)

        x_obs_init_1 = x_obs_trajectory_1[:,0]
        x_obs_init_2 = x_obs_trajectory_2[:,0]
        x_obs_init_3 = x_obs_trajectory_3[:,0]
        y_obs_init_1 = y_obs_trajectory_1[:,0]
        y_obs_init_2 = y_obs_trajectory_2[:,0]
        y_obs_init_3 = y_obs_trajectory_3[:,0]
        z_obs_init_1 = z_obs_trajectory_1[:,0]
        z_obs_init_2 = z_obs_trajectory_2[:,0]
        z_obs_init_3 = z_obs_trajectory_3[:,0]

        # figure = mlab.figure(bgcolor=(1, 1, 1), size=(800, 600))
        # for i in range(16):
        #     x = x_obs_init_1.flatten()[i] + (self.a_obs_1-0.03) * jnp.outer(jnp.cos(u), jnp.sin(v))
        #     y = y_obs_init_1.flatten()[i] + (self.b_obs_1-0.03) * jnp.outer(jnp.sin(u), jnp.sin(v))
        #     z = z_obs_init_1.flatten()[i] + (self.c_obs_1-0.03) * jnp.outer(jnp.ones(jnp.size(u)), jnp.cos(v))
        #     mlab.mesh(x, y, z, color=(0.5, 0.5, 0.5))

        # for i in range(2):
        #     x = x_obs_init_2.flatten()[i] + (self.a_obs_2-0.03) * jnp.outer(jnp.cos(u), jnp.sin(v))
        #     y = y_obs_init_2.flatten()[i] + (self.b_obs_2-0.03) * jnp.outer(jnp.sin(u), jnp.sin(v))
        #     z = z_obs_init_2.flatten()[i] + (self.c_obs_2-0.03) * jnp.outer(jnp.ones(jnp.size(u)), jnp.cos(v))
        #     # mlab.mesh(x, y, z, color=(0.9, 0.9, 0.9))

        # for i in range(32):
        #     x = x_obs_init_3.flatten()[i] + (self.a_obs_3-0.03) * jnp.outer(jnp.cos(u), jnp.sin(v))
        #     y = y_obs_init_3.flatten()[i] + (self.b_obs_3-0.03) * jnp.outer(jnp.sin(u), jnp.sin(v))
        #     z = z_obs_init_3.flatten()[i] + (self.c_obs_3-0.03) * jnp.outer(jnp.ones(jnp.size(u)), jnp.cos(v))
        #     mlab.mesh(x, y, z, color=(0.1, 0.1, 0.1))
        #     mlab.mesh(x, y, z, color=(0.5, 0.5, 0.5))
        # for j in range(110):
        #     mlab.plot3d(x_samples_init_1[j,:], y_samples_init_1[j,:], z_samples_init_1[j,:], color=(0,0,0))###navi
        #     mlab.plot3d(x_samples_init_2[j,:], y_samples_init_2[j,:], z_samples_init_2[j,:], color=(1,0,0))###navi
        #     mlab.plot3d(x_samples_init_3[j,:], y_samples_init_3[j,:], z_samples_init_3[j,:], color=(0,1,0))###navi
        #     # mlab.plot3d(x_guess[j,:], y_guess[j,:], z_guess[j,:], color=(1,0,0))##olive



        # mlab.view(azimuth=45, elevation=60, distance=6, focalpoint=(0.5, 0.5, 0.5))
        # mlab.show()
        return sol_x_bar, sol_y_bar, sol_z_bar, x_guess, y_guess, z_guess, xdot_guess, ydot_guess, zdot_guess, xddot_guess, yddot_guess, zddot_guess, c_mean, c_cov


    ################################################

    @partial(jit, static_argnums=(0,),backend="gpu")
    def compute_projection(self, x_obs_trajectory_1, y_obs_trajectory_1, z_obs_trajectory_1, x_obs_trajectory_2, y_obs_trajectory_2, z_obs_trajectory_2, x_obs_trajectory_3, y_obs_trajectory_3, z_obs_trajectory_3,  d_obs, alpha_obs, beta_obs, alpha_a, beta_a, d_a, alpha_v, beta_v, d_v, x_fin, y_fin, z_fin, lamda_x, lamda_y, lamda_z, b_eq_x, b_eq_y, b_eq_z, sol_x_bar, sol_y_bar, sol_z_bar, s_x, s_y, s_z ):

        b_projection_x = sol_x_bar
        b_projection_y = sol_y_bar
        b_projection_z = sol_z_bar

        radi_a = self.a_obs_1        
        radi_b = self.b_obs_1#jnp.hstack(( self.b_obs_1*jnp.ones((self.num_batch , self.num * self.num_obs_1)) , self.b_obs_2 * jnp.ones((self.num_batch , self.num * self.num_obs_2)) ))#, self.b_obs_3*jnp.ones((self.num_batch , self.num * self.num_obs_3)) ))
        radi_c = self.c_obs_1#jnp.hstack(( self.c_obs_1*jnp.ones((self.num_batch , self.num * self.num_obs_1)) , self.c_obs_2 * jnp.ones((self.num_batch , self.num * self.num_obs_2)) ))#, self.c_obs_3*jnp.ones((self.num_batch , self.num * self.num_obs_3)) ))

        temp_x_obs = d_obs*jnp.cos(alpha_obs)*jnp.sin(beta_obs)* radi_a
        b_obs_x =  x_obs_trajectory_1.reshape(self.num*self.num_obs_1 )+temp_x_obs
            
        temp_y_obs = d_obs*jnp.sin(alpha_obs)*jnp.sin(beta_obs)* radi_b
        b_obs_y = y_obs_trajectory_1.reshape(self.num*self.num_obs_1 ) +temp_y_obs

        temp_z_obs = d_obs*jnp.cos(beta_obs)* radi_c
        b_obs_z = z_obs_trajectory_1.reshape(self.num*self.num_obs_1 ) +temp_z_obs

        b_ax_ineq = d_a*jnp.cos(alpha_a)*jnp.sin(beta_a)
        b_ay_ineq = d_a*jnp.sin(alpha_a)*jnp.sin(beta_a)
        b_az_ineq = d_a*jnp.cos(beta_a)

        b_vx_ineq = d_v*jnp.cos(alpha_v)*jnp.sin(beta_v)
        b_vy_ineq = d_v*jnp.sin(alpha_v)*jnp.sin(beta_v)
        b_vz_ineq = d_v*jnp.cos(beta_v)

        b_x_bound = jnp.hstack(( self.x_max*jnp.ones((self.num_batch, self.num)), -self.x_min*jnp.ones((self.num_batch, self.num)) ))  - s_x
        b_y_bound = jnp.hstack(( self.y_max*jnp.ones((self.num_batch, self.num)), -self.y_min*jnp.ones((self.num_batch, self.num)) ))  - s_y
        b_z_bound = jnp.hstack(( self.z_max*jnp.ones((self.num_batch, self.num)), -self.z_min*jnp.ones((self.num_batch, self.num)) ))  - s_z

        lincost_x =  - self.rho_proj * jnp.dot(self.A_projection.T, b_projection_x.T).T -lamda_x-self.rho_obs*jnp.dot(self.A_obs.T, b_obs_x.T).T-self.rho_ineq*jnp.dot(self.A_acc.T, b_ax_ineq.T).T-self.rho_ineq*jnp.dot(self.A_vel.T, b_vx_ineq.T).T #- self.rho_b * jnp.dot(self.A_bound.T, b_x_bound.T).T# - self.rho_f * jnp.dot(self.A_f.T, b_x_f.T).T
        lincost_y =  - self.rho_proj * jnp.dot(self.A_projection.T, b_projection_y.T).T -lamda_y-self.rho_obs*jnp.dot(self.A_obs.T, b_obs_y.T).T-self.rho_ineq*jnp.dot(self.A_acc.T, b_ay_ineq.T).T-self.rho_ineq*jnp.dot(self.A_vel.T, b_vy_ineq.T).T #- self.rho_b * jnp.dot(self.A_bound.T, b_y_bound.T).T# - self.rho_f * jnp.dot(self.A_f.T, b_y_f.T).T
        lincost_z =  - self.rho_proj * jnp.dot(self.A_projection.T, b_projection_z.T).T -lamda_z-self.rho_obs*jnp.dot(self.A_obs.T, b_obs_z.T).T-self.rho_ineq*jnp.dot(self.A_acc.T, b_az_ineq.T).T-self.rho_ineq*jnp.dot(self.A_vel.T, b_vz_ineq.T).T #- self.rho_b * jnp.dot(self.A_bound.T, b_z_bound.T).T# - self.rho_f * jnp.dot(self.A_f.T, b_z_f.T).T

        sol_x = jnp.dot(self.cost_mat_inv_x_projection, jnp.hstack(( -lincost_x, b_eq_x )).T).T
        sol_y = jnp.dot(self.cost_mat_inv_y_projection, jnp.hstack(( -lincost_y, b_eq_y )).T).T
        sol_z = jnp.dot(self.cost_mat_inv_y_projection, jnp.hstack(( -lincost_z, b_eq_z )).T).T

        primal_sol_x = sol_x[:,0:self.nvar]
        primal_sol_y = sol_y[:,0:self.nvar]
        primal_sol_z = sol_z[:,0:self.nvar]

        x = jnp.dot(self.P_jax, primal_sol_x.T).T
        xdot = jnp.dot(self.Pdot_jax, primal_sol_x.T).T
        xddot = jnp.dot(self.Pddot_jax, primal_sol_x.T).T

        y = jnp.dot(self.P_jax, primal_sol_y.T).T
        ydot = jnp.dot(self.Pdot_jax, primal_sol_y.T).T
        yddot = jnp.dot(self.Pddot_jax, primal_sol_y.T).T

        z = jnp.dot(self.P_jax, primal_sol_z.T).T
        zdot = jnp.dot(self.Pdot_jax, primal_sol_z.T).T
        zddot = jnp.dot(self.Pddot_jax, primal_sol_z.T).T

        s_x = jnp.maximum( jnp.zeros(( self.num_batch, 2*self.num )), -jnp.dot(self.A_bound, primal_sol_x.T).T+b_x_bound  )
        s_y = jnp.maximum( jnp.zeros(( self.num_batch, 2*self.num )), -jnp.dot(self.A_bound, primal_sol_y.T).T+b_y_bound  )
        s_z = jnp.maximum( jnp.zeros(( self.num_batch, 2*self.num )), -jnp.dot(self.A_bound, primal_sol_z.T).T+b_z_bound  )

        res_x_vec_b = jnp.dot(self.A_bound, primal_sol_x.T).T-b_x_bound+s_x
        res_y_vec_b = jnp.dot(self.A_bound, primal_sol_y.T).T-b_y_bound+s_y
        res_z_vec_b = jnp.dot(self.A_bound, primal_sol_z.T).T-b_z_bound+s_z


        return primal_sol_x, primal_sol_y, primal_sol_z, x, y, z, xdot, ydot, zdot, xddot, yddot, zddot, s_x, s_y, s_z, res_x_vec_b, res_y_vec_b, res_z_vec_b

    #########################################################
    @partial(jit, static_argnums=(0,),backend="gpu")	
    def initial_alpha_d(self, x_fin, y_fin, z_fin, x_guess, y_guess, z_guess, xdot_guess, ydot_guess, zdot_guess, xddot_guess, yddot_guess, zddot_guess, x_obs_trajectory_1, y_obs_trajectory_1, z_obs_trajectory_1, x_obs_trajectory_2, y_obs_trajectory_2, z_obs_trajectory_2, x_obs_trajectory_3, y_obs_trajectory_3, z_obs_trajectory_3, lamda_x, lamda_y, lamda_z):
        
        wc_alpha_temp = (x_guess - x_obs_trajectory_1[:,jnp.newaxis])
        ws_alpha_temp = (y_guess - y_obs_trajectory_1[:,jnp.newaxis])

        wc_alpha = wc_alpha_temp.transpose(1, 0, 2)
        ws_alpha = ws_alpha_temp.transpose(1, 0, 2)

        wc_alpha = wc_alpha.reshape(self.num_batch, self.num*(self.num_obs_1  ))
        ws_alpha = ws_alpha.reshape(self.num_batch, self.num*(self.num_obs_1  ))

        radi_a = self.a_obs_1#jnp.hstack(( self.a_obs_1*jnp.ones((self.num_batch , self.num * self.num_obs_1)) , self.a_obs_2 * jnp.ones((self.num_batch , self.num * self.num_obs_2)) ))
        radi_b = self.b_obs_1#jnp.hstack(( self.b_obs_1*jnp.ones((self.num_batch , self.num * self.num_obs_1)) , self.b_obs_2 * jnp.ones((self.num_batch , self.num * self.num_obs_2))))
        radi_c = self.c_obs_1#jnp.hstack(( self.c_obs_1*jnp.ones((self.num_batch , self.num * self.num_obs_1)) , self.c_obs_2 * jnp.ones((self.num_batch , self.num * self.num_obs_2)) ))

        alpha_obs = jnp.arctan2( ws_alpha* radi_a, wc_alpha* radi_b)

        wc_beta_temp = (z_guess - z_obs_trajectory_1[:,jnp.newaxis] )
        wc_beta = wc_beta_temp.transpose(1, 0, 2)
        wc_beta = wc_beta.reshape(self.num_batch, self.num*(self.num_obs_1 ))
        ws_beta = wc_alpha / jnp.cos(alpha_obs)

        beta_obs = jnp.arctan2( ws_beta* radi_c, wc_beta* radi_a )

        c1_d = 1.0*self.rho_obs * radi_a**2 *jnp.sin(beta_obs)**2 + 1.0*self.rho_obs * radi_c**2*jnp.cos(beta_obs)**2
        c2_d = 1.0*radi_a*self.rho_obs*wc_alpha*jnp.cos(alpha_obs)*jnp.sin(beta_obs) + radi_b*self.rho_obs*ws_alpha*jnp.sin(alpha_obs)*jnp.sin(beta_obs) + radi_c*self.rho_obs*wc_beta*jnp.cos(beta_obs)

        d_obs = c2_d/c1_d
        d_obs = jnp.maximum(jnp.ones((self.num_batch,  self.num*(self.num_obs_1 )   )), d_obs   )

        ####################### velocity terms

        wc_alpha_v = xdot_guess
        ws_alpha_v = ydot_guess

        alpha_v = jnp.arctan2( ws_alpha_v, wc_alpha_v)

        wc_beta_v = zdot_guess
        ws_beta_v = wc_alpha_v/jnp.cos(alpha_v)
        beta_v = jnp.arctan2( ws_beta_v, wc_beta_v)

        c1_d_v = 1.0*self.rho_ineq
        c2_d_v = 1.0*self.rho_ineq*(wc_alpha_v*jnp.cos(alpha_v)*jnp.sin(beta_v) + ws_alpha_v*jnp.sin(alpha_v)*jnp.sin(beta_v) +wc_beta_v *jnp.cos(beta_v) )

        d_v = c2_d_v/c1_d_v

        d_v = jnp.minimum(self.v_max*jnp.ones((self.num_batch, self.num)), d_v   )

        ################# acceleration terms

        wc_alpha_a = xddot_guess
        ws_alpha_a = yddot_guess

        alpha_a = jnp.arctan2( ws_alpha_a, wc_alpha_a)	

        wc_beta_a = zddot_guess
        ws_beta_a = wc_alpha_a/jnp.cos(alpha_a)
        beta_a = jnp.arctan2( ws_beta_a, wc_beta_a)

        c1_d_a = 1.0*self.rho_ineq
        c2_d_a = 1.0*self.rho_ineq*(wc_alpha_a*jnp.cos(alpha_a)*jnp.sin(beta_a) + ws_alpha_a*jnp.sin(alpha_a)*jnp.sin(beta_a) +wc_beta_a *jnp.cos(beta_a) )

        d_a = c2_d_a/c1_d_a

        d_a = jnp.minimum(self.a_max*jnp.ones((self.num_batch, self.num)), d_a   )
       
        ##############################################################
        b_x_bound = jnp.hstack(( self.x_max*jnp.ones((self.num_batch, self.num)), -self.x_min*jnp.ones((self.num_batch, self.num)) ))  
        b_y_bound = jnp.hstack(( self.y_max*jnp.ones((self.num_batch, self.num)), -self.y_min*jnp.ones((self.num_batch, self.num)) ))  
        b_z_bound = jnp.hstack(( self.z_max*jnp.ones((self.num_batch, self.num)), -self.z_min*jnp.ones((self.num_batch, self.num)) ))  

        res_x_b = jnp.hstack(( x_guess, x_guess)) - b_x_bound
        res_y_b = jnp.hstack(( y_guess, y_guess)) - b_y_bound
        res_z_b = jnp.hstack(( z_guess, z_guess)) - b_z_bound

        res_ax_vec = xddot_guess-d_a*jnp.cos(alpha_a) * jnp.sin(beta_a)
        res_ay_vec = yddot_guess-d_a*jnp.sin(alpha_a) * jnp.sin(beta_a)
        res_az_vec = zddot_guess-d_a*jnp.cos(beta_a)

        res_vx_vec = xdot_guess-d_v*jnp.cos(alpha_v) * jnp.sin(beta_v)
        res_vy_vec = ydot_guess-d_v*jnp.sin(alpha_v) * jnp.sin(beta_v)
        res_vz_vec = zdot_guess-d_v*jnp.cos(beta_v)


        res_x_obs_vec = wc_alpha-radi_a*d_obs*jnp.cos(alpha_obs)*jnp.sin(beta_obs)
        res_y_obs_vec = ws_alpha-radi_b*d_obs*jnp.sin(alpha_obs)*jnp.sin(beta_obs)
        res_z_obs_vec = wc_beta-radi_c*d_obs*jnp.cos(beta_obs)

        lamda_x = lamda_x -self.rho_obs*jnp.dot(self.A_obs.T, res_x_obs_vec.T).T-self.rho_ineq*jnp.dot(self.A_acc.T, res_ax_vec.T).T-self.rho_ineq*jnp.dot(self.A_vel.T, res_vx_vec.T).T# -self.rho_b*jnp.dot(self.A_bound.T, res_x_b.T).T
        lamda_y = lamda_y -self.rho_obs*jnp.dot(self.A_obs.T, res_y_obs_vec.T).T-self.rho_ineq*jnp.dot(self.A_acc.T, res_ay_vec.T).T-self.rho_ineq*jnp.dot(self.A_vel.T, res_vy_vec.T).T# -self.rho_b*jnp.dot(self.A_bound.T, res_y_b.T).T
        lamda_z = lamda_z -self.rho_obs*jnp.dot(self.A_obs.T, res_z_obs_vec.T).T-self.rho_ineq*jnp.dot(self.A_acc.T, res_az_vec.T).T-self.rho_ineq*jnp.dot(self.A_vel.T, res_vz_vec.T).T# -self.rho_b*jnp.dot(self.A_bound.T, res_z_b.T).T

        return alpha_obs, beta_obs, d_obs, alpha_a, beta_a, d_a, alpha_v, beta_v, d_v, lamda_x, lamda_y, lamda_z
         
    ####################################################

    @partial(jit, static_argnums=(0,))	
    def compute_alph_d_proj(self, x_fin, y_fin, z_fin, x, y, z, xdot, ydot, zdot, xddot, yddot, zddot, x_obs_trajectory_1, y_obs_trajectory_1, z_obs_trajectory_1, x_obs_trajectory_2, y_obs_trajectory_2, z_obs_trajectory_2, x_obs_trajectory_3, y_obs_trajectory_3, z_obs_trajectory_3, lamda_x, lamda_y, lamda_z, s_x, s_y, s_z, res_x_vec_b, res_y_vec_b, res_z_vec_b):
                                
    
        wc_alpha_temp = (x-  x_obs_trajectory_1[:,jnp.newaxis] )
        ws_alpha_temp = (y- y_obs_trajectory_1[:,jnp.newaxis] )

        wc_alpha = wc_alpha_temp.transpose(1, 0, 2)
        ws_alpha = ws_alpha_temp.transpose(1, 0, 2)

        wc_alpha = wc_alpha.reshape(self.num_batch, self.num*(self.num_obs_1  ))
        ws_alpha = ws_alpha.reshape(self.num_batch, self.num*(self.num_obs_1  ))

        radi_a = self.a_obs_1#jnp.hstack(( self.a_obs_1*jnp.ones((self.num_batch , self.num * self.num_obs_1)) , self.a_obs_2 * jnp.ones((self.num_batch , self.num * self.num_obs_2)) ))#, self.a_obs_3*jnp.ones((self.num_batch , self.num * self.num_obs_3)) ))
        radi_b = self.b_obs_1#jnp.hstack(( self.b_obs_1*jnp.ones((self.num_batch , self.num * self.num_obs_1)) , self.b_obs_2 * jnp.ones((self.num_batch , self.num * self.num_obs_2)) ))#, self.b_obs_3*jnp.ones((self.num_batch , self.num * self.num_obs_3)) ))
        radi_c = self.c_obs_1#jnp.hstack(( self.c_obs_1*jnp.ones((self.num_batch , self.num * self.num_obs_1)) , self.c_obs_2 * jnp.ones((self.num_batch , self.num * self.num_obs_2)) ))#, self.c_obs_3*jnp.ones((self.num_batch , self.num * self.num_obs_3)) ))


        alpha_obs = jnp.arctan2( ws_alpha*radi_a, wc_alpha*radi_b)

        wc_beta_temp = (z-   z_obs_trajectory_1[:,jnp.newaxis]  )
        wc_beta = wc_beta_temp.transpose(1, 0, 2)
        wc_beta = wc_beta.reshape(self.num_batch, self.num*(self.num_obs_1  ))
        ws_beta = wc_alpha / jnp.cos(alpha_obs)

        beta_obs = jnp.arctan2( ws_beta*radi_c, wc_beta*radi_a )

        c1_d = 1.0*self.rho_obs * radi_a**2 *jnp.sin(beta_obs)**2 + 1.0*self.rho_obs * radi_c**2*jnp.cos(beta_obs)**2
        c2_d = 1.0*radi_a*self.rho_obs*wc_alpha*jnp.cos(alpha_obs)*jnp.sin(beta_obs) + radi_b*self.rho_obs*ws_alpha*jnp.sin(alpha_obs)*jnp.sin(beta_obs) + radi_c*self.rho_obs*wc_beta*jnp.cos(beta_obs)

        d_obs = c2_d/c1_d
        d_obs = jnp.maximum(jnp.ones((self.num_batch,  self.num*(self.num_obs_1 )   )), d_obs   )
        ####################### velocity terms

        wc_alpha_vx = xdot
        ws_alpha_vy = ydot
        alpha_v = jnp.arctan2( ws_alpha_vy, wc_alpha_vx)		
        c1_d_v = 1.0*self.rho_ineq*(jnp.cos(alpha_v)**2 + jnp.sin(alpha_v)**2 )
        c2_d_v = 1.0*self.rho_ineq*(wc_alpha_vx*jnp.cos(alpha_v) + ws_alpha_vy*jnp.sin(alpha_v)  )
        
        wc_beta_v = zdot
        ws_beta_v = wc_alpha_vx/jnp.cos(alpha_v)
        beta_v = jnp.arctan2( ws_beta_v, wc_beta_v)

        c1_d_v = 1.0*self.rho_ineq
        c2_d_v = 1.0*self.rho_ineq*(wc_alpha_vx*jnp.cos(alpha_v)*jnp.sin(beta_v) + ws_alpha_vy*jnp.sin(alpha_v)*jnp.sin(beta_v) +wc_beta_v *jnp.cos(beta_v) )


        d_v = c2_d_v/c1_d_v
        d_v = jnp.minimum(self.v_max*jnp.ones((self.num_batch, self.num)), d_v   )

        ################# acceleration terms

        wc_alpha_a = xddot
        ws_alpha_a = yddot

        alpha_a = jnp.arctan2( ws_alpha_a, wc_alpha_a)	

        wc_beta_a = zddot
        ws_beta_a = wc_alpha_a/jnp.cos(alpha_a)
        beta_a = jnp.arctan2( ws_beta_a, wc_beta_a)

        c1_d_a = 1.0*self.rho_ineq
        c2_d_a = 1.0*self.rho_ineq*(wc_alpha_a*jnp.cos(alpha_a)*jnp.sin(beta_a) + ws_alpha_a*jnp.sin(alpha_a)*jnp.sin(beta_a) +wc_beta_a *jnp.cos(beta_a) )

        d_a = c2_d_a/c1_d_a

        d_a = jnp.minimum(self.a_max*jnp.ones((self.num_batch, self.num)), d_a   )

        ##############################################################
        res_ax_vec = xddot-d_a*jnp.cos(alpha_a) * jnp.sin(beta_a)
        res_ay_vec = yddot-d_a*jnp.sin(alpha_a) * jnp.sin(beta_a)
        res_az_vec = zddot-d_a*jnp.cos(beta_a)

        res_vx_vec = xdot-d_v*jnp.cos(alpha_v) * jnp.sin(beta_v)
        res_vy_vec = ydot-d_v*jnp.sin(alpha_v) * jnp.sin(beta_v)
        res_vz_vec = zdot-d_v*jnp.cos(beta_v)

        res_x_obs_vec = wc_alpha - radi_a*d_obs*jnp.cos(alpha_obs)*jnp.sin(beta_obs)
        res_y_obs_vec = ws_alpha - radi_b*d_obs*jnp.sin(alpha_obs)*jnp.sin(beta_obs)
        res_z_obs_vec = wc_beta - radi_c*d_obs*jnp.cos(beta_obs)

        lamda_x = lamda_x -self.rho_obs*jnp.dot(self.A_obs.T, res_x_obs_vec.T).T-self.rho_ineq*jnp.dot(self.A_acc.T, res_ax_vec.T).T-self.rho_ineq*jnp.dot(self.A_vel.T, res_vx_vec.T).T #-self.rho_b*jnp.dot(self.A_bound.T, res_x_vec_b.T).T
        lamda_y = lamda_y -self.rho_obs*jnp.dot(self.A_obs.T, res_y_obs_vec.T).T-self.rho_ineq*jnp.dot(self.A_acc.T, res_ay_vec.T).T-self.rho_ineq*jnp.dot(self.A_vel.T, res_vy_vec.T).T #-self.rho_b*jnp.dot(self.A_bound.T, res_y_vec_b.T).T
        lamda_z = lamda_z -self.rho_obs*jnp.dot(self.A_obs.T, res_z_obs_vec.T).T-self.rho_ineq*jnp.dot(self.A_acc.T, res_az_vec.T).T-self.rho_ineq*jnp.dot(self.A_vel.T, res_vz_vec.T).T #-self.rho_b*jnp.dot(self.A_bound.T, res_z_vec_b.T).T

        res_obs_vec = jnp.hstack(( res_x_obs_vec, res_y_obs_vec, res_z_obs_vec  ))
        res_acc_vec = jnp.hstack(( res_ax_vec,  res_ay_vec, res_az_vec  ))
        res_vel_vec = jnp.hstack(( res_vx_vec,  res_vy_vec, res_vz_vec  ))
        res_bound = jnp.hstack(( res_x_vec_b, res_y_vec_b, res_z_vec_b  ))
        

        res_norm_batch = 1*jnp.linalg.norm(res_obs_vec, axis =1)+jnp.linalg.norm(res_acc_vec, axis =1)+jnp.linalg.norm(res_vel_vec, axis =1) #+1.0*jnp.linalg.norm(res_bound, axis =1)# + 1*jnp.linalg.norm(res_f_vec, axis =1)

        return alpha_obs, beta_obs, d_obs, alpha_a, beta_a, d_a, alpha_v, beta_v, d_v,  lamda_x, lamda_y, lamda_z, res_norm_batch
    

    @partial(jit, static_argnums=(0, ))	
    def compute_projection_sampling(self, key, sol_x_bar, sol_y_bar, sol_z_bar, x_obs_trajectory_1, y_obs_trajectory_1, z_obs_trajectory_1, x_obs_trajectory_2, y_obs_trajectory_2, z_obs_trajectory_2, x_obs_trajectory_3, y_obs_trajectory_3, z_obs_trajectory_3, lamda_x, lamda_y, lamda_z, x_guess, y_guess, z_guess,  xdot_guess, ydot_guess, zdot_guess, xddot_guess, yddot_guess, zddot_guess, initial_state, x_fin, y_fin, z_fin):

        b_eq_x, b_eq_y, b_eq_z = self.compute_boundary_vec(initial_state, x_fin, y_fin, z_fin)
        alpha_obs_init, beta_obs_init, d_obs_init, alpha_a_init, beta_a_init, d_a_init, alpha_v_init, beta_v_init, d_v_init, lamda_x_init, lamda_y_init, lamda_z_init = self.initial_alpha_d( x_fin, y_fin, z_fin, x_guess, y_guess, z_guess,  xdot_guess, ydot_guess, zdot_guess, xddot_guess, yddot_guess, zddot_guess, x_obs_trajectory_1, y_obs_trajectory_1, z_obs_trajectory_1, x_obs_trajectory_2, y_obs_trajectory_2, z_obs_trajectory_2, x_obs_trajectory_3, y_obs_trajectory_3, z_obs_trajectory_3, lamda_x, lamda_y, lamda_z)
        
        s_x_init = jnp.zeros(( self.num_batch, 2*self.num ))
        s_y_init = jnp.zeros(( self.num_batch, 2*self.num ))
        s_z_init = jnp.zeros(( self.num_batch, 2*self.num ))

        def lax_projection(carry, proj_iter ):

            c_x, c_y, c_z, x, xdot, xddot, y, ydot, yddot, z, zdot, zddot, res_norm_batch, alpha_obs, beta_obs, d_obs, alpha_a, beta_a, d_a, alpha_v, beta_v, d_v, lamda_x, lamda_y, lamda_z, s_x, s_y, s_z, res_x_vec_b, res_y_vec_b, res_z_vec_b = carry

            c_x, c_y, c_z, x, y, z, xdot, ydot, zdot, xddot, yddot, zddot, s_x, s_y, s_z, res_x_vec_b, res_y_vec_b, res_z_vec_b = self.compute_projection(x_obs_trajectory_1, y_obs_trajectory_1, z_obs_trajectory_1, x_obs_trajectory_2, y_obs_trajectory_2, z_obs_trajectory_2, x_obs_trajectory_3, y_obs_trajectory_3, z_obs_trajectory_3, d_obs, alpha_obs, beta_obs, alpha_a, beta_a, d_a, alpha_v, beta_v, d_v, x_fin, y_fin, z_fin, lamda_x, lamda_y, lamda_z, b_eq_x, b_eq_y, b_eq_z, sol_x_bar, sol_y_bar, sol_z_bar, s_x, s_y, s_z )
            alpha_obs, beta_obs, d_obs, alpha_a, beta_a, d_a, alpha_v, beta_v, d_v, lamda_x, lamda_y, lamda_z, res_norm_batch = self.compute_alph_d_proj( x_fin, y_fin, z_fin, x, y, z, xdot, ydot, zdot, xddot, yddot, zddot, x_obs_trajectory_1, y_obs_trajectory_1, z_obs_trajectory_1, x_obs_trajectory_2, y_obs_trajectory_2, z_obs_trajectory_2, x_obs_trajectory_3, y_obs_trajectory_3, z_obs_trajectory_3, lamda_x, lamda_y, lamda_z, s_x, s_y, s_z, res_x_vec_b, res_y_vec_b, res_z_vec_b)

            return (c_x, c_y, c_z, x, xdot, xddot, y, ydot, yddot, z, zdot, zddot, res_norm_batch, alpha_obs, beta_obs, d_obs, alpha_a, beta_a, d_a, alpha_v, beta_v, d_v, lamda_x, lamda_y, lamda_z,  s_x, s_y, s_z, res_x_vec_b, res_y_vec_b, res_z_vec_b), x

        carry_init =  jnp.zeros((self.num_batch,self.nvar)), jnp.zeros((self.num_batch,self.nvar)), jnp.zeros((self.num_batch,self.nvar)), jnp.zeros((self.num_batch,self.num)),  jnp.zeros((self.num_batch,self.num)), jnp.zeros((self.num_batch,self.num)),  jnp.zeros((self.num_batch,self.num)),  jnp.zeros((self.num_batch,self.num)), jnp.zeros((self.num_batch,self.num)), jnp.zeros((self.num_batch,self.num)),  jnp.zeros((self.num_batch,self.num)), jnp.zeros((self.num_batch,self.num)), jnp.zeros(self.num_batch),alpha_obs_init, beta_obs_init, d_obs_init, alpha_a_init, beta_a_init, d_a_init, alpha_v_init, beta_v_init, d_v_init,  lamda_x_init, lamda_y_init, lamda_z_init, s_x_init, s_y_init, s_z_init, jnp.zeros(( self.num_batch, 2*self.num )), jnp.zeros(( self.num_batch, 2*self.num )), jnp.zeros(( self.num_batch, 2*self.num ))

        carry_fin, result = lax.scan(lax_projection, carry_init, jnp.arange(self.maxitet_proj))
        
        c_x, c_y, c_z, x, xdot, xddot, y, ydot, yddot, z, zdot, zddot, res_norm_batch, alpha_obs, beta_obs, d_obs, alpha_a, beta_a, d_a, alpha_v, beta_v, d_v, lamda_x, lamda_y, lamda_z, s_x, s_y, s_z, res_x_vec_b, res_y_vec_b, res_z_vec_b  = carry_fin
       
        return c_x, c_y, c_z, x, xdot, xddot, y, ydot, yddot, z, zdot, zddot, res_norm_batch, s_x, s_y, s_z
        


    ###############################################
    @partial(jit, static_argnums=(0,))
    def compute_cost_batch(self, x, y, z, xdot, ydot, zdot, xddot, yddot, zddot, x_project, y_project, z_project, res_norm_batch, x_fin, y_fin, z_fin,  x_obs_trajectory_1, y_obs_trajectory_1, z_obs_trajectory_1, x_obs_trajectory_2, y_obs_trajectory_2, z_obs_trajectory_2, x_obs_trajectory_3, y_obs_trajectory_3, z_obs_trajectory_3):

        dist_x_temp_1 = (x-x_obs_trajectory_1[:,jnp.newaxis])
        dist_y_temp_1 = (y-y_obs_trajectory_1[:,jnp.newaxis])
        dist_z_temp_1 = (z-z_obs_trajectory_1[:,jnp.newaxis])

        dist_x_temp_2 = (x-x_obs_trajectory_2[:,jnp.newaxis])
        dist_y_temp_2 = (y-y_obs_trajectory_2[:,jnp.newaxis])
        dist_z_temp_2 = (z-z_obs_trajectory_2[:,jnp.newaxis])

        dist_x_temp_3 = (x-x_obs_trajectory_3[:,jnp.newaxis])
        dist_y_temp_3 = (y-y_obs_trajectory_3[:,jnp.newaxis])
        dist_z_temp_3 = (z-z_obs_trajectory_3[:,jnp.newaxis])

        dist_x_1 = dist_x_temp_1.transpose(1, 0, 2)
        dist_y_1 = dist_y_temp_1.transpose(1, 0, 2)
        dist_z_1 = dist_z_temp_1.transpose(1, 0, 2)

        dist_x_2 = dist_x_temp_2.transpose(1, 0, 2)
        dist_y_2 = dist_y_temp_2.transpose(1, 0, 2)
        dist_z_2 = dist_z_temp_2.transpose(1, 0, 2)

        dist_x_3 = dist_x_temp_3.transpose(1, 0, 2)
        dist_y_3 = dist_y_temp_3.transpose(1, 0, 2)
        dist_z_3 = dist_z_temp_3.transpose(1, 0, 2)

        dist_x_1 = dist_x_1.reshape(self.ellite_num_const, self.num*self.num_obs_1)
        dist_y_1 = dist_y_1.reshape(self.ellite_num_const, self.num*self.num_obs_1)
        dist_z_1 = dist_z_1.reshape(self.ellite_num_const, self.num*self.num_obs_1)

        dist_x_2 = dist_x_2.reshape(self.ellite_num_const, self.num*self.num_obs_2)
        dist_y_2 = dist_y_2.reshape(self.ellite_num_const, self.num*self.num_obs_2)
        dist_z_2 = dist_z_2.reshape(self.ellite_num_const, self.num*self.num_obs_2)

        dist_x_3 = dist_x_3.reshape(self.ellite_num_const, self.num*self.num_obs_3)
        dist_y_3 = dist_y_3.reshape(self.ellite_num_const, self.num*self.num_obs_3)
        dist_z_3 = dist_z_3.reshape(self.ellite_num_const, self.num*self.num_obs_3)

        dist_obs_1 = -(dist_x_1**2/(self.a_obs_1**2))-(dist_y_1**2/(self.b_obs_1**2))-((dist_z_1**2)/(self.c_obs_1**2))+1
        dist_obs_2 = -(dist_x_2**2/(self.a_obs_2**2))-(dist_y_2**2/(self.b_obs_2**2))-((dist_z_2**2)/(self.c_obs_2**2))+1
        dist_obs_3 = -(dist_x_3**2/(self.a_obs_3**2))-(dist_y_3**2/(self.b_obs_3**2))-((dist_z_3**2)/(self.c_obs_1**2))+1
        
        dist_obs = dist_obs_1

        clearance_cost = - jnp.min(dist_obs)
        cost_obs = jnp.linalg.norm(jnp.maximum(jnp.zeros(( self.ellite_num_const, self.num*(self.num_obs_1 ) )), dist_obs), axis = 1)

        cost_smoothness =  ( (jnp.linalg.norm(xddot, axis = 1)) + (jnp.linalg.norm(yddot, axis = 1))+ (jnp.linalg.norm(zddot, axis = 1)))
        cost_track = jnp.linalg.norm(x-x_project, axis = 1)+jnp.linalg.norm(y-y_project, axis = 1)+2*jnp.linalg.norm(z-z_project, axis = 1) 

        cost_max_x = jnp.linalg.norm( jnp.maximum( jnp.zeros((self.ellite_num_const, self.num)),  (x-self.x_max) ), axis = 1)#+jnp.linalg.norm(y-y_project, axis = 1)+2*jnp.linalg.norm(z-z_project, axis = 1) 
        cost_max_y = jnp.linalg.norm( jnp.maximum( jnp.zeros((self.ellite_num_const, self.num)),  (y-self.y_max) ), axis = 1)
        cost_max_z = jnp.linalg.norm( jnp.maximum( jnp.zeros((self.ellite_num_const, self.num)),  (z-self.z_max) ), axis = 1)
        cost_max = cost_max_x + cost_max_y + 7*cost_max_z

        cost_min_x = jnp.linalg.norm( jnp.maximum( jnp.zeros((self.ellite_num_const, self.num)),  (-x+self.x_min) ), axis = 1)#+jnp.linalg.norm(y-y_project, axis = 1)+2*jnp.linalg.norm(z-z_project, axis = 1) 
        cost_min_y = jnp.linalg.norm( jnp.maximum( jnp.zeros((self.ellite_num_const, self.num)),  (-y+self.y_min) ), axis = 1)
        cost_min_z = jnp.linalg.norm( jnp.maximum( jnp.zeros((self.ellite_num_const, self.num)),  (-z+self.z_min) ), axis = 1)
        cost_min = cost_min_x + cost_min_y + 7*cost_min_z

        cost_batch = 1.0* res_norm_batch + 0.2*cost_smoothness + 0.0* cost_track + 8.0*cost_obs + 4*cost_max + 4*cost_min#+ 1.0*clearance_cost#+ 0.1*cost_fin# + cost_fin_2 #+ 1.0*clearance_cost#+ 0.1 * cost_velocity#+ 0.1* cost_fin 
       
        return cost_batch
    ########################

    @partial(jit, static_argnums=(0,))
    def comp_prod(self, diffs, d ):

        term_1 = jnp.expand_dims(diffs, axis = 1)
        term_2 = jnp.expand_dims(diffs, axis = 0)

        prods = d * jnp.outer(term_1,term_2)

        return prods 

    @partial(jit, static_argnums=(0,))	
    def compute_shifted_samples(self, key, cost_batch, c_x_ellite, c_y_ellite, c_z_ellite, x_obs_trajectory_1, y_obs_trajectory_1, z_obs_trajectory_1, x_obs_trajectory_2, y_obs_trajectory_2, z_obs_trajectory_2, x_obs_trajectory_3, y_obs_trajectory_3, z_obs_trajectory_3, i, c_mean_prev, c_cov_prev):

                                        
        c_ellite = jnp.hstack(( c_x_ellite, c_y_ellite, c_z_ellite  ))

        beta_param = jnp.min(cost_batch)
        d = cost_batch 
        d = jnp.exp(-(1/self.lamda) * (d-beta_param) )
        sum_d= jnp.sum(d, axis = 0) 
        c_mean = (1- self.alpha) * c_mean_prev + self.alpha * jnp.sum( (c_ellite * d[:,jnp.newaxis]) , axis= 0)/ sum_d

        diffs = (c_ellite - c_mean)
        prod_result = self.vec_product(diffs, d)
        c_cov =   (1- self.alpha) * c_cov_prev + self.alpha * ( (jnp.sum( prod_result , axis = 0)) / sum_d) 


        c_ellite_shift_temp = jax.random.multivariate_normal(key, c_mean, c_cov, (self.initial_up_sampling*self.num_batch, ))

        c_x_ellite_shift_temp = c_ellite_shift_temp[:, 0:11]
        c_y_ellite_shift_temp = c_ellite_shift_temp[:, 11:22]
        c_z_ellite_shift_temp = c_ellite_shift_temp[:, 22:33]

        x_guess_temp = jnp.dot(self.P, c_x_ellite_shift_temp.T).T 
        y_guess_temp = jnp.dot(self.P, c_y_ellite_shift_temp.T).T 
        z_guess_temp = jnp.dot(self.P, c_z_ellite_shift_temp.T).T 

        dist_x_temp_1 = (x_guess_temp-x_obs_trajectory_1[:,jnp.newaxis])
        dist_y_temp_1 = (y_guess_temp-y_obs_trajectory_1[:,jnp.newaxis])
        dist_z_temp_1 = (z_guess_temp-z_obs_trajectory_1[:,jnp.newaxis])

        dist_x_temp_2 = (x_guess_temp-x_obs_trajectory_2[:,jnp.newaxis])
        dist_y_temp_2 = (y_guess_temp-y_obs_trajectory_2[:,jnp.newaxis])
        dist_z_temp_2 = (z_guess_temp-z_obs_trajectory_2[:,jnp.newaxis])

        dist_x_temp_3 = (x_guess_temp-x_obs_trajectory_3[:,jnp.newaxis])
        dist_y_temp_3 = (y_guess_temp-y_obs_trajectory_3[:,jnp.newaxis])
        dist_z_temp_3 = (z_guess_temp-z_obs_trajectory_3[:,jnp.newaxis])

        dist_x_1 = dist_x_temp_1.transpose(1, 0, 2)
        dist_y_1 = dist_y_temp_1.transpose(1, 0, 2)
        dist_z_1 = dist_z_temp_1.transpose(1, 0, 2)

        dist_x_2 = dist_x_temp_2.transpose(1, 0, 2)
        dist_y_2 = dist_y_temp_2.transpose(1, 0, 2)
        dist_z_2 = dist_z_temp_2.transpose(1, 0, 2)

        dist_x_3 = dist_x_temp_3.transpose(1, 0, 2)
        dist_y_3 = dist_y_temp_3.transpose(1, 0, 2)
        dist_z_3 = dist_z_temp_3.transpose(1, 0, 2)

        dist_x_1 = dist_x_1.reshape(self.num_batch * self.initial_up_sampling, self.num*self.num_obs_1)
        dist_y_1 = dist_y_1.reshape(self.num_batch * self.initial_up_sampling, self.num*self.num_obs_1)
        dist_z_1 = dist_z_1.reshape(self.num_batch * self.initial_up_sampling, self.num*self.num_obs_1)

        dist_x_2 = dist_x_2.reshape(self.num_batch * self.initial_up_sampling, self.num*self.num_obs_2)
        dist_y_2 = dist_y_2.reshape(self.num_batch * self.initial_up_sampling, self.num*self.num_obs_2)
        dist_z_2 = dist_z_2.reshape(self.num_batch * self.initial_up_sampling, self.num*self.num_obs_2)

        dist_x_3 = dist_x_3.reshape(self.num_batch * self.initial_up_sampling, self.num*self.num_obs_3)
        dist_y_3 = dist_y_3.reshape(self.num_batch * self.initial_up_sampling, self.num*self.num_obs_3)
        dist_z_3 = dist_z_3.reshape(self.num_batch * self.initial_up_sampling, self.num*self.num_obs_3)

        dist_obs_1 = -(dist_x_1**2/(self.a_obs_1**2))-(dist_y_1**2/(self.b_obs_1**2))-((dist_z_1**2)/(self.c_obs_1**2))+1
        dist_obs_2 = -(dist_x_2**2/(self.a_obs_2**2))-(dist_y_2**2/(self.b_obs_2**2))-((dist_z_2**2)/(self.c_obs_2**2))+1
        dist_obs_3 = -(dist_x_3**2/(self.a_obs_3**2))-(dist_y_3**2/(self.b_obs_3**2))-((dist_z_3**2)/(self.c_obs_3**2))+1

        dist_obs =  dist_obs_1

        cost_obs_penalty = jnp.linalg.norm(jnp.maximum(jnp.zeros(( self.initial_up_sampling*self.num_batch, self.num*(self.num_obs_1) )), dist_obs), axis = 1)#jnp.linalg.norm(1.6*jnp.minimum(jnp.zeros(( self.initial_up_sampling*self.num_batch, self.num*self.num_obs_1 )), -dist_obs_1), axis = 1)

        idx_ellite = jnp.argsort(cost_obs_penalty)

        c_x_ellite_shift = c_x_ellite_shift_temp[idx_ellite[0:self.num_batch-self.ellite_num_const]]
        c_y_ellite_shift = c_y_ellite_shift_temp[idx_ellite[0:self.num_batch-self.ellite_num_const]]
        c_z_ellite_shift = c_z_ellite_shift_temp[idx_ellite[0:self.num_batch-self.ellite_num_const]]

        sol_x_bar = jnp.vstack(( c_x_ellite, c_x_ellite_shift ))
        sol_y_bar = jnp.vstack(( c_y_ellite, c_y_ellite_shift ))
        sol_z_bar = jnp.vstack(( c_z_ellite, c_z_ellite_shift ))

        x_guess = jnp.dot(self.P_jax, sol_x_bar.T).T
        xdot_guess = jnp.dot(self.Pdot_jax, sol_x_bar.T).T
        xddot_guess = jnp.dot(self.Pddot_jax, sol_x_bar.T).T

        y_guess = jnp.dot(self.P_jax, sol_y_bar.T).T
        ydot_guess = jnp.dot(self.Pdot_jax, sol_y_bar.T).T
        yddot_guess = jnp.dot(self.Pddot_jax, sol_y_bar.T).T

        z_guess = jnp.dot(self.P_jax, sol_z_bar.T).T
        zdot_guess = jnp.dot(self.Pdot_jax, sol_z_bar.T).T
        zddot_guess = jnp.dot(self.Pddot_jax, sol_z_bar.T).T
            

        return sol_x_bar, sol_y_bar, sol_z_bar, x_guess, y_guess, z_guess, xdot_guess, ydot_guess, zdot_guess, xddot_guess, yddot_guess, zddot_guess, c_mean, c_cov



    @partial(jit, static_argnums=(0, ))	
    def compute_cem(self, key, initial_state, x_fin, y_fin, z_fin, lamda_x, lamda_y, lamda_z, x_obs_trajectory_1, y_obs_trajectory_1,  z_obs_trajectory_1,  x_obs_trajectory_2, y_obs_trajectory_2, z_obs_trajectory_2, x_obs_trajectory_3, y_obs_trajectory_3, z_obs_trajectory_3, sol_x_bar, sol_y_bar, sol_z_bar, x_guess, y_guess, z_guess, xdot_guess, ydot_guess, zdot_guess, xddot_guess, yddot_guess, zddot_guess, x_waypoint,  y_waypoint, z_waypoint, arc_vec, c_mean, c_cov):
        
        c_mean_prev = c_mean
        c_cov_prev = c_cov

        
            
        for i in range(0, self.maxiter_cem):

            c_x, c_y, c_z, x, xdot, xddot, y, ydot, yddot, z, zdot, zddot, res_norm_batch, s_x, s_y, s_z =  self.compute_projection_sampling(key, sol_x_bar, sol_y_bar, sol_z_bar, x_obs_trajectory_1, y_obs_trajectory_1, z_obs_trajectory_1, x_obs_trajectory_2, y_obs_trajectory_2, z_obs_trajectory_2, x_obs_trajectory_3, y_obs_trajectory_3, z_obs_trajectory_3, lamda_x, lamda_y, lamda_z, x_guess, y_guess, z_guess, xdot_guess, ydot_guess, zdot_guess, xddot_guess, yddot_guess, zddot_guess, initial_state, x_fin, y_fin, z_fin)

            idx_ellite_projection = jnp.argsort(res_norm_batch)
              
           
            x_ellite = x[idx_ellite_projection[0:self.ellite_num_const]]
            y_ellite = y[idx_ellite_projection[0:self.ellite_num_const]]
            z_ellite = z[idx_ellite_projection[0:self.ellite_num_const]]

            xdot_ellite = xdot[idx_ellite_projection[0:self.ellite_num_const]]
            ydot_ellite = ydot[idx_ellite_projection[0:self.ellite_num_const]]
            zdot_ellite = zdot[idx_ellite_projection[0:self.ellite_num_const]]
        
            xddot_ellite = xddot[idx_ellite_projection[0:self.ellite_num_const]]
            yddot_ellite = yddot[idx_ellite_projection[0:self.ellite_num_const]]
            zddot_ellite = zddot[idx_ellite_projection[0:self.ellite_num_const]]

            c_x_ellite = c_x[idx_ellite_projection[0:self.ellite_num_const]]
            c_y_ellite = c_y[idx_ellite_projection[0:self.ellite_num_const]]
            c_z_ellite = c_z[idx_ellite_projection[0:self.ellite_num_const]]
        
            res_norm_batch_proj = res_norm_batch[idx_ellite_projection[0:self.ellite_num_const]]

            x_guess_flat = x_ellite.reshape(self.ellite_num_const * self.num)
            y_guess_flat = y_ellite.reshape(self.ellite_num_const * self.num)
            z_guess_flat = z_ellite.reshape(self.ellite_num_const * self.num)

            normal_distance, x_project, y_project, z_project = self.vectorized_projection(x_waypoint,  y_waypoint, z_waypoint, x_guess_flat, y_guess_flat, z_guess_flat, arc_vec)

            normal_distance = normal_distance.reshape(self.ellite_num_const , self.num)

            x_project_ellite = x_project.reshape(self.ellite_num_const , self.num)
            y_project_ellite = y_project.reshape(self.ellite_num_const , self.num)
            z_project_ellite = z_project.reshape(self.ellite_num_const , self.num)

            cost_batch  = self.compute_cost_batch(x_ellite, y_ellite, z_ellite, xdot_ellite, ydot_ellite, zdot_ellite,  xddot_ellite, yddot_ellite, zddot_ellite, x_project_ellite, y_project_ellite, z_project_ellite, res_norm_batch_proj, x_fin, y_fin, z_fin, x_obs_trajectory_1, y_obs_trajectory_1, z_obs_trajectory_1, x_obs_trajectory_2, y_obs_trajectory_2, z_obs_trajectory_2, x_obs_trajectory_3, y_obs_trajectory_3, z_obs_trajectory_3 )
                    
            key, subkey = random.split(key)
            sol_x_bar, sol_y_bar, sol_z_bar, x_guess, y_guess, z_guess, xdot_guess, ydot_guess, zdot_guess, xddot_guess, yddot_guess, zddot_guess, c_mean_prev, c_cov_prev = self.compute_shifted_samples(key, cost_batch, c_x_ellite, c_y_ellite, c_z_ellite, x_obs_trajectory_1, y_obs_trajectory_1, z_obs_trajectory_1, x_obs_trajectory_2, y_obs_trajectory_2, z_obs_trajectory_2, x_obs_trajectory_3, y_obs_trajectory_3, z_obs_trajectory_3,  i, c_mean_prev, c_cov_prev)

            idx_min = jnp.argmin(cost_batch)
                  
        x_best = x_ellite[idx_min]
        y_best = y_ellite[idx_min]
        z_best = z_ellite[idx_min]


        return   x_best, y_best, z_best