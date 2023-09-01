import numpy as np
import jax.numpy as jnp
from functools import partial
import jax
from jax import jit, random
from jax.ops import index_update, index
import bernstein_coeff_order10_arbitinterval
from jax.ops import index_update, index
import time
import matplotlib.pyplot as plt 
from scipy.linalg import block_diag
from scipy.interpolate import UnivariateSpline
from jax import lax
import jax.lax as lax




class Optim():

    def __init__(self, way_point_shape, max_proj, num_goal, v_max, a_max, weight_smoothness,  nvar, num, num_obs, num_batch, a_obs, b_obs, t_fin, vx_obs, vy_obs, vx_des, vy_des ):

        self.v_max = v_max
        self.a_max = a_max
        
        self.vx_obs = vx_obs
        self.vy_obs = vy_obs

        self.vx_des = vx_des
        self.vy_des = vy_des

        self.t_fin = t_fin
        self.num = num
        self.num_obs = num_obs
        self.num_batch = num_batch
        self.num_goal = int(self.num_batch*0.6)
        self.nvar = nvar
        self.maxitet_proj = 15
        self.weight_smoothness =  weight_smoothness

        self.a_obs = a_obs
        self.b_obs = b_obs
    
        self.tot_time = np.linspace(0, t_fin, self.num)
        self.tot_time_copy = self.tot_time.reshape(self.num, 1)

        self.rho_obs = 10
        self.rho_ineq = 1
        self.rho_track = 1

        self.P, self.Pdot, self.Pddot = bernstein_coeff_order10_arbitinterval.bernstein_coeff_order10_new(10, self.tot_time_copy[0], self.tot_time_copy[-1], self.tot_time_copy)
        self.P_jax, self.Pdot_jax, self.Pddot_jax = jnp.asarray(self.P), jnp.asarray(self.Pdot), jnp.asarray(self.Pddot)
        
        self.num_up = 100
        dt_up = self.t_fin/self.num_up
        tot_time_up = np.linspace(0, self.t_fin, self.num_up)
        self.tot_time_up = tot_time_up
        tot_time_copy_up = self.tot_time_up.reshape(self.num_up, 1)
        P_up, Pdot_up, Pddot_up = bernstein_coeff_order10_arbitinterval.bernstein_coeff_order10_new(10, tot_time_copy_up[0], tot_time_copy_up[-1], tot_time_copy_up)
        self.P_up_jax = jnp.asarray(P_up)
        self.Pdot_up_jax = jnp.asarray(Pdot_up)
        self.Pddot_up_jax = jnp.asarray(Pddot_up)

        self.A_obs = jnp.tile(self.P_jax, (self.num_obs, 1))
        self.A_vel = self.Pdot_jax 
        self.A_acc = self.Pddot_jax
        self.A_track = self.P_jax
        self.A_eq = jnp.vstack(( self.P_jax[0], self.Pdot_jax[0], self.Pddot_jax[0], self.P_jax[-1] ))
    
        self.cost_smoothness = self.weight_smoothness*jnp.dot(self.Pddot_jax.T, self.Pddot_jax)
        self.vectorized_projection = jit(jax.vmap(self.compute_contouring_error, in_axes=(None, None, 0, 0, None, ) ))
        self.way_point_shape = way_point_shape
    
    
    @partial(jit, static_argnums=(0,))
    def compute_contouring_error(self, x_waypoint, y_waypoint, x_target_point, y_target_point, arc_vec):

        dist = jnp.sqrt( (x_waypoint - x_target_point)**2 + (y_waypoint - y_target_point)**2  )
        index = jnp.argmin(dist)

        arc_point = arc_vec[index]

        x_project = x_waypoint[index]
        y_project = y_waypoint[index]

        return arc_point, x_project, y_project


    ###############################################################
    @partial(jit, static_argnums=(0,))	
    def compute_boundary_vec(self, initial_state, x_fin, y_fin ):		

        x_init, y_init, vx_init, vy_init, ax_init, ay_init = initial_state

        x_init_vec = x_init*jnp.ones((self.num_goal, 1))
        y_init_vec = y_init*jnp.ones((self.num_goal, 1)) 

        vx_init_vec = vx_init*jnp.ones((self.num_goal, 1))
        vy_init_vec = vy_init*jnp.ones((self.num_goal, 1))

        ax_init_vec = ax_init*jnp.ones((self.num_goal, 1))
        ay_init_vec = ay_init*jnp.ones((self.num_goal, 1))

        x_fin_vec = x_fin*jnp.ones((self.num_goal, 1))
        y_fin_vec = y_fin*jnp.ones((self.num_goal, 1)) 
    
        b_eq_x = jnp.hstack(( x_init_vec, vx_init_vec, ax_init_vec, x_fin_vec ))
        b_eq_y = jnp.hstack(( y_init_vec, vy_init_vec, ay_init_vec, y_fin_vec  ))
        
        return b_eq_x, b_eq_y

    #######################################
    @partial(jit, static_argnums=(0,))
    def compute_x(self, x_obs_trajectory, y_obs_trajectory, d_obs, alpha_obs, alpha_a, d_a, alpha_v, d_v, lamda_x, lamda_y, b_eq_x, b_eq_y ):

        # b_projection_x = sol_x_bar
        # b_projection_y = sol_y_bar

        temp_x_obs = d_obs*jnp.cos(alpha_obs)*self.a_obs
        b_obs_x = x_obs_trajectory.reshape(self.num*(self.num_obs))+temp_x_obs
            
        temp_y_obs = d_obs*jnp.sin(alpha_obs)*self.b_obs
        b_obs_y = y_obs_trajectory.reshape(self.num*(self.num_obs))+temp_y_obs

        b_ax_ineq = d_a*jnp.cos(alpha_a)
        b_ay_ineq = d_a*jnp.sin(alpha_a)

        b_vx_ineq = d_v*jnp.cos(alpha_v)
        b_vy_ineq = d_v*jnp.sin(alpha_v)

        lincost_x =  -lamda_x-self.rho_obs*jnp.dot(self.A_obs.T, b_obs_x.T).T-self.rho_ineq*jnp.dot(self.A_acc.T, b_ax_ineq.T).T-self.rho_ineq*jnp.dot(self.A_vel.T, b_vx_ineq.T).T
        lincost_y =  -lamda_y-self.rho_obs*jnp.dot(self.A_obs.T, b_obs_y.T).T-self.rho_ineq*jnp.dot(self.A_acc.T, b_ay_ineq.T).T-self.rho_ineq*jnp.dot(self.A_vel.T, b_vy_ineq.T).T

        cost = self.cost_smoothness + self.rho_obs * jnp.dot(self.A_obs.T, self.A_obs) + self.rho_ineq * jnp.dot(self.A_vel.T, self.A_vel) + self.rho_ineq * jnp.dot(self.A_acc.T, self.A_acc)
        cost_mat = jnp.vstack(( jnp.hstack(( cost, self.A_eq.T )), jnp.hstack((self.A_eq, jnp.zeros(( 4, 4 )) )) ))
        cost_mat_inv = jnp.linalg.inv(cost_mat)

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


    ###############################################
    @partial(jit, static_argnums=(0,))	
    def initial_alpha_d(self, x_guess, y_guess, xdot_guess, ydot_guess, xddot_guess, yddot_guess, x_obs_trajectory, y_obs_trajectory, lamda_x, lamda_y):

        wc_alpha_temp = (x_guess-x_obs_trajectory[:,jnp.newaxis])
        ws_alpha_temp = (y_guess-y_obs_trajectory[:,jnp.newaxis])

        wc_alpha = wc_alpha_temp.transpose(1, 0, 2)
        ws_alpha = ws_alpha_temp.transpose(1, 0, 2)

        wc_alpha = wc_alpha.reshape(self.num_goal, self.num*(self.num_obs))
        ws_alpha = ws_alpha.reshape(self.num_goal, self.num*(self.num_obs))

        alpha_obs = jnp.arctan2( ws_alpha*self.a_obs, wc_alpha*self.b_obs)
    
        c1_d = 1.0*self.rho_obs*(self.a_obs**2*jnp.cos(alpha_obs)**2 + self.b_obs**2*jnp.sin(alpha_obs)**2 )
        c2_d = 1.0*self.rho_obs*(self.a_obs*wc_alpha*jnp.cos(alpha_obs) + self.b_obs*ws_alpha*jnp.sin(alpha_obs)  )

        d_obs = c2_d/c1_d
        d_obs = jnp.maximum(jnp.ones((self.num_goal,  self.num*(self.num_obs)   )), d_obs   )
        ####################### velocity terms

        wc_alpha_vx = xdot_guess
        ws_alpha_vy = ydot_guess
        alpha_v = jnp.arctan2( ws_alpha_vy, wc_alpha_vx)		
        c1_d_v = 1.0*self.rho_ineq*(jnp.cos(alpha_v)**2 + jnp.sin(alpha_v)**2 )
        c2_d_v = 1.0*self.rho_ineq*(wc_alpha_vx*jnp.cos(alpha_v) + ws_alpha_vy*jnp.sin(alpha_v)  )

        d_v = c2_d_v/c1_d_v
        d_v = jnp.minimum(self.v_max*jnp.ones((self.num_goal, self.num)), d_v   )

        ################# acceleration terms

        wc_alpha_ax = xddot_guess
        ws_alpha_ay = yddot_guess
        alpha_a = jnp.arctan2( ws_alpha_ay, wc_alpha_ax)		
        c1_d_a = 1.0*self.rho_ineq*(jnp.cos(alpha_a)**2 + jnp.sin(alpha_a)**2 )
        c2_d_a = 1.0*self.rho_ineq*(wc_alpha_ax*jnp.cos(alpha_a) + ws_alpha_ay*jnp.sin(alpha_a)  )

        d_a = c2_d_a/c1_d_a
        d_a = jnp.minimum(self.a_max*jnp.ones((self.num_goal, self.num)), d_a   )
        #########################################33
        res_ax_vec = xddot_guess-d_a*jnp.cos(alpha_a)
        res_ay_vec = yddot_guess-d_a*jnp.sin(alpha_a)

        res_vx_vec = xdot_guess-d_v*jnp.cos(alpha_v)
        res_vy_vec = ydot_guess-d_v*jnp.sin(alpha_v)

        res_x_obs_vec = wc_alpha-self.a_obs*d_obs*jnp.cos(alpha_obs)
        res_y_obs_vec = ws_alpha-self.b_obs*d_obs*jnp.sin(alpha_obs)

        lamda_x = lamda_x -self.rho_obs*jnp.dot(self.A_obs.T, res_x_obs_vec.T).T-self.rho_ineq*jnp.dot(self.A_acc.T, res_ax_vec.T).T-self.rho_ineq*jnp.dot(self.A_vel.T, res_vx_vec.T).T
        lamda_y = lamda_y -self.rho_obs*jnp.dot(self.A_obs.T, res_y_obs_vec.T).T-self.rho_ineq*jnp.dot(self.A_acc.T, res_ay_vec.T).T-self.rho_ineq*jnp.dot(self.A_vel.T, res_vy_vec.T).T

        return alpha_obs, d_obs, alpha_a, d_a, alpha_v, d_v, lamda_x, lamda_y
    ###########################################3
    @partial(jit, static_argnums=(0,))	
    def compute_alph_d_proj(self, x, y, xdot, ydot, xddot, yddot, x_obs_trajectory, y_obs_trajectory, lamda_x, lamda_y):
        
        wc_alpha_temp = (x-x_obs_trajectory[:,jnp.newaxis])
        ws_alpha_temp = (y-y_obs_trajectory[:,jnp.newaxis])

        wc_alpha = wc_alpha_temp.transpose(1, 0, 2)
        ws_alpha = ws_alpha_temp.transpose(1, 0, 2)

        wc_alpha = wc_alpha.reshape(self.num_goal, self.num*(self.num_obs))
        ws_alpha = ws_alpha.reshape(self.num_goal, self.num*(self.num_obs))

        alpha_obs = jnp.arctan2( ws_alpha*self.a_obs, wc_alpha*self.b_obs)
        c1_d = 1.0*self.rho_obs*(self.a_obs**2*jnp.cos(alpha_obs)**2 + self.b_obs**2*jnp.sin(alpha_obs)**2 )
        c2_d = 1.0*self.rho_obs*(self.a_obs*wc_alpha*jnp.cos(alpha_obs) + self.b_obs*ws_alpha*jnp.sin(alpha_obs)  )

        d_obs = c2_d/c1_d
        d_obs = jnp.maximum(jnp.ones((self.num_goal,  self.num*(self.num_obs)   )), d_obs   )
        ####################### velocity terms

        wc_alpha_vx = xdot
        ws_alpha_vy = ydot
        alpha_v = jnp.arctan2( ws_alpha_vy, wc_alpha_vx)		
        c1_d_v = 1.0*self.rho_ineq*(jnp.cos(alpha_v)**2 + jnp.sin(alpha_v)**2 )
        c2_d_v = 1.0*self.rho_ineq*(wc_alpha_vx*jnp.cos(alpha_v) + ws_alpha_vy*jnp.sin(alpha_v)  )

        d_v = c2_d_v/c1_d_v
        d_v = jnp.minimum(self.v_max*jnp.ones((self.num_goal, self.num)), d_v   )

        ################# acceleration terms

        wc_alpha_ax = xddot
        ws_alpha_ay = yddot
        alpha_a = jnp.arctan2( ws_alpha_ay, wc_alpha_ax)		
        c1_d_a = 1.0*self.rho_ineq*(jnp.cos(alpha_a)**2 + jnp.sin(alpha_a)**2 )
        c2_d_a = 1.0*self.rho_ineq*(wc_alpha_ax*jnp.cos(alpha_a) + ws_alpha_ay*jnp.sin(alpha_a)  )

        d_a = c2_d_a/c1_d_a

        d_a = jnp.minimum(self.a_max*jnp.ones((self.num_goal, self.num)), d_a   )

        #########################################33
        res_ax_vec = xddot-d_a*jnp.cos(alpha_a)
        res_ay_vec = yddot-d_a*jnp.sin(alpha_a)

        res_vx_vec = xdot-d_v*jnp.cos(alpha_v)
        res_vy_vec = ydot-d_v*jnp.sin(alpha_v)

        res_x_obs_vec = wc_alpha-self.a_obs*d_obs*jnp.cos(alpha_obs)
        res_y_obs_vec = ws_alpha-self.b_obs*d_obs*jnp.sin(alpha_obs)

        lamda_x = lamda_x -self.rho_obs*jnp.dot(self.A_obs.T, res_x_obs_vec.T).T-self.rho_ineq*jnp.dot(self.A_acc.T, res_ax_vec.T).T-self.rho_ineq*jnp.dot(self.A_vel.T, res_vx_vec.T).T
        lamda_y = lamda_y -self.rho_obs*jnp.dot(self.A_obs.T, res_y_obs_vec.T).T-self.rho_ineq*jnp.dot(self.A_acc.T, res_ay_vec.T).T-self.rho_ineq*jnp.dot(self.A_vel.T, res_vy_vec.T).T

        res_obs_vec = jnp.hstack(( res_x_obs_vec, res_y_obs_vec  ))
        res_acc_vec = jnp.hstack(( res_ax_vec,  res_ay_vec  ))
        res_vel_vec = jnp.hstack(( res_vx_vec,  res_vy_vec  ))

        res_norm_batch = 1*jnp.linalg.norm(res_obs_vec, axis =1)+jnp.linalg.norm(res_acc_vec, axis =1)+jnp.linalg.norm(res_vel_vec, axis =1)

        return alpha_obs, d_obs, alpha_a, d_a, alpha_v, d_v, lamda_x, lamda_y, res_norm_batch

    @partial(jit, static_argnums = (0,) )
    def compute_cost_batch(self, x, y, xddot, yddot, x_obs, y_obs, res_norm_batch , x_project, y_project):

        dist_x_temp = (x-x_obs[:,jnp.newaxis])
        dist_y_temp = (y-y_obs[:,jnp.newaxis])
        
        dist_x = dist_x_temp.transpose(1, 0, 2)#[:, :, 0:30]
        dist_y = dist_y_temp.transpose(1, 0, 2)#[:, :, 0:30]

        dist_x = dist_x.reshape(self.num_goal, self.num*self.num_obs)
        dist_y = dist_y.reshape(self.num_goal, self.num*self.num_obs)

        dist_obs = -(dist_x**2)/(self.a_obs**2)- (dist_y**2)/(self.b_obs**2)+1

        cost_obs = jnp.linalg.norm(jnp.maximum(jnp.zeros(( self.num_goal, self.num*self.num_obs  )), dist_obs), axis = 1)

        cost_smoothness =  ( (jnp.linalg.norm(xddot, axis = 1)) + (jnp.linalg.norm(yddot, axis = 1))  )
        cost_track = jnp.linalg.norm(x-x_project, axis = 1)+jnp.linalg.norm(y-y_project, axis = 1) #(x-x_project)**2+(y-y_project)**2
        clearance_cost = - jnp.min(dist_obs, axis = 1)
        cost_batch = res_norm_batch + 10.0*cost_obs + 0.02*cost_smoothness + 0.2*cost_track #+ clearance_cost

        return cost_batch

    ###############################################################################

    @partial( jit, static_argnums = (0,) )
    def SolveOpt1(self, x_waypoint,  y_waypoint, arc_vec, x_samples, y_samples, xdot_samples, ydot_samples, xddot_samples, yddot_samples, x_obs, y_obs,  x_des_traj_init, y_des_traj_init, initial_state, x_fin, y_fin, lamda_x, lamda_y):

        b_eq_x, b_eq_y = self.compute_boundary_vec(initial_state, x_fin, y_fin)
        alpha_obs_init, d_obs_init, alpha_a_init, d_a_init, alpha_v_init, d_v_init, lamda_x_init, lamda_y_init = self.initial_alpha_d(x_samples, y_samples,  xdot_samples, ydot_samples, xddot_samples, yddot_samples, x_obs, y_obs , lamda_x, lamda_y)
        #alpha_obs, d_obs, alpha_a, d_a, alpha_v, d_v, lamda_x, lamda_y = self.initial_alpha_d(x_samples, y_samples,  xdot_samples, ydot_samples, xddot_samples, yddot_samples, x_obs, y_obs , lamda_x, lamda_y)

        def lax_projection(carry, proj_iter ):

            c_x, c_y, x, xdot, xddot, y, ydot, yddot, res_norm_batch, alpha_obs, d_obs, alpha_a, d_a, alpha_v, d_v, lamda_x, lamda_y = carry
            c_x, c_y, x, y, xdot, ydot, xddot, yddot = self.compute_x(x_obs, y_obs, d_obs, alpha_obs, alpha_a, d_a, alpha_v, d_v, lamda_x, lamda_y, b_eq_x, b_eq_y )
            alpha_obs, d_obs, alpha_a, d_a, alpha_v, d_v, lamda_x, lamda_y, res_norm_batch = self.compute_alph_d_proj( x, y, xdot, ydot, xddot, yddot, x_obs, y_obs, lamda_x, lamda_y)
            
            # min_res = jnp.argmin(res_norm_batch)
            # res.append(res_norm_batch)
            # plt.figure(2)
            # plt.figure(x.T, y.T)
            return (c_x, c_y, x, xdot, xddot, y, ydot, yddot, res_norm_batch, alpha_obs, d_obs, alpha_a, d_a, alpha_v, d_v,  lamda_x, lamda_y), x

        carry_init =  jnp.zeros((self.num_goal,self.nvar)), jnp.zeros((self.num_goal,self.nvar)), jnp.zeros((self.num_goal,self.num)),  jnp.zeros((self.num_goal,self.num)),  jnp.zeros((self.num_goal,self.num)),  jnp.zeros((self.num_goal,self.num)),  jnp.zeros((self.num_goal,self.num)),  jnp.zeros((self.num_goal,self.num)),  jnp.zeros(self.num_goal),alpha_obs_init, d_obs_init, alpha_a_init, d_a_init, alpha_v_init, d_v_init, lamda_x_init, lamda_y_init

        carry_fin, result = lax.scan(lax_projection, carry_init, jnp.arange(self.maxitet_proj))
        
        c_x, c_y, x, xdot, xddot, y, ydot, yddot, res_norm_batch, alpha_obs, d_obs, alpha_a, d_a, alpha_v, d_v, lamda_x, lamda_y = carry_fin

        x_sample_flat = x_samples.reshape(self.num_goal* self.num)
        y_sample_flat = y_samples.reshape(self.num_goal* self.num)
        
        normal_distance, x_project, y_project = self.vectorized_projection(x_waypoint,  y_waypoint, x_sample_flat , y_sample_flat, arc_vec)

        x_project = x_project.reshape(self.num_goal, self.num)
        y_project = y_project.reshape(self.num_goal, self.num)
        
        cost_func  = self.compute_cost_batch(x, y, xddot, yddot, x_obs, y_obs, res_norm_batch, x_project, y_project )
        idx_ellite = jnp.argsort(cost_func)

        idx_min = jnp.argmin(cost_func)
        
        x_best = x[idx_min]
        y_best = y[idx_min]

        sol_x_best = c_x[idx_min]
        sol_y_best = c_y[idx_min]


        return sol_x_best, sol_y_best, x_best, y_best, x, y
