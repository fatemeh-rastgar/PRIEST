import rockit
from rockit import *
from casadi import vertcat, sumsqr, horzcat
from scipy.io import loadmat
import scipy
from pylab import *
import casadi as cs
import string
import sys
import os
import numpy as np

class stats1():

    def __init__(self):
        self.time_FE = 0.
        self.no_iterations = 0
        self.time_solver = 0. # without FE
        self.problem_name = ""
        self.solver = ""


    #################################################################
    def __str__(self) -> str:
        return  str(self.problem_name) + " solved with " + str(self.solver) + \
        "\nno_iterations: " + str(self.no_iterations) + \
        "\ntime_FE:       " + str(self.time_FE) + \
        "\ntime_solver:   " + str(self.time_solver) + \
        "\ntime_total:   " + str(self.time_solver + self.time_FE) 

class Solvers_func():

    def __init__(self, t_fin, a_max, v_max, a_obs, num, num_obs):

        self.t_fin = t_fin
        self.a_max = a_max
        self.v_max = v_max
        self.a_obs = a_obs
        self.num = num
        self.time_FE = 0.
        self.no_iterations = 0
        self.time_solver = 0. # without FE
        self.problem_name = ""
        self.solver = ""
        self.num_obs = num_obs



    #################################
    def SolveRockit(self, initial_state, x_fin, y_fin, x_initial_compare, y_initial_compare, x_obs_init, y_obs_init):

        print("Rockit")
        x_init, y_init, vx_init, vy_init, ax_init, ay_init = initial_state
        
        ocp = rockit.Ocp(T=self.t_fin)

        x_rockit = ocp.state()
        y_rockit = ocp.state()
        Vx_rockit = ocp.state()
        Vy_rockit = ocp.state()
        ax_rockit = ocp.control()
        ay_rockit = ocp.control()

        ocp.set_der(x_rockit, Vx_rockit)
        ocp.set_der(y_rockit, Vy_rockit)
        ocp.set_der(Vx_rockit, ax_rockit)
        ocp.set_der(Vy_rockit, ay_rockit)

        # Initial constraints
        ocp.subject_to(ocp.at_t0(x_rockit)== x_init)
        ocp.subject_to(ocp.at_t0(y_rockit)== y_init)
        ocp.subject_to(ocp.at_t0(Vx_rockit)== vx_init)
        ocp.subject_to(ocp.at_t0(Vy_rockit)== vy_init)
        ocp.subject_to(ocp.at_t0(ax_rockit)== ax_init)
        ocp.subject_to(ocp.at_t0(ay_rockit)== ay_init)

        # Final constraint
        ocp.subject_to(ocp.at_tf(x_rockit)== x_fin)
        ocp.subject_to(ocp.at_tf(y_rockit)== y_fin)

        ##initial guess for states and control
        ocp.set_initial(x_rockit, x_initial_compare)
        ocp.set_initial(y_rockit, y_initial_compare)

        # r0 = self.a_obs
        # ts = np.linspace(0,2*pi,100)
        # p = vertcat(x_rockit,y_rockit)
        # p_obs = horzcat(x_obs_init, y_obs_init)
        # figure( )

        # plot(x_initial_compare, y_initial_compare, "b")
        # for i in range(x_obs_init.shape[0]):
        #     plot(p_obs[i,0]+r0*cos(ts),p_obs[i, 1]+r0*sin(ts),'r-')

        # axis('equal')
        # show(block=True)

        ocp.set_initial(Vx_rockit, vx_init)
        ocp.set_initial(Vy_rockit, vy_init)
        ocp.set_initial(ax_rockit, ax_init)
        ocp.set_initial(ay_rockit, ay_init)

        #####Boundary conditions
        ocp.subject_to((0.02<= np.sqrt(Vx_rockit**2 + Vy_rockit**2)) <= self.v_max)
        ocp.subject_to((ax_rockit**2 + ay_rockit**2) <= self.a_max)

        r0 = self.a_obs
        p = vertcat(x_rockit,y_rockit)
        p_obs = horzcat(x_obs_init, y_obs_init)

        for i in range(x_obs_init.shape[0]):
            ocp.subject_to(np.sqrt(sumsqr(p-p_obs[i,:].T))>=r0)
        
        ###########
        ocp.add_objective(ocp.sum( ax_rockit**2 + ay_rockit**2)) 

        # Pick a solution method
        ocp.solver('ipopt', {'expand' : True, 'ipopt':{'hessian_approximation': 'limited-memory', 'tol':1e-3, 'hessian_constant':"no"}})

        # Make it concrete for this ocp
        
        ocp.method(MultipleShooting(N=self.num,M=1,intg='expl_euler'))

        # solve
        try:
            sol = ocp.solve()

            ts, xs = sol.sample(x_rockit, grid='control')
            ts, ys = sol.sample(y_rockit, grid='control')
        except:
            ocp.show_infeasibilities(1e-4)
            print("infeasible results")

            sol = ocp.non_converged_solution
            ts, xs = sol.sample(x_rockit, grid='control')
            ts, ys = sol.sample(y_rockit, grid='control')

        return sol, xs, ys
    ############################################
    def distsq(self, p1, p2):
        return cs.sum1((p2-p1)**2)  
    def csmax(self, x1, x2):
        return cs.if_else(x1>x2, x1, x2)
    def sqrt_special(self, x):
        return cs.if_else(x>0.0, cs.sqrt(x), x)
    #################################################################
    def SolveFatrop(self,initial_state, x_fin, y_fin, x_initial_compare, y_initial_compare, x_obs_init, y_obs_init ):

        print("Fatrop")
        x_init, y_init, vx_init, vy_init, ax_init, ay_init = initial_state

        ocp_fatrop = rockit.Ocp(T=self.t_fin)
    
        x_fatrop = ocp_fatrop.state()
        y_fatrop = ocp_fatrop.state()

        Vx_fatrop = ocp_fatrop.state()
        Vy_fatrop = ocp_fatrop.state()

        ax_fatrop = ocp_fatrop.control()
        ay_fatrop = ocp_fatrop.control()

        ocp_fatrop.set_der(x_fatrop, Vx_fatrop)
        ocp_fatrop.set_der(y_fatrop, Vy_fatrop)
        ocp_fatrop.set_der(Vx_fatrop, ax_fatrop)
        ocp_fatrop.set_der(Vy_fatrop, ay_fatrop)

        # Initial constraints
        ocp_fatrop.subject_to(ocp_fatrop.at_t0(x_fatrop)== x_init)
        ocp_fatrop.subject_to(ocp_fatrop.at_t0(y_fatrop)== y_init)
        ocp_fatrop.subject_to(ocp_fatrop.at_t0(Vx_fatrop)== vx_init)
        ocp_fatrop.subject_to(ocp_fatrop.at_t0(Vy_fatrop)== vy_init)
        ocp_fatrop.subject_to(ocp_fatrop.at_t0(ax_fatrop)== ax_init)
        ocp_fatrop.subject_to(ocp_fatrop.at_t0(ay_fatrop)== ay_init)

        # Final constraint
        ocp_fatrop.subject_to(ocp_fatrop.at_tf(x_fatrop)== x_fin)
        ocp_fatrop.subject_to(ocp_fatrop.at_tf(y_fatrop)== y_fin)

        ##initial guess for states and control
        x_guess = x_initial_compare 
        y_guess = y_initial_compare
        ocp_fatrop.set_initial(x_fatrop, x_guess)
        ocp_fatrop.set_initial(y_fatrop, y_guess)
        ocp_fatrop.set_initial(Vx_fatrop, vx_init)
        ocp_fatrop.set_initial(Vy_fatrop, vy_init)
        ocp_fatrop.set_initial(ax_fatrop, ax_init)
        ocp_fatrop.set_initial(ay_fatrop, ay_init)

        ####Boundary conditions
        ocp_fatrop.subject_to((0.02< np.sqrt(Vx_fatrop**2 + Vy_fatrop**2)) < self.v_max)
        ocp_fatrop.subject_to((ax_fatrop**2 + ay_fatrop**2) < self.a_max)

        ##########################################################################
        r0 = self.a_obs 
        p = vertcat(x_fatrop,y_fatrop)
        p_obs = horzcat(x_obs_init, y_obs_init)

        softconstr_1 = self.sqrt_special((self.distsq(p[:2], transpose(p_obs[0,:]) )))-(r0)
        softconstr_2 = self.sqrt_special((self.distsq(p[:2], transpose(p_obs[1,:]) )))-(r0)
        softconstr_3 = self.sqrt_special((self.distsq(p[:2], transpose(p_obs[2,:]) )))-(r0)
        softconstr_4 = self.sqrt_special((self.distsq(p[:2], transpose(p_obs[3,:]) )))-(r0)
        softconstr_5 = self.sqrt_special((self.distsq(p[:2], transpose(p_obs[4,:]) )))-(r0)
        softconstr_6 = self.sqrt_special((self.distsq(p[:2], transpose(p_obs[5,:]) )))-(r0)
        softconstr_7 = self.sqrt_special((self.distsq(p[:2], transpose(p_obs[6,:]) )))-(r0)
        softconstr_8 = self.sqrt_special((self.distsq(p[:2], transpose(p_obs[7,:]) )))-(r0)
        softconstr_9 = self.sqrt_special((self.distsq(p[:2], transpose(p_obs[8,:]) )))-(r0)
        softconstr_10 = self.sqrt_special((self.distsq(p[:2], transpose(p_obs[9,:]) )))-(r0)
        softconstr_11 = self.sqrt_special((self.distsq(p[:2], transpose(p_obs[10,:]) )))-(r0)
        softconstr_12 = self.sqrt_special((self.distsq(p[:2], transpose(p_obs[11,:]) )))-(r0)
        softconstr_13 = self.sqrt_special((self.distsq(p[:2], transpose(p_obs[12,:]) )))-(r0)
        softconstr_14 = self.sqrt_special((self.distsq(p[:2], transpose(p_obs[13,:]) )))-(r0)
        softconstr_15 = self.sqrt_special((self.distsq(p[:2], transpose(p_obs[14,:]) )))-(r0)
        softconstr_16 = self.sqrt_special((self.distsq(p[:2], transpose(p_obs[15,:]) )))-(r0)
        softconstr_17 = self.sqrt_special((self.distsq(p[:2], transpose(p_obs[16,:]) )))-(r0)
        softconstr_18 = self.sqrt_special((self.distsq(p[:2], transpose(p_obs[17,:]) )))-(r0)
        softconstr_19 = self.sqrt_special((self.distsq(p[:2], transpose(p_obs[18,:]) )))-(r0)
        softconstr_20 = self.sqrt_special((self.distsq(p[:2], transpose(p_obs[19,:]) )))-(r0)
        softconstr_21 = self.sqrt_special((self.distsq(p[:2], transpose(p_obs[20,:]) )))-(r0)
        softconstr_22 = self.sqrt_special((self.distsq(p[:2], transpose(p_obs[21,:]) )))-(r0)
        softconstr_23 = self.sqrt_special((self.distsq(p[:2], transpose(p_obs[22,:]) )))-(r0)
        softconstr_24 = self.sqrt_special((self.distsq(p[:2], transpose(p_obs[23,:]) )))-(r0)
        softconstr_25 = self.sqrt_special((self.distsq(p[:2], transpose(p_obs[24,:]) )))-(r0)
        softconstr_26 = self.sqrt_special((self.distsq(p[:2], transpose(p_obs[25,:]) )))-(r0)
        softconstr_27 = self.sqrt_special((self.distsq(p[:2], transpose(p_obs[26,:]) )))-(r0)
        softconstr_28 = self.sqrt_special((self.distsq(p[:2], transpose(p_obs[27,:]) )))-(r0)
        softconstr_29 = self.sqrt_special((self.distsq(p[:2], transpose(p_obs[28,:]) )))-(r0)
        softconstr_30 = self.sqrt_special((self.distsq(p[:2], transpose(p_obs[29,:]) )))-(r0)
        softconstr_31 = self.sqrt_special((self.distsq(p[:2], transpose(p_obs[30,:]) )))-(r0)
        softconstr_32 = self.sqrt_special((self.distsq(p[:2], transpose(p_obs[31,:]) )))-(r0)
        softconstr_33 = self.sqrt_special((self.distsq(p[:2], transpose(p_obs[32,:]) )))-(r0)
        softconstr_34 = self.sqrt_special((self.distsq(p[:2], transpose(p_obs[33,:]) )))-(r0)
        softconstr_35 = self.sqrt_special((self.distsq(p[:2], transpose(p_obs[34,:]) )))-(r0)
        softconstr_36 = self.sqrt_special((self.distsq(p[:2], transpose(p_obs[35,:]) )))-(r0)
        softconstr_37 = self.sqrt_special((self.distsq(p[:2], transpose(p_obs[36,:]) )))-(r0)
        softconstr_38 = self.sqrt_special((self.distsq(p[:2], transpose(p_obs[37,:]) )))-(r0)
        softconstr_39 = self.sqrt_special((self.distsq(p[:2], transpose(p_obs[38,:]) )))-(r0)
        softconstr_40 = self.sqrt_special((self.distsq(p[:2], transpose(p_obs[39,:]) )))-(r0)
        softconstr_41 = self.sqrt_special((self.distsq(p[:2], transpose(p_obs[40,:]) )))-(r0)
        softconstr_42 = self.sqrt_special((self.distsq(p[:2], transpose(p_obs[41,:]) )))-(r0)
        softconstr_43 = self.sqrt_special((self.distsq(p[:2], transpose(p_obs[42,:]) )))-(r0)
        softconstr_44 = self.sqrt_special((self.distsq(p[:2], transpose(p_obs[43,:]) )))-(r0)
        softconstr_45 = self.sqrt_special((self.distsq(p[:2], transpose(p_obs[44,:]) )))-(r0)
        softconstr_46 = self.sqrt_special((self.distsq(p[:2], transpose(p_obs[45,:]) )))-(r0)
        softconstr_47 = self.sqrt_special((self.distsq(p[:2], transpose(p_obs[46,:]) )))-(r0)
        softconstr_48 = self.sqrt_special((self.distsq(p[:2], transpose(p_obs[47,:]) )))-(r0)
        softconstr_49 = self.sqrt_special((self.distsq(p[:2], transpose(p_obs[48,:]) )))-(r0)
        softconstr_50 = self.sqrt_special((self.distsq(p[:2], transpose(p_obs[49,:]) )))-(r0)

        constr = cs.vertcat(softconstr_1, softconstr_2, softconstr_3,softconstr_4, softconstr_5, softconstr_6, softconstr_7, softconstr_8, softconstr_9, softconstr_10, softconstr_11, softconstr_12, softconstr_13, softconstr_14, softconstr_15, softconstr_16, softconstr_17, softconstr_18, softconstr_19, softconstr_20, 
                            softconstr_21, softconstr_22, softconstr_23,softconstr_24, softconstr_25, softconstr_26, softconstr_27, softconstr_28, softconstr_29, softconstr_30, softconstr_31, softconstr_32, softconstr_33, softconstr_34, softconstr_35, softconstr_36, softconstr_37, softconstr_38, softconstr_39, softconstr_40, 
                            softconstr_41, softconstr_42, softconstr_43,softconstr_44, softconstr_45, softconstr_46, softconstr_47, softconstr_48, softconstr_49, softconstr_50)
        n_constr = constr.shape[0]
        n = ocp_fatrop.control(n_constr)
        ocp_fatrop.add_objective(ocp_fatrop.sum(1e2*cs.sum1(n), include_last=False))
        ocp_fatrop.subject_to(n>0, include_last=False)
        ocp_fatrop.subject_to(constr+n > 0, include_last=False)

        ocp_fatrop.add_objective(ocp_fatrop.sum( ax_fatrop**2 + ay_fatrop**2))
        
        method = rockit.external_method('fatrop', N=100-1)
        ocp_fatrop.method(method)
        ocp_fatrop.solve()
        sol = ocp_fatrop.solve()

        # xx = ocp_fatrop.method(method)
        # print(dir(xx))
        
        # stats = stats1()
        # fatropstats = ocp_fatrop.myOCP.GetStats()
        # stats.solver = "fatrop"
        # stats.time_FE = fatropstats.eval_hess_time + fatropstats.eval_jac_time + fatropstats.eval_cv_time + fatropstats.eval_grad_time + fatropstats.eval_obj_time
        # stats.time_solver = fatropstats.time_total - stats.time_FE

        ts, xs = sol.sample(x_fatrop, grid='control')
        ts, ys = sol.sample(y_fatrop, grid='control')

        return sol, xs, ys#, stats.time_solver
    
    ##########################################