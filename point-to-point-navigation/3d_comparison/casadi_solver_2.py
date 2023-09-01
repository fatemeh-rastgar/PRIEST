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

    def __init__(self, t_fin, a_max, v_max, a_obs_1, b_obs_1, c_obs_1, a_obs_2, b_obs_2, c_obs_2, a_obs_3, b_obs_3, c_obs_3, num, num_obs_1, num_obs_2, num_obs_3):

        self.t_fin = t_fin
        self.a_max = a_max
        self.v_max = v_max

        self.a_obs_1 = a_obs_1
        self.b_obs_1 = b_obs_1
        self.c_obs_1 = c_obs_1
        
        self.a_obs_2 = a_obs_2
        self.b_obs_2 = b_obs_2
        self.c_obs_2 = c_obs_2

        self.a_obs_3 = a_obs_3
        self.b_obs_3 = b_obs_3
        self.c_obs_3 = c_obs_3


        self.num = num
        self.time_FE = 0.
        self.no_iterations = 0
        self.time_solver = 0. # without FE
        self.problem_name = ""
        self.solver = ""

        self.num_obs_1 = num_obs_1
        self.num_obs_2 = num_obs_2
        self.num_obs_3 = num_obs_3


        self.a_bound = 3.5
        self.b_bound = 3.5
        self.c_bound = 1.2

        self.x_center = 3.5
        self.y_center = 3.5
        self.z_center = 1.2





    #################################
    def SolveRockit(self, initial_state, x_fin, y_fin, z_fin, x_initial_compare, y_initial_compare, z_initial_compare, x_obs_init_1, y_obs_init_1, z_obs_init_1, x_obs_init_2, y_obs_init_2, z_obs_init_2, x_obs_init_3, y_obs_init_3, z_obs_init_3):

        print("Rockit")
        x_init, y_init, z_init, vx_init, vy_init, vz_init, ax_init, ay_init, az_init = initial_state
        
        ocp = rockit.Ocp(T=self.t_fin)

        x_rockit = ocp.state()
        y_rockit = ocp.state()
        z_rockit = ocp.state()

        Vx_rockit = ocp.state()
        Vy_rockit = ocp.state()
        Vz_rockit = ocp.state()

        ax_rockit = ocp.control()
        ay_rockit = ocp.control()
        az_rockit = ocp.control()

        ocp.set_der(x_rockit, Vx_rockit)
        ocp.set_der(y_rockit, Vy_rockit)
        ocp.set_der(z_rockit, Vz_rockit)

        ocp.set_der(Vx_rockit, ax_rockit)
        ocp.set_der(Vy_rockit, ay_rockit)
        ocp.set_der(Vz_rockit, az_rockit)

        # Initial constraints
        print(z_init)
        ocp.subject_to(ocp.at_t0(x_rockit)== x_init)
        ocp.subject_to(ocp.at_t0(y_rockit)== y_init)
        ocp.subject_to(ocp.at_t0(z_rockit)== z_init)

        ocp.subject_to(ocp.at_t0(Vx_rockit)== vx_init)
        ocp.subject_to(ocp.at_t0(Vy_rockit)== vy_init)
        ocp.subject_to(ocp.at_t0(Vz_rockit)== vz_init)

        ocp.subject_to(ocp.at_t0(ax_rockit)== ax_init)
        ocp.subject_to(ocp.at_t0(ay_rockit)== ay_init)
        ocp.subject_to(ocp.at_t0(az_rockit)== az_init)

        # Final constraint
        ocp.subject_to(ocp.at_tf(x_rockit)== x_fin)
        ocp.subject_to(ocp.at_tf(y_rockit)== y_fin)
        ocp.subject_to(ocp.at_tf(z_rockit)== z_fin)

        ##initial guess for states and control
        ocp.set_initial(x_rockit, x_initial_compare)
        ocp.set_initial(y_rockit, y_initial_compare)
        ocp.set_initial(z_rockit, z_initial_compare)

        ocp.set_initial(Vx_rockit, vx_init)
        ocp.set_initial(Vy_rockit, vy_init)
        ocp.set_initial(Vz_rockit, vz_init)

        ocp.set_initial(ax_rockit, ax_init)
        ocp.set_initial(ay_rockit, ay_init)
        ocp.set_initial(az_rockit, az_init)

        #####Boundary conditions
        ocp.subject_to((0.02<= np.sqrt(Vx_rockit**2 + Vy_rockit**2+ Vz_rockit**2)) <= self.v_max)
        ocp.subject_to((ax_rockit**2 + ay_rockit**2+ az_rockit**2) <= self.a_max)

        r_wall = np.sqrt( self.a_obs_1 **2 + self.b_obs_1**2 + self.c_obs_1**2)
        r2 = np.sqrt( self.a_obs_2 **2 + self.b_obs_2**2 + self.c_obs_2**2)
        r_obs = np.sqrt( self.a_obs_3 **2 + self.b_obs_3**2 + self.c_obs_3**2)

        p = vertcat(x_rockit,y_rockit, z_rockit)


        for i in range(x_obs_init_1.shape[0]):

            dist_x = -(x_rockit - x_obs_init_1[i])**2/ self.a_obs_1**2
            dist_y = -(y_rockit - y_obs_init_1[i])**2/ self.b_obs_1**2
            dist_z = -(z_rockit - z_obs_init_1[i])**2/ self.c_obs_1**2
            ocp.subject_to( dist_x + dist_y + dist_z + 1 <=0 )

       
        ocp.subject_to( (0.0<= x_rockit) <= 6.2)
        ocp.subject_to( (0.0<= y_rockit) <=6.2)
        ocp.subject_to( (0.0<= z_rockit) <= 2.8)
            
        # for i in range(x_obs_init_2.shape[0]):

        #     dist_x2 = -(x_rockit - x_obs_init_2[i])**2/ self.a_obs_2**2
        #     dist_y2 = -(y_rockit - y_obs_init_2[i])**2/ self.b_obs_2**2
        #     dist_z2 = -(z_rockit - z_obs_init_2[i])**2/ self.c_obs_2**2
        #     ocp.subject_to( dist_x2 + dist_y2 + dist_z2 + 1 <=0 )
        
        # for i in range(x_obs_init_3.shape[0]):

        #     dist_x3 = -(x_rockit - x_obs_init_3[i])**2/ self.a_obs_3**2
        #     dist_y3 = -(y_rockit - y_obs_init_3[i])**2/ self.b_obs_3**2
        #     dist_z3 = -(z_rockit - z_obs_init_3[i])**2/ self.c_obs_3**2
        #     ocp.subject_to( dist_x3 + dist_y3 + dist_z3 + 1 <=0 )

        
        # for i in range(x_obs_init_2.shape[0]):
        #     ocp.subject_to(np.sqrt(sumsqr(p-p_obs_2[i,:].T))>=r2)
        # for i in range(x_obs_init_3.shape[0]):
        #     ocp.subject_to(np.sqrt(sumsqr(p-p_obs_3[i,:].T))>=r_obs)
 
        # ###########
        ocp.add_objective(ocp.sum( ax_rockit**2 + ay_rockit**2+ az_rockit**2)) 

        # Pick a solution method
        ocp.solver('ipopt', {'expand' : True, 'ipopt':{'hessian_approximation': 'limited-memory', 'tol':1e-3, 'hessian_constant':"no"}})

        # Make it concrete for this ocp
        
        ocp.method(MultipleShooting(N=self.num,M=1,intg='expl_euler'))

        # solve
        try:
            sol = ocp.solve()

            ts, xs = sol.sample(x_rockit, grid='control')
            ts, ys = sol.sample(y_rockit, grid='control')
            ts, zs = sol.sample(z_rockit, grid='control')
        except:
            ocp.show_infeasibilities(1e-4)
            print("infeasible results")

            sol = ocp.non_converged_solution
            ts, xs = sol.sample(x_rockit, grid='control')
            ts, ys = sol.sample(y_rockit, grid='control')
            ts, zs = sol.sample(z_rockit, grid='control')

            # xs = np.ones(self.num)
            # ys = np.linspace(0,10,self.num)
            # zs = 7*np.ones(self.num)

        return sol, xs, ys, zs
    ############################################
    def distsq(self, p1, p2, r):
        # new_p1 = p1/ r[0]
        # new_p2 = 
        dist_x = (p1[0] - p2[0])**2 / r[0]**2
        dist_y = (p1[1] - p2[1])**2 / r[1]**2
        dist_z = (p1[2] - p2[2])**2 / r[2]**2
        return dist_x+dist_y+dist_z   
    def csmax(self, x1, x2):
        return cs.if_else(x1>x2, x1, x2)
    def sqrt_special(self, x):
        return cs.if_else(x>0.0, cs.sqrt(x), x)
    
    def sqrt_special2(self, x):
        return cs.if_else(x>0.0, cs.sqrt(x), x)
    #################################################################
    def SolveFatrop(self,initial_state, x_fin, y_fin, z_fin, x_initial_compare, y_initial_compare, z_initial_compare, x_obs_init_1, y_obs_init_1, z_obs_init_1, x_obs_init_2, y_obs_init_2, z_obs_init_2, x_obs_init_3, y_obs_init_3, z_obs_init_3  ):

        print("Fatrop")
        x_init, y_init, z_init, vx_init, vy_init, vz_init, ax_init, ay_init, az_init = initial_state

        ocp_fatrop = rockit.Ocp(T=self.t_fin)
    
        x_fatrop = ocp_fatrop.state()
        y_fatrop = ocp_fatrop.state()
        z_fatrop = ocp_fatrop.state()

        Vx_fatrop = ocp_fatrop.state()
        Vy_fatrop = ocp_fatrop.state()
        Vz_fatrop = ocp_fatrop.state()

        ax_fatrop = ocp_fatrop.control()
        ay_fatrop = ocp_fatrop.control()
        az_fatrop = ocp_fatrop.control()

        ocp_fatrop.set_der(x_fatrop, Vx_fatrop)
        ocp_fatrop.set_der(y_fatrop, Vy_fatrop)
        ocp_fatrop.set_der(z_fatrop, Vz_fatrop)

        ocp_fatrop.set_der(Vx_fatrop, ax_fatrop)
        ocp_fatrop.set_der(Vy_fatrop, ay_fatrop)
        ocp_fatrop.set_der(Vz_fatrop, az_fatrop)

        # Initial constraints
        ocp_fatrop.subject_to(ocp_fatrop.at_t0(x_fatrop)== x_init)
        ocp_fatrop.subject_to(ocp_fatrop.at_t0(y_fatrop)== y_init)
        ocp_fatrop.subject_to(ocp_fatrop.at_t0(z_fatrop)== z_init)

        ocp_fatrop.subject_to(ocp_fatrop.at_t0(Vx_fatrop)== vx_init)
        ocp_fatrop.subject_to(ocp_fatrop.at_t0(Vy_fatrop)== vy_init)
        ocp_fatrop.subject_to(ocp_fatrop.at_t0(Vz_fatrop)== vz_init)

        ocp_fatrop.subject_to(ocp_fatrop.at_t0(ax_fatrop)== ax_init)
        ocp_fatrop.subject_to(ocp_fatrop.at_t0(ay_fatrop)== ay_init)
        ocp_fatrop.subject_to(ocp_fatrop.at_t0(az_fatrop)== az_init)

        # Final constraint
        ocp_fatrop.subject_to(ocp_fatrop.at_tf(x_fatrop)== x_fin)
        ocp_fatrop.subject_to(ocp_fatrop.at_tf(y_fatrop)== y_fin)
        ocp_fatrop.subject_to(ocp_fatrop.at_tf(z_fatrop)== z_fin)

        ##initial guess for states and control
        x_guess = x_initial_compare 
        y_guess = y_initial_compare
        z_guess = z_initial_compare

        ocp_fatrop.set_initial(x_fatrop, x_guess)
        ocp_fatrop.set_initial(y_fatrop, y_guess)
        ocp_fatrop.set_initial(z_fatrop, z_guess)

        ocp_fatrop.set_initial(Vx_fatrop, vx_init)
        ocp_fatrop.set_initial(Vy_fatrop, vy_init)
        ocp_fatrop.set_initial(Vz_fatrop, vz_init)

        ocp_fatrop.set_initial(ax_fatrop, ax_init)
        ocp_fatrop.set_initial(ay_fatrop, ay_init)
        ocp_fatrop.set_initial(az_fatrop, az_init)

        ####Boundary conditions
        ocp_fatrop.subject_to((0.02< np.sqrt(Vx_fatrop**2 + Vy_fatrop**2+ Vz_fatrop**2)) < self.v_max)
        ocp_fatrop.subject_to((ax_fatrop**2 + ay_fatrop**2+ az_fatrop**2) < self.a_max)

        ##########################################################################
        r_obs = horzcat( self.a_obs_1 ,  self.b_obs_1, self.c_obs_1)
        r2 = horzcat( self.a_bound ,  self.b_bound, self.c_bound)
        #r_wall = horzcat( self.a_obs_3 ,  self.b_obs_3, self.c_obs_3)

        p = vertcat(x_fatrop,y_fatrop, z_fatrop)
        p_obs_3 = horzcat(x_obs_init_1, y_obs_init_1, z_obs_init_1)
        p_center = horzcat(self.x_center, self.y_center, self.z_center)
        # p_obs_2 = horzcat(x_obs_init_2, y_obs_init_2, z_obs_init_2)
        # p_obs_1 = horzcat(x_obs_init_3, y_obs_init_3, z_obs_init_3)

        # softconstr_1 = (-self.distsq(p[:3], transpose(p_center[0,:]), r2  ))+(1)

        # softconstr_1 = self.sqrt_special((self.distsq(p[:3], transpose(p_obs_1[0,:]), r_wall  )))-(1)
        # softconstr_2 = self.sqrt_special((self.distsq(p[:3], transpose(p_obs_1[1,:]), r_wall )))-(1)
        # softconstr_3 = self.sqrt_special((self.distsq(p[:3], transpose(p_obs_1[2,:]), r_wall )))-(1)
        # softconstr_4 = self.sqrt_special((self.distsq(p[:3], transpose(p_obs_1[3,:]), r_wall )))-(1)
        # softconstr_5 = self.sqrt_special((self.distsq(p[:3], transpose(p_obs_1[4,:]), r_wall )))-(1)
        # softconstr_6 = self.sqrt_special((self.distsq(p[:3], transpose(p_obs_1[5,:]), r_wall )))-(1)
        # softconstr_7 = self.sqrt_special((self.distsq(p[:3], transpose(p_obs_1[6,:]), r_wall )))-(1)
        # softconstr_8 = self.sqrt_special((self.distsq(p[:3], transpose(p_obs_1[7,:]), r_wall )))-(1)
        # softconstr_9 = self.sqrt_special((self.distsq(p[:3], transpose(p_obs_1[8,:]), r_wall )))-(1)
        # softconstr_10 = self.sqrt_special((self.distsq(p[:3], transpose(p_obs_1[9,:]), r_wall )))-(1)
        # softconstr_11 = self.sqrt_special((self.distsq(p[:3], transpose(p_obs_1[10,:]), r_wall )))-(1)
        # softconstr_12 = self.sqrt_special((self.distsq(p[:3], transpose(p_obs_1[11,:]), r_wall )))-(1)
        # softconstr_13 = self.sqrt_special((self.distsq(p[:3], transpose(p_obs_1[12,:]), r_wall )))-(1)
        # softconstr_14 = self.sqrt_special((self.distsq(p[:3], transpose(p_obs_1[13,:]), r_wall )))-(1)
        # softconstr_15 = self.sqrt_special((self.distsq(p[:3], transpose(p_obs_1[14,:]), r_wall )))-(1)
        # # softconstr_16 = self.sqrt_special((self.distsq(p[:3], transpose(p_obs_1[15,:]), r_wall )))-(1)
        softconstr_17 = self.sqrt_special((self.distsq(p[:3], transpose(p_obs_3[16,:]), r_obs )))-(1)
        softconstr_18 = self.sqrt_special((self.distsq(p[:3], transpose(p_obs_3[17,:]), r_obs )))-(1)
        softconstr_19 = self.sqrt_special((self.distsq(p[:3], transpose(p_obs_3[18,:]), r_obs )))-(1)
        softconstr_20 = self.sqrt_special((self.distsq(p[:3], transpose(p_obs_3[19,:]), r_obs )))-(1)
        softconstr_21 = self.sqrt_special((self.distsq(p[:3], transpose(p_obs_3[20,:]), r_obs )))-(1)
        softconstr_22 = self.sqrt_special((self.distsq(p[:3], transpose(p_obs_3[21,:]), r_obs )))-(1)
        softconstr_23 = self.sqrt_special((self.distsq(p[:3], transpose(p_obs_3[22,:]), r_obs )))-(1)
        softconstr_24 = self.sqrt_special((self.distsq(p[:3], transpose(p_obs_3[23,:]), r_obs )))-(1)
        softconstr_25 = self.sqrt_special((self.distsq(p[:3], transpose(p_obs_3[24,:]), r_obs )))-(1)
        # softconstr_26 = self.sqrt_special((self.distsq(p[:3], transpose(p_obs_1[25,:]), r_wall )))-(1)
        # softconstr_27 = self.sqrt_special((self.distsq(p[:3], transpose(p_obs_1[26,:]), r_wall )))-(1)
        # softconstr_28 = self.sqrt_special((self.distsq(p[:3], transpose(p_obs_1[27,:]), r_wall )))-(1)
        # softconstr_29 = self.sqrt_special((self.distsq(p[:3], transpose(p_obs_1[28,:]), r_wall )))-(1)
        # softconstr_30 = self.sqrt_special((self.distsq(p[:3], transpose(p_obs_1[29,:]) , r_wall)))-(1)
        # softconstr_31 = self.sqrt_special((self.distsq(p[:3], transpose(p_obs_1[30,:]), r_wall )))-(1)
        softconstr_32 = self.sqrt_special((self.distsq(p[:3], transpose(p_obs_3[0,:]), r_obs )))-(1)
        softconstr_33 = self.sqrt_special((self.distsq(p[:3], transpose(p_obs_3[1,:]), r_obs )))-(1)
        softconstr_34 = self.sqrt_special((self.distsq(p[:3], transpose(p_obs_3[2,:]), r_obs )))-(1)
        softconstr_35 = self.sqrt_special((self.distsq(p[:3], transpose(p_obs_3[3,:]), r_obs )))-(1)
        softconstr_36 = self.sqrt_special((self.distsq(p[:3], transpose(p_obs_3[4,:]), r_obs )))-(1)
        softconstr_37 = self.sqrt_special((self.distsq(p[:3], transpose(p_obs_3[5,:]), r_obs )))-(1)
        softconstr_38 = self.sqrt_special((self.distsq(p[:3], transpose(p_obs_3[6,:]), r_obs )))-(1)
        softconstr_39 = self.sqrt_special((self.distsq(p[:3], transpose(p_obs_3[7,:]), r_obs )))-(1)
        softconstr_40 = self.sqrt_special((self.distsq(p[:3], transpose(p_obs_3[8,:]), r_obs )))-(1)
        softconstr_41 = self.sqrt_special((self.distsq(p[:3], transpose(p_obs_3[9,:]), r_obs )))-(1)
        softconstr_42 = self.sqrt_special((self.distsq(p[:3], transpose(p_obs_3[10,:]), r_obs )))-(1)
        softconstr_43 = self.sqrt_special((self.distsq(p[:3], transpose(p_obs_3[11,:]), r_obs )))-(1)
        softconstr_44 = self.sqrt_special((self.distsq(p[:3], transpose(p_obs_3[12,:]), r_obs )))-(1)
        softconstr_45 = self.sqrt_special((self.distsq(p[:3], transpose(p_obs_3[13,:]), r_obs )))-(1)
        softconstr_46 = self.sqrt_special((self.distsq(p[:3], transpose(p_obs_3[14,:]), r_obs )))-(1)
        softconstr_47 = self.sqrt_special((self.distsq(p[:3], transpose(p_obs_3[15,:]), r_obs )))-(1)
        # softconstr_48 = self.sqrt_special((self.distsq(p[:3], transpose(p_obs_1[31,:]), r_obs )))-(1)
        # softconstr_49 = self.sqrt_special((self.distsq(p[:3], transpose(p_obs_2[0,:]), r2 )))-(1)
        # softconstr_50 = self.sqrt_special((self.distsq(p[:3], transpose(p_obs_2[1,:]), r2 )))-(1)

        #constr = cs.vertcat(softconstr_1, softconstr_2, softconstr_3,softconstr_4, softconstr_5, softconstr_6, softconstr_7, softconstr_8, softconstr_9, softconstr_10, softconstr_11, softconstr_12, softconstr_13, softconstr_14, softconstr_15, softconstr_16, softconstr_17, softconstr_18, softconstr_19, softconstr_20, 
                            #softconstr_21, softconstr_22, softconstr_23,softconstr_24, softconstr_25, softconstr_26, softconstr_27, softconstr_28, softconstr_29, softconstr_30, softconstr_31, softconstr_32, softconstr_33, softconstr_34, softconstr_35, softconstr_36, softconstr_37, softconstr_38, softconstr_39, softconstr_40, 
                            #softconstr_41, softconstr_42, softconstr_43,softconstr_44, softconstr_45, softconstr_46, softconstr_47, softconstr_48, softconstr_49, softconstr_50)
        constr = cs.vertcat(softconstr_17, softconstr_18, softconstr_19, softconstr_20, 
                            softconstr_21, softconstr_22, softconstr_23,softconstr_24, softconstr_25, 
            softconstr_32, softconstr_33, softconstr_34, softconstr_35, softconstr_36, softconstr_37, softconstr_38, softconstr_39, softconstr_40, 
                            softconstr_41, softconstr_42, softconstr_43,softconstr_44, softconstr_45, softconstr_46, softconstr_47)

        n_constr = constr.shape[0]
        n = ocp_fatrop.control(n_constr)
        ocp_fatrop.add_objective(ocp_fatrop.sum(1e1*cs.sum1(n), include_last=False))
        ocp_fatrop.subject_to(n>0, include_last=False)
        ocp_fatrop.subject_to(constr+n > 0, include_last=False)
        ocp_fatrop.subject_to( (0.0<= x_fatrop) <= 6.2)
        ocp_fatrop.subject_to( (0.0<= y_fatrop) <= 6.2)
        ocp_fatrop.subject_to( (0.0<= z_fatrop) <= 2.8)

        ocp_fatrop.add_objective(ocp_fatrop.sum( ax_fatrop**2 + ay_fatrop**2 + az_fatrop**2))
        
        method = rockit.external_method('fatrop', N=100-1)
        ocp_fatrop.method(method)
        ocp_fatrop.solve()
        sol = ocp_fatrop.solve()

        ts, xs = sol.sample(x_fatrop, grid='control')
        ts, ys = sol.sample(y_fatrop, grid='control')
        ts, zs = sol.sample(z_fatrop, grid='control')


        return sol, xs, ys, zs#, stats.time_solver
    
    ##########################################