from numpy import random,zeros,argpartition,diag,ones,inf,roots,tensordot,eye,ceil,argmax,\
    array,fill_diagonal,outer,copy,emath,trace,sign,maximum,linspace,where,savez
import time
from scipy.linalg import norm

import cvxpy as cp

from get_dsm import cw_linear_BM_seq,cw_linear_BM,cw_quadratic_BM_seq,cw_quadratic_BM

#%% test code

T = 10
up_to = 20

cpu_time = zeros((6,up_to,T))
wall_time = zeros((6,up_to,T))

for i in range(up_to):
    for t in range(T):
        print(" i = ",i," retry = ",t)
        n = 250 * (i+1)
        m = int(n//8)
        
        # random
        A = random.randn(m,n)
        
        # A2 = A.T.dot(A) + eye(n) * n ** 0.5
        A2 = A.T.dot(A)
        
        c = ones(n)
        if n <= 2500:
            # cvxpy linear

            lineer = cp.Variable(n)
            
            obj_l = cp.Minimize(c @ lineer)
            constraints_l = [cp.diag(lineer) - A2 >> 0]
            prob_l = cp.Problem(obj_l, constraints_l)
            
            wall0 = time.time()
            cpu0 = time.process_time()
            
            prob_l.solve(solver=cp.MOSEK)
            
            cpu1 = time.process_time()
            wall1 = time.time()
            
            lineer0 = lineer.value
            
            print("CVX lineer wall",wall1-wall0)
            print("CVX lineer cpu",cpu1-cpu0)
    
            # cvxpy quadratic
            quad = cp.Variable(n)
            
            obj_q = cp.Minimize(cp.norm(quad,2))
            constraints_q = [cp.diag(quad)-A2 >> 0]
            
            prob_q = cp.Problem(obj_q, constraints_q)
            
            wall2 = time.time()
            cpu2 = time.process_time()
            
            prob_q.solve(solver=cp.MOSEK)
            
            cpu3 = time.process_time()
            wall3 = time.time()
            
            print("CVX quad wall",wall3-wall2)
            print("CVX quad cpu",cpu3-cpu2)
            
            quad0 = quad.value
        
        # Block Coordinate Maximization (BCM) Linear
        
        cpu4 = time.process_time()
        wall4 = time.time()
        GL_seq,ZL_seq,primals_L_seq,obj_L_seq = cw_linear_BM_seq(A2,c ** 0.5)
        cpu5 = time.process_time()
        wall5 = time.time()
            
        # Conditional Gradient (CG) (parallel BCM for linear)
        
        cpu6 = time.process_time()
        wall6 = time.time()
        GL,ZL,primals_L,obj_L = cw_linear_BM(A2,c ** 0.5)
        cpu7 = time.process_time()
        wall7 = time.time()
        
        # Block Coordinate Maximization Quadratic
        
        cpu8 = time.process_time()
        wall8 = time.time()
        GQ_seq,ZQ_seq,primals_Q_seq,obj_Q_seq = cw_quadratic_BM_seq(A2)
        cpu9 = time.process_time()
        wall9 = time.time()
        
        # CG & BCM Hybrid (parallel BCM for quadratic)
        
        cpu10 = time.process_time()
        wall10 = time.time()
        GQ,ZQ,primals_Q,obj_Q = cw_quadratic_BM(A2)
        cpu11 = time.process_time()
        wall11 = time.time()
        
        cpu_time[0,i,t] = cpu1-cpu0
        cpu_time[1,i,t] = cpu3-cpu2
        cpu_time[2,i,t] = cpu5-cpu4
        cpu_time[3,i,t] = cpu7-cpu6
        cpu_time[4,i,t] = cpu9-cpu8
        cpu_time[5,i,t] = cpu11-cpu10
        
        wall_time[0,i,t] = wall1-wall0
        wall_time[1,i,t] = wall3-wall2
        wall_time[2,i,t] = wall5-wall4
        wall_time[3,i,t] = wall7-wall6
        wall_time[4,i,t] = wall9-wall8
        wall_time[5,i,t] = wall11-wall10

order = ["cp_linear","cp_quadratic","BCM_linear","CG_linear","BCM_quadratic","BCM_parallel"]
# savez("compute_dsm_T_"+str(T)+"up_to"+str(up_to),cpu_time = cpu_time,wall_time = wall_time,order = order)
savez("compute_dsm_2500_T_"+str(T)+"up_to"+str(up_to),cpu_time = cpu_time,wall_time = wall_time,order = order)