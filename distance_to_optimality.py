from numpy import random,zeros,argpartition,diag,ones,inf,roots,tensordot,eye,ceil,argmax,\
    array,fill_diagonal,outer,copy,emath,trace,sign,maximum,linspace,where,savez,cumsum,hstack
import time
from scipy.linalg import solve,norm,eigh,cholesky,lstsq,inv

import matplotlib.pyplot as pylot
import cvxpy as cp

from matplotlib.pyplot import *

#%% same algorithm as in get_dsm but checks distane to optimal solution

def cw_linear_BM(M,w,vector,matrix):
    m,n = M.shape
    B = random.randn(min(int(ceil((2* n +1/4) ** 0.5 - 1/2)  * 1),n),n)
    
    cpu_timer = time.process_time()
    wall_timer = time.time()
    
    cpu_vals = zeros(50)
    wall_vals = zeros(50)
    vector_distance = zeros(50)
    matrix_distance = zeros(50)
    
    Z = B.T.dot(B)
    primals = ((Z * (Z.dot(M))).sum(axis = 0))/norm(Z,axis = 0) ** 2
    vd = norm(primals-vector)/norm(vector)
    md = norm(Z-matrix)/norm(matrix)
    
    
    for t in range(500):
        # print('iter',t)
        B = B.dot(M)
        B = B/norm(B,axis = 0) * w
        
        if t % 10 == 9:
            cpu_stop = time.process_time()
            wall_stop = time.time()
            
            cpu_vals[t//10] = cpu_stop - cpu_timer
            wall_vals[t//10] = wall_stop - wall_timer
            
            Z = B.T.dot(B)
            primals = ((Z * (Z.dot(M))).sum(axis = 0))/norm(Z,axis = 0) ** 2
            
            vector_distance[t//10] = norm(primals-vector)/norm(vector)
            matrix_distance[t//10] = norm(Z-matrix)/norm(matrix)
            
            cpu_timer = time.process_time()
            wall_timer = time.time()
        
    Z = B.T.dot(B)
    obj = tensordot(M,Z)
    primals = ((Z * (Z.dot(M))).sum(axis = 0))/norm(Z,axis = 0) ** 2
    
    return B,Z,primals,obj,cumsum(cpu_vals),cumsum(wall_vals),vector_distance,matrix_distance

#%% same algorithm as in get_dsm but checks distane to optimal solution

def cw_linear_BM_seq(M,w,vector,matrix):
    # do not keep memory for gradient
    m,n = M.shape
    B = random.randn(min(int(ceil((2* n +1/4) ** 0.5 - 1/2))  * 1,n),n)
    B = B/norm(B,axis = 0)
    diag_M = diag(M).copy()
    fill_diagonal(M,0)
    
    cpu_timer = time.process_time()
    wall_timer = time.time()
    
    cpu_vals = zeros(50)
    wall_vals = zeros(50)
    vector_distance = zeros(50)
    matrix_distance = zeros(50)
    
    Z = B.T.dot(B)
    primals = ((Z * (Z.dot(M))).sum(axis = 0))/norm(Z,axis = 0) ** 2
    vd = norm(primals-vector)/norm(vector)
    md = norm(Z-matrix)/norm(matrix)

    for t in range(200):
        for i in range(n):
            c = M[:,[i]]
            g = B.dot(c)
            B[:,[i]] = g/norm(g) * w[i]
        
        if t % 4 == 3:
            cpu_stop = time.process_time()
            wall_stop = time.time()
            cpu_vals[t//4] = cpu_stop - cpu_timer
            wall_vals[t//4] = wall_stop - wall_timer
            
            Z = B.T.dot(B)
            fill_diagonal(M,diag_M)
            primals = ((Z * (Z.dot(M))).sum(axis = 0))/norm(Z,axis = 0) ** 2
            
            fill_diagonal(M,0)
            vector_distance[t//4] = norm(primals-vector)/norm(vector)
            matrix_distance[t//4] = norm(Z-matrix)/norm(matrix)
            
            cpu_timer = time.process_time()
            wall_timer = time.time()
            
            
    fill_diagonal(M,diag_M)
    Z = B.T.dot(B)
    obj = tensordot(M,Z)
    primals = ((Z * (Z.dot(M))).sum(axis = 0))/norm(Z,axis = 0) ** 2
    
    return B,Z,primals,obj,cumsum(cpu_vals),cumsum(wall_vals),vector_distance,matrix_distance

#%% same algorithm as in get_dsm but checks distane to optimal solution

def cw_quadratic_BM(M,vector,matrix):
    m,n = M.shape
    B = random.randn(min(int(ceil((2* n +1/4) ** 0.5 - 1/2) * 1 ),n),n)
    
    rho = 0.0
    # for certain problems, using a small exponential smoothing works better
    # rho = 0.1 
    diag_M = diag(M).copy()
    fill_diagonal(M,0)
    
    
    cpu_timer = time.process_time()
    wall_timer = time.time()
    
    cpu_vals = zeros(50)
    wall_vals = zeros(50)
    vector_distance = zeros(50)
    matrix_distance = zeros(50)
    
    Z = B.T.dot(B)
    primals = ((Z * (Z.dot(M))).sum(axis = 0))/norm(Z,axis = 0) ** 2
    vd = norm(primals-vector)/norm(vector)
    md = norm(Z-matrix)/norm(matrix)
    
    for t in range(500):
        g = B.dot(M)
        ng = norm(g,axis = 0)
        r = (emath.sqrt(3 * (4 * -diag_M ** 3 + 27 * ng ** 2)) + 9 * ng) ** (1/3)
        sigma1 = r/((2 * 9) ** (1/3)) + (2/3) ** (1/3) * diag_M/r
        
        B = rho * B + (1 - rho) * g/ng * sigma1.real
        
        if t % 10 == 9:
            cpu_stop = time.process_time()
            wall_stop = time.time()
            cpu_vals[t//10] = cpu_stop - cpu_timer
            wall_vals[t//10] = wall_stop - wall_timer
            
            Z = B.T.dot(B)
            primals = diag(Z)
            
            vector_distance[t//10] = norm(primals-vector)/norm(vector)
            matrix_distance[t//10] = norm(Z-matrix)/norm(matrix)
            
            cpu_timer = time.process_time()
            wall_timer = time.time()
            
        # if t % 2 == 1:
        #     cpu_stop = time.process_time()
        #     wall_stop = time.time()
        #     cpu_vals[t//2] = cpu_stop - cpu_timer
        #     wall_vals[t//2] = wall_stop - wall_timer
            
        #     Z = B.T.dot(B)
        #     primals = diag(Z)
            
        #     vector_distance[t//2] = norm(primals-vector)/norm(vector)
        #     matrix_distance[t//2] = norm(Z-matrix)/norm(matrix)
            
        #     cpu_timer = time.process_time()
        #     wall_timer = time.time()
            
    fill_diagonal(M,diag_M)
    Z = B.T.dot(B)
    obj = tensordot(M,Z) -1/2 * sum(norm(B,axis = 0) ** 4)
    # print("objective")
    # print(obj)
    primals = diag(Z)
    
    # vector_distance = hstack((vd,vector_distance))
    # matrix_distance = hstack((md,matrix_distance))
    # cpu_vals        = hstack((0,cpu_vals))
    # wall_vals       = hstack((0,wall_vals))
    return B,Z,primals,obj,cumsum(cpu_vals),cumsum(wall_vals),vector_distance,matrix_distance

#%% same algorithm as in get_dsm but checks distane to optimal solution

def cw_quadratic_BM_seq(M,vector,matrix):
    m,n = M.shape
    B = random.randn(min(int(ceil((2* n +1/4) ** 0.5 - 1/2) * 1 ),n),n)
    
    diag_M = diag(M).copy()
    fill_diagonal(M,0)
    
    cpu_timer = time.process_time()
    wall_timer = time.time()
    
    cpu_vals = zeros(50)
    wall_vals = zeros(50)
    vector_distance = zeros(50)
    matrix_distance = zeros(50)
    
    Z = B.T.dot(B)
    primals = ((Z * (Z.dot(M))).sum(axis = 0))/norm(Z,axis = 0) ** 2
    vd = norm(primals-vector)/norm(vector)
    md = norm(Z-matrix)/norm(matrix)
    
    for t in range(200):
        for i in range(n):
            c = M[:,[i]]
            a = diag_M[i]
            g = B.dot(c)
            ng = norm(g)
            r = (emath.sqrt(3 * (4 * -a ** 3 + 27 * ng ** 2)) + 9 * ng) ** (1/3)
            sigma1 = r/((2 * 9) ** (1/3)) + (2/3) ** (1/3) * a/r
            B[:,[i]] = g/ng * sigma1

        if t % 4 == 3:
            cpu_stop = time.process_time()
            wall_stop = time.time()
            cpu_vals[t//4] = cpu_stop - cpu_timer
            wall_vals[t//4] = wall_stop - wall_timer
            
            Z = B.T.dot(B)
            primals = diag(Z)
            
            vector_distance[t//4] = norm(primals-vector)/norm(vector)
            matrix_distance[t//4] = norm(Z-matrix)/norm(matrix)
            
            cpu_timer = time.process_time()
            wall_timer = time.time()
            
            
    fill_diagonal(M,diag_M)
    Z = B.T.dot(B)
    obj = tensordot(M,Z) -1/2 * sum(norm(B,axis = 0) ** 4)
    primals = diag(Z)
    
    return B,Z,primals,obj,cumsum(cpu_vals),cumsum(wall_vals),vector_distance,matrix_distance


#%% test code
 

problem_size = [500,1000,2000]
figs = []
random.seed(0)

for i in range(len(problem_size)):
    print("current i",i)
    n = problem_size[i]
    m = int(n//8)
    
    # random
    A = random.randn(m,n)
    
    A2 = A.T.dot(A)
    c = ones(n)
    if n <= 2000:
        
        # cvxpy linear
        c = ones(n)
        lineer = cp.Variable(n)
        
        obj_l = cp.Minimize(c @ lineer)
        constraints_l = [cp.diag(lineer) - A2 >> 0]
        prob_l = cp.Problem(obj_l, constraints_l)
        
        wall0 = time.time()
        cpu0 = time.process_time()
        
        prob_l.solve(solver=cp.MOSEK)
        
        cpu1 = time.process_time()
        wall1 = time.time()
        
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
        
        linear_vector = lineer.value
        linear_matrix = constraints_l[0].dual_value
        
        quad_vector = quad.value
        quad_matrix = constraints_q[0].dual_value
        quad_matrix = quad_matrix * (quad_vector/diag(quad_matrix)).mean()
    
    # Block Coordinate Maximization (BCM) Linear
    
    GL_seq,ZL_seq,primals_L_seq,obj_L_seq,cpu_L_seq,wall_L_seq,vector_L_seq,matrix_L_seq\
        = cw_linear_BM_seq(A2,c ** 0.5,linear_vector,linear_matrix)
        
    # Conditional Gradient (CG) (parallel BCM for linear)
    GL,ZL,primals_L,obj_L,cpu_L,wall_L,vector_L,matrix_L = cw_linear_BM(A2,c ** 0.5,linear_vector,linear_matrix)
    # Block Coordinate Maximization Quadratic
    GQ_seq,ZQ_seq,primals_Q_seq,obj_Q_seq,cpu_Q_seq,wall_Q_seq,vector_Q_seq,matrix_Q_seq\
        = cw_quadratic_BM_seq(A2,quad_vector,quad_matrix)
    # CG & BCM Hybrid (parallel BCM for quadratic)
    GQ,ZQ,primals_Q,obj_Q,cpu_Q,wall_Q,vector_Q,matrix_Q = cw_quadratic_BM(A2,quad_vector,quad_matrix)

    
    print("BCM L",wall_L_seq[-1])
    print("CG L",wall_L[-1])
    print("BCM Q",wall_Q_seq[-1])
    print("CG Q",wall_Q[-1])
#%% figures

    fig0 = figure(2 * i)
    
    plot(wall_L_seq,vector_L_seq,color = "gold")
    plot(wall_L,vector_L,color = "gold",linestyle = "dashed")
    plot(wall_Q_seq,vector_Q_seq,color = "red")
    plot(wall_Q,vector_Q,color = "red",linestyle = "dashed")

    grid(True)
    legend(["BCM $D_L$","Parallel BCM $D_L$ (CG)","BCM $D_Q$","Parallel BCM $D_Q$"])
    xlabel("Wall Time")
    ylabel(r"$\frac{||w_k-w^*||_2}{||w^*||_2}$")
    title("Relative Error Against Wall Time n = "+str(n))
    xscale("log")
    yscale("log")
    
    
    fig0.savefig("rel_err_ll_w_n_"+str(n)+".png",format = "png")
    fig0.savefig("rel_err_ll_w_n_"+str(n)+".pdf",format = "pdf")
    fig0.savefig("rel_err_ll_w_n_"+str(n)+".svg",format = "svg")
    
    fig1 = figure(2 * i + 1)
    
    plot(wall_L_seq,matrix_L_seq,color = "gold")
    plot(wall_L,matrix_L,color = "gold",linestyle = "dashed")
    plot(wall_Q_seq,matrix_Q_seq,color = "red")
    plot(wall_Q,matrix_Q,color = "red",linestyle = "dashed")

    grid(True)
    legend(["BCM $D_L$","Parallel BCM $D_L$ (CG)","BCM $D_Q$","Parallel BCM $D_Q$"])
    xlabel("Wall Time")
    ylabel(r"$\frac{||Z_k-Z^*||_F}{||Z^*||_F}$")
    title("Relative Error Against Wall Time n = " + str(n))
    xscale("log")
    yscale("log")
    
    
    fig1.savefig("rel_err_ll_Z_n_"+str(n)+".png",format = "png")
    fig1.savefig("rel_err_ll_Z_n_"+str(n)+".pdf",format = "pdf")
    fig1.savefig("rel_err_ll_Z_n_"+str(n)+".svg",format = "svg")
    
    figs += [fig0,fig1]
