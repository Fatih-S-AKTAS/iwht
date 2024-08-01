from numpy import zeros,argpartition,log2,ceil,count_nonzero,std,sort,random,array,where,eye,ones,diag,inf,\
    savez,load,arange,sign,linspace
from scipy.linalg import norm,eigh

from get_dsm import cw_linear_BM,cw_quadratic_BM
from random import sample

from newton_algs import Newton_L2,Newton_H
from non_newton_algs import IWHT,CIWHT,CIWHT_LS_R,MFISTA

#%% generate a compressed sensing problem instance

# problem size
[m,n] = [64,256]
s = 20

# generate design/data matrix
A = random.randn(m,n)
A = A/norm(A,axis = 0)
A2 = A.T.dot(A)

# generate sparse vector
x0 = zeros(n)
support_set = sample(range(n),s)
x0[support_set] = random.randn(s)

# noiseless setting
b = A.dot(x0) + random.randn(m) * 0.0

#%% compute DSM

# compute Lipschtiz constant L
L0,v0 = eigh(A.T.dot(A),subset_by_index = [n-1,n-1])

# compute DSM D_L
B_l,Z_l,primals_l,obj_l = cw_linear_BM(A2,ones(n))
# compute DSM D_Q
B_q,Z_q,primals_q,obj_q = cw_quadratic_BM(A2)

# shrink parameter for taking large step sizes
shrink = 0.155

dal = ((primals_l) ** 0.5 ) * shrink
daq = ((primals_q) ** 0.5 ) * shrink
dac = (ones(n) * L0 ** 0.5 ) * shrink


# GPNP algorithm
x1,fx1,k1,wall1,cpu1 = Newton_L2(A,b,s,1/(L0 * shrink ** 2))

# not needed but let us explicitly initialize from x_0 = 0
x = zeros(n)

# set up the list of DSM
H1 = [daq,dal,dac]
# set up the list of DSM with safe step size
H1L = [primals_q ** 0.5,primals_l ** 0.5,ones(n) * L0 ** 0.5]


# Apply GPNP with different DSMs
x2,fx2,k2,wall2,cpu2 = Newton_H(A,b,s,H1,x)


# Apply basic gradient descent with theoretically safe step size
x3,fx3,cost3,wall3,cpu3 = IWHT(A,b,s,ones(n) * L0 ** 0.5,x,K = 5000)


# Apply multiple DSM that are theoretically safe
x4,fx4,cost4,wall4,cpu4 = CIWHT(A,b,s,H1L,x,K = 5000,period = 1)


# Apply gradient descent with line search for large step sizes and gradient based restarts for 
# extensive search
x11,fx11,cost11,wall11,cpu11 = CIWHT_LS_R(A,b,s,[dac],x,K = 15000,period = 1)


# Apply multiple DSM with line search for large step sizes and gradient based restarts for 
# extensive search
x7,fx7,cost7,wall7,cpu7 = CIWHT_LS_R(A,b,s,H1,x,K = 15000,period = 1)


# Apply multiple DSM with line search for large step sizes and gradient based restarts for 
# extensive search and first order monotone acceleration technique whenever feasible
x9,fx9,cost9,wall9,cpu9 = MFISTA(A,b,s,H1,x,K = 15000,period = 1)
