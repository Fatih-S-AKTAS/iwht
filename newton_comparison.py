from numpy import zeros,argpartition,log2,ceil,count_nonzero,std,sort,random,array,where,eye,ones,diag,inf,\
    savez,load,arange,sign
import time
from scipy.sparse.linalg import cg
from scipy.linalg import solve,norm,eigh,cholesky

from scipy.io import savemat

from get_dsm import cw_linear_BM,cw_quadratic_BM
from newton_algs import Newton_L2,Newton_D,Newton_H,Newton_H_P
from random import sample

#%% generate instance

C = 100

S = [5,35]

algo_count = 11
recovery = zeros((S[1]-S[0]+1,5 * C,algo_count))
cpu = zeros((S[1]-S[0]+1,5 * C,algo_count))
wall = zeros((S[1]-S[0]+1,5 * C,algo_count))
fval = zeros((S[1]-S[0]+1,5 * C,algo_count))
xset = zeros((S[1]-S[0]+1,5 * C,algo_count + 1),dtype = object)
for s in range(S[0],S[1]+1):
  for century in range(C):
      [m,n] = [64,256]
      A = random.randn(m,n)
      A = A/norm(A,axis = 0)
      A2 = A.T.dot(A)
      
      L0,v0 = eigh(A.T.dot(A),subset_by_index = [n-1,n-1])

      B_l,Z_l,primals_l,obj_l = cw_linear_BM(A2,ones(n))
      B_q,Z_q,primals_q,obj_q = cw_quadratic_BM(A2)

      shrink = 0.155
      da = diag(A2) ** 0.5
      dal = ((primals_l) ** 0.5 ) * shrink
      daq = ((primals_q) ** 0.5 ) * shrink
      dac = (ones(n) * L0 ** 0.5 ) * shrink
      
      for accelerate in range(5):
        t = 5 * century + accelerate
        print("-----Sparsity",s,"-----Trial-----",t,"---------")
        x0 = zeros(n)
        support_set = sample(range(n),s)
        x0[support_set] = random.randn(s)
        b = A.dot(x0) + random.randn(m) * 0.0
        xset[s - S[0],t,-1] = x0
        
        x_initial = zeros(n)
        
        x1,fx1,k1,wall1,cpu1 = Newton_L2(A,b,s,1/(L0 * shrink ** 2))
        if norm(x1-x0)/norm(x0) < 1e-4:
          recovery[s - S[0],t,0] += 1
        cpu[s - S[0],t,0] = cpu1
        wall[s - S[0],t,0] = wall1
        fval[s - S[0],t,0] = fx1
        xset[s - S[0],t,0] = x1
        
        x2,fx2,k2,wall2,cpu2 = Newton_D(A,b,s,dal)
        if norm(x2-x0)/norm(x0) < 1e-4:
          recovery[s - S[0],t,1] += 1
        cpu[s - S[0],t,1] += cpu2
        wall[s - S[0],t,1] += wall2
        fval[s - S[0],t,1] = fx2
        xset[s - S[0],t,1] = x2
        
        x3,fx3,k3,wall3,cpu3 = Newton_D(A,b,s,daq)
        if norm(x3-x0)/norm(x0) < 1e-4:
          recovery[s - S[0],t,2] += 1
        cpu[s - S[0],t,2] += cpu3
        wall[s - S[0],t,2] += wall3
        fval[s - S[0],t,2] = fx3
        xset[s - S[0],t,2] = x3
    
        x = zeros(n)
        H1 = [daq,dal,dac]
    
        x5,fx5,k5,wall5,cpu5 = Newton_H(A,b,s,H1,x)
        if norm(x5-x0)/norm(x0) < 1e-4:
          recovery[s - S[0],t,3] += 1
        cpu[s - S[0],t,3] += cpu5
        wall[s - S[0],t,3] += wall5
        fval[s - S[0],t,3] = fx5
        xset[s - S[0],t,3] = x5
        
        x6,fx6,k6,wall6,cpu6 = Newton_H_P(A,b,s,H1,x)
        if norm(x6-x0)/norm(x0) < 1e-4:
          recovery[s - S[0],t,4] += 1
        cpu[s - S[0],t,4] += cpu6
        wall[s - S[0],t,4] += wall6
        fval[s - S[0],t,4] = fx6
        xset[s - S[0],t,4] = x6
    
        H2 = [dal,daq,dac]
    
        x7,fx7,k7,wall7,cpu7 = Newton_H(A,b,s,H2,x)
        if norm(x7-x0)/norm(x0) < 1e-4:
          recovery[s - S[0],t,5] += 1
        cpu[s - S[0],t,5] += cpu7
        wall[s - S[0],t,5] += wall7
        fval[s - S[0],t,5] = fx7
        xset[s - S[0],t,5] = x7
        
        x8,fx8,k8,wall8,cpu8 = Newton_H_P(A,b,s,H2,x)
        if norm(x8-x0)/norm(x0) < 1e-4:
          recovery[s - S[0],t,6] += 1
        cpu[s - S[0],t,6] += cpu8
        wall[s - S[0],t,6] += wall8
        fval[s - S[0],t,6] = fx8
        xset[s - S[0],t,6] = x8
    
        H3 = [dal,dac,daq]
    
        x9,fx9,k9,wall9,cpu9 = Newton_H(A,b,s,H3,x)
        if norm(x9-x0)/norm(x0) < 1e-4:
          recovery[s - S[0],t,7] += 1
        cpu[s - S[0],t,7] += cpu9
        wall[s - S[0],t,7] += wall9
        fval[s - S[0],t,7] = fx9
        xset[s - S[0],t,7] = x9
        
        x10,fx10,k10,wall10,cpu10 = Newton_H_P(A,b,s,H3,x)
        if norm(x10-x0)/norm(x0) < 1e-4:
          recovery[s - S[0],t,8] += 1
        cpu[s - S[0],t,8] += cpu10
        wall[s - S[0],t,8] += wall10
        fval[s - S[0],t,8] = fx10
        xset[s - S[0],t,8] = x10
        
        H4 = [daq,dac,dal]
    
        x11,fx11,k11,wall11,cpu11 = Newton_H(A,b,s,H4,x)
        if norm(x11-x0)/norm(x0) < 1e-4:
          recovery[s - S[0],t,9] += 1
        cpu[s - S[0],t,9] += cpu11
        wall[s - S[0],t,9] += wall11
        fval[s - S[0],t,9] = fx11
        xset[s - S[0],t,9] = x11
        
        x12,fx12,k12,wall12,cpu12 = Newton_H_P(A,b,s,H4,x)
        if norm(x12-x0)/norm(x0) < 1e-4:
          recovery[s - S[0],t,10] += 1
        cpu[s - S[0],t,10] += cpu12
        wall[s - S[0],t,10] += wall12
        fval[s - S[0],t,10] = fx12
        xset[s - S[0],t,10] = x12
        

#%% save file

savez("newton_comparison",recovery = recovery,cpu = cpu,wall = wall,fval = fval,xset = xset)