from numpy import zeros,argpartition,log2,ceil,count_nonzero,std,sort,random,array,where,eye,ones,diag,inf,\
    savez,load,arange,sign,linspace
import time
from scipy.sparse.linalg import cg
from scipy.linalg import solve,norm,eigh,cholesky

from scipy.io import savemat

from get_dsm import cw_linear_BM,cw_quadratic_BM
from random import sample

from newton_algs import Newton_L2,Newton_D,Newton_H,Newton_H_P
from non_newton_algs import IWHT,CIWHT,CIWHT_LS_R,MFISTA

#%% test code

C = 40
S = [5,35]

algo_count = 7
recovery = zeros((S[1]-S[0]+1,5 * C,algo_count))
cpu = zeros((S[1]-S[0]+1,5 * C,algo_count))
wall = zeros((S[1]-S[0]+1,5 * C,algo_count))
fval = zeros((S[1]-S[0]+1,5 * C,algo_count))
xset = zeros((S[1]-S[0]+1,5 * C,algo_count + 1),dtype = object)
for s in range(S[0],S[1]+1,2):
  for century in range(C * 5):
      [m,n] = [64,256]
      A = random.randn(m,n)
      A = A/norm(A,axis = 0)
      A2 = A.T.dot(A)
      
      L0,v0 = eigh(A.T.dot(A),subset_by_index = [n-1,n-1])

      B_l,Z_l,primals_l,obj_l = cw_linear_BM(A2,ones(n))
      B_q,Z_q,primals_q,obj_q = cw_quadratic_BM(A2)
      B_l2,Z_l2,primals_l2,obj_l2 = cw_linear_BM(A2,norm(A2,axis = 0))
      
      shrink = 0.155
      
      da = diag(A2) ** 0.5
      dal = ((primals_l) ** 0.5 ) * shrink
      daq = ((primals_q) ** 0.5 ) * shrink
      dac = (ones(n) * L0 ** 0.5 ) * shrink
      # dac = ones(n) * 1/5 ** 0.5
      dal2 = ((primals_l2) ** 0.5 ) * shrink
      
      for accelerate in range(1):
        t = 5 * century + accelerate
        t = century
        print("-----Sparsity",s,"-----Trial-----",t,"---------")
        # x0 = random.randn(n) * 10
        x0 = zeros(n)
        support_set = sample(range(n),s)
        x0[support_set] = random.randn(s)
        # x0[:s] = random.randn(s)
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
        
        x = zeros(n)
        H1 = [daq,dal,dac]
        # H1 = [daq,dal,dal2,dac]
        H1L = [primals_q ** 0.5,primals_l ** 0.5,ones(n) * L0 ** 0.5]
        
    
        x5,fx5,k5,wall5,cpu5 = Newton_H(A,b,s,H1,x)
        if norm(x5-x0)/norm(x0) < 1e-4:
          recovery[s - S[0],t,1] += 1
        cpu[s - S[0],t,1] += cpu5
        wall[s - S[0],t,1] += wall5
        fval[s - S[0],t,1] = fx5
        xset[s - S[0],t,1] = x5
        
        
        x3,fx3,cost3,wall3,cpu3 = IWHT(A,b,s,ones(n) * L0 ** 0.5,x,K = 5000)
        if norm(x3-x0)/norm(x0) < 1e-4:
          recovery[s - S[0],t,2] += 1
        cpu[s - S[0],t,2] += cpu3
        wall[s - S[0],t,2] += wall3
        fval[s - S[0],t,2] = fx3
        xset[s - S[0],t,2] = x3

        x6,fx6,cost6,wall6,cpu6 = CIWHT(A,b,s,H1L,x,K = 5000,period = 1)
        if norm(x6-x0)/norm(x0) < 1e-4:
          recovery[s - S[0],t,3] += 1
        cpu[s - S[0],t,3] += cpu6
        wall[s - S[0],t,3] += wall6
        fval[s - S[0],t,3] = fx6
        xset[s - S[0],t,3] = x6
    
    
        x11,fx11,cost11,wall11,cpu11 = CIWHT_LS_R(A,b,s,[dac],x,K = 15000,period = 1)
        if norm(x11-x0)/norm(x0) < 1e-4:
          recovery[s - S[0],t,4] += 1
        cpu[s - S[0],t,4] += cpu11
        wall[s - S[0],t,4] += wall11
        fval[s - S[0],t,4] = fx11
        xset[s - S[0],t,4] = x11
        
    
        x7,fx7,cost7,wall7,cpu7 = CIWHT_LS_R(A,b,s,H1,x,K = 15000,period = 1)
        if norm(x7-x0)/norm(x0) < 1e-4:
          recovery[s - S[0],t,5] += 1
        cpu[s - S[0],t,5] += cpu7
        wall[s - S[0],t,5] += wall7
        fval[s - S[0],t,5] = fx7
        xset[s - S[0],t,5] = x7
        

        x9,fx9,cost9,wall9,cpu9 = MFISTA(A,b,s,H1,x,K = 15000,period = 1)
        if norm(x9-x0)/norm(x0) < 1e-4:
          recovery[s - S[0],t,6] += 1
        cpu[s - S[0],t,6] += cpu9
        wall[s - S[0],t,6] += wall9
        fval[s - S[0],t,6] = fx9
        xset[s - S[0],t,6] = x9
        
        

#%% save file
savez("Result_newton_grad",recovery = recovery,cpu = cpu,wall = wall,fval = fval,xset = xset)
