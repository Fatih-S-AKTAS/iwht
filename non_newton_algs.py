from numpy import zeros,argpartition,log2,ceil,count_nonzero,std,sort,random,array,where,eye,ones,diag,inf,\
    savez,load,arange,sign
import time
from scipy.sparse.linalg import cg
from scipy.linalg import solve,norm

#%% some comments

#   A = m by n data matrix, in array form with float64
#   b = m observation vector, array form with float64, note that np array differentiates (m,) with (m,1) vectors
#   s = integer, the cardinality level
#   x = initial guess, if not given, taken as all zero vector
#   alpha0 = a constant, float64, initial guess of L, this is assumed to be a small constant and descent conditions are checked, it is
#       increased until descent conditions are satisfied (in reverse it is large step size and shrunk until descent
#        conditions are met)
#   da0   = same as alpha0, except it is a diagonal vector (array) in shape (m,) and it is stored as the square root of
#            the actual D matrix i.e it is the D^(1/2) matrix since in the algorithm that how it is applied
#   H   = a list of da0's to apply in the algorithm


# Note that sigma parameter is fixed for all algorithms, it may need to be changed or chosen
# adaptively for varying problem sizes


#%% Safe Step Size

def IWHT(A,b,s,da,x_h,K = 5000):
    # weighted $\ell_2$ norm proximal gradient algorithm with safe step size
    cpu0 = time.process_time()
    wall0 = time.time()
    
    k = 0
    cost = zeros(K+1)
    while k <= K:
        g = A.T.dot(A.dot(x_h)-b)
        x_h = da * x_h - g/da
        partition = argpartition(abs(x_h),-s)
        T = partition[-s:]
        Tc = partition[:-s]
        x_h[Tc] = 0
        x_h = x_h / da
        gcost = norm(A.dot(x_h)-b)
        cost[k] = gcost
        k += 1
        # since no restart, if no progress is made, terminate the algorithm
        if std(cost[k-20:k],ddof = 1) < 1e-8:
            break
    cost = cost[:k]
    return x_h,gcost,cost,time.time()-wall0,time.process_time()-cpu0

def CIWHT(A,b,s,H,x_h,K = 5000,period = 5):
    # list of weighted $\ell_2$ norm proximal 
    # gradient algorithm with safe step size 
    cpu0 = time.process_time()
    wall0 = time.time()
    
    m,n = A.shape
    len_h = len(H)
    h_index = 0
    k = 0
    T = argpartition(abs(x_h),-s)
    da = H[h_index]
    cost = zeros(K+1)
    g = A.T.dot(A.dot(x_h)-b)
    while k <= K:
        if k % period == 0:
            h_index = (h_index+1) % len_h
            da = H[h_index]
        
        g = A.T.dot(A.dot(x_h)-b)
        x_h = da * x_h - g/da
        partition = argpartition(abs(x_h),-s)
        T = partition[-s:]
        Tc = partition[:-s]
        x_h[Tc] = 0
        x_h = x_h / da
        gcost = norm(A.dot(x_h)-b)
        cost[k] = gcost
        k += 1
        # since no restart, if no progress is made, terminate the algorithm
        if std(cost[k-20:k],ddof = 1) < 1e-8:
            break
    cost = cost[:k]
    return x_h,gcost,cost,time.time()-wall0,time.process_time()-cpu0

#%% Line Search and Restart

def CIWHT_LS_R(A,b,s,H,x_h,K = 15000,period = 4):
    # CIWHT algorithm with line search and gradient based restarts
    
    cpu0 = time.process_time()
    wall0 = time.time()
    
    m,n = A.shape
    len_h = len(H)
    h_index = 0
    k = 0
    T = argpartition(abs(x_h),-s)
    da = H[h_index]
    cost = zeros(K+1)
    gamma = 0.5 ** 0.5
    sigma = 1e-4
    fx = norm(A.dot(x_h)-b) ** 2
    g = A.T.dot(A.dot(x_h)-b)
    J = 8
    while k <= K:
        if k % period == 0:
            h_index = (h_index+1) % len_h
            da = H[h_index]
        # step size
        das = da.copy()
        for j in range(J):
            # print("chosen j",j)
            partition = argpartition(abs(das * x_h-g/das),-s)
            T = partition[-s:]
            Tc = partition[:-s]
            subu = (das * x_h-g/das)[T]/das[T]
            u = zeros(n)
            u[T] = (das * x_h-g/das)[T]/das[T]
            fu = norm(A[:,T].dot(u[T])-b) ** 2
            if fu < fx - sigma * norm(u-x_h) ** 2: # optimize
                break
            das = das / gamma
        x_h  = u.copy()
        fx = fu
        
        
        g = A.T.dot(A.dot(x_h)-b)
        error = norm(g[T])
        J         = 8;
        # restart condition
        if error  < 1e-6 and norm(g)>1e-2 and k < K-10:
           J      = 1
        
        gcost = norm(A.dot(x_h)-b)
        cost[k] = gcost
        k += 1
        if gcost < 1e-10:
            break
    cost = cost[:k]
    return x_h,gcost,cost,time.time()-wall0,time.process_time()-cpu0

#%% First order acceleration

def MFISTA(A,b,s,H,x_h,K = 15000,period = 5):
    # CIWHT algorithm with line search and gradient based restarts
    # additionally, if the sparsity pattern does not change, monotone version of 
    # FISTA algorithm is applied to accelerate the convergence
    cpu0 = time.process_time()
    wall0 = time.time()
    
    m,n = A.shape
    len_h = len(H)
    k = 0
    T = argpartition(abs(x_h),-s)
    y = x_h.copy()
    t = 1
    cost = zeros(K+1)
    fx = norm(A.dot(x_h)-b) ** 2
    fy = norm(A.dot(y)-b) ** 2
    h_index = 0
    sigma = 1e-4
    
    gamma = 0.5 ** 0.5
    J = 8

    while k <= K:
        x_h_old = x_h.copy()
        T_old = T.copy()
        if k % period == 0:
            h_index = (h_index+1) % len_h
            da = H[h_index]
        g = A.T.dot(A.dot(y)-b)
        
        
        # step size
        das = da.copy()
        for j in range(J):
            partition = argpartition(abs(das * y-g/das),-s)
            T = partition[-s:]
            Tc = partition[:-s]
            subu = (das * y-g/das)[T]/das[T]
            u = zeros(n)
            u[T] = (das * y-g/das)[T]/das[T]
            fu = norm(A[:,T].dot(u[T])-b) ** 2
            if fu < fy - sigma * norm(u-y) ** 2: # optimize
                break
            das = das / gamma
        # print("step j",j)
        if fu < fx:
            x_h = u.copy()
            fx = fu
        
        # acceleration condition
        if sorted(T_old) == sorted(T):
            t_old = t
            t = (1 + (1+ 4 * t** 2) ** 0.5 ) * 0.5
            y = x_h + t_old/t * (u - x_h) + (t_old-1)/t * (x_h-x_h_old)
            fy = norm(A[:,T].dot(y[T])-b) ** 2
        else:
            t = 1
            y = x_h.copy()
            fy = fx
        
        g = A.T.dot(A.dot(y)-b)
        error = norm(g[T])

        # restart condition
        if error  < 1e-4 and norm(g)>1e-2 and k < K-10:
            restart = H[h_index]
            x_h = restart * x_h - g/restart
            partition = argpartition(abs(x_h),-s)
            T = partition[-s:]
            Tc = partition[:-s]
            x_h[Tc] = 0
            x_h = x_h / restart
            fx = norm(A.dot(x_h)-b) ** 2
            fy = fx
            y = x_h.copy()
            t = 1
        
        gcost = norm(A.dot(x_h)-b)
        cost[k] = gcost
        

        k += 1
        if gcost < 1e-10:
            break
    cost = cost[:k]
    return x_h,gcost,cost,time.time()-wall0,time.process_time()-cpu0

