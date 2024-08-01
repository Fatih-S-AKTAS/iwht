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
#   H   = a list of da0's to apply in the algorithm, 



# Note that sigma parameter is fixed for all algorithms, it may need to be changed or chosen
# adaptively for varying problem sizes

#%% Lipschitz constant

def Newton_L(A,b,s,x= None):
    # original code is GPNP in MATLAB
    # this code is a python equivalent
    fmin = inf
    m,n = A.shape
    if x is None:
        x = zeros(n)
    x0 = zeros(n)
    cpu0 = time.process_time()
    wall0 = time.time()

    OBJ   = [0] * 5

    sigma = 1e-4
    J = 1
    alpha0 = 5
    gamma = 0.5

    if m/n   >= 1/6 and s/n <= 0.05 and n >= 1e4:
       alpha0 = 1;
       gamma  = 0.1;
    if s/n   <= 0.05:
       thd    = int(ceil(log2(2+s)*50));
    else:
        if  n    > 1e3:
            thd  = 100
        elif n > 500:
            thd  = 500
        else:
            thd  = int(ceil(log2(2+s)*750))

    tol = 1e-10
    tolF = 1e-20
    
    maxit = 15000;

    fx = norm(A.dot(x)-b) ** 2
    grad = A.T.dot(A.dot(x)-b)

    Tx = argpartition(abs(x-alpha0*grad),-s)[-s:]

    minobj    = zeros(int(maxit)+2)
    minobj[0] = fx
    print(' Start to run the solver -- GPNP')
    print(" ----------------------------------------------------------")
    print(' Iter         ||Ax-b||          Wall Time          CPU Time')
    print(' ----------------------------------------------------------')

    k = 0
    while k <= maxit:
        k += 1

        # step size 
        alpha = alpha0
        for j in range(J):
            Tu = argpartition(abs(x-alpha*grad),-s)[-s:]
            subu = (x-alpha*grad)[Tu]
            u = x0.copy()
            u[Tu] = (x-alpha*grad)[Tu]        
            fu = norm(A[:,Tu].dot(u[Tu])-b) ** 2
            if fu < fx - sigma * norm(u-x) ** 2: # optimize ?
                break
            alpha = alpha * gamma
        grad = A.T.dot(A[:,Tu].dot(u[Tu])-b)
        normg = norm(grad) ** 2
        x  = u.copy()
        fx = fu;

        # Newton
        sT   = sorted(Tu);
        mark = count_nonzero(sort(Tu)-sort(Tx))==0;
        Tx   = sT.copy()
        eps  = 1e-4;
        if  mark or normg < 1e-4 or alpha0==1:
            v = x0.copy()
            if s < 2000 and m <= 2e4:
               subv = solve(A[:,Tu].T.dot(A[:,Tu]),A[:,Tu].T.dot(b))
               eps  = 1e-10;
            else:
               cgit = min(20,2*iter);
               subv = cg(A[:,Tu].T.dot(A[:,Tu]),A[:,Tu].dot(b),tol = 1e-30,maxiter = cgit,x0 = zeros(s,1))

            v[Tu] = subv
            fv  = norm(A[:,Tu].dot(subv)-b) ** 2
            if fv      <= fu  - sigma * norm(subu-subv) ** 2:
               x        = v
               fx       = fv
               subu     = subv
               grad      = A.T.dot(A[:,Tu].dot(subv)-b)
               normg    = norm(grad) ** 2

        error = norm(grad[Tu])
        obj  = fx ** 0.5
        OBJ  = OBJ[1:] + [obj]
        if k % 100 == 0:
            print('{:4d}         {:9.4e}      {:7.3f}sec      {:7.3f} sec'.format(k,fx,time.time()-wall0,time.process_time()-cpu0))

        maxg      = max(abs(grad))
        minx      = min(abs(subu))
        J         = 8;
        # restart condition
        if error  < tol*1e3 and normg>1e-2 and k < maxit-10:
           J      = int(min(8,max(1,ceil(maxg/minx)-1)))

        minobj[k] = min(minobj[k-1],fx);
        if fx    < minobj[k-1]:
            xmin = x
            fmin = fx

        if k  > thd:
           count = std(minobj[k-thd:k+1],ddof = 1)<1e-10
        else:
           count = 0

        if  normg<tol or fx < tolF or count  or (std(OBJ,ddof = 1)<eps*(1+obj)):
            if count and fmin < fx:
                x = xmin
                fx = fmin
            break

    if  fx<1e-10:
        print('---------------------------------------\n')
        print(' A global optimal solution may be found');
        print(' because of ||Ax-b|| = {:5.3e}!'.format(fx ** 0.5))
        print('---------------------------------------');

    return x,fx,k,time.time()-wall0,time.process_time()-cpu0

def Newton_L2(A,b,s,alpha0,x= None):
    # GPNP with initial step size candidate given as an input
    # this is for fairness, as playing with alpha0 changes algorithm behaviour
    fmin = inf
    m,n = A.shape
    if x is None:
        x = zeros(n)
    x0 = zeros(n)
    cpu0 = time.process_time()
    wall0 = time.time()

    OBJ   = [0] * 5

    sigma = 1e-4
    J = 1
    gamma = 0.5

    if m/n   >= 1/6 and s/n <= 0.05 and n >= 1e4:
       alpha0 = 1;
       gamma  = 0.1;
    if s/n   <= 0.05:
       thd    = int(ceil(log2(2+s)*50));
    else:
        if  n    > 1e3:
            thd  = 100
        elif n > 500:
            thd  = 500
        else:
            thd  = int(ceil(log2(2+s)*750))

    tol = 1e-10
    tolF = 1e-20
    maxit = 15000;

    fx = norm(A.dot(x)-b) ** 2
    grad = A.T.dot(A.dot(x)-b)

    Tx = argpartition(abs(x-alpha0*grad),-s)[-s:]

    minobj    = zeros(int(maxit)+2)
    minobj[0] = fx
    print(' Start to run the solver -- GPNP')
    print(" ----------------------------------------------------------")
    print(' Iter         ||Ax-b||          Wall Time          CPU Time')
    print(' ----------------------------------------------------------')

    k = 0
    while k <= maxit:
        k += 1
        # step size 
        alpha = alpha0
        for j in range(J):
            Tu = argpartition(abs(x-alpha*grad),-s)[-s:]
            subu = (x-alpha*grad)[Tu]
            u = x0.copy()
            u[Tu] = (x-alpha*grad)[Tu]
            fu = norm(A[:,Tu].dot(u[Tu])-b) ** 2
            if fu < fx - sigma * norm(u-x) ** 2: # optimize
                break
            alpha = alpha * gamma
        grad = A.T.dot(A[:,Tu].dot(u[Tu])-b)
        normg = norm(grad) ** 2
        x  = u.copy()
        fx = fu;

        # Newton
        sT   = sorted(Tu);
        mark = count_nonzero(sort(Tu)-sort(Tx))==0;
        Tx   = sT.copy()
        eps  = 1e-4;
        if  mark or normg < 1e-4 or alpha0==1:
            v = x0.copy()
            if s < 2000 and m <= 2e4:
               subv = solve(A[:,Tu].T.dot(A[:,Tu]),A[:,Tu].T.dot(b))
               eps  = 1e-10;
            else:
               cgit = min(20,2*iter);
               subv = cg(A[:,Tu].T.dot(A[:,Tu]),A[:,Tu].dot(b),tol = 1e-30,maxiter = cgit,x0 = zeros(s,1))

            v[Tu] = subv
            fv  = norm(A[:,Tu].dot(subv)-b) ** 2
            if fv      <= fu  - sigma * norm(subu-subv) ** 2:
               x        = v
               fx       = fv
               subu     = subv
               grad      = A.T.dot(A[:,Tu].dot(subv)-b)
               normg    = norm(grad) ** 2

        error = norm(grad[Tu])
        obj  = fx ** 0.5
        OBJ  = OBJ[1:] + [obj]
        if k % 100 == 0:
            print('{:4d}         {:9.4e}      {:7.3f}sec      {:7.3f} sec'.format(k,fx,time.time()-wall0,time.process_time()-cpu0))

        maxg      = max(abs(grad))
        minx      = min(abs(subu))
        J         = 8;
        # restart condition
        if error  < tol*1e3 and normg>1e-2 and k < maxit-10:
           J      = int(min(8,max(1,ceil(maxg/minx)-1)))

        minobj[k] = min(minobj[k-1],fx);
        if fx    < minobj[k-1]:
            xmin = x
            fmin = fx

        if k  > thd:
           count = std(minobj[k-thd:k+1],ddof = 1)<1e-10
        else:
           count = 0

        if  normg<tol or fx < tolF or count  or (std(OBJ,ddof = 1)<eps*(1+obj)):
            if count and fmin < fx:
                x = xmin
                fx = fmin
            break

    if  fx<1e-10:
        print('---------------------------------------\n')
        print(' A global optimal solution may be found');
        print(' because of ||Ax-b|| = {:5.3e}!'.format(fx ** 0.5))
        print('---------------------------------------');

    return x,fx,k,time.time()-wall0,time.process_time()-cpu0

#%% weighted $\ell_2$ norm proximal descent

def Newton_D(A,b,s,da0,x= None):
    # instead of L stationarity, D stationarity is imposed, whatever is given as an input
    m,n = A.shape
    if x is None:
        x = zeros(n)
    x0 = zeros(n)
    cpu0 = time.process_time()
    wall0 = time.time()

    OBJ   = [0] * 5

    sigma = 1e-4
    J = 1
    gamma = 0.5 ** 0.5

    if m/n   >= 1/6 and s/n <= 0.05 and n >= 1e4:
       gamma  = 0.1;
    if s/n   <= 0.05:
       thd    = int(ceil(log2(2+s)*50));
    else:
        if  n    > 1e3:
            thd  = 100
        elif n > 500:
            thd  = 500
        else:
            thd  = int(ceil(log2(2+s)*750))

    tol = 1e-10
    tolF = 1e-20

    maxit = 15000;

    fx = norm(A.dot(x)-b) ** 2
    grad = A.T.dot(A.dot(x)-b)

    Tx = argpartition(abs(da0 * x-grad/da0),-s)[-s:]

    minobj    = zeros(int(maxit)+2)
    minobj[0] = fx
    print(' Start to run the solver -- GPNP')
    print(" ----------------------------------------------------------")
    print(' Iter         ||Ax-b||          Wall Time          CPU Time')
    print(' ----------------------------------------------------------')

    k = 0
    while k <= maxit:
        k += 1

        # step size 
        da = da0.copy()
        for j in range(J):
            Tu = argpartition(abs(da * x-grad/da),-s)[-s:]
            subu = (da * x-grad/da)[Tu]/da[Tu]
            u = x0.copy()
            u[Tu] = (da * x-grad/da)[Tu]/da[Tu]
            fu = norm(A[:,Tu].dot(u[Tu])-b) ** 2
            if fu < fx - sigma * norm(u-x) ** 2: # optimize
                break
            da = da / gamma
        grad = A.T.dot(A[:,Tu].dot(u[Tu])-b)
        normg = norm(grad) ** 2
        x  = u.copy()
        fx = fu;

        # Newton
        sT   = sorted(Tu);
        mark = count_nonzero(sort(Tu)-sort(Tx))==0;
        Tx   = sT.copy()
        eps  = 1e-4;
        if  mark or normg < 1e-4:
            v = x0.copy()
            if s < 2000 and m <= 2e4:
               subv = solve(A[:,Tu].T.dot(A[:,Tu]),A[:,Tu].T.dot(b))
               eps  = 1e-10;
            else:
               cgit = min(20,2*iter);
               subv = cg(A[:,Tu].T.dot(A[:,Tu]),A[:,Tu].dot(b),tol = 1e-30,maxiter = cgit,x0 = zeros(s,1))

            v[Tu] = subv
            fv  = norm(A[:,Tu].dot(subv)-b) ** 2
            if fv      <= fu  - sigma * norm(subu-subv) ** 2:
               x        = v
               fx       = fv
               subu     = subv
               grad      = A.T.dot(A[:,Tu].dot(subv)-b)
               normg    = norm(grad) ** 2

        error = norm(grad[Tu])
        obj  = fx ** 0.5
        OBJ  = OBJ[1:] + [obj]
        if k % 100 == 0:
            print('{:4d}         {:9.4e}      {:7.3f}sec      {:7.3f} sec'.format(k,fx,time.time()-wall0,time.process_time()-cpu0))

        maxg      = max(abs(grad))
        minx      = min(abs(subu))
        J         = 8;
        # restart condition
        if error  < tol*1e3 and normg>1e-2 and k < maxit-10:
           J      = int(min(8,max(1,ceil(maxg/minx)-1)))

        minobj[k] = min(minobj[k-1],fx);
        if fx    < minobj[k-1]:
            xmin = x
            fmin = fx

        if k  > thd:
           count = std(minobj[k-thd:k+1],ddof = 1)<1e-10
        else:
           count = 0

        if  normg<tol or fx < tolF or count  or (std(OBJ,ddof = 1)<eps*(1+obj)):
            if count and fmin < fx:
                x = xmin
                fx = fmin
            break

    if  fx<1e-10:
        print('---------------------------------------\n')
        print(' A global optimal solution may be found');
        print(' because of ||Ax-b|| = {:5.3e}!'.format(fx ** 0.5))
        print('---------------------------------------');

    return x,fx,k,time.time()-wall0,time.process_time()-cpu0

#%% multiple weighted $\ell_2$ norm proximal descent

# mathematically correct algorithms

def Newton_H(A,b,s,H,x= None):
    # H is a list of diagonal matrices stored in a vector form
    # instead of L stationarity, for all D in H, D stationarity is imposed
    # also, diagonal matrix to apply is changed at every iteration

    fmin = inf
    m,n = A.shape
    if x is None:
        x = zeros(n)
    x0 = zeros(n)
    cpu0 = time.process_time()
    wall0 = time.time()

    OBJ   = [0] * 5

    sigma = 1e-4
    J = 1
    gamma = 0.5 ** 0.5

    if m/n   >= 1/6 and s/n <= 0.05 and n >= 1e4:
       gamma  = 0.1;
    if s/n   <= 0.05:
       thd    = int(ceil(log2(2+s)*50));
    else:
        if  n    > 1e3:
            thd  = 100
        elif n > 500:
            thd  = 500
        else:
            thd  = int(ceil(log2(2+s)*750))


    tol = 1e-10
    tolF = 1e-20
    # maxit = 10000
    maxit = (n<=1e3)*1e4 + (n>1e3)*5e3;maxit = 15000;

    fx = norm(A.dot(x)-b) ** 2
    grad = A.T.dot(A.dot(x)-b)

    h0 = H[0]
    len_h = len(H)

    Tx = argpartition(abs(h0 * x-grad/h0),-s)[-s:]

    minobj    = zeros(int(maxit)+2)
    minobj[0] = fx
    print(' Start to run the solver -- GPNP')
    print(" ----------------------------------------------------------")
    print(' Iter         ||Ax-b||          Wall Time          CPU Time')
    print(' ----------------------------------------------------------')

    k = 0
    h_index = -1
    while k <= maxit:
        # x_old = x.copy()
        k += 1
        h_index = (h_index+1) % len_h
        h0 = H[h_index]
        # step size ?
        h = h0.copy()
        for j in range(J):
            Tu = argpartition(abs(h * x-grad/h),-s)[-s:]
            subu = (h * x-grad/h)[Tu]/h[Tu]
            u = x0.copy()
            u[Tu] = (h * x-grad/h)[Tu]/h[Tu]      # play with this
            fu = norm(A[:,Tu].dot(u[Tu])-b) ** 2
            if fu < fx - sigma * norm(u-x) ** 2: # optimize
                break
            h = h / gamma
        grad = A.T.dot(A[:,Tu].dot(u[Tu])-b)
        normg = norm(grad) ** 2
        x  = u.copy()
        fx = fu;

        # Newton
        sT   = sorted(Tu);
        mark = count_nonzero(sort(Tu)-sort(Tx))==0;
        Tx   = sT.copy()
        eps  = 1e-4;
        if  mark or normg < 1e-4:
            v = x0.copy()
            if s < 2000 and m <= 2e4:
               subv = solve(A[:,Tu].T.dot(A[:,Tu]),A[:,Tu].T.dot(b))
               eps  = 1e-10;
               # eps  = 1e-4;
            else:
               cgit = min(20,2*iter);
               subv = cg(A[:,Tu].T.dot(A[:,Tu]),A[:,Tu].dot(b),tol = 1e-30,maxiter = cgit,x0 = zeros(s,1))

            v[Tu] = subv
            fv  = norm(A[:,Tu].dot(subv)-b) ** 2
            if fv      <= fu  - sigma * norm(subu-subv) ** 2:
               x        = v
               fx       = fv
               subu     = subv
               grad      = A.T.dot(A[:,Tu].dot(subv)-b)
               normg    = norm(grad) ** 2

        error = norm(grad[Tu])
        obj  = fx ** 0.5
        OBJ  = OBJ[1:] + [obj]
        if k % 100 == 0:
            print('{:4d}         {:9.4e}      {:7.3f}sec      {:7.3f} sec'.format(k,fx,time.time()-wall0,time.process_time()-cpu0))

        maxg      = max(abs(grad))
        minx      = min(abs(subu))
        J         = 8;
        # restart condition
        if error  < tol*1e3 and normg>1e-2 and k < maxit-10:
           J      = int(min(8,max(1,ceil(maxg/minx)-1)))

        minobj[k] = min(minobj[k-1],fx);
        if fx    < minobj[k-1]:
            xmin = x
            fmin = fx

        if k  > thd:
           count = std(minobj[k-thd:k+1],ddof = 1)<1e-10
        else:
           count = 0

        if  normg<tol or fx < tolF or count  or (std(OBJ,ddof = 1)<eps*(1+obj)):
            if count and fmin < fx:
                x = xmin
                fx = fmin
            break

    if  fx<1e-10:
        print('---------------------------------------\n')
        print(' A global optimal solution may be found');
        print(' because of ||Ax-b|| = {:5.3e}!'.format(fx ** 0.5))
        print('---------------------------------------');

    return x,fx,k,time.time()-wall0,time.process_time()-cpu0

def Newton_H_P(A,b,s,H,x= None,period = 10):
    # H is a list of diagonal matrices stored in a vector form
    # instead of L stationarity, for all D in H, D stationarity is imposed
    # also, diagonal matrix to apply is not changed every iteration but every P = 10 iteration
    fmin = inf
    m,n = A.shape
    if x is None:
        x = zeros(n)
    x0 = zeros(n)
    cpu0 = time.process_time()
    wall0 = time.time()

    OBJ   = [0] * 5

    sigma = 1e-4
    # sigma = 1e-2
    J = 1
    gamma = 0.5 ** 0.5

    if m/n   >= 1/6 and s/n <= 0.05 and n >= 1e4:
       gamma  = 0.1;
    if s/n   <= 0.05:
       thd    = int(ceil(log2(2+s)*50));
    else:
        if  n    > 1e3:
            thd  = 100
        elif n > 500:
            thd  = 500
        else:
            thd  = int(ceil(log2(2+s)*750))


    # L0,v0 = eigh(A.T.dot(A),subset_by_index = [n-1,n-1])

    # ls = estimate_restricted_lipschitz(2 * L0 * eye(n) - A.T.dot(A),min(s,n))
    # ls = (2 * L0 - ls) * 0.1
    # sigma = ls/4

    tol = 1e-10
    tolF = 1e-20
    # maxit = 10000
    maxit = (n<=1e3)*1e4 + (n>1e3)*5e3;maxit = 15000;

    fx = norm(A.dot(x)-b) ** 2
    grad = A.T.dot(A.dot(x)-b)

    h0 = H[0]
    len_h = len(H)

    Tx = argpartition(abs(h0 * x-grad/h0),-s)[-s:]

    minobj    = zeros(int(maxit)+2)
    minobj[0] = fx
    print(' Start to run the solver -- GPNP')
    print(" ----------------------------------------------------------")
    print(' Iter         ||Ax-b||          Wall Time          CPU Time')
    print(' ----------------------------------------------------------')

    k = 0
    h_index = 0
    while k <= maxit:
        # x_old = x.copy()
        k += 1
        if k % period == 0:
            h_index = (h_index+1) % len_h
            h0 = H[h_index]
        # step size ?
        h = h0.copy()
        for j in range(J):
            Tu = argpartition(abs(h * x-grad/h),-s)[-s:]
            subu = (h * x-grad/h)[Tu]/h[Tu]
            u = x0.copy()
            u[Tu] = (h * x-grad/h)[Tu]/h[Tu]      # play with this
            fu = norm(A[:,Tu].dot(u[Tu])-b) ** 2
            if fu < fx - sigma * norm(u-x) ** 2: # optimize
                break
            h = h / gamma
        grad = A.T.dot(A[:,Tu].dot(u[Tu])-b)
        normg = norm(grad) ** 2
        x  = u.copy()
        fx = fu;

        # Newton
        sT   = sorted(Tu);
        mark = count_nonzero(sort(Tu)-sort(Tx))==0;
        Tx   = sT.copy()
        eps  = 1e-4;
        if  mark or normg < 1e-4:
            v = x0.copy()
            if s < 2000 and m <= 2e4:
               subv = solve(A[:,Tu].T.dot(A[:,Tu]),A[:,Tu].T.dot(b))
               eps  = 1e-10;
               # eps  = 1e-4;
            else:
               cgit = min(20,2*iter);
               subv = cg(A[:,Tu].T.dot(A[:,Tu]),A[:,Tu].dot(b),tol = 1e-30,maxiter = cgit,x0 = zeros(s,1))

            v[Tu] = subv
            fv  = norm(A[:,Tu].dot(subv)-b) ** 2
            if fv      <= fu  - sigma * norm(subu-subv) ** 2:
               x        = v
               fx       = fv
               subu     = subv
               grad      = A.T.dot(A[:,Tu].dot(subv)-b)
               normg    = norm(grad) ** 2

        error = norm(grad[Tu])
        obj  = fx ** 0.5
        OBJ  = OBJ[1:] + [obj]
        if k % 100 == 0:
            print('{:4d}         {:9.4e}      {:7.3f}sec      {:7.3f} sec'.format(k,fx,time.time()-wall0,time.process_time()-cpu0))

        maxg      = max(abs(grad))
        minx      = min(abs(subu))
        J         = 8;
        # restart condition
        if error  < tol*1e3 and normg>1e-2 and k < maxit-10:
           J      = int(min(8,max(1,ceil(maxg/minx)-1)))

        minobj[k] = min(minobj[k-1],fx);
        if fx    < minobj[k-1]:
            xmin = x
            fmin = fx

        if k  > thd:
           count = std(minobj[k-thd:k+1],ddof = 1)<1e-10
        else:
           count = 0

        if  normg<tol or fx < tolF or count  or (std(OBJ,ddof = 1)<eps*(1+obj)):
            if count and fmin < fx:
                x = xmin
                fx = fmin
            break

    if  fx<1e-10:
        print('---------------------------------------\n')
        print(' A global optimal solution may be found');
        print(' because of ||Ax-b|| = {:5.3e}!'.format(fx ** 0.5))
        print('---------------------------------------');

    return x,fx,k,time.time()-wall0,time.process_time()-cpu0