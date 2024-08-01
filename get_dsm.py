from numpy import random,zeros,argpartition,diag,ones,inf,roots,tensordot,eye,ceil,argmax,\
    array,fill_diagonal,outer,copy,emath,trace,sign,maximum,linspace,where
from numpy.linalg import norm


#%% sequential algorithms

# M  = n by n symmetric matrix. it should be positive semi definite to make sense out of it. Otherwise algorithms
#       may compute an irrelevant solution, although it may make sense as a SDP template. 
# w  = vector of size (n,). It determines the coefficient in the objective, or equivalently diagonal elements of the 
#      Z matrix. It should be given in square root form, i.e. instead of original weights in the SDP model, the square
#      root of the weights is used directly.

def cw_quadratic_BM_seq(M):
    m,n = M.shape
    B = random.randn(min(int(ceil((2* n +1/4) ** 0.5 - 1/2) * 1 ),n),n)
    diag_M = diag(M).copy()
    fill_diagonal(M,0)
    for t in range(200):
        for i in range(n):
            c = M[:,[i]]
            a = diag_M[i]
            g = B.dot(c)
            ng = norm(g)
            r = (emath.sqrt(3 * (4 * -a ** 3 + 27 * ng ** 2)) + 9 * ng) ** (1/3)
            sigma1 = r/((2 * 9) ** (1/3)) + (2/3) ** (1/3) * a/r
            B[:,[i]] = g/ng * sigma1

    fill_diagonal(M,diag_M)
    Z = B.T.dot(B)
    obj = tensordot(M,Z) -1/2 * sum(norm(B,axis = 0) ** 4)
    primals = diag(Z)
    return B,Z,primals,obj

def cw_linear_BM_seq(M,w):
    m,n = M.shape
    B = random.randn(min(int(ceil((2* n +1/4) ** 0.5 - 1/2)  * 1),n),n)
    B = B/norm(B,axis = 0)
    diag_M = diag(M).copy()
    fill_diagonal(M,0)
    for t in range(200):
        for i in range(n):
            c = M[:,[i]]
            g = B.dot(c)
            B[:,[i]] = g/norm(g) * w[i]
    fill_diagonal(M,diag_M)
    Z = B.T.dot(B)
    obj = tensordot(M,Z)
    primals = ((Z * (Z.dot(M))).sum(axis = 0))/norm(Z,axis = 0) ** 2
    return B,Z,primals,obj

#%% parallel algorithms

def cw_linear_BM(M,w):
    m,n = M.shape
    B = random.randn(min(int(ceil((2* n +1/4) ** 0.5 - 1/2)  * 1),n),n)
    for t in range(500):
        # print('iter',t)
        B = B.dot(M)
        B = B/norm(B,axis = 0) * w
    Z = B.T.dot(B)
    obj = tensordot(M,Z)
    primals = ((Z * (Z.dot(M))).sum(axis = 0))/norm(Z,axis = 0) ** 2
    return B,Z,primals,obj

def cw_quadratic_BM(M):
    m,n = M.shape
    B = random.randn(min(int(ceil((2* n +1/4) ** 0.5 - 1/2) * 1 ),n),n)
    # for certain type of problems it does help when smoothing is used
    # rho = 0.1
    rho = 0.0
    diag_M = diag(M).copy()
    fill_diagonal(M,0)
    for t in range(500):
        g = B.dot(M)
        ng = norm(g,axis = 0)
        r = (emath.sqrt(3 * (4 * -diag_M ** 3 + 27 * ng ** 2)) + 9 * ng) ** (1/3)
        sigma1 = r/((2 * 9) ** (1/3)) + (2/3) ** (1/3) * diag_M/r
        B = rho * B + (1 - rho) * g/ng * sigma1.real
    fill_diagonal(M,diag_M)
    Z = B.T.dot(B)
    obj = tensordot(M,Z) -1/2 * sum(norm(B,axis = 0) ** 4)
    primals = diag(Z)
    return B,Z,primals,obj