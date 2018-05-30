import numpy as np
import itertools
import scipy.linalg as scl

#def formH(P,K,h,V):
    

np.set_printoptions(precision=2,suppress=True)

def antisymMat(n):
    Z = (np.random.rand(n,n)-0.5)*0.5
    Z = Z+Z.T
    for i in range(n):
        Z[i,i] = 0.
        for j in range(i):
            Z[i,j] *= -1.
    return Z

def unitMatGen(n):
    Z = antisymMat(n)
    U = scl.expm(Z)
    B = 2.*(np.random.rand(n)-0.5)
    B = np.diag(B)
    A = U.dot(B.dot(U.T))
    return Z,U,B,A


n = 8
m = 4
nf = 2
F = np.random.rand(n,n)-0.5
F = F + F.T #F is a random symmetric matrix
Delta = antisymMat(n)
Delta *= 0. #let's just 0 this for now, like a good HF state. This should also make k=0

H = np.zeros((2*n,2*n))
H[:n,:n] = F
H[n:,n:] = -F
H[:n,n:] = Delta 
H[n:,:n] = -Delta

#print H
#print np.linalg.norm(H-H.T) #H is hermitian, because the block off-diagonals are neg of antisym

eF,wF = np.linalg.eigh(F)
print eF
print np.sum(eF[:m])
C_occ = wF[:,:m]
rho = C_occ.dot(C_occ.T)

e,W = np.linalg.eigh(H) #This has all the right eigenvalues, but they're mixed with the positive ones

#Is this to get the order of eigenvalues and eigenvectors to match? Maybe needed for schmidt?
#print W.dot(W.T) #W is unitary before this transformation
#W[:,n:] = np.fliplr(W[:,n:]) #Why did I think this was necessary? It isn't what makes G idempotent?
#for i in range(n):
#    if np.linalg.norm(W[:n,i]-W[n:,n+i]) > 10**-6:
#        W[:,n+i] *= -1.

Wocc = W[:,n:]
G = Wocc.dot(Wocc.T)

#print W.dot(W.T) #W is Unitary
#print np.linalg.norm(G.dot(G)-G) #G is idempotent
#print G #with Delta=0, indeed k=0

#E = mu*N + tr(h*p - k**Delta) - /\phi|V|phi/\
#E = mu*N - /\phi|V|phi/\ + Ec
#Ec = 0.5*tr(h)-0.5*Sum(Ev) = -Sum(Ev|Yv|^2)

print np.trace(rho.dot(F))
print np.trace(rho)










