import numpy as np
import itertools
import scipy.linalg as scl
import c1c2

#def formH(P,K,h,V):
    

np.set_printoptions(precision=4,suppress=True)

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

def embed_energy(nf,h,V,rho):
    n = len(h)
    ri = range(n)
    rf = range(nf)

    e1 = 0
    for f,j in itertools.product(rf,ri):
        e1 += 2.*h[f,j]*rho[j,f]

    e2 = 0
    gamma = np.zeros((n,n,n,n))
    for i,j,k,l in itertools.product(ri,ri,ri,ri):
        gamma[i,j,k,l] = rho[k,i]*rho[l,j]
    for f,j,k,l in itertools.product(rf,ri,ri,ri):
        e2 += (2*V[f,j,k,l]-V[f,j,l,k])*gamma[k,l,f,j]
    print "e1: ", e1
    print "e2: ", e2

    return e1 + e2

n = 12
nf = 2

z = antisymMat(n)
Zm = np.zeros((2*n,2*n))
Zm[:n,n:] = z
Zm[n:,:n] = z

W = scl.expm(Zm) #Thouless form of an HFB state; arbitrary because z is random and antisymmetric

Wocc = W[:,n:] #In HFB, it's always "half full" because we treat particle and hole states kind of the same
G = Wocc.dot(Wocc.T) #G = [[1-r,k],[k.T,r]]

#reorganize generalized density matrix G into impurity and env blocks (imp rho and k together)
#Gswp=[[Gimp,GcT],[Gc,Genv]]
#Gimp=[[1-r_f,k_f],[k_f.T,r_f]]
Gswp = G.copy() 
for i in range(nf):
    Gswp[:,[n+i,nf+i]] = Gswp[:,[nf+i,n+i]]
    Gswp[[n+i,nf+i]] = Gswp[[nf+i,n+i]]
    Hswp[:,[n+i,nf+i]] = Hswp[:,[nf+i,n+i]]
    Hswp[[n+i,nf+i]] = Hswp[[nf+i,n+i]]

Gimp = Gswp[:2*nf,:2*nf] #pull Gimp out of re-organized density matrix
Gic = Gswp[:,:2*nf] #pull out [[Gimp],[Gc]]

eGi,vGi = np.linalg.eigh(Gimp) #diagonalize Gimp so we can get A
sq_eGi = np.diag(np.sqrt(eGi)) #take sqrt of eigenvalues then turn it into a matrix
A = np.dot(vGi,sq_eGi) #AA.T = Gimp, and A = U.dot(e^0.5)

ATinv = np.linalg.inv(A.T)
C2 = Gic.dot(ATinv)

h_f = C2.T.dot(h).dot(C2)
G_f = C2.T.dot(Gswp).dot(C2)

#is calculating fragment E the same for HF as HFB?
#is calculating total E something I even know how to do? let's start with that
#if I make an HF state and put it into HFB form, I should be able to get the energy back out





















