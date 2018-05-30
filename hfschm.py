import numpy as np
import itertools

import schmidt
import xform
import hf
import dmetenergy

np.set_printoptions(precision=4,suppress=True)

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

def hf_energy(h,V,rho):
    e1 = np.trace(h.dot(rho)) * 2
    e2 = 0

    n = len(h)
    ri = range(n)
    gamma = np.zeros((n,n,n,n))
    for i,j,k,l in itertools.product(ri,ri,ri,ri):
        gamma[i,j,k,l] = rho[k,i]*rho[l,j]
    for i,j,k,l in itertools.product(ri,ri,ri,ri):
        e2 += (2*V[i,j,k,l]-V[i,j,l,k])*gamma[k,l,i,j]

    print "e1: ", e1
    print "e2: ", e2
    return e1 + e2

def V_hubbard(n, U):
    V = np.zeros((n,n,n,n))
    for i in range(n):
        V[i,i,i,i] = U
    return V


n = 12
m = 6
U = 0.
nf = 2
V = V_hubbard(n,U)

h = np.diag(np.ones(n-1),1)
h[0,-1] = -1
h += h.T
h *= -1

hart = hf.HF(h,V,m)
psi = hart.get_Cocc()
rho = hart.get_rho()
print hart.get_energy()/n
print hf_energy(h,V,rho)/n

M = psi[:nf,:]
S_M = np.dot(M.T,M)

u,s,v = np.linalg.svd(M)

C = psi.dot(v.T)
T = np.zeros((n,2*nf))
T[:nf,:nf] = C[:nf,:nf]
T[nf:,nf:] = C[nf:,:nf]
Tc = C[:,nf:]
print Tc

hc = xform.core(Tc,V)
h_f = xform.one(T,h+hc)
V_f = xform.two(T,V)

rho_f = T.T.dot(np.dot(rho,T))
print embed_energy(nf,h_f,V_f,rho_f)/nf

Tfull = schmidt.schmidt(psi,range(nf),allv=True)
T = Tfull[:,:2*nf]
Tc = Tfull[:,2*nf:]
rho_f = xform.one(T,rho)
hc = xform.core(Tc,V)
h_f = xform.one(T,h+hc)
V_f = xform.two(T,V)
print embed_energy(nf,h_f,V_f,rho_f)/nf

hart_f = hf.HF(h_f,V,nf)
print hart_f.get_energy()/nf























