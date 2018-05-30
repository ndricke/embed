import numpy as np

import hf
import hfb
import xform
import schmidt
import c1c2

def rowcolSwp(mtrx,i,j):
    mtrx[:,[i,j]] = mtrx[:,[j,i]]
    mtrx[[i,j]] = mtrx[[j,i]]

n = 12
U = 1.
m = 4
nf = 3
h,V = c1c2.randHV(n,U=U,scale=0.5)
#D_rnsym = antisymMat(n)
#bogil = hfb.HFB(h,V,m,Delta=D_rnsym)
bogil = hfb.HFB(h,V,m)

#make a list of range(nf) that tile n
fraglist = []
for i in range(n/nf): #assumes n divisible by nf
    fraglist.append(range(nf*i,nf*(1+i)))
print fraglist

H_app,G = bogil.occOpt()
app_pot = bogil.mu
H = bogil.H

h_e = np.zeros((2*n,2*n))
F_red = c1c2.redFock(n,G[:n,:n],h,V) #Fock matrix with all the 2ei contributions reduced
h_e[:n,:n] = F_red
h_e[n:,n:] = -F_red

e_list = []
for frag_sites in fraglist:
    Gswp = G.copy(); Hswp = H.copy(); h_swp = h_e.copy()
    for rc,i in enumerate(frag_sites): #swap fragment to be in upper-left of p-p and p-h blocks
        rowcolSwp(Gswp,rc,i)
        rowcolSwp(Hswp,rc,i)
        rowcolSwp(h_swp,rc,i)
        rowcolSwp(Gswp,rc+n,i+n)
        rowcolSwp(Hswp,rc+n,i+n)
        rowcolSwp(h_swp,rc+n,i+n)

    for i in range(nf): #swap blocks from hole-hole to be next to fragment particle-particle block
        rowcolSwp(Gswp,n+i,nf+i)
        rowcolSwp(Hswp,n+i,nf+i)
        rowcolSwp(h_swp,n+i,nf+i)

    Gimp = Gswp[:2*nf,:2*nf] #pull Gimp out of re-organized density matrix
    Gic = Gswp[:,:2*nf] #pull out [[Gimp],[Gc]]

    eGi,vGi = np.linalg.eigh(Gimp) #diagonalize Gimp so we can get A
    sq_eGi = np.diag(np.sqrt(eGi)) #take sqrt of eigenvalues then turn it into a matrix
    A = np.dot(vGi,sq_eGi) #AA.T = Gimp, and A = U.dot(e^0.5)

    ATinv = np.linalg.inv(A.T)
    C2 = Gic.dot(ATinv)
    G_f = C2.T.dot(Gswp).dot(C2) #Why is this the identity matrix? (at least for half-filling)
    h_f = c1c2.hRedHalf(nf,h_swp,C2) #0's the rows below nf

    efrag = 2.*np.trace(G_f.dot(h_f))/nf #Works for arbitrary fragment size, half filling, V=0
    e_list.append(efrag)
    print "Efrag %s: %s" % (" ".join([str(item) for item in frag_sites]),str(efrag))

e_arr = np.array(e_list)
print np.sum(e_arr)/(n/nf)


























"""
#make a list of range(nf) that tile n
fraglist = []
for i in range(n/nf): #assumes n divisible by nf
    fraglist.append(range(nf*i,nf*(1+i)))
print fraglist

h,V = c1c2.randHV(n,U=U,scale=0.)

print "Beginning HF"
hart = hf.HF(h,V,m)
psi = hart.get_Cocc()
rho = hart.get_rho()
F = hart.F
print hart.get_energy()/n

eF,vF = np.linalg.eigh(F)
Delta = antisymMat(n)
Delta *= 0. #let's just 0 this for now, like a good HF state. This should also make k=0

H = np.zeros((2*n,2*n))
H[:n,:n] = F
H[n:,n:] = -F
H[:n,n:] = Delta 
H[n:,:n] = -Delta
e,W = np.linalg.eigh(H) #This has all the right eigenvalues, but they're mixed with the positive ones

Wocc = W[:,:n] #If I choose W[:,n:], then I'll get a positive energy for the trace of G and H
#G = Wocc.dot(Wocc.T) #sorry bud, we're going to make sure we can recover the right energy w/o you atm
G = np.zeros((2*n,2*n))
G[:n,:n] = rho
G[n:,n:] = np.eye(n) - rho

e_list = []
for frag_sites in fraglist:
    T = schmidt.schmidt(psi,frag_sites,allv=True)
    C1 = T[:,:2*nf]
    Tc = T[:,2*nf:]
    h_f1 = xform.one(C1,h)
    V_f1 = xform.two(C1,V)
    core = xform.core(Tc,V)
    rho_f1 = xform.one(C1,rho+core)

    efrag = c1c2.embed_energy(nf,h_f1,V_f1,rho_f1)/nf
    e_list.append(efrag)
    print "Efrag %s: %s" % (" ".join([str(item) for item in frag_sites]),str(efrag))

e_arr = np.array(e_list)
print np.sum(e_arr)/(n/nf)

###hfb.py



Gswp = G.copy(); Hswp = H.copy()
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

h_f = c1c2.hRedHalf(nf,Hswp,C2) #0's the rows below nf
#V_f = c1c2.VRedHalf(nf,V,C2)
G_f = C2.T.dot(Gswp).dot(C2) #Why is this the identity matrix? (at least for half-filling)

h_e = np.zeros((2*n,2*n))
F_red = redFock(n,h,V)
h_e[:n,:n] = F_red
h_e[n:,n:] = -F_red
#h_e[:n,:n] = h
#h_e[n:,n:] = -h
h_swp = h_e.copy()
for i in range(nf):
    h_swp[:,[n+i,nf+i]] = h_swp[:,[nf+i,n+i]]
    h_swp[[n+i,nf+i]] = h_swp[[nf+i,n+i]]
h_f = c1c2.hRedHalf(nf,h_swp,C2) #0's the rows below nf
print "HFB fragment energy: ",2.*np.trace(G_f.dot(h_f))/nf #Works for arbitrary fragment size, half filling, V=0
"""
