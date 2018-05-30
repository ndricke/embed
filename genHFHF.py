import numpy as np
import sys

import hf
import xform
import schmidt
import c1c2

def rowcolSwp(mtrx,i,j):
    mtrx[:,[i,j]] = mtrx[:,[j,i]]
    mtrx[[i,j]] = mtrx[[j,i]]

#Swap indices in a 4-tensor
def rcSwp4(V,i,j):
    V[[i,j],:,:,:] = V[[j,i],:,:,:]    
    V[:,[i,j],:,:] = V[:,[j,i],:,:]    
    V[:,:,[i,j],:] = V[:,:,[j,i],:]    
    V[:,:,:,[i,j]] = V[:,:,:,[j,i]]    

n = 8
m = 4
U = 2.
nf = 2

#make a list of range(nf) that tile n
fraglist = []
for i in range(n//nf): #assumes n divisible by nf
    fraglist.append(range(nf*i,nf*(1+i)))
print(fraglist)

h,V = c1c2.randHV(n,U=U,h_scale=0.5,V_scale=0.5)
#h,V = c1c2.coulV(n, U=U, cutoff=1)

print("Beginning HF")
hart = hf.HF(h,V,m)
psi = hart.get_Cocc()
rho = hart.get_rho()
F = hart.F
print("HF Energy:", hart.get_energy())


e_list = []; e_list2 = []; e_list3 = []; e_list4 = []

for frag_sites in fraglist:

    #Test for generating C1 (via wavefunction rather than density matrix)
    C1, Tc, hc, h_f, V_f = c1c2.C1Proj(nf, psi, frag_sites, h, V)
    P1_f = xform.one(C1, rho)
    P2_f = c1c2.genP2f(P1_f)
    hc_f = xform.one(C1, hc)

    F_f = xform.one(C1,h+F)
    efrag = np.trace(np.dot(F_f[:,:nf].T,P1_f[:,:nf]))/nf
    e1_1 = np.trace(np.dot(2.*h_f[:,:nf].T,P1_f[:,:nf]))/nf
    e1_2 = efrag - e1_1
    e_list4.append(efrag)
    print("Efrag %s: %f, e1: %f, e2: %f" % (" ".join([str(item) for item in frag_sites]),efrag,e1_1,e1_2))

#    efrag2 = c1c2.embed_e1(nf,F_f,P1_f)/nf
#    e_list2.append(efrag2)
#    print("Efrag2 %s: %s" % (" ".join([str(item) for item in frag_sites]),str(efrag2)))
#
#    efrag3 = c1c2.embed_energy(nf, h_f, hc_f, V_f, P1_f, P2_f)/nf
#    e_list3.append(efrag3)
#    print("Efrag3 %s: %s" % (" ".join([str(item) for item in frag_sites]),str(efrag3)))
#
#    efrag4, e4_1, e4_2 = c1c2._calc_energy(nf, h_f, hc_f, V_f, P1_f, P2_f)
#    e_list4.append(efrag4/nf)
#    print("Efrag4: %f, e1: %f, e2: %f" % (efrag4, e4_1, e4_2))

    C1, Tc, hc, h_f, V_f = c1c2.C1svd(nf, psi, frag_sites, h, V)
    P1_f = xform.one(C1, rho)
    P2_f = c1c2.genP2f(P1_f)
    hc_f = xform.one(C1, hc)

    F_f = xform.one(C1,h+F)
    efrag = np.trace(np.dot(F_f[:,:nf].T,P1_f[:,:nf]))
    e1_1 = np.trace(np.dot(2.*h_f[:,:nf].T,P1_f[:,:nf]))
    e1_2 = efrag - e1_1
    e_list.append(efrag)
    print("Efrag %s: %f, e1: %f, e2: %f" % (" ".join([str(item) for item in frag_sites]),efrag,e1_1,e1_2))

    efrag2 = c1c2.embed_e1(nf,F_f,P1_f)/nf
    e_list2.append(efrag2)
    print("Efrag2 %s: %s" % (" ".join([str(item) for item in frag_sites]),str(efrag2)))

    efrag3 = c1c2.embed_energy(nf, h_f, hc_f, V_f, P1_f, P2_f)/nf
    e_list3.append(efrag3)
    print("Efrag3 %s: %s" % (" ".join([str(item) for item in frag_sites]),str(efrag3)))
    

#    sys.exit(-1)

#e_arr = np.array(e_list)
#print(np.sum(e_arr)/(n/nf))

e_arr = np.array(e_list)
e_1 = np.sum(e_arr)
print("Embed energy 1:", e_1)

e_arr2 = np.array(e_list2)
e_2 = np.sum(e_arr2)/(n/nf)
print("Embed energy 2:", e_2)

e_arr3 = np.array(e_list3)
e_3 = np.sum(e_arr3)/(n/nf)
print("Embed energy 3:", e_3)

e_arr4 = np.array(e_list4)
e_4 = np.sum(e_arr4)/(n/nf)
print("Embed energy 4:", e_4)

##In-line C2 generation
#    rho_imp = rho_swp[:nf,:nf] #pull Gimp out of re-organized density matrix
#    rho_ic = rho_swp[:,:nf] #pull out [[Gimp],[Gc]]
#    erhoi,vrhoi = np.linalg.eigh(rho_imp) #diagonalize Gimp so we can get A
#    sq_erhoi = np.diag(np.sqrt(erhoi)) #take sqrt of eigenvalues then turn it into a matrix
#    A = np.dot(vrhoi,sq_erhoi) #AA.T = Gimp, and A = U.dot(e^0.5)
#    ATinv = np.linalg.inv(A.T)
#    C2 = rho_ic.dot(ATinv)

##In-line C1 generation
#    T = schmidt.schmidt(psi,frag_sites,allv=True)
#    C1 = T[:,:2*nf]
#    Tc = T[:,2*nf:]
#    h_f1 = xform.one(C1,h)
#    V_f1 = xform.two(C1,V)
#    core = xform.core(Tc,V)
#    rho_f1 = xform.one(C1,rho+core)

##Test for generating C1 (via wavefunction rather than density matrix)
#e_list = []
#for frag_sites in fraglist:
#    T = schmidt.schmidt(psi,frag_sites,allv=True)
#    C1 = T[:,:2*nf]
#    Tc = T[:,2*nf:]
#    h_f1 = xform.one(C1,h)
#    V_f1 = xform.two(C1,V)
#    core = xform.core(Tc,V)
#    rho_f1 = xform.one(C1,rho+core)
#
#    efrag = c1c2.embed_energy(nf,h_f1,V_f1,rho_f1)/nf
#    e_list.append(efrag)
#    print "Efrag %s: %s" % (" ".join([str(item) for item in frag_sites]),str(efrag))
#
#e_arr = np.array(e_list)
#print np.sum(e_arr)/(n/nf)



