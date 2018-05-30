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

#h,V = c1c2.hubb(n,U=U)
h,V = c1c2.randHV(n,U=U,h_scale=0.1,V_scale=0.1)
#h,V = c1c2.coulV(n, U=U, cutoff=1)

print("Beginning HF")
hart = hf.HF(h,V,m)
psi = hart.get_Cocc()
rho = hart.get_rho()
F = hart.F
print("HF Energy:", hart.get_energy())

e_list = []; e_list2 = []; e_list3 = []; e_list4 = []
for frag_sites in fraglist:


    rho_swp = rho.copy(); F_swp = F.copy(); h_swp = h.copy(); V_swp = V.copy()
    for rc,i in enumerate(frag_sites): #swap fragment to be in upper-left of p-p and p-h blocks
        rowcolSwp(rho_swp,rc,i)
        rowcolSwp(F_swp,rc,i)
        rowcolSwp(h_swp,rc,i)
    #for rc,i in enumerate(frag_sites):
    #    rcSwp4(V_swp,rc,i)

    rho_swp_hole = np.eye(n) - rho_swp

    C2 = c1c2.C2swp(rho_swp,nf) #C2swp relies on the density matrix having been rearranged
    C2_hole = c1c2.C2swp(rho_swp_hole,nf) #C2swp relies on the density matrix having been rearranged

    #T_2 = np.zeros((n,2*nf))
    #T_2[:nf,:nf] = C2[:nf,:]
    #T_2[nf:,nf:] = C2[nf:,:]
    
    T_2 = np.column_stack((C2,C2_hole))
#    print(C2)
#    print(C2_hole)
#    print(T_2)

    TdT = np.dot(T_2.T,T_2)
    T_2 *= np.diag(TdT)**-0.5
#    print(TdT)
#    print(T_2)
#    print(np.linalg.det(TdT))

    P1_f = xform.one(T_2, rho_swp)
    h_f = xform.one(T_2,h_swp)
    F_f = xform.one(T_2,h_swp+F_swp)

    F_r = c1c2.reduceBath(h_swp+F_swp, nf=2)
    F_rf = xform.one(T_2,F_r)

    #F_f = c1c2.hRedHalf(nf,F_swp+h_swp,T_2) #0's the rows below nf
    #P2_f = c1c2.genP2f(rho_f)
    #V_f = xform.two(T_2,V)
    #    V_f = c1c2.genP2f(P1_f)

    efrag = np.trace(np.dot(F_f[:,:nf].T,P1_f[:,:nf]))
    e_list.append(efrag)
    efrag = np.trace(np.dot(F_rf.T,P1_f))
    e_list3.append(efrag)


    C1, Tc, hc, h_f, V_f = c1c2.C1svd(nf, psi, frag_sites, h, V)
    P1_f = xform.one(C1, rho)
    h_f = xform.one(C1,h)
    F_f = xform.one(C1,h+F)
    efrag = np.trace(np.dot(F_f[:,:nf].T,P1_f[:,:nf]))
    e1_1 = np.trace(np.dot(2.*h_f[:,:nf].T,P1_f[:,:nf]))
    e1_2 = efrag - e1_1
    e_list2.append(efrag)


print(np.sum(e_list))
print(np.sum(e_list2))
print(np.sum(e_list3))





