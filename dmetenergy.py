import numpy
import itertools
import sys

import est.xform

def _calc_energy(s,ni,center,Pf,hf,hc,V):
    e1 = 0
    e1c = 0

    nz = range(2*ni)
    niz = range(ni) #we always organize fragment, then bath

#        for f,j in itertools.product(niz,nz):
#            frag.e1 += hf[f,j] * P1f[j,f]
#        for f,j in itertools.product(center,nz):
#            frag.e1c += hf[f,j] * P1f[j,f]

    core = 0.0
    core_cen = 0.0
    for f,j in itertools.product(niz,nz):
        e1 += (hf[f,j]-hc[f,j]) * P1f[j,f]
    for f,j in itertools.product(center,nz):
        e1c += (hf[f,j]-hc[f,j]) * P1f[j,f]
    for f in niz:
        core += hc[f,f]*P1f[f,f]
    for c in center:
        core_cen += hc[c,c]*P1f[c,c]
    core /= 2.*ni
    core_cen /= 2.*len(center)

    e1 /= ni
    e1c /= len(center)

    #2 electron component
    if type(V) == float: e2,e2c = e2_Ucalc()
    elif len(V.shape) == 4: e2,e2c = e2_V4calc(frag,center)

    e = e1 + e2 + core
    ec = e1c + e2c + core_cen

#Calculates 2e- energy when V is a 4-tensor
def e2_V4calc(s,ni,center,V,P2f):
    e2 = 0.0
    e2c = 0.0
    f_index = range(ni)
    niz = range(2*ni) 
    
    #Fragment 2e Energy
    for f,j,k,l in itertools.product(f_index,niz,niz,niz):
        e2 += V[f,j,k,l]*P2f[f,j,k,l]
    e2 /= ni*4.
    
    #Center 2e Energy
    for c,j,k,l in itertools.product(center,niz,niz,niz):
        e2c += V[c,j,k,l]*P2f[c,j,k,l]
    e2c /= len(center)*4.

    return e2,e2c

#Calculates 2e- energy when V is a hubbard U, so we assume V is diagonal
def e2_Ucalc(s,ni,center,V,P2f):
    e2 = 0.0; e2c = 0.0
    for i in range(ni):
        e2 += P2f[i,i,i,i]/4.*V[i,i,i,i]
    for i in center_fragdex:
        e2c += P2f[i,i,i,i]/4.*V[i,i,i,i]
    e2 /= ni
    e2c /= len(center_fragdex)
    return e2,e2c

def pr_P2(s,diag=True,off_diag=False):
    if diag == True:
        print("P2 Diagonal Entries:")
        for i in range(s.ni):
            print(s.P2_r[i,i,i,i])
    if off_diag == True:
        print("P2 Off-Diagonal Entries:")
        for i in range(s.ni-1):
            print(str(s.P2_r[i,i+1,i,i+1]) + " " + str(s.P2_r[i+1,i,i+1,i]))



