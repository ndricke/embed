import numpy as np

import hf
import hfb
import c1c2
import occopt as oc


#def parab(x):
#    return 3*x**2+2*x-12
#x = 5.
#for i in range(15):
#    print "Iteration ",i
#    x_new = oc.newton_iter(x,parab)
#    print x_new
#    if np.abs(x_new-x) < 10**-8:
#        break
#    x = x_new

n = 4
nf = 1
m = 2
U = 0.

h,V = c1c2.randHV(n,U=U,scale=0.)
hart = hf.HF(h,V,m)
hart._do_hf()
rho = hart.rho
F = hart.F
Delta = np.zeros((n,n))

H = hfb.HFB.genH(n,F,Delta)
print H

G = hfb.HFB.genG(n,H)
print
print rho
print
print G














