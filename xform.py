import numpy
import itertools
import logging

kennyloggins = logging.getLogger("xform")

def one(T,h):
    return T.T.dot(h).dot(T)

def two(T,V):
    n, ni = T.shape

    Vi = numpy.zeros((ni,ni,ni,ni))
    
    imps = range(ni)

    if len(V.shape) == 1:
        for i,j,k,l in itertools.product(imps,imps,imps,imps):
            for mu in range(n):
                Vi[i,j,k,l] += T[mu,i]*T[mu,j]*T[mu,k]*T[mu,l]*V[mu]
    elif len(V.shape) == 4:
        Vhalf = numpy.zeros((ni,ni,n,n ))
        for i in range(n):
            for j in range(n):
                Vhalf[:,:,i,j] = T.T.dot(V[:,:,i,j]).dot(T)
        for i in imps:
            for j in imps:
                Vi[i,j,:,:]  = T.T.dot(Vhalf[i,j,:,:]).dot(T)

    else: 
        kennyloggins.error( "V shape wrong")
        import sys; sys.exit(1)
    return Vi

def twoU(T,U):
    n, ni = T.shape

    Vi = numpy.zeros((ni,ni,ni,ni))
    
    imps = range(ni)

    for i,j,k,l in itertools.product(imps,imps,imps,imps):
        for mu in range(n):
            Vi[i,j,k,l] += T[mu,i]*T[mu,j]*T[mu,k]*T[mu,l]*U
    return Vi





def rdm(T,psi):
    rho = psi.dot(psi.T)
    return T.T.dot(rho).dot(T)

def one_hub(T):
    n = T.shape[0]
    h = numpy.diag(numpy.ones(n-1),1)
    h[-1,0] = 1
    h += h.T
    h *= -1
    return T.T.dot(h).dot(T)

def two_hub(T):
    n,m = T.shape
    Uimp = numpy.zeros((m,m,m,m))
    #for i in range(m):
    #    Uimp[i,i,i,i] = 1
    for i in range(m):
     for j in range(m):
      for k in range(m):
       for l in range(m):
        for mu in range(n):
            Uimp[i,j,k,l] += T[mu,i]*T[mu,j]*T[mu,k]*T[mu,l]
    return Uimp

def core(T, V):
    n,m = T.shape
    rc = T.dot(T.T)
    if type(V) == float: #if V comes as an int, we want it to throw an error
        hc = numpy.diag([rc[i,i]*V for i in range(n)])
    elif len(V.shape) == 4:
        hc = coreFock(rc,V)
    return hc

def coreFock(rho,V):
    sz = len(V)
    F = numpy.zeros((sz,sz))
    ri = range(sz)
    for mu,nu,la,si in itertools.product(ri,ri,ri,ri):
        F[mu,nu] += (2*V[mu,si,nu,la] - V[mu,si,la,nu])*rho[la,si]
    return F

