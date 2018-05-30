import numpy

def schmidt(psi,sites,allv=False):
    ns = len(sites)
    n,m = psi.shape
    Pf = numpy.zeros((n,n))
    for i in sites:
        Pf[i,i] = 1
    Pb = numpy.eye(n) - Pf

    M = numpy.zeros((m,m))
    for q in range(m):
        for p in range(m):
            M[p,q] = psi[:,q].dot(Pf.dot(psi[:,p]))
    w,V = numpy.linalg.eigh(M)

    F = numpy.zeros((n,ns))
    B = numpy.zeros((n,ns))
    for i in range(ns):
        for p in range(m):
            #F[:,i] += V[p,-i-1]/numpy.sqrt(w[-i-1]) * Pf.dot(psi[:,p])
            B[:,i] += V[p,-i-1]/numpy.sqrt(1-w[-i-1]) * Pb.dot(psi[:,p])
    if allv:
        E = numpy.zeros((n,m-ns))
        for i in range(m-ns):
            for p in range(m):
                E[:,i] += V[p,i]*psi[:,p]

    for i,site in enumerate(sites):
        F[site,i] = 1 #Easier to apply the potential in this basis
    if not allv:
        T = numpy.zeros((n,2*ns))
        T[:,:ns] = F
        T[:,ns:] = B
    else:
        T = numpy.zeros((n,m+ns))
        T[:,:ns] = F
        T[:,ns:2*ns] = B
        T[:,2*ns:] = E


    return T

def fast_schmidt(psi,sites):
    rho = psi.dot(psi.T)
    n,m = psi.shape

    ns = len(sites)
    F = numpy.zeros((n,ns))
    B = numpy.zeros((n,ns))

    for i,site in enumerate(sites):
        B[:,i] = rho[:,site]
        F[site,i] = 1
    for site in sites:
        B[site,:] = 0
    for i in range(len(sites)):
        B[:,i] /= numpy.linalg.norm(B[:,i])
    T = numpy.zeros((n,2*ns))
    T[:,:ns] = F
    T[:,ns:] = B

    S = T.T.dot(T)
    w,v = numpy.linalg.eigh(S)
    whalf = 1/numpy.sqrt(w)
    Shalf = v.dot(numpy.diag(whalf)).dot(v.T)
    T = T.dot(Shalf)


    return T






if __name__ == "__main__":
    numpy.set_printoptions(suppress=True, precision=5)
    n = 5
    m = 3

    H = numpy.random.randn(n,n)
    H+=H.T

    w,v = numpy.linalg.eigh(H)
    psi = v[:,:m]

    print("Psi overlap")
    print(psi.T.dot(psi))
    print()

    sites = [0,2]
    ns = len(sites)
    F,B = schmidt(psi, sites)

    T = numpy.zeros((n,2*ns))
    T[:,:ns] = F
    T[:,ns:] = B
    print("T vector")
    print(T)
    print()

    print("T vector overlap")
    print(T.T.dot(T))
    print()

    himp = xform_hub1(T)
    print("Impurity one electron integrals:")
    print(himp)
    print()

    Uimp = xform_hub2(T)
    print("Impurity two electron integrals:")
    print(Uimp)
    print()




