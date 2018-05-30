import numpy as np
import itertools
import logging

class UHF:
    def __init__(s,h,V,m_a,m_b,diis=True,tol=1e-8,rho_guess=None):
        s.h = h
        s.V = V
        s.m_a = m_a
        s.m_b = m_b
        s.diis = diis
        s.tol = tol

        s.n = s.h.shape[0] #Num rows is the dimension of the density matrix to be
        if type(rho_guess) == np.ndarray:
            s.rho_a = rho_guess[0]
            s.rho_b = rho_guess[1]
            if s.rho.shape[0] != s.n: raise ValueError("Input HF guess has wrong shape")
        else:
            s.rho_a = np.eye(s.n)
            s.rho_b = np.eye(s.n)

        s.C = None
        s.e = None

    @staticmethod
    def genFock(rho_a,rho_b,h,V):
        F_a = h.copy()
        F_b = h.copy()
        ri = range(len(h))
        for mu,nu,la,si in itertools.product(ri,ri,ri,ri):
            F_a[mu,nu] += (V[mu,si,nu,la] - V[mu,si,la,nu])*rho_a[la,si] + V[mu,si,nu,la]*rho_b[la,si]
            F_b[mu,nu] += (V[mu,si,nu,la] - V[mu,si,la,nu])*rho_b[la,si] + V[mu,si,nu,la]*rho_a[la,si]
        return F_a, F_b

    def checkHomoLumoDegen(s,w,m):
            homo = w[m-1]
            lumo = w[m]
            if abs(lumo-homo) < 1e-4:
                print "DANGER ZONE: HOMO and LUMO are degenerate"

    def genRho(s,F,m):
            w,C = np.linalg.eigh(F)
            s.checkHomoLumoDegen(w,m)
            Cocc = C[:,:m]
            rho = Cocc.dot(Cocc.T)
            return rho

    def _do_scf(s):
        n = s.h.shape[0]
        F_a, F_b = s.genFock(s.rho_a,s.rho_b,s.h,s.V)

        converged = False
        s.count = 0
        while not converged:
            s.count += 1; print s.count;
            rho_a_new = s.genRho(F_a, s.m_a)
            rho_b_new = s.genRho(F_b, s.m_b)
            F_a_new, F_b_new = s.genFock(rho_a_new,rho_b_new,s.h,s.V)
            err = F_a_new.dot(rho_a_new) - rho_a_new.dot(F_a_new) + \
                  F_b_new.dot(rho_b_new) - rho_b_new.dot(F_b_new)

            F_a, F_b = (F_a_new, F_b_new)
            converged = np.linalg.norm(err)<s.tol*n

        s.rho_a, s.rho_b = (rho_a_new, rho_b_new)
        s.F_a, s.F_b = (F_a,F_b)

    @staticmethod
    def _next_diis(errs, Fs):
        n = len(errs)
        B = np.zeros((n,n))
        for i,j in itertools.product(range(n), range(n)):
            B[i,j] = np.dot(errs[i].ravel(), errs[j].ravel())
        A = np.zeros((n+1, n+1))
        A[:n,:n] = B
        A[n,:] = -1
        A[:,n] = -1
        A[n,n] = 0
        b = np.zeros(n+1)
        b[n] = -1
        try:
            x = np.linalg.solve(A,b)
        except (np.linalg.linalg.LinAlgError):
            print "lin solver fails! Using pinv..."
            P = np.linalg.pinv(A)
            x = P.dot(b)
        w = x[:n]

        F = np.zeros(Fs[0].shape)
        for i in range(n):
            F += w[i] * Fs[i]
        return F

if __name__ == "__main__":
    from hf import HF
    np.set_printoptions(precision=3, suppress = True)

    #Hubbard ABAB model
    n = 8
    ma = 4
    mb = 4
    U = 2.0
    V = np.zeros((n,n,n,n))
    for i in range(n):
        V[i,i,i,i] = U

    h = np.diag(np.ones(n-1),1)
    h[0,-1] = 1
    h += h.T
    h*=-1
    for i in range(0,n,2):
        h[i,i] = 1

    uhf = UHF(h,V,ma,mb)
    uhf._do_scf()
    rhoa = uhf.rho_a
    rhob = uhf.rho_b
    rhoT = rhoa + rhob
    print rhoT
#    print np.trace(uhf.rho_a.dot(uhf.F_a)+uhf.rho_b.dot(uhf.F_b))
#    print np.trace(uhf.rho_a)
#    print np.trace(uhf.rho_b)


    hf = HF(h,V,ma)
    hf._do_hf()
#    print hf.e
#    print np.trace(hf.rho)

    print hf.rho



















