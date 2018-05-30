import numpy as np
import itertools
import logging
import scipy.optimize as sco

if __name__=="__main__": logging.basicConfig()
kennyloggins = logging.getLogger('hf')

class HF:
    def __init__(s,h,V,m,scf='diis',tol=1e-8,rho_guess=None):
        s.h = h
        s.V = V
        s.m = m
        s.scf = scf
        s.tol = tol

        s.n = s.h.shape[0] #Num rows is the dimension of the density matrix to be
        if type(rho_guess) == np.ndarray:
            s.rho = rho_guess
            if s.rho.shape[0] != s.n: raise ValueError("Input HF guess has wrong shape")
        else:
            s.rho = np.eye(s.n)

        s.C = None
        s.e = None

    @staticmethod
    def genFock(rho,h,V):
        if type(V) == float:
            F = h + np.diag(np.diag(rho)*V)
        elif len(V.shape)==4:
            F = h.copy()
            ri = range(len(h))
            for mu,nu,la,si in itertools.product(ri,ri,ri,ri):
                F[mu,nu] += (2*V[mu,si,nu,la] - V[mu,si,la,nu])*rho[la,si]
        else:
            print "Unknown V shape"
            import sys
            sys.exit(1)
        return F

    def _do_hf(s):
        n = s.h.shape[0]
        s.F = s.genFock(s.rho,s.h,s.V)
        s.e = np.trace(s.rho.dot(s.h+s.F))
        s.count = 0

        converged = False
#        if s.scf == 'diis':
        if 'diis' in s.scf:
            errs = []; Fs = [];
        while not converged:
            s.count += 1; print s.count,s.e
#            if s.count % 10 == 0: print s.count;

            w,C = np.linalg.eigh(s.F)
            s.checkDegen(w) #Warn if homo-lumo degeneracy
            Cocc = C[:,:s.m]
            s.rho_new = Cocc.dot(Cocc.T)
            s.F_new = s.genFock(s.rho_new,s.h,s.V)
            s.e_new = np.trace(s.rho_new.dot(s.h+s.F_new))

            if s.scf == 'rca':
                s.rho_new = s._rca_iter()
                s.F_new = s.genFock(s.rho_new,s.h,s.V)
            err = s.F_new.dot(s.rho_new) - s.rho_new.dot(s.F_new)
            converged = np.linalg.norm(err)<s.tol*n
            if s.scf == 'diis':
                errs.append(err)
                Fs.append(s.F_new)
                s.F_new = HF._next_diis(errs, Fs)
            if s.scf == 'rcadiis':
                s.rho_new = s._rca_iter()
                s.F_new = s.genFock(s.rho_new,s.h,s.V)
                err = s.F_new.dot(s.rho_new) - s.rho_new.dot(s.F_new)
                errs.append(err)
                Fs.append(s.F_new)
                s.F_new = HF._next_diis(errs, Fs)

            s.rho = s.rho_new.copy()
            s.F = s.F_new.copy()
            s.e = s.e_new.copy()

        print s.count
        s.C = C


    def _rca_lc(s,a):
        rho_rca = a*s.rho_new + (1.-a)*s.rho
        F = s.genFock(rho_rca,s.h,s.V)
        return rho_rca, F

    def _rca_E(s,a):
        #E = s.e_new*a + s.e*(1.-a) + 0.5*a*(s.rho_new-s.rho).dot((1.-a)*(s.F_new-s.F))
        rho_rca = a*s.rho_new + (1.-a)*s.rho
        F = s.genFock(rho_rca,s.h,s.V)
        return np.trace(rho_rca.dot(s.h+F))

    def _rca_iter(s):
        opt_res = sco.minimize_scalar(s._rca_E,bounds=[0.,1.],method='bounded',tol=1e-08) 
        a = opt_res.x
        return a*s.rho_new + (1.-a)*s.rho

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

    def get_Cocc(s):
        if s.C is None: s._do_hf()
        return s.C[:,:s.m]

    def get_C(s):
        if s.C is None: s._do_hf()
        return s.C

    def get_energy(s):
        if s.e is None: s._do_hf()
        return s.e

    def get_rho(s):
        C = s.get_Cocc()
        return C.dot(C.T)

    def get_P2(s):
        P1 = s.get_rho()
        s.P2 = s.pdm2(P1)
        return s.P2

    def pdm2(s,P1):
        n = P1.shape[0]
        P2 = np.zeros((n,n,n,n))
        sz = range(n)
        for i,j,k,l in itertools.product(sz,sz,sz,sz):
            P2[i,j,k,l] = P1[k,i]*P1[j,l]
        return P2

    #Check for HOMO-LUMO degeneracy
    def checkDegen(s,w):
        homo = w[s.m-1]
        lumo = w[s.m]
        if abs(lumo-homo) < 1e-4:
            kennyloggins.warning("DANGER ZONE: HOMO and LUMO are degenerate")

if __name__ == "__main__":
#    import lattice
    import sys
    import argparse

#    parser = argparse.ArgumentParser()
#    parser.add_argument('-m', help='Number of electron pairs', type=int, required=True)
#    parser.add_argument('-l', help='Lattice file', type=str, required=True)
#    parser.add_argument('-o', help='Output file', type=str, required=True)
#    parser.add_argument('-guess', help='HF Guess', type=str)
#    args = parser.parse_args()
#
#    rho = np.genfromtxt(args.guess)
#
#    lat = lattice.Lattice.fromfile(args.l,args.m)
#    if args.guess != None: fock = HF(lat.h,lat.V,lat.m,rho_guess = rho) 
#    else: fock = HF(lat.h,lat.V,lat.m)
#    rho = fock.get_rho()
#    np.savetxt(args.o,rho)


    np.set_printoptions(precision=3, suppress = True)
    n = 42
    m = 21
    U = 8.0
    h = np.diag(np.ones(n-1),1)
    h[0,-1] = 1
    h += h.T
    h*=-1
    for i in range(0,n,2):
        h[i,i] = 1

    hf = HF(h,U,m,scf='diis')
    print "Energy: ", hf.get_energy()

#    hf2 = HF(h,U,m,scf='rca')
#    print "Energy: ", hf2.get_energy()

    hf3 = HF(h,U,m,scf='rcadiis')
    print "Energy: ", hf3.get_energy()

#    hf2 = HF(h,U,m,scf='diis')
#    psi2 = hf2.get_Cocc()
#    print hf2.get_energy()





