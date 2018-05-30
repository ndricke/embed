import numpy as np
import itertools
import scipy.linalg as scl
import scipy as sc

import xform
import c1c2 
import hf #def formH(P,K,h,V):
    
np.set_printoptions(precision=2,suppress=True)

class HFB(object):
    def __init__(s,h,V,m,rho=None,Delta=None):
        s.h = h
        s.V = V
        s.m = m
        s.n = h.shape[0]

        if rho is None: #generate HF density matrix instead
            hart = hf.HF(h,V,m)
            s.rho = hart.get_rho()
            print hart.e/s.n
        else: s.rho = rho
        if Delta is None: #just set to 0's, as it would be for HF solution
            s.Delta = np.zeros((s.n,s.n))
        else: s.Delta = Delta
        s.F = hf.HF.genFock(s.rho,s.h,s.V) 
        s.H = s.genH(s.F,s.Delta)
        s.G = s.genG(s.H)

    def occOpt(s,stepsz=0.2,max_iter=50):
        H_app = s.H.copy()
        pots = np.zeros((s.n,s.n))
        for i in range(max_iter):
            G = s.genG(H_app)
            n_G = np.trace(G[:s.n,:s.n])
            err_new = n_G - s.m
            print "Info (err,i): ", err_new,i
            if np.abs(err_new) >= 10**-14:
                pots = np.eye(s.n)*stepsz*np.sign(err_new)
                H_app[:s.n,:s.n] += pots
                H_app[s.n:,s.n:] -= pots
                err = err_new
                if np.sign(err) != np.sign(err_new):
                    stepsz *= 0.5
                else:
                    stepsz *= 1.2
            else:
                return pots,G,H_app

    def occDif(s,pot):
        app = np.eye(s.n)*pot
        H_app[:s.n,:s.n] += app
        H_app[s.n:,s.n:] -= app
        G = s.genG(H_app)
        n_G = np.trace(G[:s.n,:s.n])
        return n_G - s.m

#Figure out two potentials that bracket the potential that must be applied for desired population
    def occBrack(s,maxiter=50,xfactor=1.4):
        #calculate whether population is above or below desired quantity with no applied potential
        Umax = 4. #the applied potentials are *probably* not bigger than 8; will still check
        err0 = s.occDif(0.)
        #set one end of bracket to 0
        if err0 < 0: #if the population is negative, we want to put a negative mu to draw particles
            static_brack = 0.
            dyn_brack = -Umax
        elif err0 > 0:
            static_brack = Umax
            dyn_brack = 0.
        elif err0 == 0: return 0.
        #iteratively increase other end of bracket until sign changes
        for i in range(maxiter):
            if s.occDif(static_brack)*s.occDif(dyn_brack) < 0:
                return sorted([static_brack,dyn_brack]),True
            else:
                static_brack = 1.0*dyn_brack
                dyn_brack *= xfactor
        return None,False

    def occOptBis(s):
        brack,success = s.occBrack()
        if success:
            return sc.optimize.brentq(s.occDif,brack[0],brack[1])
        else:
            raise ValueError("Failed to find brackets for root")

    @staticmethod
    def genH(F,Delta):
        n = F.shape[0]
        H = np.zeros((2*n,2*n))
        H[:n,:n] = F
        H[n:,n:] = -F
        H[:n,n:] = Delta 
        H[n:,:n] = -Delta
        return H

    @staticmethod
    def genG(H):
        n = int(H.shape[0]//2)
        e,W = np.linalg.eigh(H) #This has all the right eigenvalues, but they're mixed with the positive ones
        Wocc = W[:,:n] #If I choose W[:,n:], then I'll get a positive energy for the trace of G and H
        G = Wocc.dot(Wocc.T) 
        #G = np.zeros((2*n,2*n))
        #G[:n,:n] = rho
        #G[n:,n:] = np.eye(n) - rho
        return G

    def _do_hfb(s):
        H = s.genH(s.F,s.Delta)
        converged = False
        niter = 0
        errs = []
        Fs = []
        while not converged:
            s.count += 1; #print s.count
            s.genH()
            s.genG(H)
            
            F_new_MO = C.T.dot(F_new).dot(C)
            #err = F_new_MO[:m,m:]
            err = F_new.dot(rho_new) - rho_new.dot(F_new)

            errs.append(err)
            Fs.append(F_new)
            if s.diis:
                F = HF._next_diis(errs, Fs)
            else:
                s.rho = rho_new
                F = F_new

            converged = numpy.linalg.norm(err)<s.tol*n
            niter += 1
            emat = Cocc.T.dot(s.h+F).dot(Cocc)
            kennyloggins.debug("Levels: %s" % str([emat[i,i] for i in range(len(emat))]))
        emat = Cocc.T.dot(s.h+F).dot(Cocc)
        e = 0
        for i in range(s.m):
            e += emat[i,i]
        s.F = F
        s.C = C
        s.e = e

    @staticmethod
    def _next_diis(errs, Fs):
        n = len(errs)
        B = numpy.zeros((n,n))
        for i,j in itertools.product(range(n), range(n)):
            B[i,j] = numpy.dot(errs[i].ravel(), errs[j].ravel())
        A = numpy.zeros((n+1, n+1))
        A[:n,:n] = B
        A[n,:] = -1
        A[:,n] = -1
        A[n,n] = 0
        b = numpy.zeros(n+1)
        b[n] = -1
        try:
            x = numpy.linalg.solve(A,b)
        except (numpy.linalg.linalg.LinAlgError):
            print "lin solver fails! Using pinv..."
            P = numpy.linalg.pinv(A)
            x = P.dot(b)
        w = x[:n]

        F = numpy.zeros(Fs[0].shape)
        for i in range(n):
            F += w[i] * Fs[i]
        return F

if __name__ == "__main__":
    n = 12
    U = 2.
    m = 6
    nf = 4
    h,V = c1c2.randHV(n,U=U,scale=0.1)
    D_rnsym = c1c2.antisymMat(n)*0.1
    bogil = HFB(h,V,m,Delta=D_rnsym)

#    pots = np.eye(n)*0.1
#    H_app = bogil.H.copy()
#    for i in range(5):
#        H_app[:n,:n] += pots
#        H_app[n:,n:] -= pots
#        print np.trace(bogil.genG(H_app)[:n,:n])

    app_pots,G,H_app = bogil.occOpt()
    H = bogil.H
    







