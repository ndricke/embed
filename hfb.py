import numpy as np
import itertools
import scipy.linalg as scl
import scipy.optimize as sco
import scipy as sc

import xform
import c1c2 
import hf #def formH(P,K,h,V):

import sys
    
np.set_printoptions(precision=4,suppress=True)

class HFB(object):
    def __init__(s,h,V,m,rho=None,Delta=None,tol=10**-7):
        s.h = h
        s.V = V
        s.m = m
        s.n = h.shape[0]
        s.tol = tol
        s.app_pot = 0.
        s.mu = 0. #each time we re-calc applied potentials, we'll keep track of the total with mu

        if rho is None: #generate HF density matrix instead
            hart = hf.HF(h,V,m,scf='diis')
            s.rho = hart.get_rho()
            s.E_hf = hart.e/s.n
            print "HF energy: ", s.E_hf
        else: s.rho = rho
        if Delta is None: #just set to 0's, as it would be for HF solution
            s.Delta = np.zeros((s.n,s.n))
        else: s.Delta = Delta
        s.F = hf.HF.genFock(s.rho,s.h,s.V) 
        s.H = np.zeros((2*s.n,2*s.n))
        s.H[:s.n,:s.n] = s.F
        s.H[s.n:,s.n:] = -s.F
        s.H[:s.n,s.n:] = s.Delta 
        s.H[s.n:,:s.n] = -s.Delta
        s.G = s.genG(s.H)
        ##I thought we could use commutability as a convergence criteria, but maybe that isn't true here
        #print "Initial distance from convergence: ",np.linalg.norm(np.dot(s.G,s.H)-np.dot(s.H,s.G))

    def occDif(s,pot):
        H_app = s.appH(s.H,pot)
        G = s.genG(H_app)
        return np.trace(G[:s.n,:s.n]) - s.m

#Figure out two potentials that bracket the potential that must be applied for desired population
    def occBrack(s,maxiter=50,xfactor=1.4):
        #calculate whether population is above or below desired quantity with no applied potential
        Umax = 8. #the applied potentials are *probably* not bigger than 8; will still check
        err0 = s.occDif(0.)
        #set one end of bracket to 0
        if err0 < 0: #if the population is negative, we want to put a negative mu to draw particles
            static_brack = 0.
            dyn_brack = -Umax
        elif err0 > 0:
            static_brack = Umax
            dyn_brack = 0.
        elif err0 == 0: return 0.,True
        #iteratively increase other end of bracket until sign changes
        for i in range(maxiter):
            if s.occDif(static_brack)*s.occDif(dyn_brack) < 0:
                print "occBrack potential range:",sorted([static_brack,dyn_brack])
                return sorted([static_brack,dyn_brack]),True
            else:
                static_brack = 1.0*dyn_brack
                dyn_brack *= xfactor
        print "occBrack: Failed to find suitable root brackets"
        return None,False

    def occOpt(s):
        brack,success = s.occBrack() #applies nec. mu to tune pop to m
        if success:
            if brack == 0: s.app_pot,conv = (0.,'Zero-Limit')
            else: s.app_pot,conv = sco.brentq(s.occDif,brack[0],brack[1],full_output=True) #XXX unreliable
        else: raise ValueError("Failed to find brackets for root")
        s.mu += s.app_pot
        H_app = HFB.appH(s.H,s.app_pot)
        G_new = HFB.genG(H_app)
        return H_app, G_new

    def _do_hfb(s):
        converged = False
        count = 0;
        while not converged:
            #s.app_pot,convergence = s.occOpt()
            brack,success = s.occBrack() #applies nec. mu to tune pop to m
            if success:
                if brack == 0: s.app_pot,conv = (0.,'Zero-Limit')
                else: s.app_pot,conv = sco.brentq(s.occDif,brack[0],brack[1],full_output=True) #XXX unreliable
            else: raise ValueError("Failed to find brackets for root")
            s.mu += s.app_pot
            H_app = HFB.appH(s.H,s.app_pot)
            G_new = HFB.genG(H_app)
            s.F = hf.HF.genFock(G_new[:s.n,:s.n],s.h,s.V) #is HFB the same as HF for this part?
            s.Delta = HFB.genDelta(G_new[:s.n,s.n:],s.V) #are we applying this correctly?
            s.H = HFB.genH(s.F,s.Delta)

            #err = H_new.dot(G_new) - G_new.dot(H_new) #Blaizot_p189: [H,R]=0
            err = G_new - s.G #I switched to this criteria b/c the other didn't seem to be working
            s.G = G_new

            print np.trace(s.G[:s.n,s.n:].dot(s.H[:s.n,s.n:]))
            converged = np.linalg.norm(err)<s.tol*s.n
            count += 1
            if count % 10 == 0: 
                print "HFB Iteration ",count
            #print brack
            #print "applying potential: ", s.app_pot
            #print "trace(G[:n,:n]) - target: ", np.trace(G_new[:s.n,:s.n])-s.m
            #print "HFB Iteration ",count
            #print "HFB error: ",np.linalg.norm(err)
            #print "Stats: counter, bracket, applied potential, norm(G-G0)"
            #print count,brack,s.app_pot,np.linalg.norm(err)
            #print s.G[:5,:5]

        print "HFB supposedly converged?"
#        s.E = np.trace(np.dot(s.G[:n,:n],s.H[:n,:n]+s.h-0.*s.mu*np.eye(s.n))-s.G[:n,n:].dot(s.H[:s.n,s.n:]))/s.n
        s.E = np.trace(np.dot(s.G[:s.n,:s.n],s.F[:s.n,:s.n]+s.h)-s.G[:s.n,s.n:].dot(s.H[:s.n,s.n:]))/s.n
        F_mod = s.H[:s.n,:s.n]-s.app_pot*np.eye(s.n)

    @staticmethod
    def genDelta(Ki,V):
        n = len(V)
        D = np.zeros((n,n))
        ri = range(n)
        for i,j,k,l in itertools.product(ri,ri,ri,ri):
            #D[i,j] += 0.5*V[i,j,k,l]*Ki[k,l]
            #D[i,j] += V[i,j,k,l]*Ki[k,l] #unsure on factor of 2

            #HFB version of: F[mu,nu] += (2*V[mu,si,nu,la] - V[mu,si,la,nu])*rho[la,si]
            D[i,j] += (V[i,k,j,l] - 0.5*V[i,k,l,j])*Ki[k,l] #check V indices for exchange
        return D

#  D_ij = <ij|V|kl>K_kl
#  h_ij = <ik|V|jl>P_lk

#   for mu,nu,la,si in itertools.product(ri,ri,ri,ri):
#       F[mu,nu] += (2*V[mu,si,nu,la] - V[mu,si,la,nu])*rho[la,si]
#       F[i,j] += V[i,k,j,l]*rho[l,k]

    @staticmethod
    def appH(H,pot):
        H_app = H.copy()
        n = H_app.shape[0]//2
        app = np.eye(n)*pot
        H_app[:n,:n] += app
        H_app[n:,n:] -= app
        return H_app

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
        Wocc = W[:,:n] #choose this over W[:,n:] because this has negative eigenvalues
        G = Wocc.dot(Wocc.T) 
        return G

        


if __name__ == "__main__":
    import sys
    n = 22
    U = -6.
    m = 11
    cl_list = []

#    h,V = c1c2.randHV(n,U=U,scale=0.1)
    h,V = c1c2.hubb(n,U=U,pbc=True)
    print "U: ", V[0,0,0,0]
    D_rnsym = c1c2.antisymMat(n)*0.2
    rho_guess = np.eye(n)

    bogil = HFB(h,V,m,Delta=D_rnsym,rho=rho_guess)
    bogil._do_hfb()
    cl_list.append(bogil)

#    bogil2 = HFB(h,V,m,Delta=D_rnsym)
#    bogil2._do_hfb()
#    cl_list.append(bogil2)

    for inst in cl_list:
        print "Trace HF rho: ", np.trace(inst.rho)
        print "Trace HFB G[:n,:n]: ",np.trace(inst.G[:inst.n,:inst.n])
        print "norm H(D):", np.linalg.norm(inst.H[:n,n:])
        print "norm G(K):", np.linalg.norm(inst.G[:n,n:])
        print "Idemp G:", np.linalg.norm(inst.G - np.dot(inst.G,inst.G))
        print "[H,G]:", np.linalg.norm(np.dot(inst.H,inst.G) - np.dot(inst.G,inst.H))
        print "E: ", inst.E
        print "HF hacktrace: ",np.trace(np.dot(inst.G[:inst.n,:inst.n],inst.F+inst.h))/inst.n #HF energy












