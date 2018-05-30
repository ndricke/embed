import numpy as np
import copy
import scipy.linalg as slg

def eigSort (e, u):
    '''
    descending order by eigenvalues
    '''
    ind = e.argsort ()[::-1]
    return e[ind], u[:, ind]

def getRandomSymmMat (K):
    h = np.random.rand (K, K)
    h = h + h.T
    return h

def getHubbardh (K, N):
    h = np.zeros ([K, K])
    for i in range (K):
        if i != K - 1:
            h[i, i + 1] = h[i + 1, i] = -1.
        else:
            if N % 2 == 0:
                h[K - 1, 0] = h[0, K - 1] = 1.
            else:
                h[K - 1, 0] = h[0, K - 1] = -1.

    return h

def fragEnergy (h, P, impList):
    '''
    given h and P, evaluate fragment energy
    '''
    K = h.shape[0]
    e = 0
    for f in impList:
        for i in range (K):
            e += P[i, f] * h[i, f]

    return e

if __name__ == "__main__":

    K = 8       # num of sites
    N = 3       # num of elec pairs
    Nimp = 2    # num of frag sites
                # implicitly assuming the first Nimp sites are frag sites

    #h = getRandomSymmMat (K)
    h = getHubbardh (K, N)

# original HF problem
    EL0, C0 = slg.eigh (h)
    C0occ = C0[:, :N]
    P0 = C0occ.dot (C0occ.T)
    E0 = np.trace (h.dot (P0))
    print "E0 = %.10f" % E0

# C1 (I stick to our terminology, denoting it T1)
    Cf = C0occ[:Nimp, :]
    e, u = slg.eigh (Cf.T.dot (Cf))
    # descending sort to make sure frag orbitals come first
    e, u = eigSort (e, u)
    # xform C0occ using u, now C1imp has form
    #   | f 0 |
    #   | b e |
    C1imp = C0occ.dot (u)
    # get T1
    T1 = np.zeros ([K, 2 * Nimp])
    T1[:Nimp, :Nimp] = C1imp[:Nimp, :Nimp]
    T1[Nimp : K, Nimp : Nimp * 2] = C1imp[Nimp : K, :Nimp]
    # normalize (sanity check: T1.T.dot (T1) == I. checked!)
    T1[:, 0] /= np.sqrt (e[0])
    T1[:, 1] /= np.sqrt (e[1])
    T1[:, 2] /= np.sqrt (1. - e[0])
    T1[:, 3] /= np.sqrt (1. - e[1])
    # xform h using T1
    h1 = T1.T.dot (h).dot (T1)
    EL1, C1 = slg.eigh (h1)
    C1occ = C1[:, :Nimp]
    P1 = C1occ.dot (C1occ.T)
    E1 = fragEnergy (h1, P1, range (Nimp))
    print "E1 = %.10f" % E1

# C2 (I stick to our terminology, denoting it T2)
    # get A, B using P0
    P0imp = P0[:Nimp, :Nimp]
    P0c = P0[Nimp : K, :Nimp]
    P0env = P0[Nimp : K, Nimp : K]
    e, u = slg.eigh (P0imp)
    A = u.dot (np.diag (np.sqrt (e)))
    B = P0c.dot (slg.inv (A.T))
    # get Ap, Bp using Ph = I - P0
    P0h = np.eye (K) - P0
    P0himp = P0h[:Nimp, :Nimp]
    P0hc = P0h[Nimp : K, :Nimp]
    P0henv = P0h[Nimp : K, Nimp : K]
    e, u = slg.eigh (P0himp)
    Ap = u.dot (np.diag (np.sqrt (e)))
    Bp = P0hc.dot (slg.inv (Ap.T))
    # get T2 (sanity check: T2.T.dot (T2) == I. checked!)
    T2 = np.zeros ([K, 2 * Nimp])
    T2[:Nimp, :Nimp] = A
    T2[Nimp : K, :Nimp] = B
    T2[:Nimp, Nimp : 2 * Nimp] = Ap
    T2[Nimp : K, Nimp : 2 * Nimp] = Bp
    # get hphys by setting h_{bb} to zero
    hphys = h
    hphys[Nimp : K, Nimp : K] = 0
    # get h2 by xforming hphys w/ T2
    h2 = T2.T.dot (hphys).dot (T2)
    EL2, C2 = slg.eigh (h2)
    C2occ = C2[:, :Nimp]
    P2 = C2occ.dot (C2occ.T)

    h2_partial = copy.copy(h2)
    h2_partial[Nimp:,:Nimp] *= 0.5
    h2_partial[:Nimp,Nimp:] *= 0.5
    h2_partial[Nimp:,Nimp:] *= 0.
    print h2
    print h2_partial


    h2_p = copy.copy(h2)
    h2_p[Nimp:,:] *= 0.
    print h2_p

    E2 = np.trace(P2.dot(h2))
    print "E2 = %.10f" % E2
    print "E2partial = ",np.trace(P2.dot(h2_partial))
    print "E2p = ",np.trace(P2.dot(h2_p))




















