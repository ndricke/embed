import numpy as np

#apply a global chemical potential to obtain correct filling of HFB state
#Do we apply a chemical potential to both rho and 1-rho?

def newton_iter(x,f,ep=10.**-8):
    fx = f(x)
    grad_f = (f(x+ep)-f(x-ep))/(2*ep)
#    print "f(x): ",fx
#    print "df/dx: ", grad_f
    return x - fx/grad_f

def _calc_obj(rho,m,ep=10.**-6):
    
    m_cur = np.trace(rho)
    dif = m_cur-m #if we are too high, dif will be positive
    m_grad = (np.trace(rho+eye(n)*ep) - m_cur)/ep # df/dx = (f(x+ep)-f(x))/(dx)

    return mu - m_cur/m_grad
#the independent variable is the chemical potential; the dependent variable is the occupation number


#takes potential, turns it into 1 and 2 e- H terms, solves FCI for frags with potentials, then calc constraints
def _dmet_iter(s, pot):
	s._make_pots(pot)

	for frag in s.part.fragments:
		frag.solver.solve(frag.h+frag.hpot, frag.V+frag.Vpot, nelec=frag.ni)

	obj = s._calc_obj()
	return obj

def optimize(s, guess=None):
	guess = numpy.zeros(s.part.optlen()) 

	optpot = nr.nr(s._dmet_iter, guess) # nr(f,x0) 
	s._dmet_iter(optpot) # Just to make sure 
	s._check() 








