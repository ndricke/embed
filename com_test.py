import numpy as np

import c1c2


A = c1c2.antisymMat(10)
B = c1c2.antisymMat(10)

print A
print B

difAB = A.dot(B)-B.dot(A)
print difAB
print np.linalg.norm(difAB)
