import numpy as np
from scipy.linalg import block_diag
import itertools

#####################################################
##
## Calculate F1 and F2 hat matrices assuming Ngen=1
##
#####################################################

#-- Define F1 function --#
def F1Hat(a, b, c, d, X):

    Xtot0 = np.linalg.multi_dot([X[c], X[a], X[d], X[b]])
    Xtot1 = np.linalg.multi_dot([X[a], X[c], X[d], X[b]])
    Xtot2 = np.linalg.multi_dot([X[c], X[a], X[b], X[d]])
    Xtot3 = np.linalg.multi_dot([X[a], X[c], X[b], X[d]])
    Xtot4 = np.linalg.multi_dot([X[a], X[b], X[c], X[d]])

    term1 = (1./4.)*(np.trace(Xtot0) + np.trace(Xtot1))
    term2 = (-1./12.)*(np.trace(Xtot2) + np.trace(Xtot3))
    term3 = (-1./3.)*(np.trace(Xtot4))

    return term1 + term2 + term3

#-- Define F2 function --#
def F2Hat(a, b, c, d, A, X):

    Xtot = np.linalg.multi_dot([A, X[a], X[b], X[c], X[d]])

    return np.trace(Xtot)

#-- Calculate F1 and F2 Matrices --#
def calcF1F2HatMatrices(X, A, DEBUG=True):
            
    # Create F1 and F2 Matrices of all possible combinations (tensor)
    F1HatMatrix = np.zeros((15,15,15,15), dtype=complex) #! Will need to change this for Ngen != 1
    F2HatMatrix = np.zeros((15,15,15,15), dtype=complex)
    
    dummyarr = np.arange(14) + 1
    
    for (a,b,c,d) in itertools.product(dummyarr, dummyarr, dummyarr, dummyarr):
        F1HatMatrix[a,b,c,d] = F1Hat(a, b, c, d, X)
        F2HatMatrix[a,b,c,d] = F2Hat(a, b, c, d, A, X)
    
    if (DEBUG):#! Make these more meaningful later
        print("F1Matrix[7, 7, 2, 2] = ", F1HatMatrix[7, 7, 2, 2]) 
        print("")
    
    if (DEBUG):
        print("F2Matrix[5, 1, 1, 5] = ", F2HatMatrix[5, 1, 1, 5])      
        print("") 

    return F1HatMatrix, F2HatMatrix