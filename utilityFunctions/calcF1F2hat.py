import numpy as np
from scipy.linalg import block_diag
import itertools

#####################################################
##
## Calculate F1 and F2 hat matrices 
##
#####################################################

#-- Define F1 function --#
def F1Hat(a, b, c, d, X):

    Xtot0 = X[c] @ X[a] @ X[d] @ X[b] 
    Xtot1 = X[a] @ X[c] @ X[d] @ X[b] 
    Xtot2 = X[c] @ X[a] @ X[b] @ X[d] 
    Xtot3 = X[a] @ X[c] @ X[b] @ X[d] 
    Xtot4 = X[a] @ X[b] @ X[c] @ X[d]

    term1 = (1./4.)*(np.trace(Xtot0) + np.trace(Xtot1))
    term2 = (-1./12.)*(np.trace(Xtot2) + np.trace(Xtot3))
    term3 = (-1./3.)*(np.trace(Xtot4))
    
    return term1 + term2 + term3

#-- Define F2 function --#
def F2Hat(a, b, c, d, A, X):

    Xtot = A @ X[a] @ X[b] @ X[c] @ X[d]

    return np.trace(Xtot)

#-- Calculate F1 and F2 Matrices --#
def calcF1F2HatMatrices(X, A, Ngen=1, DEBUG=True):
            
    # Create F1 and F2 Matrices of all possible combinations (tensor)
    if(Ngen==1):
        n = 15 # 15 = eta' + 14 pions, 14 = 2Nf^2 − Nf − 1 = 2 (3)^2 - 3 - 1
    elif(Ngen==3):
        n = 91 # 91 = eta' + 90 pions, 90 = 2Nf^2 − Nf − 1 = 2 (7)^2 - 7 - 1
    else:
        print("Error: Invalid Ngen. Please use either Ngen=1 or Ngen=3.")
        return 
        
    F1HatMatrix = np.zeros((n,n,n,n), dtype=complex)
    F2HatMatrix = np.zeros((n,n,n,n), dtype=complex)
  
    dummyarr = np.arange(n) 
    
    it = 0
    for (a,b,c,d) in itertools.product(dummyarr, dummyarr, dummyarr, dummyarr):
        
        if(a == it):
            it+=1
            print("Now processing: ",a,b,c,d)
        
        F1HatMatrix[a,b,c,d] = F1Hat(a, b, c, d, X)
        F2HatMatrix[a,b,c,d] = F2Hat(a, b, c, d, A, X)     
        
    
    if (DEBUG):#! Make these more meaningful later
        print("F1Matrix[7, 7, 2, 2] = ", F1HatMatrix[7, 7, 2, 2]) 
        print("")
    
    if (DEBUG):
        print("F2Matrix[5, 1, 1, 5] = ", F2HatMatrix[5, 1, 1, 5])      
        print("") 

    return F1HatMatrix, F2HatMatrix