import numpy as np
from scipy.linalg import block_diag

#-- Transform FMatrix into DM basis --#
def transformF(vMatrix, FMatrix, DEBUG=True):
    
    FDMchargeBasisMatrix = np.einsum('ae,bf,cg,dh,efgh -> abcd',vMatrix,vMatrix,vMatrix,vMatrix,FMatrix, optimize='greedy')

    if (DEBUG):
        #print("Transformed F matrix shape:")
        #print(FDMchargeBasisMatrix.shape)
        #! Other checks?
        print("")
        
    return FDMchargeBasisMatrix

#-- Calculate Transformation Matrix --#
def calcDMTransformMatrix(DEBUG=True):    

    vSubMatrix = np.zeros((8, 8), dtype=complex)
    vSubMatrix[0,0] = 1
    vSubMatrix[0,1] = 1
    vSubMatrix[1,2] = 1
    vSubMatrix[1,3] = 1
    vSubMatrix[2,2] = 0+1j
    vSubMatrix[2,3] = 0-1j
    vSubMatrix[3,0] = 0+1j
    vSubMatrix[3,1] = 0-1j
    vSubMatrix[4,4] = 1
    vSubMatrix[4,5] = 1
    vSubMatrix[5,6] = 1
    vSubMatrix[5,7] = 1
    vSubMatrix[6,6] = 0+1j
    vSubMatrix[6,7] = 0-1j
    vSubMatrix[7,4] = 0+1j
    vSubMatrix[7,5] = 0-1j

    vSubMatrix = (1./np.sqrt(2))*vSubMatrix

    if (DEBUG):
        print("Transformation sub matrix:")
        print(vSubMatrix)
        print("")

    id5 = np.diag([1., 1., 1., 1., 1.]) 
    id2 = np.diag([1., 1.])

    vMatrix = block_diag(id5, vSubMatrix, id2)
    
    return vMatrix

#-- Convert F1Matrix, F2Matrix in definite DM basis --#
def convertToDMBasis(F1Matrix, F2Matrix, DEBUG=True):
    
    #-- Create transformation matrix --#
    vMatrix = calcDMTransformMatrix(DEBUG)
    
    #-- Transform F1 --#
    F1DMchargeBasisMatrix = transformF(vMatrix, F1Matrix, DEBUG)

    #-- Transform F2 --#
    F2DMchargeBasisMatrix = transformF(vMatrix, F2Matrix, DEBUG)

    return F1DMchargeBasisMatrix, F2DMchargeBasisMatrix