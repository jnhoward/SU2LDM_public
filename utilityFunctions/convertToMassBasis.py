import numpy as np
from scipy.linalg import block_diag
from os import path

#################################################################
##
## All functions necessary to transform F (or Fhat) Matrices into 
## definite mass basis. Valid for Ngen=3 only since for Ngen=1 the 
## mass and interaction bases ~coincide.
##
#################################################################

#-- Calculate Transformation Matrix --#
def calcMassTransformMatrix(lamW, fpi, kappa, mDM, Ngen=3, DEBUG=True):    

    #-- Load core matrix as base for complete matrix --#
    filename = "../npyFiles/WcoreMatrix_intToMass_Ngen3.npy" # If function is called from this directory
    if (path.exists(filename) == False): 
        filename = "npyFiles/WcoreMatrix_intToMass_Ngen3.npy" # If function is called from main directory
    
    if (path.exists(filename) == False): 
        print("Error: %s does not exist. Please run preScan.py before proceeding."%filename)
        return
    else: 
        Wmatrix = np.load(filename)[0]
    
    #-- Calculate elements which depend on parameters --#
    # Import necessary "OFF" (off-diagonal) and "ON" (on-diagonal) functions
    from ON_OFF_diagonalFunctions import off_1, on_1, on_10 
    off1    = off_1(fpi, lamW, mDM)
    off1inv = 1./off1
    off1sq  = off1**2
    on1     = on_1(fpi, lamW, mDM, kappa)
    on1sq   = on1**2
    on10    = on_10(fpi, lamW, mDM)
    on10sq  = on10**2
    
    #-- Assign these elements to their proper place --#
    Wmatrix[89,0] = -0.5 * off1inv * (-on1 + on10 + ( 4.*off1sq + on1sq - 2.*on1*on10 + on10sq )**(1/2))
    Wmatrix[90,0] = -0.5 * off1inv * (-on1 + on10 - ( 4.*off1sq + on1sq - 2.*on1*on10 + on10sq )**(1/2))
    
    return Wmatrix

#-- Calculate Core Transformation Matrix --#
def calcCoreWmatrix(Ngen=3, DEBUG=True):
    
    #-- Load sparse matrix from file and convert to python notation --#
    txtFilename = "WcoreMatrix_intToMass_sparse.txt" # If function is called from this directory
    if (path.exists(txtFilename) == False): 
        txtFilename = "utilityFunctions/WcoreMatrix_intToMass_sparse.txt" # If function is called from main directory
    
    Wsparse = np.loadtxt(txtFilename, dtype=np.int, delimiter=', ')
    
    convArr = np.ones(Wsparse.shape, dtype=np.int)
    convArr[:,0] = np.zeros(Wsparse.shape[0], dtype=np.int)
    Wsparse = Wsparse - convArr
    
    nPions = 91 # 91 = eta' + 90 pions, 90 = 2Nf^2 − Nf − 1 = 2 (7)^2 - 7 - 1 
    Wcore  = np.zeros((nPions, nPions), dtype=complex)
    
    value = Wsparse[:,0]
    row   = Wsparse[:,1]
    col   = Wsparse[:,2]
    
    Wcore[row, col] = value
    
    return Wcore

#-- Convert F1Matrix, F2Matrix in definite DM basis --#
def convertToMassBasis(F1Matrix, F2Matrix, Wmatrix, Ngen=3, DEBUG=True):

    #-- Transform Fs from interaction to mass basis --#
    from transformFs import transformF
    F1MassBasisMatrix = transformF(Wmatrix, F1Matrix, DEBUG)
    F2MassBasisMatrix = transformF(Wmatrix, F2Matrix, DEBUG)

    return F1MassBasisMatrix, F2MassBasisMatrix