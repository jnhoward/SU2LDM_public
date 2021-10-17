import numpy as np
from scipy.linalg import block_diag
from os import path

#################################################################
##
## Function necessary to transform F (or Fhat) Matrices into 
## definite mass basis. Valid for Ngen=3 only since for Ngen=1 the 
## mass and interaction bases ~coincide.
##
#################################################################

#-- Convert F1Matrix, F2Matrix in definite DM basis --#
def convertToMassBasis(F1Matrix, F2Matrix, Wmatrix, Ngen=3, DEBUG=True):

    #-- Transform Fs from interaction to mass basis --#
    from transformFs import transformF
    F1MassBasisMatrix = transformF(Wmatrix, F1Matrix, DEBUG)
    F2MassBasisMatrix = transformF(Wmatrix, F2Matrix, DEBUG)

    return F1MassBasisMatrix, F2MassBasisMatrix