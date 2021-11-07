import numpy as np
from scipy.linalg import block_diag

#################################################################
##
## Transform F (or Fhat) Matrices into definite DM charge basis
##
#################################################################

#-- Calculate Transformation Matrix --#
def calcDMTransformMatrix(Ngen, DEBUG=True):    
    # \Pi^{mass}_i = V_ij \Pi^{DM charge basis}_j
    
    if(Ngen==1):
        Npions = 15 # Total number of pions
        
        #-- Set arrays that show how mass and DM charged states relate --#
        """
        DM array: 
        $[\Pi_0,   ...,   \Pi_4,   \Pi_5,   \Pi_6, ..., \Pi_{11}, \Pi_{12},   \Pi_{13},   \Pi_{14}]$
        $[\Pi_0^0, ..., \Pi_4^0, \Pi_1^+, \Pi_1^-, ...,  \Pi_4^+,  \Pi_4^-, \Pi_{13}^0, \Pi_{14}^0]$
                    
        Define vectors A, B, D as follows:

        D | D+1:  5 |  6 |  7 |  8 |  9 | 10 | 11 | 12 | <- DM charged pions in DM charge Basis
        A      :    5    |    6    |    9    |    10   | <- Pions in mass basis with V = 1/sqrt(2)
        B      :    8    |    7    |    12   |    11   | <- Pions in mass basis with V = +-i/sqrt(2)
        """
        D = np.array([5, 7,  9, 11])
        A = np.array([5, 6,  9, 10])
        B = np.array([8, 7, 12, 11])
    elif(Ngen==3):        
        Npions = 91 # Total number of pions
        
        #-- Set arrays that show how mass and DM charged states relate --#
        """
        DM array: 
        $[\Pi_0,     \Pi_1,   \Pi_2, ...,   \Pi_{23},   \Pi_{24},  \Pi_{25}, ...,   \Pi_{90}]$
        $[\Pi_0^0, \Pi_1^+, \Pi_1^-, ..., \Pi_{12}^+, \Pi_{12}^-, \Pi_{1}^0, ..., \Pi_{66}^0]$
        
        Note: 
         - As in Ngen=1 case, \Pi_0^0 is the eta' particle
         - \Pi_{66}^0 analogous to \Pi_{14}^0 in the Ngen=1 case
         - We obtain the mass array basis through numerical diagonalization, but the order is consistent and 
           only depends on the structure of the non-diagonal matrix which does not change when changing the 
           scan parameters
        
        Define vectors A, B, D as follows:
        
        D <- DM charged pions in DM charge Basis
        A <- Pions in mass basis with V = 1/sqrt(2)
        B <- Pions in mass basis with V = +-i/sqrt(2)
        
        D | D+1:  1 |  2 |  3 |  4 |  5 |  6 |  7 |  8 |  9 | 10 | 11 | 12 |
        A      :    38   |    39   |    50   |    51   |    62   |    63   |
        B      :    41   |    40   |    53   |    52   |    65   |    64   |

        (cont.)
        D | D+1: 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 | 21 | 22 | 23 | 24 |
        A      :    70   |    71   |    78   |    79   |    82   |    83   |
        B      :    73   |    72   |    81   |    80   |    85   |    84   |
        
        """
        D = np.array([ 1,  3,  5,  7,  9, 11, 13, 15, 17, 19, 21, 23])
        A = np.array([38, 39, 50, 51, 62, 63, 70, 71, 78, 79, 82, 83]) 
        B = np.array([41, 40, 53, 52, 65, 64, 73, 72, 81, 80, 85, 84])
    else:
        print("Error: Invalid Ngen. Please use either Ngen=1 or Ngen=3.")
        return 
    
    #-- Create V matrix --#
    Vmatrix = np.zeros((Npions,Npions), dtype=complex)
    normFactor = 1./(np.sqrt(2))
    for i in range(len(A)):
        a = A[i]
        d = D[i]
        Vmatrix[a,d]   = normFactor
        Vmatrix[a,d+1] = normFactor

    for i in range(len(B)):
        b = B[i]
        d = D[i]
        Vmatrix[b,d]   = (0+1j)*normFactor
        Vmatrix[b,d+1] = (0-1j)*normFactor

    Iarr = np.arange(Npions)
    sans = np.concatenate((A, B, D, D+1))
    Iarr = np.delete(Iarr, sans)

    for i in Iarr:
        Vmatrix[i,i] = 1.

    return Vmatrix

#-- Convert F1Matrix, F2Matrix in definite DM basis --#
def convertToDMBasis(F1Matrix, F2Matrix, Vmatrix, DEBUG=True):
    
    #-- Transform Fs from interaction to DM charge basis --#
    from transformFs import transformF
    F1DMchargeBasisMatrix = transformF(Vmatrix, F1Matrix, DEBUG)
    F2DMchargeBasisMatrix = transformF(Vmatrix, F2Matrix, DEBUG)

    return F1DMchargeBasisMatrix, F2DMchargeBasisMatrix