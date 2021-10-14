import numpy as np
from numpy import linalg as LA

#################################################################
##
## Calculate Pion Mass Squared values
##
#################################################################

def calcPionMassSq_1gen(CA, CG, CW, CZ, eQ, gs, sQsq, lamW, fpi, mD, kappa, DEBUG=True):
    """
    """    
    fsq  = fpi**2 
    pi3  = np.pi**3
    pi2  = np.pi**2
    eQsq = eQ**2
    gssq = gs**2     
    
    # NOTE: The \ continues the expression onto the next line for readability
    # https://stackoverflow.com/questions/53162/how-can-i-do-a-line-break-line-continuation-in-python  

    Msq0     = 64.*fpi*(mD*pi3 + 6.*fpi*kappa*pi2)
    Msq1to4  = (fsq*(-3.*CA*eQsq + 2.*CZ*eQsq - 9.*CG*gssq + 6.*CW*gssq))/6. + \
                    (CZ*eQsq*fsq)/(6.*sQsq) - (CZ*eQsq*fsq*sQsq)/2.
    Msq5and8 =  64.*fpi*mD*pi3
    Msq6and7 =  (-2.*fpi*(3.*CA*eQsq*fpi - CZ*eQsq*fpi - 96.*mD*pi3))/3. - 2.*CZ*eQsq*fsq*sQsq

    Msq9to12 =  -0.5*(fpi*(CA*eQsq*fpi + 3*CG*fpi*gssq - 128.*mD*pi3)) + \
                    (CZ*eQsq*fsq)/(18.*sQsq) - (CZ*eQsq*fsq*sQsq)/2.
    Msq14    =  64.*fpi*mD*pi3

    M2arr_mass = np.array([Msq0, Msq1to4, Msq1to4, Msq1to4, Msq1to4, Msq5and8, Msq6and7, \
                   Msq6and7, Msq5and8, Msq9to12, Msq9to12, Msq9to12, Msq9to12, 0., Msq14]) # Mass Squared Array in mass basis
    
    #-- Convert to mass array in DM charge basis --#
    #  8 ->  6,  6 ->  7,  7 ->  8
    # 12 -> 10, 10 -> 11, 11 -> 12
    indxArr = np.array([0, 1, 2, 3, 4, 5, 8, 6, 7, 9, 12, 10, 11, 13, 14])
    M2arr_DMcharge = M2arr_mass[indxArr]
 
    #-- Identify pions which contain constituent DM --#
    DMindexArr = np.arange(8)+5
    #SMindexArr = np.delete(np.arange(15), DMindexArr)
    
    #-- Find M2DMarr --#
    M2DMarr = M2arr_DMcharge[DMindexArr]
    
    return M2arr_DMcharge, M2arr_mass, M2DMarr#, DMindexArr, SMindexArr

def calcUniqueVals(CA, CG, CW, CZ, eQ, gs, sQsq, lamW, fpi, mD, kappa, DEBUG=True):    
    
    fpiSq = fpi*fpi
    fpiMD = fpi*mD 
    
    #-- Assuming BP1 --# #!
    uV1  = 896.*(np.pi**2)*(fpiSq) + (128./7.)*(np.pi**3)*fpiMD
    uV2  = -(128./7.)*(6**(1./2.))*(np.pi**3)*fpiMD
    uV3  = 1.2944*(fpiSq)
    uV4  = 0.706493*(fpiSq)
    uV5  = 0.654401*(fpiSq)
    uV6  = 0.64*(fpiSq)
    uV7  = -0.64*(fpiSq)
    uV8  = 64.*(np.pi**3)*fpiMD
    uV9  = 0.390937*(fpiSq) + 64.*(np.pi**3)*fpiMD
    uV10 = 2.56*(fpiSq)
    uV11 = 0.978845*(fpiSq) + 64.*(np.pi**3)*fpiMD
    uV12 = (768./7.)*(np.pi**3)*fpiMD
#     uV1  = 8843.163857479234*fpiSq + 283.485958219884*fpiMD
#     uV2  = 0.0016858968310591513*fpiSq
#     uV3  = -694.3959468826664*fpiMD
#     uV4  = -0.2821204399661114*fpiSq
#     uV5  = 0.7064928301181443*fpiSq
#     uV6  = -0.2915119880661114*fpiSq
#     uV7  = 0.0093915481*fpiSq
#     uV8  = -0.0093915481*fpiSq
#     uV9  = 1984.4017075391882*fpiMD
#     uV10 = 0.3909369757111462*fpiSq + 1984.4017075391882*fpiMD
#     uV11 = 0.0375661924*fpiSq
#     uV12 = 0.03293260247603701*fpiSq + 1984.4017075391882*fpiMD
#     uV13 = -0.0016858968310591443*fpiSq + 283.485958219884*fpiMD
#     uV14 = 3401.831498638608*fpiMD

    return np.array([uV1, uV2, uV3, uV4, uV5, uV6, uV7, uV8, uV9, uV10, uV11, uV12])

def calcPionMassSq_3gen(CA, CG, CW, CZ, eQ, gs, sQsq, lamW, fpi, mD, kappa, DEBUG=True):
    
    #-- Define empty non-diagonal M2 matrix --#
    M2_nondiag = np.zeros((91,91), dtype=complex)
    
    #-- Get array of unique values --#
    uVs = calcUniqueVals(CA, CG, CW, CZ, eQ, gs, sQsq, lamW, fpi, mD, kappa, DEBUG)
    
    #-- Define indices of non-zero values --#
    # There are 14  unique values
    # g[i] contains list of these index pairs in Mathematica notation (1 to 91)
    # Ex. g[0] -> [[1,1]] -> M2_nondiag[0,0] = uV1
    gIndex = [[[1, 1]], 
             [[1, 91], [91, 1]], 
             [[2, 2], [3, 3], [4, 4], [5, 5], [46, 46], [47, 47], [48, 48], [49, 49], [74, 74], [75, 75], [76, 76], [77, 77]], 
             [[7, 7], [8, 8], [15, 15], [16, 16], [51, 51], [52, 52]], 
             [[10, 10], [11, 11], [12, 12], [13, 13], [18, 18], [19, 19], [20, 20], [21, 21], [26, 26], [27, 27], [28, 28], 
              [29, 29],[34, 34], [35, 35], [36, 36], [37, 37], [54, 54], [55, 55], [56, 56], [57, 57], [62, 62], [63, 63], 
              [64, 64], [65, 65]], 
             [[10, 26], [18, 34], [26, 10], [34, 18], [54, 62], [62, 54]], 
             [[11, 27], [12, 28], [13, 29], [19, 35], [20, 36], [21, 37], [27, 11], [28, 12], [29, 13], [35, 19], [36, 20], 
              [37, 21], [55, 63], [56, 64], [57, 65], [63, 55], [64, 56], [65, 57]], 
             [[22, 22], [25, 25], [58, 58], [61, 61], [78, 78], [81, 81]], 
             [[23, 23], [24, 24], [59, 59], [60, 60], [79, 79], [80, 80]], 
             [[31, 31], [32, 32], [33, 33], [39, 39], [40, 40], [41, 41], [67, 67], [68, 68], [69, 69]], 
             [[42, 42], [43, 43], [44, 44], [45, 45], [70, 70], [71, 71], [72, 72], [73, 73], [82, 82], [83, 83], [84, 84], 
              [85, 85]], 
             [[91, 91]]]
#     gIndex = [[[1, 1]], 
#               [[1, 86], [86, 1]], 
#               [[1, 91], [86, 91], [91, 1], [91, 86]], 
#               [[2, 2], [3, 3], [4, 4], [5, 5], [46, 46], [47, 47], [48, 48], [49, 49], [74, 74], [75, 75], [76, 76], [77, 77]],
#               [[7, 7], [8, 8], [15, 15], [16, 16], [51, 51], [52, 52]], 
#               [[10, 10], [11, 11], [12, 12], [13, 13], [18, 18], [19, 19], [20, 20], [21, 21], [26, 26], [27, 27], [28, 28], 
#                [29, 29], [34, 34], [35, 35], [36, 36], [37, 37], [54, 54], [55, 55], [56, 56], [57, 57], [62, 62], [63, 63], 
#                [64, 64], [65, 65]], 
#               [[10, 26], [18, 34], [26, 10], [34, 18], [54, 62], [62, 54]], 
#               [[11, 27], [12, 28], [13, 29], [19, 35], [20, 36], [21, 37], [27, 11], [28, 12], [29, 13], [35, 19], [36, 20], 
#                [37, 21], [55, 63], [56, 64], [57, 65], [63, 55], [64, 56], [65, 57]], 
#               [[22, 22], [25, 25], [58, 58], [61, 61], [78, 78], [81, 81]], 
#               [[23, 23], [24, 24], [59, 59], [60, 60], [79, 79], [80, 80]], 
#               [[31, 31], [32, 32], [33, 33], [39, 39], [40, 40], [41, 41], [67, 67], [68, 68], [69, 69]], 
#               [[42, 42], [43, 43], [44, 44], [45, 45], [70, 70], [71, 71], [72, 72], [73, 73], [82, 82], [83, 83], [84, 84], 
#                [85, 85]], 
#               [[86, 86]], 
#               [[91, 91]]]
    
    #-- Loop over unique values and assign them to proper locations --#
    for i in range(uVs.shape[0]):
        arr = np.array(gIndex[i])
        M2_nondiag[arr[:,0]-1, arr[:,1]-1] = uVs[i]
    
    #-- Numerically diagonalize M2_nondiag --#
    # M2arr_mass are eigenvalues in the mass basis order
    # Wmatrix are normalized eigenvectors 
    M2arr_mass, Wmatrix_mass = LA.eig(M2_nondiag)
    
    #-- Convert M2arr_mass to M2arr_DMcharge --#
    # New index: 0,  1, ..., 89, 90 <- np.arange(91)
    # Old index: 0, 38, ..., 90,  1 <- indxArr
    # Note: This assumes that we know how eigenvectors/eigenvalues are sorted when returned. 
    #       This should be consistent for the same structured matrix
    index_0 = np.array([0,  38,41,39,40,  50,53,51,52,  62,65,63,64,  70,73,71,72,  78,81,79,80,  82,85,83,84])
    index_1 = np.arange(35+1)+2
    index_2 = np.arange(7+1) +42
    index_3 = np.arange(7+1) +54
    index_4 = np.arange(3+1) +66
    index_5 = np.arange(3+1) +74
    index_6 = np.arange(4+1) +86
    index_7 = np.array([1])
    
    indxArr = np.concatenate((index_0, index_1, index_2, index_3, index_4, index_5, index_6, index_7))

    M2arr_DMcharge = M2arr_mass[indxArr]
 
    #-- Identify pions which contain constituent DM --#
    DMindexArr = np.arange(24)+1
    
    #-- Find M2DMarr --#
    M2DMarr = M2arr_DMcharge[DMindexArr]
    
    return M2arr_DMcharge, M2arr_mass, M2DMarr, Wmatrix_mass#, DMindexArr, SMindexArr

# def calcPionMassSq_3gen_old(CA, CG, CW, CZ, eQ, gs, sQsq, lamW, fpi, mD, kappa, DEBUG=True):
#     #()
#     """
#     """
#     #-- Import necessary "OFF" (off-diagonal) and "ON" (on-diagonal) functions --#
#     from ON_OFF_diagonalFunctions import on_1, on_2, on_4, on_5, on_6, on_7, on_8, on_9, on_10, off_1, off_2, off_3

#     #-- Create array of mass squared values --#
#     M2arr = np.zeros(91)
    
#     on2Index      = np.arange(12)
#     on3Index      = np.arange(14) + 12
#     on4Index      = np.arange(6)  + 26
#     mOFF2ON5Index = np.arange(3)  + 32
#     pOFF2ON5Index = np.arange(3)  + 35
#     mOFF3ON5Index = np.arange(9)  + 38
#     pOFF3ON5Index = np.arange(9)  + 47
#     on6Index      = np.arange(6)  + 56
#     on7Index      = np.arange(6)  + 62
#     on8Index      = np.arange(9)  + 68
#     on9Index      = np.arange(12) + 77
    
#     M2arr[on2Index]      = on_2(CA, CG, CW, CZ, eQ, gs, sQsq, lamW)
#     M2arr[on3Index]      = 0. #on_3()
#     M2arr[on4Index]      = on_4(CA, CZ, eQ, sQsq, lamW)
    
#     on5  = on_5(CA, CG, CZ, eQ, gs, sQsq, lamW)
#     off2 = off_2(CW, gs, lamW)
#     off3 = off_3(CW, gs, lamW)
#     M2arr[mOFF2ON5Index] = -off2 + on5
#     M2arr[pOFF2ON5Index] = off2  + on5
#     M2arr[mOFF3ON5Index] = -off3 + on5
#     M2arr[pOFF3ON5Index] = off3  + on5
    
#     M2arr[on6Index]      = on_6(fpi, lamW, mD)
#     M2arr[on7Index]      = on_7(CA, CZ, eQ, sQsq, lamW, fpi, mD)
#     M2arr[on8Index]      = on_8(CG, gs, lamW)
#     M2arr[on9Index]      = on_9(CA, CG, CZ, eQ, gs, sQsq, lamW, fpi, mD)
    
    
#     """
#     $\frac{1}{2} \left( -ON_{1} - ON_{10} + \left[ 4OFF_{1}^2 + ON_{1}^2 -2ON_{1}ON_{10} + ON_{10}^2 \right]^{1/2} \right)$
#     $\frac{1}{2} \left( ON_{1} + ON_{10} + \left[ 4OFF_{1}^2 + ON_{1}^2 -2ON_{1}ON_{10} + ON_{10}^2 \right]^{1/2} \right)$
#     """
#     on1    = on_1(fpi, lamW, mD, kappa)
#     on10   = on_10(fpi, lamW, mD)
#     off1Sq = (off_1(fpi, lamW, mD))**2
#     term1  = on1 + on10 
#     term2  = np.sqrt( 4*off1Sq + (on1**2) - 2*on1*on10 + (on10**2) )
#     M2arr[89] = 0.5*(term1 - term2)
#     M2arr[90] = 0.5*(term1 + term2)
    
#     #-- Identify pions which contain constituent DM --#
#     DMindexArr = np.concatenate((on6Index, on7Index, on9Index))
#     #SMindexArr = np.delete(np.arange(91), DMindexArr)
    
#     #-- Find M2DMarr --#
#     M2DMarr = M2arr[DMindexArr]
    
#     if (DEBUG):
#         #! Other checks?
#         print("")
        
#     return M2arr, M2DMarr#, DMindexArr, SMindexArr

def calcPionMassSq(Ngen, CA, CG, CW, CZ, eQ, gs, sQsq, lamW, fpi, mD, kappa, DEBUG=True):
    if(Ngen==1):
        return calcPionMassSq_1gen(CA, CG, CW, CZ, eQ, gs, sQsq, lamW, fpi, mD, kappa, DEBUG)
    elif(Ngen==3):
        return calcPionMassSq_3gen(CA, CG, CW, CZ, eQ, gs, sQsq, lamW, fpi, mD, kappa, DEBUG)
    else:
        print("Error: Invalid Ngen. Please use either Ngen=1 or Ngen=3.")
        return 0
    
    