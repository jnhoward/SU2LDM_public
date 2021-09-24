import numpy as np

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

    M2arr = np.array([Msq0, Msq1to4, Msq1to4, Msq1to4, Msq1to4, Msq5and8, Msq6and7, \
                   Msq6and7, Msq5and8, Msq9to12, Msq9to12, Msq9to12, Msq9to12, 0., Msq14]) # Mass Squared Array
 
    #-- Identify pions which contain constituent DM --#
    DMindexArr = np.arange(8)+5
    #SMindexArr = np.delete(np.arange(15), DMindexArr)
    
    #-- Find M2DMarr --#
    M2DMarr = M2arr[DMindexArr]
    
    return M2arr, M2DMarr#, DMindexArr, SMindexArr
    
def calcPionMassSq_3gen(CA, CG, CW, CZ, eQ, gs, sQsq, lamW, fpi, mD, kappa, DEBUG=True):
    #()
    """
    """
    #-- Import necessary "OFF" (off-diagonal) and "ON" (on-diagonal) functions --#
    from ON_OFF_diagonalFunctions import on_1, on_2, on_4, on_5, on_6, on_7, on_8, on_9, on_10, off_1, off_2, off_3

    #-- Create array of mass squared values --#
    M2arr = np.zeros(91)
    
    on2Index      = np.arange(12)
    on3Index      = np.arange(14) + 12
    on4Index      = np.arange(6)  + 26
    mOFF2ON5Index = np.arange(3)  + 32
    pOFF2ON5Index = np.arange(3)  + 35
    mOFF3ON5Index = np.arange(9)  + 38
    pOFF3ON5Index = np.arange(9)  + 47
    on6Index      = np.arange(6)  + 56
    on7Index      = np.arange(6)  + 62
    on8Index      = np.arange(9)  + 68
    on9Index      = np.arange(12) + 77
    
    M2arr[on2Index]      = on_2(CA, CG, CW, CZ, eQ, gs, sQsq, lamW)
    M2arr[on3Index]      = 0. #on_3()
    M2arr[on4Index]      = on_4(CA, CZ, eQ, sQsq, lamW)
    
    on5  = on_5(CA, CG, CZ, eQ, gs, sQsq, lamW)
    off2 = off_2(CW, gs, lamW)
    off3 = off_3(CW, gs, lamW)
    M2arr[mOFF2ON5Index] = -off2 + on5
    M2arr[pOFF2ON5Index] = off2  + on5
    M2arr[mOFF3ON5Index] = -off3 + on5
    M2arr[pOFF3ON5Index] = off3  + on5
    
    M2arr[on6Index]      = on_6(fpi, lamW, mD)
    M2arr[on7Index]      = on_7(CA, CZ, eQ, sQsq, lamW, fpi, mD)
    M2arr[on8Index]      = on_8(CG, gs, lamW)
    M2arr[on9Index]      = on_9(CA, CG, CZ, eQ, gs, sQsq, lamW, fpi, mD)
    
    
    """
    $\frac{1}{2} \left( -ON_{1} - ON_{10} + \left[ 4OFF_{1}^2 + ON_{1}^2 -2ON_{1}ON_{10} + ON_{10}^2 \right]^{1/2} \right)$
    $\frac{1}{2} \left( ON_{1} + ON_{10} + \left[ 4OFF_{1}^2 + ON_{1}^2 -2ON_{1}ON_{10} + ON_{10}^2 \right]^{1/2} \right)$
    """
    on1    = on_1(fpi, lamW, mD, kappa)
    on10   = on_10(fpi, lamW, mD)
    off1Sq = (off_1(fpi, lamW, mD))**2
    term1  = on1 + on10 
    term2  = np.sqrt( 4*off1Sq + (on1**2) - 2*on1*on10 + (on10**2) )
    M2arr[89] = 0.5*(-term1 + term2)
    M2arr[90] = 0.5*( term1 + term2)
    
    #-- Identify pions which contain constituent DM --#
    DMindexArr = np.concatenate((on6Index, on7Index, on9Index))
    #SMindexArr = np.delete(np.arange(91), DMindexArr)
    
    #-- Find M2DMarr --#
    M2DMarr = M2arr[DMindexArr]
    
    if (DEBUG):
        #! Other checks?
        print("")
        
    return M2arr, M2DMarr#, DMindexArr, SMindexArr

def calcPionMassSq(Ngen, CA, CG, CW, CZ, eQ, gs, sQsq, lamW, fpi, mD, kappa, DEBUG=True):
    if(Ngen==1):
        return calcPionMassSq_1gen(CA, CG, CW, CZ, eQ, gs, sQsq, lamW, fpi, mD, kappa, DEBUG)
    elif(Ngen==3):
        return calcPionMassSq_3gen(CA, CG, CW, CZ, eQ, gs, sQsq, lamW, fpi, mD, kappa, DEBUG)
    else:
        print("Error: Invalid Ngen. Please use either Ngen=1 or Ngen=3.")
        return 0
    
    