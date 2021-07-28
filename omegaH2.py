import numpy as np
import time
import itertools
import os.path
from os import path

#-- Add utilityFunctions/ to easily use utility .py files --#
import sys
sys.path.append("utilityFunctions/")

DEBUG = False  # Turn off DEBUG statements by default
TIME  = False  # Turn off printing time statements
PLOT  = False  # Turn off plotting

Ngen = 1 # Hyper parameter

def omegaH2(gs, fpi, kappa, asmall, bsmall, sQsq, DEBUG=False):
    
    #---------------------------------#
    #-- Load precalculated matrices --#
    #---------------------------------#
    if (path.exists("npyFiles/Fmatrices_Ngen1.npy") == False): 
        print("Error: npyFiles/Fmatrices_Ngen1.npy does not exists. Please run preScan.py before proceeding.")
        return
    else: 
        F1HatDMchargeBasisMatrix, F2HatDMchargeBasisMatrix = np.load("npyFiles/Fmatrices_Ngen1.npy")
            
    #-----------------------#
    #-- Define parameters --#
    #-----------------------#
    
    #-- Define fixed parameters --#
    garr = np.array([1.,1.,1.,1.,1.,1.,1.,1.]) 
    x    = 20.
    CA   = -1.
    CZ   = -1.
    CG   = -1.
    CW   = 1.
    
    #-- Parameters from arguments --#
    # gs:     
    # fpi:     lamW = 4*pi*fpi
    # kappa:   
    # asmall:  eQ = asmall*gs, 1e-3
    # bsmall:  mD = bsmall*lamW, at most ~0.1
    # sQsq:    
  
    eQ   = asmall*gs
    fsq  = fpi**2
    lamW = 4.*np.pi*fpi
    pi3  = np.pi**3
    pi2  = np.pi**2
    eQsq = eQ**2
    gssq = gs**2
    mD   = bsmall*lamW
    
    #-- Create Mass Squared Array of Pions --#

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

    M2 = np.array([Msq0, Msq1to4, Msq1to4, Msq1to4, Msq1to4, Msq5and8, Msq6and7, Msq6and7, Msq5and8, Msq9to12, Msq9to12, Msq9to12, Msq9to12, 0., Msq14]) # Mass Squared Array

    # Create Array of DM pions only
    M2DMarr = M2[5:13] 

    if (DEBUG):        
        print("Hyperparameter Settings:")
        print("Mass of DM (in GeV ?):          ", mD)
        print("gs:                             ", gs)
        print("")
        print("Weak confinement scale:         ", lamW)
        print("Number of generations:          ", Ngen)
        print("Pion decay constant equivalent: ", fpi)
        print("Array of dof of DM pions:       ", garr)
        print("x = m_1/T value:                ", x)
        print("CA, CZ, CG, CW:                 ", CA, CZ, CG, CW)
        print("")
        print("Mass Squared Array:             ")
        print(M2)
        print("Mass Squared DM Array:          ")
        print(M2DMarr)
        print("")

    #-- Calculate F1DMchargeBasisMatrix and F2DMchargeBasisMatrix --#
    F1const = 4./(fpi**2)         
    F2const = -2.*mD*(lamW**3)/(3*(fpi**4))

    F1DMchargeBasisMatrix = F1const*F1HatDMchargeBasisMatrix
    F2DMchargeBasisMatrix = F2const*F2HatDMchargeBasisMatrix

    #-- Calculate aeff --#

    start = time.process_time()
    from coannihilation import calcSigma_ij, calcaEff

    # Calculate sigma_ij matrix 
    sigij = calcSigma_ij(M2, F1DMchargeBasisMatrix, F2DMchargeBasisMatrix, aeff= True, DEBUG=DEBUG)
    end   = time.process_time()

    if (TIME):
        print("------------------------------------------")
        print("Calculate sigma_ij matrix ")
        print("Time elapsed: ", end - start)
        print("")

    # Calculate aeff
    start = time.process_time()
    aeff  = calcaEff(sigij, M2DMarr, garr, x, DEBUG)
    end   = time.process_time()

    if (DEBUG):
        print("mD, gs, aeff: ", mD, gs, aeff)
        print("")

    if (TIME):
        print("------------------------------------------")
        print("Calculate aeff")
        print("Time elapsed: ", end - start)
        print("")

        
    #-- Calculate omegaH2 --#
    from relicAbundance import calcOmegaH2
    
    m1 = np.sqrt(np.min(M2DMarr))
    omegaH2 = calcOmegaH2(m1, np.real(aeff), DEBUG)
                    
    end_paramScanTime = time.process_time()

    if(DEBUG):
        print("Final omegaH2: ", omegaH2[-1])
        print("")
    
    if (TIME):
        print("------------------------------------------")
        print("Single param calculation time: ", end_paramScanTime - start_paramScanTime)
        print("")
    
    return omegaH2[-1]
    
    
    
if __name__ == "__main__":
    
    kwargs = { "fpi":100, "gs":1e-3, "kappa":1, "asmall":1e-3, "bsmall":1e-5, "sQsq":0.001}
    oH2 = omegaH2(**kwargs, DEBUG=DEBUG)
    print(oH2)
    
