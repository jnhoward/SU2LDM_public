import numpy as np
import time
import os.path
from os import path

#-- Add utilityFunctions/ to easily use utility .py files --#
import sys
sys.path.append("utilityFunctions/")

#-- Define default settings --#
DEBUG = False  # Turn off DEBUG statements by default
TIME  = True  # Turn off printing time statements
#Ngen  = 1      # Number of generations = 1

###################################################################################################
##
##  How to run from command line for Ngen=1:
##  $ python omegaH2.py 1
##  How to run from command line for Ngen=3:
##  $ python omegaH2.py 3
##
##  OR
##
##  $ from omegah2 import omegaH2
##  $ kwargs = { "gs":0.8, "fpi":155.*1000., "kappa":1.0, "asmall":0.625, "bsmall":0.01, "sQsq":0.3}
##  $ omegaH2(**kwargs)
##
###################################################################################################

def omegaH2(Ngen, gs, fpi, kappa, asmall, bsmall, sQsq, F1HatMatrix=None, F2HatMatrix=None, DEBUG=False):
    
    start_paramScanTime = time.process_time()
    
    #---------------------------------------#
    #-- Set overall setings based on Ngen --#
    #---------------------------------------#
    if(Ngen==1):
        FhatFilename = "npyFiles/FhatMatrices_DMBasis_Ngen1.npy"
        nDMPions     = 8  
    elif(Ngen==3):
        FhatFilename    = "npyFiles/FhatMatrices_IntBasis_Ngen3.npy"
        #WcoreFilename   = "npyFiles/WcoreMatrix_intToMass_Ngen3.npy"
        VmatrixFilename = "npyFiles/VMatrix_massToDM_Ngen3.npy"
        nDMPions        = 24
    else:
        print("Error: Invalid Ngen. Please use either Ngen=1 or Ngen=3.")
        return 
        
        
    #-----------------------#
    #-- Define parameters --#
    #-----------------------#
    
    #-- Define fixed parameters --#
    garr = np.ones(nDMPions)
    x    = 20.
    CA   = -1.
    CZ   = -1.
    CG   = -1.
    CW   =  1.
    
    #-- Parameters from arguments --#
    eQ   = asmall*gs
    lamW = 4.*np.pi*fpi
    mD   = bsmall*lamW
    
    fsq  = fpi**2 
#     pi3  = np.pi**3
#     pi2  = np.pi**2
#     eQsq = eQ**2
#     gssq = gs**2     
    
    #-----------------------------------------------------------#
    #-- Load precalculated matrices if not passed to function --#
    #-----------------------------------------------------------#
    if (F1HatMatrix is None) or (F2HatMatrix is None):  
        if (path.exists(FhatFilename) == False): 
            print("Error: %s does not exists. Please run preScan.py before proceeding."%FhatFilename)
            return
        else: 
            F1HatMatrix, F2HatMatrix = np.load(FhatFilename)
    
    #-------------------------------------------#
    #-- Transform Fhat matrices if applicable --#
    #-------------------------------------------#
    if(Ngen==1):
        F1HatMatrix_DMbasis, F2HatMatrix_DMbasis = F1HatMatrix, F2HatMatrix
    elif(Ngen==3):
         
        #-- Transform from interaction to mass basis --#
        from convertToMassBasis import calcMassTransformMatrix, convertToMassBasis
        
        # Calculate mass transformation matrix and perform transformation
        Wmatrix = calcMassTransformMatrix(lamW, fpi, kappa, mD, Ngen, DEBUG)
        F1HatMatrix_mass, F2HatMatrix_mass = convertToMassBasis(F1HatMatrix, F2HatMatrix, Wmatrix, Ngen, DEBUG)
        
        #-- Transform from mass to DM charge basis --#
        from convertToDMBasis import calcDMTransformMatrix, convertToDMBasis
        
        # Load pre-calculated DM transformation matrix and perform transformation
        Vmatrix = np.load(VmatrixFilename)[0]
        F1HatMatrix_DMbasis, F2HatMatrix_DMbasis = convertToDMBasis(F1HatMatrix, F2HatMatrix, Vmatrix, DEBUG)

    #---------------------------------------------------------------#
    #-- Calculate F1DMchargeBasisMatrix and F2DMchargeBasisMatrix --#
    #---------------------------------------------------------------#
    F1const = 4./fsq
    F2const = -2.*mD*(lamW*lamW*lamW)/(3*(fsq*fsq))

    F1DMchargeBasisMatrix = F1const*F1HatMatrix_DMbasis
    F2DMchargeBasisMatrix = F2const*F2HatMatrix_DMbasis
        
    #----------------------------------------#
    #-- Create Mass Squared Array of Pions --#
    #----------------------------------------#
    from calcPionMassSq import calcPionMassSq
    M2, M2DMarr = calcPionMassSq(Ngen, CA, CG, CW, CZ, eQ, gs, sQsq, lamW, fpi, mD, kappa, DEBUG)
    #M2 = np.ones(M2.shape)#!
    #M2DMarr = np.ones(M2DMarr.shape)#!
    # Note need to redefine M2 to be returned as masses in DM charge basis
    
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

    #-- Calculate aeff --#
    
    start = time.process_time()
    from coannihilation import calcSigma_ij, calcaEff

    # Calculate sigma_ij matrix 
    sigij = calcSigma_ij(M2, F1DMchargeBasisMatrix, F2DMchargeBasisMatrix, Ngen, aeff=True, DEBUG=DEBUG)
    end   = time.process_time()

    if (TIME):
        print("------------------------------------------")
        print("Calculate sigma_ij matrix ")
        print("Time elapsed: ", end - start)
        print("")

    # Calculate aeff
    start = time.process_time()
    aeff  = calcaEff(sigij, M2DMarr, garr, x, Ngen, DEBUG)
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
    from relicDMAbundance import calcOmegaH2
    
    m1 = np.sqrt(np.min(M2DMarr))
    omegaH2, therm = calcOmegaH2(m1, mD, np.real(aeff))
                    
    end_paramScanTime = time.process_time()

    if(DEBUG):
        print("Final omegaH2: ", omegaH2[-1])
        print("")
    
    if (TIME):
        print("------------------------------------------")
        print("Single param calculation time: ", end_paramScanTime - start_paramScanTime)
        print("")
    
    return omegaH2[-1], therm

    
if __name__ == "__main__":
    
    Ngen = int(sys.argv[1])
    kwargs = { "gs":0.8, "fpi":155.*1000., "kappa":1.0, "asmall":0.625, "bsmall":0.01, "sQsq":0.3}
    oH2, _ = omegaH2(Ngen, **kwargs, DEBUG=DEBUG)
    print(oH2)
    
