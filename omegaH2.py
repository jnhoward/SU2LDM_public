import numpy as np
import time
import os.path
from os import path

#-- Add utilityFunctions/ to easily use utility .py files --#
import sys
sys.path.append("utilityFunctions/")

#-- Define default settings --#
DEBUG = False  # Turn off DEBUG statements by default
TIME  = False  # Turn off printing time statements

###################################################################################################
##
##  How to run from command line for Ngen=1:
##  $ python omegaH2.py 1
##  How to run from command line for Ngen=3:
##  $ python omegaH2.py 3
##
##  OR
##
##  $ from omegaH2 import omegaH2
##  $ kwargs = { "Ngen": 1, "gs":0.8, "fpi":155.*1000., "kappa":1.0, "eQ":0.5, "bsmall":0.01, "sQsq":0.3}
##  $ omegaH2(**kwargs)
##
###################################################################################################

def omegaH2(Ngen, gs, fpi, kappa, eQ, bsmall, sQsq, F1HatMatrix=None, F2HatMatrix=None, DEBUG=False, RETURN=None):
    
    start_paramScanTime = time.process_time()
    
    #---------------------------------------#
    #-- Make sure RETURN flag makes sense --#
    #---------------------------------------#
    if RETURN not in [None,'M2_DMbasis','m1_aeff','sigij']:
        print("Error: Invalid RETURN option. Please use one of the following.")
        print("     None: Will cause omegaH2 to return oh2, therm.") 
        print("'m1_aeff': Will cause omegaH2 to return m1, aeff.")
        print("  'sigij': Will cause omegaH2 to return sigij matrix.")
        return
    
    #---------------------------------------#
    #-- Set overall setings based on Ngen --#
    #---------------------------------------#
    if(Ngen==1):
        FhatFilename = "Data/npyFiles/FhatMatrices_DMBasis_Ngen1.npy"
        nDMPions     = 8  
    elif(Ngen==3):
        FhatFilename    = "Data/npyFiles/FhatMatrices_IntBasis_Ngen3.npy"
        VmatrixFilename = "Data/npyFiles/VMatrix_massToDM_Ngen3.npy"
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
    lamW = 4.*np.pi*fpi
    mD   = bsmall*lamW   
    
    #------------------------------------------------------------------------------------------------#
    #-- Create Mass Squared Array of Pions And Get Interaction to Mass Basis Transformation Matrix --#
    #------------------------------------------------------------------------------------------------#
    # Note: M2 array is given in DM charge basis order
    from calcPionMassSq import calcPionMassSq
    if(Ngen==1):
        # M2arr_DMcharge, M2arr_mass, M2DMarr 
        M2, _, M2DMarr = calcPionMassSq(Ngen, CA, CG, CW, CZ, eQ, gs, sQsq, lamW, fpi, mD, kappa, DEBUG)
    elif(Ngen==3):
        # M2arr_DMcharge, M2arr_mass, M2DMarr, Wmatrix_mass
        M2, _, M2DMarr, Wmatrix = calcPionMassSq(Ngen, CA, CG, CW, CZ, eQ, gs, sQsq, lamW, fpi, mD, kappa, DEBUG)
    else:
        print("Error: Invalid Ngen. Please use either Ngen=1 or Ngen=3.")
        return    
    
    if RETURN == 'M2_DMbasis':
        return M2
    
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
        
        #-- Transform from interaction to mass to DM charge basis in one go --#
        Vmatrix = np.load(VmatrixFilename)[0]
        WVmatrix = Wmatrix @ Vmatrix
        
        from transformFs import transformF
        F1HatMatrix_DMbasis = transformF(WVmatrix, F1HatMatrix, DEBUG)
        F2HatMatrix_DMbasis = transformF(WVmatrix, F2HatMatrix, DEBUG)

    #---------------------------------------------------------------#
    #-- Calculate F1DMchargeBasisMatrix and F2DMchargeBasisMatrix --#
    #---------------------------------------------------------------#
    fsq  = fpi**2 
    F1const = 4./fsq
    F2const = -2.*mD*(lamW*lamW*lamW)/(3*(fsq*fsq))

    F1DMchargeBasisMatrix = F1const*F1HatMatrix_DMbasis
    F2DMchargeBasisMatrix = F2const*F2HatMatrix_DMbasis

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
 
    #--------------------#    
    #-- Calculate aeff --#
    #--------------------#  
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

    if RETURN =='sigij':
        return sigij
    
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

    #-----------------------#     
    #-- Calculate omegaH2 --#
    #-----------------------#   
    from relicDMAbundance import calcOmegaH2
    
    m1 = np.sqrt(np.min(M2DMarr))
    if RETURN == 'm1_aeff':
        return m1, aeff

    if RETURN == None:
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
    kwargs = { "gs":0.8, "fpi":155.*1000., "kappa":1.0, "eQ":0.5, "bsmall":0.01, "sQsq":0.3}
    oH2, _ = omegaH2(Ngen, **kwargs, DEBUG=DEBUG, RETURN=None)
    print(oH2)
    
