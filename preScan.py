import numpy as np
import time
import os.path
from os import path

#-- Add utilityFunctions/ to path to easily use utility .py files --#
import sys
sys.path.append("utilityFunctions/")

#-- Define default settings --#
DEBUG = False  # Turn off DEBUG statements by default
TIME  = True  # Turn off printing time statements by default
PLOT  = False  # Turn off plotting by default

#################################
##
##  How to run from command line for Ngen=1:
##  $ python preScan.py 1 
##  How to run from command line for Ngen=3:
##  $ python preScan.py 3 
##
#################################

def preScan(Ngen, DEBUG=False):

    #-- Define filename based on Ngen --#
    if(Ngen==1):
        filename = "npyFiles/FhatMatrices_DMBasis_Ngen1.npy"
    elif(Ngen==3):
        filename = "npyFiles/FhatMatrices_IntBasis_Ngen3.npy"
    else:
        print("Error: Invalid Ngen. Please use either Ngen=1 or Ngen=3.")
        return 0        

    #-- Check if file already exists --#
    if(path.exists(filename)): # If matrices have been calculated already, raise error
        print("Error: %s already exists. Please remove before rerunning preScan.py."%filename)
        return
    else:
        start_preTime = time.process_time()
    
    #------------------------#
    #-- Calculate matrices --#
    #------------------------#
    
        #-- Calculate A and X matrices --#        
        start = time.process_time()
        from calcMatrices import calcXs, calcA
        X = calcXs(Ngen, DEBUG)
        A = calcA(Ngen, DEBUG)
        end   = time.process_time()

        if (TIME):
            print("------------------------------------------")
            print("Calculate A and X")
            print("Time elapsed: ", end - start)
            print("")

        #-- Calculate F1Hat and F2Hat --#
        print("Calculating F1Hat and F2Hat matrices")
        # Here we only calculate the part which only depends on Ngen (through X and A defs)
        # F1Matrix and F2Matrix with appropriate factors will be calculated later
        start = time.process_time()
        from calcF1F2hat import calcF1F2HatMatrices
        #F1HatMatrix, F2HatMatrix = calcF1F2HatMatrices(X, A, Ngen, DEBUG) #! commented out to test rest of functions
        F1HatMatrix = np.zeros((91,91,91,91), dtype=complex) #!
        F2HatMatrix = np.zeros((91,91,91,91), dtype=complex) #!
        end   = time.process_time()
        
        if (DEBUG):
            print("------------------------------------------")
            print("F1HatMatrix Shape: ",F1HatMatrix.shape())
            print("F2HatMatrix Shape: ",F2HatMatrix.shape())
            print("")
        
        if (TIME):
            print("------------------------------------------")
            print("Calculate F1Hat and F2Hat")
            print("Time elapsed: ", end - start)
            print("")

        #-- Transform F1HatMatrix, F2HatMatrix if applicable and save file --#
        # In general we want to transform from interaction to mass to DM charge bases
        # For Ngen = 1: 
        #    Mass ~ interaction and mass to DM transformation is independent of scan/benchmark parameters.
        #    Therefore we perform the transformation here before scanning.
        #    We store F*HatMatrices in the DM charge basis.
        # For Ngen = 3:
        #    Interaction to mass transformation is NOT independent of scan/benchmark parameters.
        #    Therefore interaction->mass->DM charge will need to happen during scan. This will be somewhat slower.
        #    We store F*HatMatrices in the interaction basis.    
        print("Transforming F1Hat and F2Hat matrices if applicable, or storing transformation matrices")
        start = time.process_time()
        if(Ngen==1):
            # Transform to definite DM charge basis
            from convertToDMBasis import calcDMTransformMatrix, convertToDMBasis
            Vmatrix = calcDMTransformMatrix(Ngen, DEBUG)
            F1HatDMchargeBasisMatrix, F2HatDMchargeBasisMatrix = convertToDMBasis(F1HatMatrix, F2HatMatrix, Vmatrix, DEBUG)  
            
            # Save file
            np.save(filename, [F1HatDMchargeBasisMatrix, F2HatDMchargeBasisMatrix])
        else:
            # Do not transform, but make sure transformation matrices are calculated and stored
            from convertToMassBasis import calcCoreWmatrix
            WcoreMatrix = calcCoreWmatrix(DEBUG)
            np.save("npyFiles/WcoreMatrix_intToMass_Ngen3.npy", [WcoreMatrix])
            
            from convertToDMBasis import calcDMTransformMatrix
            Vmatrix = calcDMTransformMatrix(Ngen, DEBUG)
            np.save("npyFiles/VMatrix_massToDM_Ngen3.npy", [Vmatrix])
            
            # Save file
            np.save(filename, [F1HatMatrix, F2HatMatrix])
        
        end   = time.process_time()
        end_preTime = time.process_time()
        
        if (TIME):
            print("------------------------------------------")
            if(Ngen==1):
                print("Transform to definite DM charge basis from mass (=interaction) basis and save to file") 
            else:
                print("Save matrices in interaction basis to file") 
            print("Time elapsed: ", end - start)
            print("")
            
            print("Pre parameter scan time: ", end_preTime - start_preTime)
            print("") 
            
        print("------------------------------------------")
        print("Prescan finished successfully!")
        print("Fhat matrices stored in: ", filename)

    
if __name__ == "__main__":   
    
    Ngen = int(sys.argv[1])
    print("Running preScan.py for Ngen=%d"%Ngen)
    preScan(Ngen=Ngen)