import numpy as np
import time
#import itertools
import os.path
from os import path

#-- Add utilityFunctions/ to path to easily use utility .py files --#
import sys
sys.path.append("utilityFunctions/")

#-- Define default settings --#
DEBUG = False  # Turn off DEBUG statements by default
TIME  = False  # Turn off printing time statements by default
PLOT  = False  # Turn off plotting by default
Ngen  = 1      # Number of generations = 1

#################################
##
##  How to run from command line:
##  $ python preScan.py
##
#################################

def preScan(DEBUG=False):
 
    #------------------------#
    #-- Calculate matrices --#
    #------------------------#
    if(path.exists("npyFiles/Fmatrices_Ngen1.npy")): # If matrices have been calculated already, raise error
        print("Error: npyFiles/Fmatrices_Ngen1.npy already exists. Please remove before rerunning preScan.py.")
        return
    else:
        start_preTime = time.process_time()
        
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

        #-- Calculate F1Hat and F2Hat and save to file--#
      
        # Here we only calculate the part which only depends on Ngen (through X and A defs)
        # F1Matrix and F2Matrix with appropriate factors will be calculated later
        start = time.process_time()
        from calcF1F2 import calcF1F2HatMatrices
        F1HatMatrix, F2HatMatrix = calcF1F2HatMatrices(X, A, DEBUG)
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

        #-- Transform to definite DM masses --# 
        start = time.process_time()
        from convertToDMBasis import convertToDMBasis
        F1HatDMchargeBasisMatrix, F2HatDMchargeBasisMatrix = convertToDMBasis(F1HatMatrix, F2HatMatrix, DEBUG)
   

        #-- Save matrices --#
        filename = "npyFiles/Fmatrices_Ngen1.npy"
        np.save(filename, [F1HatDMchargeBasisMatrix, F2HatDMchargeBasisMatrix])
        end   = time.process_time()
        end_preTime = time.process_time()
        
        if (TIME):
            print("------------------------------------------")
            print("Transform to definite DM masses and save to file")    
            print("Time elapsed: ", end - start)
            print("")
            
            print("Pre parameter scan time: ", end_preTime - start_preTime)
            print("") 
            
        print("------------------------------------------")
        print("Prescan finished successfully!")
        print("Matrices stored in: ", filename)

    
if __name__ == "__main__":   
    preScan()