import numpy as np
import time
import os.path
import os
from os import path
import ulysses

#-- Add utilityFunctions/ to easily use utility .py files --#
import sys
sys.path.append("utilityFunctions/")

#-- Define default settings --#
DEBUG = False  # Turn off DEBUG statements by default
TIME  = False  # Turn off printing time statements
Ngen  = 1      # Number of generations = 1

class SU2LDM(ulysses.ULSBase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        #--------------------------------#
        #-- Initialize core parameters --#
        #--------------------------------#
        self.gs            = None # Log10[M/1g]
        self.fpi           = None # a_star
        self.kappa         = None # Log10[beta']
        #self.asmall       = None
        self.eQ            = None
        self.bsmall        = None
        self.sQsq          = None

        self.pnames = ['m',  'M1', 'M2', 'M3', 'delta', 'a21', 'a31', 'x1', 'x2', 'x3', 'y1', 'y2', 'y3',
                       't12', 't13', 't23', 'gs', 'fpi', 'kappa', 'eQ', 'bsmall', 'sQsq']
        
        #---------------------------------#
        #-- Load precalculated matrices --#
        #---------------------------------#
        if(Ngen==1):
            FmatFilePath = "Data/npyFiles/FhatMatrices_DMBasis_Ngen1.npy"
        elif(Ngen==3):
            FmatFilePath = "Data/npyFiles/FhatMatrices_IntBasis_Ngen3.npy"
        else:
            print("Error: Invalid Ngen. Please use either Ngen=1 or Ngen=3.")
            return     
        
        if (path.exists(FmatFilePath) == False):
            print("Error: %s does not exists. Please run preScan.py before proceeding."%FmatFilePath)
            os.abort()
        else:
            self.F1HatMatrix, self.F2HatMatrix = np.load(FmatFilePath)   

    def setParams(self, pdict):
        """
        This set the model parameters. pdict is expected to be a dictionary
        """
        super().setParams(pdict)
        self.gs         = pdict["gs"]
        self.fpi        = pdict["fpi"]
        self.kappa      = pdict["kappa"]
        #self.asmall     = pdict["asmall"]
        self.eQ         = pdict["eQ"]
        self.bsmall     = pdict["bsmall"]
        self.sQsq       = pdict["sQsq"]

    def shortname(self): return "SU2LDM"

    @property
    def EtaB(self):
        
        # Note that the function name EtaB is necessary to utilize ulysses
        # See section 5 of arxiv:2007.09150
        
        #-----------------------#
        #-- Calculate omegaH2 --#
        #-----------------------#
        
        #-- Set core parameter values --#
        gs       = 10.**self.gs
        fpi      = 10.**self.fpi
        kappa    = 10.**self.kappa
        #asmall   = 10.**self.asmall
        eQ       = 10.**self.eQ
        bsmall   = 10.**self.bsmall
        sQsq     = 10.**self.sQsq
        
        #-- Pass these to omegah2() --#
        from omegaH2 import omegaH2
        
        oh2, _ = omegaH2(Ngen, gs, fpi, kappa, eQ, bsmall, sQsq, \
                             self.F1HatMatrix, self.F2HatMatrix, DEBUG)
        
        return oh2
