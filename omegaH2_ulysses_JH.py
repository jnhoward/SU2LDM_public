import numpy as np
import time
import os.path
from os import path
import ulysses

#-- Add utilityFunctions/ to easily use utility .py files --#
import sys
sys.path.append("utilityFunctions/")

#-- Define default settings --#
DEBUG = False  # Turn off DEBUG statements by default
TIME  = False  # Turn off printing time statements
#Ngen  = 1      # Number of generations = 1

class SU2LDM(ulysses.ULSBase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        #--------------------------------#
        #-- Initialize core parameters --#
        #--------------------------------#
        self.gs            = None # Log10[M/1g]
        self.fpi           = None # a_star
        self.kappa         = None # Log10[beta']
        self.asmall       = None
        self.bsmall        = None
        self.sQsq          = None

        self.pnames = ['m',  'M1', 'M2', 'M3', 'delta', 'a21', 'a31', 'x1', 'x2', 'x3', 'y1', 'y2', 'y3',
                       't12', 't13', 't23', 'gs', 'fpi', 'kappa', 'asmall', 'bsmall', 'sQsq']
        
        #---------------------------------#
        #-- Load precalculated matrices --#
        #---------------------------------#
        if (path.exists("npyFiles/Fmatrices_Ngen1.npy") == False):
            print("Error: npyFiles/Fmatrices_Ngen1.npy does not exists. Please run preScan.py before proceeding.")
            abort()
        else:
            self.F1HatDMchargeBasisMatrix, self.F2HatDMchargeBasisMatrix = np.load("npyFiles/Fmatrices_Ngen1.npy")
   

    def setParams(self, pdict):
        """
        This set the model parameters. pdict is expected to be a dictionary
        """
        super().setParams(pdict)
        self.gs         = pdict["gs"]
        self.fpi        = pdict["fpi"]
        self.kappa      = pdict["kappa"]
        self.asmall     = pdict["asmall"]
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
        asmall   = 10.**self.asmall
        bsmall   = 10.**self.bsmall
        sQsq     = 10.**self.sQsq
        
        #-- Pass these to omegah2() --#
        from omegah2 import omegaH2
        
        return omegaH2(gs, fpi, kappa, asmall, bsmall, sQsq, \
                       self.F1HatDMchargeBasisMatrix, self.F2HatDMchargeBasisMatrix, DEBUG)
