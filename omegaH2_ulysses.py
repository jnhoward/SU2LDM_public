import numpy as np
import time
import itertools
import os.path
from os import path
import ulysses

#-- Add utilityFunctions/ to easily use utility .py files --#
import sys
sys.path.append("utilityFunctions/")

DEBUG = False  # Turn off DEBUG statements by default
TIME  = False  # Turn off printing time statements
PLOT  = False  # Turn off plotting

Ngen = 1 # Hyper parameter

class SU2LDM(ulysses.ULSBase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.gs            = None # Log10[M/1g]
        self.fpi           = None # a_star
        self.kappa         = None # Log10[beta']
        self.asmall       = None
        self.bsmall        = None
        self.sQsq          = None

        self.pnames = ['m',  'M1', 'M2', 'M3', 'delta', 'a21', 'a31', 'x1', 'x2', 'x3', 'y1', 'y2', 'y3',
                       't12', 't13', 't23', 'gs', 'fpi', 'kappa', 'asmall', 'bsmall', 'sQsq']
                       
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
    
        #-- Define fixed parameters --#
        garr = np.array([1.,1.,1.,1.,1.,1.,1.,1.])
        x    = 20.
        CA   = -1.
        CZ   = -1.
        CG   = -1.
        CW   =  1.
        
        asmall   = 10.**self.asmall
        bsmall   = 10.**self.bsmall
        eQ       = 10.**self.asmall*10.**self.gs
        sQsq     = 10.**self.sQsq
        kappa    = 10.**self.kappa
        gs       = 10.**self.gs
        fpi      = 10.**self.fpi
        
        
        fsq    = fpi*(fpi)
        lamW   = 4.*np.pi*fpi
        pi3    = np.pi*np.pi*np.pi
        pi2    = np.pi*np.pi
        eQsq   = eQ*eQ
        gssq   = (10**self.gs)*(10**self.gs)
        mD     =  10**(self.bsmall)*lamW
        
        #-- Create Mass Squared Array of Pions --#
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

     
        #-- Calculate F1DMchargeBasisMatrix and F2DMchargeBasisMatrix --#
        F1const = 4./fsq
        F2const = -2.*mD*(lamW*lamW*lamW)/(3*(fsq * fsq))
        
        F1DMchargeBasisMatrix = F1const*self.F1HatDMchargeBasisMatrix
        F2DMchargeBasisMatrix = F2const*self.F2HatDMchargeBasisMatrix

        #-- Calculate aeff --#

        from coannihilation import calcSigma_ij, calcaEff

        # Calculate sigma_ij matrix
        sigij = calcSigma_ij(M2, F1DMchargeBasisMatrix, F2DMchargeBasisMatrix, aeff= True, DEBUG=DEBUG)
       

        # Calculate aeff
        aeff  = calcaEff(sigij, M2DMarr, garr, x, DEBUG)

            
        #-- Calculate omegaH2 --#
        from relicAbundance_scalefactor_new import calcOmegaH2

        m1 = np.min(M2DMarr)
        omegaH2 = calcOmegaH2(np.sqrt(m1), mD, np.real(aeff))
        
              
        #deconfineconf  = (2 * mD)/np.sqrt(m1)  ##          we need the factor 2* m_constituent/m_pion
        print( mD, np.sqrt(m1))
        return omegaH2 #* deconfineconf #Conversion now done in calcOmegaH2()

#       return omegaH2


    
    
    

    
