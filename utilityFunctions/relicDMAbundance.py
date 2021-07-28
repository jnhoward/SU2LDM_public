import ulysses
import numpy as np
import pandas as pd
from scipy import interpolate
import scipy.integrate as integrate
from scipy.integrate import quad, ode, solve_ivp, odeint
from scipy.optimize import root
from scipy.special import zeta
from scipy.special import kn

# Constants

c     = 299792.458       # in km/s
gamma = 0.1924500897298753 # Collapse factor
GCF   = 6.70883e-39      # Gravitational constant in GeV^-2
mPL   = GCF**-0.5        # Planck mass in GeV
v     = 174              # Higgs vev
csp   = 0.35443          # sphaleron conversion factor
GF    = 1.1663787e-5     # Fermi constant in GeV^-2

# Conversion factors
cm_in_invkeV = 5.067730938543699e7       # 1cm in keV^-1
GeV_in_g     = 1.782661907e-24
Mpc_in_cm    = 3.085677581e24
year_in_s    = 3.168808781402895e-8
GeV_in_invs  = cm_in_invkeV * c * 1.e11


#-------------------------------------#
#    g*(T) and g*S(T) interpolation   #
#-------------------------------------#

gTab = pd.read_table("./Data/gstar.dat",  names=['T','gstar'])

Ttab = gTab.iloc[:,0]
gtab = gTab.iloc[:,1]
tck  = interpolate.splrep(Ttab, gtab, s=0)

def gstar(T): return interpolate.splev(T, tck, der=0)

gSTab = pd.read_table("./Data/gstarS.dat",  names=['T','gstarS'])

TStab = gSTab.iloc[:,0]
gstab = gSTab.iloc[:,1]
tckS  = interpolate.splrep(TStab, gstab, s=0)

def gstarS(T): return interpolate.splev(T, tckS, der = 0)

def dgstarSdT(T): return interpolate.splev(T, tckS, der = 1)

#------------------------------------------------------------------------------------------------------------------#
#                                            Input parameters                                                      #
#------------------------------------------------------------------------------------------------------------------#

def FBEqs( a, v, nphi, mDM, sv):

    #nphi   = 0.24358765646*mDM**3          # Initial photon number density
    rRAD  = v[0] # Radiation energy density
    NDM   = v[1] # DM number density
    Tp    = v[2] # Temperature

    #----------------#
    #   Parameters   #
    #----------------#

    H   = np.sqrt(25.13274122871834 * GCF * (rRAD * 10.**(-4*a))/3.)    # Hubble parameter
    Del = 1. + Tp * dgstarSdT(Tp)/(3. * gstarS(Tp))             # Temperature parameter

    #----------------------------------------#
    #    Radiation + Temperature equations   #
    #----------------------------------------#
    
    drRADda = 0.
    dTda    = - Tp/Del

    #----------------------------------------#
    #              Freeze-out DM             #
    #----------------------------------------#

    NDMeq = (10.**(3*a) * mDM*mDM * Tp * kn(2, mDM/Tp))/(9.8696044)/nphi
    
    dNDMda = -(NDM*NDM - NDMeq*NDMeq)*sv*nphi/(H*10.**(3.*a))
    
    dEqsda = [drRADda, dNDMda, dTda]

    dEqsda = [x * 2.3025 for x in dEqsda]
    
    return dEqsda


#-----------------------------------------#
#          relic density calculation      #
#-----------------------------------------#

def calcOmegaH2(mDM, mDMcon, sv):
    Ti     = mDM   # Initial Universe temperature
    rRadi  = 0.3289868133696 * gstar(Ti) * Ti*Ti*Ti*Ti  # Initial radiation energy density -- assuming a radiation Universe
    nphi   = 0.24358828401821708*Ti*Ti*Ti #0.24358765646*Ti**3
    
    v0 = [rRadi, nphi, Ti]
    
    solFBE = solve_ivp(lambda t, z: FBEqs( t, z,  nphi, mDM, sv),
                                   [0, 2.0], v0, method='Radau',  rtol=1.e-10, atol=1.e-20, dense_output=True)

    t    = solFBE.t[:]
    Rad  = solFBE.y[0]
    NDM  = solFBE.y[1]
    T    = solFBE.y[2]
    H   = np.sqrt(25.13274122871834* GCF * (Rad * 10.**(-4*t))/3.)
    
    alphaW = 0.0338
    sigmaW = (alphaW * alphaW * 3.1415)/ (mDMcon * mDMcon)
    
    rc = 1.053672e-23*cm_in_invkeV**-3  # Critical density in GeV^3
    T0 = 2.34865e-13  # Temperature today in GeV
    Conf = (gstarS(T0)/gstarS(T[-1]))*(T0/T[-1])*(T0/T[-1])*(T0/T[-1])*(1/rc)

    deconf = 2. * mDMcon/mDM # converts the energy density of the confined phase to the energy density of the deconfined phase
    Oh2   =  (NDM * nphi * 10.**(-3.*t) * mDM * Conf) * deconf #relic abundance AFTER deconfinement
    
    ##-- Test thermalization --#
    therm = min(sigmaW * 10.**(-3.*t) * NDM * nphi * 2 / H)
    if  therm > 1.:
        test_thermalise  = True
        return 1000.0
    else:
        test_thermalise = False
        return mDMcon#Oh2[-1]#sv#mDM/T[-1]#Oh2[-1]
    #print(test_thermalise)
        
