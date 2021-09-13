import ulysses
import numpy as np
import pandas as pd
from scipy import interpolate
import scipy.integrate as integrate
from scipy.integrate import quad, ode, solve_ivp, odeint
from scipy.optimize import root
from scipy.special import zeta
from scipy.special import kn

######################################
##
## Calculates the relic DM abundance
##
######################################

#-- Constants --#
c     = 299792.458         # Speed of light in km/s
gamma = 0.1924500897298753 # Collapse factor #?
GCF   = 6.70883e-39        # Gravitational constant in GeV^-2
mPL   = GCF**-0.5          # Planck mass in GeV
v     = 174                # Higgs vev
csp   = 0.35443            # Sphaleron conversion factor
GF    = 1.1663787e-5       # Fermi constant in GeV^-2

#-- Conversion factors --#
cm_in_invkeV = 5.067730938543699e7       # cm   in 1 keV^-1
GeV_in_g     = 1.782661907e-24           # GeV  in 1 g #?
Mpc_in_cm    = 3.085677581e24            # Mpc  in 1 cm
year_in_s    = 3.168808781402895e-8      # year in 1 s
GeV_in_invs  = cm_in_invkeV * c * 1.e11  # GeV  in 1 s^-1


##--------------------------------------------##
##  Calculate g*(T) and g*S(T) interpolation  ##
##--------------------------------------------##

# g*(T) is ... #?
# g*S(T) is ... #?

#-- Read in g*(T) data and define an interpolated function of g*(T)--# #?
gTab = pd.read_table("./Data/gstar.dat",  names=['T','gstar'])

Ttab = gTab.iloc[:,0]
gtab = gTab.iloc[:,1]
tck  = interpolate.splrep(Ttab, gtab, s=0)

def gstar(T): return interpolate.splev(T, tck, der=0)

#-- Read in g*S(T) data and define an interpolated function of g*S(T) and its derivative w.r.t T --# #?
gSTab = pd.read_table("./Data/gstarS.dat",  names=['T','gstarS'])

TStab = gSTab.iloc[:,0]
gstab = gSTab.iloc[:,1]
tckS  = interpolate.splrep(TStab, gstab, s=0)

def gstarS(T): return interpolate.splev(T, tckS, der = 0)

def dgstarSdT(T): return interpolate.splev(T, tckS, der = 1)

##----------------------------##
##  Define Boltzman equation  ##  #?
##----------------------------##
def FBEqs( a, v, nphi, mDM, sv):
    """ 
    Calculate ... #?
    
    a:    #?
    v:    List of parameters rRAD, NDM, Tp. See definitions below.
    nphi: Initial photon number density
    mDM:  Mass of the lightest DM pion #?
    sv:   Coannihilation cross section
    """
    #-- Define Parameters --#
    
    rRAD  = v[0]                            # Radiation energy density
    NDM   = v[1]                            # DM number density
    Tp    = v[2]                            # Temperature
    
    H   = np.sqrt(25.13274122871834 * GCF * (rRAD * 10.**(-4*a))/3.)    # Hubble parameter
    Del = 1. + Tp * dgstarSdT(Tp)/(3. * gstarS(Tp))                     # Temperature parameter

    #-- Radiation + Temperature equations --#
    
    drRADda = 0.
    dTda    = - Tp/Del

    #-- Calculate freeze-out of DM pion --#

    NDMeq = (10.**(3*a) * mDM*mDM * Tp * kn(2, mDM/Tp))/(9.8696044)/nphi
    
    dNDMda = -(NDM*NDM - NDMeq*NDMeq)*sv*nphi/(H*10.**(3.*a))
    
    dEqsda = [drRADda, dNDMda, dTda]

    dEqsda = [x * 2.3025 for x in dEqsda]
    
    return dEqsda

##----------------------------##
##  Test thermalization value ##
##----------------------------##
def testThermalization(sigmaW, t, NDM, nphi, H):
    return min(sigmaW * 10.**(-3.*t) * NDM * nphi * 2 / H) # Should be <=1

##-------------------------------------------------------------##
##  Calculate relic density of DM constituent post deconfiment ##
##-------------------------------------------------------------##
def calcOmegaH2(mDM, mDMcon, sv):
    """
    mDM:    Mass of lightest DM pion
    mDMcon: Mass of DM constituent (DM candidate)
    sv:     Coannihilation thermally averaged cross section, relative velocity v=0, non-relativistic colliding particles 
    """
    Ti     = mDM                                        # Initial Universe temperature
    rRadi  = 0.3289868133696 * gstar(Ti) * Ti*Ti*Ti*Ti  # Initial radiation energy density -- assuming a radiation Universe
    nphi   = 0.24358828401821708 * Ti*Ti*Ti             # Initial photon number density
    
    v0 = [rRadi, nphi, Ti]
    
    #-- Solve Boltzmann Equation --#
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
    T0 = 2.34865e-13                    # Temperature today in GeV
    Conf = (gstarS(T0)/gstarS(T[-1]))*(T0/T[-1])*(T0/T[-1])*(T0/T[-1])*(1/rc)

    #-- Convert energy density of the confined phase to the energy density of the deconfined phase --#
    # Note there is 1 DM constituent in the lightest DM pion
    deconf = mDMcon/mDM 
    Oh2   =  (NDM * nphi * 10.**(-3.*t) * mDM * Conf) * deconf #Relic abundance AFTER deconfinement
    
    #-- Test thermalization and store value --#
    therm = testThermalization(sigmaW, t, NDM, nphi, H)
    
    return Oh2, therm
