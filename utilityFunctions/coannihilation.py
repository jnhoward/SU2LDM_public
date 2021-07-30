import numpy as np
import itertools

from crossSection import calcCrossSection

#############################
##
## Calculate coannihilation
##
#############################

#-- Calculate sigma_ij (2D array) (assuming v=0) --#
def calcSigma_ij(M2, F1Mat, F2Mat, aeff=True, DEBUG=True):
    # Calculates DM_i DM_j -> SM SM
    # Sums over all possible SM states

    n = 15 # Number of DM + SM particles, purely for notational convenience
    sig = np.zeros((n, n), dtype=complex)
    
    DMlist = [5,6,7,8,9,10,11,12] # List of DM pions
    allDM = itertools.product(DMlist, DMlist) 
    # Note more efficient way if we account for the ij reaction symmetry <-- Maybe do this later #?

    SMlist = [1,2,3,4,13,14] # List of SM pions, ignoring eta'
    allSM = itertools.product(SMlist, SMlist)

    total = itertools.product(allDM, allSM) # Need this step because nesting loops over itertools doesn't work

    # Get masses of particles (not mass-squareds)
    particleMasses = np.sqrt(M2)

    #-- If we are calculating a_eff --# 
    if (aeff):
        for ((i,j),(c,d)) in total:
            sig[i,j] += calcCrossSection(i, j, c, d, M2, F1Mat, F2Mat, DEBUG)
    else:
        print("Function not set up to handle aeff=False case.")

    return sig

#-- Calculate Delta --#
def calcDelta(mi, m1):
    assert (m1 != 0.)
    return (mi - m1)/m1

#-- Calculate g_eff --#
def calcGeff2(g, delta, x):
    dummyArr = g*(1+delta)**(3./2.)*np.exp(-x*delta)
    return np.sum(dummyArr)**2

#-- Calculate effective cross-section to zeroth order in v --#
def calcaEff(sigma, mDMarr, g, x, DEBUG=True):

    m1 = np.min(mDMarr) # Whichever DM particle is the lightest

    # Get delta values
    delta = calcDelta(mDMarr, m1)

    # Get g_eff squared
    geff2 = calcGeff2(g, delta, x)
    assert (geff2 != 0.)

    # Calculate aeff
    sum = 0.
    DMindexlist = [5,6,7,8,9,10,11,12]

    for (i,j) in itertools.product(DMindexlist, DMindexlist):
        l = i-5
        k = j-5
        sum += sigma[i,j]*g[l]*g[k]*(1+delta[l])**(3./2.)*(1+delta[k])**(3./2.)*np.exp(-x*(delta[l] + delta[k]))
       
    return sum*(1./geff2)
    
