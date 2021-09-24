import numpy as np
import itertools

#############################
##
## Calculate coannihilation
##
#############################

#-- Calculate sigma_ij (2D array) (assuming v=0) --#
def calcSigma_ij(M2, F1Mat, F2Mat, Ngen, aeff=True, DEBUG=True):
    # Calculates DM_i DM_j -> SM SM
    # Sums over all possible SM states
    if(Ngen==1):
        n      = 15                        # Number of total pions (DM+SM)
        DMindexlist = np.arange(8)+5       # Indices of DM charged pions in DM charge basis, 5 to 12 by definition
        SMindexlist = np.delete(np.arange(n), DMindexlist) # Indices of SM charged pions in DM charge basis
        SMindexlist = np.delete(SMindexlist, 0) # Ignore eta' (index 0) -> 1,2,3,4,13,14
    elif(Ngen==3):
        n      = 91                        # Number of total pions (DM+SM)
        DMindexlist = np.arange(24)+1      # Indices of DM charged pions in DM charge basis, 1 to 24 by definition
        SMindexlist = np.delete(np.arange(n), DMindexlist) # Indices of SM charged pions in DM charge basis
        SMindexlist = np.delete(SMindexlist, 0) # Ignore eta' (index 0)
    else:
        print("Error: Invalid Ngen. Please use either Ngen=1 or Ngen=3.")
        return         
    
    sig = np.zeros((n, n), dtype=complex)    
    
    allDM = itertools.product(DMindexlist, DMindexlist) 
    # Note more efficient way if we account for the ij reaction symmetry <-- Maybe do this later #?

    allSM = itertools.product(SMindexlist, SMindexlist)

    total = itertools.product(allDM, allSM) # Need this step because nesting loops over itertools doesn't work

    # Get masses of particles (not mass-squareds)
    particleMasses = np.sqrt(M2)

    #-- If we are calculating a_eff --# 
    from crossSection import calcCrossSection
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
def calcaEff(sigma, mDMarr, g, x, Ngen, DEBUG=True):

    m1 = np.min(mDMarr) # Whichever DM particle is the lightest

    # Get delta values
    delta = calcDelta(mDMarr, m1)

    # Get g_eff squared
    geff2 = calcGeff2(g, delta, x)
    assert (geff2 != 0.)

    # Calculate aeff
    if(Ngen==1):
        DMindexlist = np.arange(8)+5  # Indices of DM charged pions in DM charge basis, 5 to 12 by definition
    elif(Ngen==3):
        DMindexlist = np.arange(24)+1 # Indices of DM charged pions in DM charge basis, 1 to 24 by definition
    else:
        print("Error: Invalid Ngen. Please use either Ngen=1 or Ngen=3.")
        return   
    
    summ = 0.
    for (i,j) in itertools.product(DMindexlist, DMindexlist):
        l = i-DMindexlist[0]
        k = j-DMindexlist[0]
        summ += sigma[i,j]*g[l]*g[k]*(1+delta[l])**(3./2.)*(1+delta[k])**(3./2.)*np.exp(-x*(delta[l] + delta[k]))
       
    return summ*(1./geff2)
    
