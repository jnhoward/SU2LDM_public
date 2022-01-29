import numpy as np
from numpy import ma
from scipy.optimize import bisect

# See https://github.com/michaelhb/superplot/blob/master/superplot/statslib/two_dim.py

def posterior_pdf(paramx, paramy, posterior, nbins=50, bin_limits=None):
    r"""
    Weighted histogram of data for two-dimensional posterior pdf.
    .. warning::
        Outliers sometimes mess up bins. So you might want to \
        specify the bin limits.
    .. warning::
        Posterior pdf normalized such that maximum value is one.
    :param paramx: Data column of parameter x
    :type paramx: numpy.ndarray
    :param paramy: Data column of parameter y
    :type paramy: numpy.ndarray
    :param posterior: Data column of posterior weight
    :type posterior: numpy.ndarray
    :param nbins: Number of bins for histogram
    :type nbins: integer
    :param bin_limits: Bin limits for histogram
    :type bin_limits: list [[xmin,xmax],[ymin,ymax]]
    :returns: Posterior pdf, x and y bin centers
    :rtype: named tuple (pdf: numpy.ndarray, bin_centers_x: \
        numpy.ndarray, bin_centers_y: numpy.ndarray)
    :Example:
    >>> nbins = 100
    >>> pdf, x, y = posterior_pdf(data[2], data[3], data[0], nbins=nbins)
    >>> assert len(pdf) == nbins
    >>> assert len(x) ==  nbins
    >>> assert len(y) == nbins
    """
    # 2D histogram the data - pdf is a matrix
    pdf, bin_edges_x, bin_edges_y = np.histogram2d(
                                        paramx,
                                        paramy,
                                        nbins,
                                        range=bin_limits,
                                        weights=posterior)

    # Normalize the pdf so that the area is one.
    pdf = pdf /pdf.sum()

    # Find centers of bins
    bin_centers_x = 0.5 * (bin_edges_x[:-1] + bin_edges_x[1:])
    bin_centers_y = 0.5 * (bin_edges_y[:-1] + bin_edges_y[1:])

    return bin_centers_x, bin_centers_y, pdf

def shift(rawBinNum, nbins, axList=None):
  
    if axList is not None:
        assert len(axList) == 2
        low  = axList[0]
        high = axList[1]
    else:
        low  = 0
        high = nbins+1

    # Remove low
    rawBinNum_sansLow = np.where(rawBinNum==low, low+1, rawBinNum)

    # Remove high
    rawBinNum_sansLowAndHigh = np.where(rawBinNum_sansLow==high, high-1, rawBinNum_sansLow)

    # Shift to python array notation
    binNum = rawBinNum_sansLowAndHigh - 1

    return binNum

def profile_like(paramx, paramy, chi_sq, nbins, bin_limits=None):
    """
    Maximizes the likelihood in each bin to obtain the profile likelihood and
    profile chi-squared.
    :param paramx: Data column of parameter x
    :type paramx: numpy.ndarray
    :param paramy: Data column of parameter y
    :type paramy: numpy.ndarray
    :param chi_sq: Data column of chi-squared
    :type chi_sq: numpy.ndarray
    :param nbins: Number of bins for histogram
    :type nbins: integer
    :param bin_limits: Bin limits for histogram
    :type bin_limits: list [[xmin,xmax],[ymin,ymax]]
    :returns: Profile chi squared, profile likelihood, x and y bin centers
    :rtype: named tuple (\
        profchi_sq: numpy.ndarray, \
        prof_like: numpy.ndarray, \
        bin_center_x: numpy.ndarray, \
        bin_center_y: numpy.ndarray)
    :Example:
    >>> nbins = 100
    >>> chi_sq, like, x, y = profile_like(data[2], data[3], data[0], nbins=nbins)
    >>> assert len(chi_sq) == nbins
    >>> assert len(like) == nbins
    >>> assert len(x) == nbins
    >>> assert len(y) == nbins
    """
    # Bin the data to find bin edges. nbins we discard the count
    _, bin_edges_x, bin_edges_y = np.histogram2d(
                                    paramx,
                                    paramy,
                                    nbins,
                                    range=bin_limits,
                                    weights=None)

    # Find centers of bins
    bin_center_x = 0.5 * (bin_edges_x[:-1] + bin_edges_x[1:])
    bin_center_y = 0.5 * (bin_edges_y[:-1] + bin_edges_y[1:])

    # Find bin number for each point in the chain
    bin_numbers_x = np.digitize(paramx, bin_edges_x)
    bin_numbers_y = np.digitize(paramy, bin_edges_y)

    # Shift bin numbers to account for outliers
    bin_numbers_x = shift(bin_numbers_x, nbins)
    bin_numbers_y = shift(bin_numbers_y, nbins)

    # Initialize the profiled chi-squared to something massive
    prof_chi_sq = np.full((nbins, nbins), float("inf"))

    # Minimize the chi-squared in each bin by looping over all the entries in
    # the chain.
    for index in range(chi_sq.size):
        bin_numbers = (bin_numbers_x[index], bin_numbers_y[index])
        if bin_numbers[0] is not None and bin_numbers[1] is not None and chi_sq[index] < prof_chi_sq[bin_numbers]:
            prof_chi_sq[bin_numbers] = chi_sq[index]

    # Subtract minimum chi-squared (i.e. minimum profile chi-squared is zero,
    # and maximum profile likelihood is one).
    prof_chi_sq = prof_chi_sq - prof_chi_sq.min()

    # Exponentiate to obtain profile likelihood
    prof_like = np.exp(- 0.5 * prof_chi_sq)

    return bin_center_x, bin_center_y, prof_chi_sq, prof_like

def getFuncOnGrid(dfDict_new, xkey, ykey, zkey, axisRange, gMesh, frequentist=True):
    
    #-- Set up preliminary parameters --#
    paramx = dfDict_new[xkey]
    paramy = dfDict_new[ykey]
    
    xmin, xmax, ymin, ymax = axisRange
    bin_limits = [[xmin, xmax], [ymin, ymax]]
    
    nbins = int(gMesh.imag)
    
    #-- Calculate function on grid depending on type --#
    if(zkey == 'posterior'):
        posterior = dfDict_new[zkey]
        X, Y, Z = posterior_pdf(paramx, paramy, posterior, nbins=nbins, bin_limits=bin_limits)
        
    elif(zkey == 'likelihood'):
        likelihood = dfDict_new[zkey]
        
        if(frequentist):
            chi_sq = -2.*np.log(likelihood)
        else:
            chi_sq = -2.*np.log((likelihood/likelihood.max()))
        
        X, Y, _, Z = profile_like(paramx, paramy, chi_sq, nbins, bin_limits=bin_limits)
        
       
    #-- Create dictionary of values --#
    gridDict = {}
    gridDict["X"] = X
    gridDict["Y"] = Y
    gridDict["Z"] = Z
    
    return gridDict

def smoothData(Z, sigma, order, normType='max'):
    
    from scipy.ndimage import gaussian_filter
    Z_smooth = gaussian_filter(Z, sigma=sigma, order=order) # sigma=1, order=0 is a good choice
    
    if normType == 'max':
        Z_smooth = Z_smooth/Z_smooth.max()
    elif normType == 'sum':    
        Z_smooth = Z_smooth/Z_smooth.sum()
    
    return Z_smooth

def calcCriticalDensity(pdf, alpha):
    
    # Normalize posterior pdf so that integral is one, if it wasn't already
    pdf = pdf / pdf.sum()
    
    # Minimize difference between amount of probability contained above a
    # particular density and that desired
    prob_desired = 1. - alpha

    def prob_contained(density):
        return ma.masked_where(pdf < density, pdf).sum()

    def delta_prob(density):
        return prob_contained(density) - prob_desired

    # Critical density cannot be greater than maximum posterior pdf and must
    # be greater than 0. The function delta_probability is monotonic on that
    # interval. Find critical density by bisection.
    critical_density = bisect(delta_prob, 0., pdf.max())

    return critical_density

def findCriticalDensityVals(pdf):
    """
    Finds 1 and 2 sigma critical density values
    alpha = 0.32 => 68% probability contained => 1 sigma curve
    alpha = 0.05 => 95% probability contained => 2 sigma curve
    """
    #-- Define alpha levels --#
    levels    = [2, 1] 
    alphaList = [0.05, 0.32]
    #! Develop fancier alpha given level function later
    
    #-- Calculate critical density --#
    critDensityList = []
    for aa in alphaList:
        critDensityList.append(calcCriticalDensity(pdf, aa))
        
    return critDensityList