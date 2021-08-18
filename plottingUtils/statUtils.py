import numpy as np
from numpy import ma
from scipy.optimize import bisect

def estimateKDE_2D(dfDict, xkey, ykey):
    
    posterior  = dfDict["posterior"]
    likelihood = dfDict["likelihood"]
    
    X = dfDict[xkey]
    Y = dfDict[ykey]
    
    from scipy import stats
    values = np.vstack([X, Y])
    
    kernel_posterior  = stats.gaussian_kde(values, weights=posterior) 
    kernel_likelihood = stats.gaussian_kde(values, weights=likelihood)
    
    return kernel_posterior, kernel_likelihood

def evaluateKernelOnGrid(kernel, axisRange, gMesh=100j):
    """
    kernel:    KDE estimate of either posterior or likelihood
    axisRange: [xmin, xmax, ymin, ymax]
    gMesh: Number of meshgrid points, default is 100j -> 100x100 grid
    """
    xmin = axisRange[0]
    xmax = axisRange[1]
    ymin = axisRange[2] 
    ymax = axisRange[3]
    
    #-- Make meshgrid --#
    X, Y = np.mgrid[xmin:xmax:gMesh, ymin:ymax:gMesh] # gMesh x gMesh grid
    positions = np.vstack([X.ravel(), Y.ravel()])
    
    #-- Evaluate kernel on meshgrid --#
    pdf = np.reshape(kernel(positions).T, X.shape) # gMesh x gMesh grid of PDF values, one for each XY point
    
    #-- Make sure pdf is normalized properly --#
    pdf = pdf/pdf.sum()
    
    #-- Return dictionary with X, Y, pdf --#
    gridDict = {}
    gridDict["X"] = X
    gridDict["Y"] = Y
    gridDict["pdf"] = pdf
    
    return gridDict

# See https://github.com/michaelhb/superplot/blob/master/superplot/statslib/two_dim.py
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