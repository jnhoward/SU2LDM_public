import numpy as np
from omegaH2 import omegaH2 

###################################################################################################
##
##  How to run from command line:
##
##
##  $ from calcAeffOnGrid import main
##  $ main(Ngen=1, BP=1, CASE=4, gMesh=10j, axisRange=[0.5, 8.5, 42, 78])
##
###################################################################################################


def calcAeffOnGrid(axisRange, gMesh, kwargs, AEFFPATH, CASE=4, COUNTER=10.):

    #-- Set up grid --#
    xmin, xmax, ymin, ymax = axisRange[0], axisRange[1], axisRange[2], axisRange[3]
    X, Y = np.mgrid[xmin:xmax:gMesh, ymin:ymax:gMesh] # gMesh x gMesh grid
    positions = np.vstack([X.ravel(), Y.ravel()]).T
    
    #-- Loop over grid points, store results in a list --#
    m1List = []
    aeffList = []
    
    i = 0 # Counter for print statements
    imax = positions.shape[0] # Total number of grid points to evaluate
    n = int(imax/COUNTER) # Print statement after COUNTER many points calculated

    for x, y in positions:

        if i % n == 0:
            print("Calculating data point %d out of %d"%(i,imax))

        if CASE == 0:
            # x = bsmall_pow, y = fpi_pow
            kwargs["fpi"]    = 10.**y # GeV
            kwargs["bsmall"] = 10.**x
        elif CASE == 1:
            # x = mD_pow, y = fpi_pow
            kwargs["fpi"]    = 10.**y # GeV
            mD               = 10.**x
            kwargs["bsmall"] = mD/(4.*np.pi*kwargs["fpi"])
        elif CASE == 2:
            # x = mD GeV, y = fpi GeV
            kwargs["fpi"]    = y # GeV
            kwargs["bsmall"] = x/(4.*np.pi*kwargs["fpi"])
        elif CASE == 3:
            # x = mD GeV, y = fpi TeV
            kwargs["fpi"]    = fpi*1000     # Convert to GeV
            kwargs["bsmall"] = x/(4.*np.pi*kwargs["fpi"])
        elif CASE ==4:
            # x = mD TeV, y = fpi TeV
            mD               = x*1000  # Convert to GeV
            kwargs["fpi"]    = y*1000     # Convert to GeV
            kwargs["bsmall"] = mD/(4.*np.pi*kwargs["fpi"])
        else:
            print("Error: Invalid CASE, select either 0,1,2,3, or 4.")
            return

        m1, aeff = omegaH2(**kwargs, RETURN='m1_aeff') #calcM1Aeff(**kwargs) 

        m1List.append(m1)
        aeffList.append(aeff.real)

        i = i+1
    
    #-- Save arrays to files --#
    # X, Y, m1, aeff
    print("Saving to file at %s"%AEFFPATH)
    np.save(AEFFPATH, [X, Y, np.reshape(np.array(m1List), X.shape ), np.reshape(np.array(aeffList), X.shape )])

def main(Ngen, BP, CASE, gMesh, axisRange):
    
    if BP == 1:
        kwargs = {'gs':0.8, 'kappa':0.0, 'eQ':0.5, 'sQsq':0.3}
        if Ngen==1:
            kwargs['Ngen'] = 1
            AEFFPATH  = 'Data/npyFiles/aeffOnGrid_Ngen1_BP1.npy'
        elif Ngen==3:
            kwargs['Ngen'] = 3
            AEFFPATH  = 'Data/npyFiles/aeffOnGrid_Ngen3_BP1.npy'
        else:
            print("Error: Invalid Ngen, must be 1 or 3.")
            return
            
    elif BP == 2:
        kwargs = {'gs':0.1, 'kappa':0.0, 'eQ':0.01,'sQsq':0.01}
        if Ngen==1:
            kwargs['Ngen'] = 1
            AEFFPATH  = 'Data/npyFiles/aeffOnGrid_Ngen1_BP2.npy'
        elif Ngen==3:
            kwargs['Ngen'] = 3
            AEFFPATH  = 'Data/npyFiles/aeffOnGrid_Ngen3_BP2.npy'
        else:
            print("Error: Invalid Ngen, must be 1 or 3.")
            return
    else:
        print("Error: Invalid BP, must be 1 or 2.")
        return
    
    COUNTER = int(gMesh.imag)
    
    calcAeffOnGrid(axisRange, gMesh, kwargs, AEFFPATH, CASE, COUNTER)
    print("Finished successfully!")
    
if __name__ == "__main__":
    
    main(Ngen=1, BP=1, CASE=4, gMesh=10j, axisRange=[0.5, 8.5, 42, 78])