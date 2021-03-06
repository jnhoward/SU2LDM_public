import numpy as np

def JacobianConvert2D(dfDict, CASE=1):
    """
    This code assumes a Ulysses scan in 'fpi_pow', 'bsmall_pow'. 
    
    This code performs a Jacobian transformation and returns a new df in different scan variables.
    - CASE==0 the variables returned are posterior_new, likelihood_new, fpi_pow,   bsmall_pow
    - CASE==1 the variables returned are posterior_new, likelihood_new, fpi_pow,   mD_pow
    - CASE==2 the variables returned are posterior_new, likelihood_new, fpi [GeV], mD [GeV]
    - CASE==3 the variables returned are posterior_new, likelihood_new, fpi [TeV], mD [GeV]
    - CASE==4 the variables returned are posterior_new, likelihood_new, fpi [TeV], mD [TeV]
    """
    #-- Check that dfDict includes posterior, likelihood, fpi_pow, bsmall_pow
    if(~np.all([True for x in ['posterior', 'likelihood', 'fpi_pow', 'bsmall_pow'] if x in list(dfDict.keys())])):
        print("Error: Incorrect dictionary format. Please make sure dictionary contains the following keys:")
        print("'posterior', 'likelihood', 'fpi_pow', 'bsmall_pow'")
        return
    
    #-- Check that CASE flag is appropriate --#    
    if(CASE not in np.array([0,1,2,3,4])):
        print("Error: Please select CASE = 0, 1, 2, 3, or 4")
        return
    
    #-- Set local definitions of scan parameters for convenience --#
    posterior  = dfDict["posterior"]
    likelihood = dfDict["likelihood"]
    fpi_pow    = dfDict["fpi_pow"]
    bsmall_pow = dfDict["bsmall_pow"]
    
    #-- Set new empty dictionary --#
    dfDict_new = {}
    
    #########################################################
    ### Transform data -- Case 0: No transformation
    ######################################################### 
    if(CASE==0):
        # In this case the jacobian is 1 so no transformation of probabilities is required
        dfDict_new["posterior"]  = posterior
        dfDict_new["likelihood"] = likelihood
        dfDict_new["jacobian"]   = np.ones(shape=posterior.shape)
        
        # fpi_pow -> fpi_pow
        dfDict_new["fpi_pow"] = fpi_pow
        
        # bsmall_pow -> mD_pow = log_10(mD/GeV)
        dfDict_new["bsmall_pow"] = bsmall_pow 

        # Return new dfDict
        return dfDict_new
    
    #########################################################
    ### Transform data -- Case 1: If plotting in log_10 still
    ######################################################### 
    if(CASE==1):
        # In this case the jacobian is 1 so no transformation of probabilities is required
        dfDict_new["posterior"]  = posterior
        dfDict_new["likelihood"] = likelihood
        dfDict_new["jacobian"]   = np.ones(shape=posterior.shape)
        
        # fpi_pow -> fpi_pow
        dfDict_new["fpi_pow"] = fpi_pow
        
        # bsmall_pow -> mD_pow = log_10(mD/GeV)
        dfDict_new["mD_pow"] = np.log10(4.*np.pi) + fpi_pow + bsmall_pow 

        # Return new dfDict
        return dfDict_new
    #########################################################
    ### Transform data -- Case 2: If NOT plotting in log_10
    #########################################################
    elif(CASE==2):
        
        # In this case the jacobian is NOT 1 so transformation of probabilities is required
        
        # fpi_pow -> fpi [GeV]
        fpi_GeV = 10.**(fpi_pow)
        
        # bsmall_pow -> mD [GeV]
        bsmall = 10.**bsmall_pow
        mD_GeV = 4.*np.pi*fpi_GeV*bsmall

        # Calculate Jacobian i.e. |\partial(fpi, mD)/\partial(fpi_pow, bsmall_pow)|
        J = (4*np.pi)*(np.log(10)**2)*(fpi_GeV**2)*bsmall

        # Store values in new dictionary
        dfDict_new["posterior"]  = posterior
        dfDict_new["likelihood"] = likelihood
        dfDict_new["jacobian"]   = J
        dfDict_new["fpi_GeV"]    = fpi_GeV
        dfDict_new["mD_GeV"]     = mD_GeV
        
        # Return new dfDict
        return dfDict_new
    ###########################################################################
    ### Transform data -- Case 3: If NOT plotting in log_10 and want fpi in TeV
    ###########################################################################
    elif(CASE==3):
        # In this case the jacobian is NOT 1 so transformation of probabilities is required
        
        # fpi_pow -> fpi [TeV]
        fpi_TeV    = (10.**-3)*10.**(fpi_pow)
        
        # bsmall_pow -> mD [GeV]
        bsmall = 10.**bsmall_pow
        mD_GeV     = 4.*np.pi*(10.**3)*fpi_TeV*bsmall

        # Calculate Jacobian i.e. |\partial(fpi, mD)/\partial(fpi_pow, bsmall_pow)|
        J = (4*np.pi)*(10**3)*(np.log(10)**2)*(fpi_TeV**2)*bsmall

        # Store values in new dictionary
        dfDict_new["posterior"]  = posterior
        dfDict_new["likelihood"] = likelihood
        dfDict_new["jacobian"]   = J
        dfDict_new["fpi_TeV"]    = fpi_TeV
        dfDict_new["mD_GeV"]     = mD_GeV

        # Return new dfDict
        return dfDict_new
    #######################################################################################
    ### Transform data -- Case 4: If NOT plotting in log_10 and want both fpi and mD in TeV
    #######################################################################################
    elif(CASE==4):
        # In this case the jacobian is NOT 1 so transformation of probabilities is required
        
        # fpi_pow -> fpi [TeV]
        fpi_TeV    = (10.**-3)*10.**(fpi_pow)
        
        # bsmall_pow -> mD [TeV]
        bsmall = 10.**bsmall_pow
        mD_TeV = 4.*np.pi*fpi_TeV*bsmall
        
        # Calculate Jacobian i.e. |\partial(fpi, mD)/\partial(fpi_pow, bsmall_pow)|
        J = fpi_TeV*mD_TeV*(np.log(10)**2)
        
        # Store values in new dictionary
        dfDict_new["posterior"]  = J*posterior
        dfDict_new["likelihood"] = likelihood
        dfDict_new["jacobian"]   = J
        dfDict_new["fpi_TeV"]    = fpi_TeV
        dfDict_new["mD_TeV"]     = mD_TeV

        # Return new dfDict
        return dfDict_new
        