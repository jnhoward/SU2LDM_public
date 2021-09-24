import numpy as np
from scipy.linalg import block_diag

####################################################################
##
## Calculate X and A matrices needed for cross section calculation
##
####################################################################

#-- Check that Xa's satisfy normalization condition --#
def normalityCheck(X):
    # X is an n x n numpy ndarray
  
    X_dagger = X.conjugate().T
    X2 = np.matmul(X, X_dagger)

    return np.trace(X2)

#-- Calculate A matrix --#
def calcA(Ngen, DEBUG=True):
    
    if(Ngen==1):
        dimMat = 2*3*Ngen   #= 6
    elif(Ngen==3):
        dimMat = 4*Ngen + 2 #= 14
    else:
        print("Error: Invalid Ngen. Please use either Ngen=1 or Ngen=3.")
        return 0
        
    A = np.zeros((dimMat, dimMat), dtype=complex)
    A[-1,-1] = 1
    A[-2,-2] = 1
    
    if(DEBUG):
        print("Ngen = ", Ngen)
        print("A Matrix: ")
        print(A)
    
    return A

#-- Calculate X matrix if Ngen == 1 --#
def calcXs_1gen(Ngen, DEBUG=True):

    #-- Set constants --# 
    Npsi   = 6*Ngen   # Number of dirac fermions
    Nf     = 3*Ngen   # Number of Weyl fermions
    dimMat = 2*Nf     # Dimension of X matrices
    
    dimSUN = 4*Nf*Nf - 1     # Number of SUN generators
    dimSpN = 2*Nf*Nf + Nf    # Number of SpN generators
    dimXa  = dimSUN - dimSpN # Number of broken generators (Xa)
    
    #-- Make Sigma Matrix --#
    Sigma0_2x2 = np.array([[0,1],[-1,0]])

    for i in range(Nf):
        if i==0:
            Sigma0 = Sigma0_2x2
        else:
            Sigma0 = block_diag(Sigma0, Sigma0_2x2) 

    if (DEBUG):
        print("Sigma0: ")
        print(Sigma0)
        print("")

    #-- Create arrays to help make Xas --#
    offDiagList = np.array([[i,j] for i in range(int(dimMat/2)) for j in range(i+1, int(dimMat/2))])
    
    if (DEBUG):
        print("Off Diagonal List:")
        print(offDiagList)
        print("")

    pauli0 = (1./np.sqrt(8.))*np.array([[1,0],[0,1]])
    pauli1 = (1./np.sqrt(8.))*np.array([[0,0+1j],[0+1j,0]])
    pauli2 = (1./np.sqrt(8.))*np.array([[0,1],[-1,0]])
    pauli3 = (1./np.sqrt(8.))*np.array([[0+1j,0],[0,0-1j]])

    pauliList = [pauli0, pauli1, pauli2, pauli3]

    if (DEBUG):
        print("Pauli Matrix List:")
        print(len(pauliList))
        print(pauliList[0])
        print(pauliList[1])
        print(pauliList[2])
        print(pauliList[3])
        
    #-- Make Xa's --#
    X=[]

    for a in range(len(offDiagList)*len(pauliList)):

        Xdummy = np.zeros((dimMat, dimMat), dtype=complex)

        # Make 0th entry a dummy entry (to help with eta' notation)
        if (a==0):
            X.append(Xdummy)

        # Get quotient and remainder: m/n = Q*n + R
        elem1 = a//len(pauliList) # Q
        elem2 = a%len(pauliList)  # R

        # Get corresponding elements
        item1 = offDiagList[elem1]
        item2 = pauliList[elem2]

        # Form X array
        Xdummy[2*item1[0], 2*item1[1]]     = item2[0,0]
        Xdummy[2*item1[0], 2*item1[1]+1]   = item2[0,1]
        Xdummy[2*item1[0]+1, 2*item1[1]]   = item2[1,0]
        Xdummy[2*item1[0]+1, 2*item1[1]+1] = item2[1,1]

        Xdummy[2*item1[1], 2*item1[0]]     = np.conj(item2[0,0])
        Xdummy[2*item1[1]+1, 2*item1[0]]   = np.conj(item2[0,1])
        Xdummy[2*item1[1], 2*item1[0]+1]   = np.conj(item2[1,0])
        Xdummy[2*item1[1]+1, 2*item1[0]+1] = np.conj(item2[1,1])

        X.append(Xdummy)

        if (a==1 and DEBUG):
            print("elem1, elem2, item1, item2: ")
            print(elem1)
            print(elem2)
            print(item1)
            print(item2)
            print("")

            print("Sparse Array coordinates and values:")
            print(2*item1[0], 2*item1[1], item2[0,0])
            print(2*item1[0], 2*item1[1]+1, item2[0,1])
            print(2*item1[0]+1, 2*item1[1], item2[1,0])
            print(2*item1[0]+1, 2*item1[1]+1, item2[1,1])
            print("")
            print(2*item1[1], 2*item1[0], np.conj(item2[0,0]))
            print(2*item1[1]+1, 2*item1[0], np.conj(item2[0,1]))
            print(2*item1[1], 2*item1[0]+1, np.conj(item2[1,0]))
            print(2*item1[1]+1, 2*item1[0]+1, np.conj(item2[1,1]))
            print("")

    X.append((1./np.sqrt(8.))*np.diag([1., 1., -1., -1., 0., 0.]))
    X.append((1./np.sqrt(8.*Nf))*np.diag([1., 1., 1., 1., -2., -2.]))
    
    # Check that all Xs meet normalization conditions
    if (DEBUG):
        for a in range(1, dimXa+1):
            assert (np.isclose(normalityCheck(X[a]), 0.5))

            print("X a = X",a)
            print("Pass normalization condition? ", np.isclose(normalityCheck(X[a]),0.5))
            print("Value: ",normalityCheck(X[0]))
    
    return X

#-- Calculate X matrix if Ngen == 3 --#
def calcXs_3gen(Ngen, DEBUG=True):
    
    #-- Set constants --# 
    Npsi   = 4*Ngen + 2  # Number of dirac fermions
    Nf     = 2*Ngen + 1  # Number of Weyl fermions
    dimMat = 2*Nf        # Dimension of X matrices
    
    dimSUN = 4*Nf*Nf - 1     # Number of SUN generators
    dimSpN = 2*Nf*Nf + Nf    # Number of SpN generators
    dimXa  = dimSUN - dimSpN # Number of broken generators (Xa)
    
    #-- Make Sigma Matrix --#
    Sigma0_2x2 = np.array([[0,1],[-1,0]])

    for i in range(int(Nf)):
        if i==0:
            Sigma0 = Sigma0_2x2
        else:
            Sigma0 = block_diag(Sigma0, Sigma0_2x2)
    
    if (DEBUG):
        print("Sigma0: ")
        print(Sigma0)
        print("")
        
    #-- Create arrays to help make Xas --#
    offDiagList = np.array([[i,j] for i in range(int(dimMat/2)) for j in range(i+1, int(dimMat/2))])

    if (DEBUG):
        print("Off Diagonal List:")
        print(offDiagList)
        print("")

    pauli0 = (1./np.sqrt(8.))*np.array([[1,0],[0,1]])
    pauli1 = (1./np.sqrt(8.))*np.array([[0,0+1j],[0+1j,0]])
    pauli2 = (1./np.sqrt(8.))*np.array([[0,1],[-1,0]])
    pauli3 = (1./np.sqrt(8.))*np.array([[0+1j,0],[0,0-1j]])

    pauliList = [pauli0, pauli1, pauli2, pauli3]

    if (DEBUG):
        print("Pauli Matrix List:")
        print(len(pauliList))
        print(pauliList[0])
        print(pauliList[1])
        print(pauliList[2])
        print(pauliList[3])

    #-- Make Xa's --#
    X=[]

    for a in range(len(offDiagList)*len(pauliList)):

        Xdummy = np.zeros((int(dimMat), int(dimMat)), dtype=complex)

        # Make 0th entry a dummy entry (to help with eta' notation)
        if (a==0):
            X.append(Xdummy)

        # Get quotient and remainder: m/n = Q*n + R
        elem1 = a//len(pauliList) # Q
        elem2 = a%len(pauliList)  # R

        # Get corresponding elements
        item1 = offDiagList[elem1]
        item2 = pauliList[elem2]

        # Form X array
        Xdummy[2*item1[0], 2*item1[1]]     = item2[0,0]
        Xdummy[2*item1[0], 2*item1[1]+1]   = item2[0,1]
        Xdummy[2*item1[0]+1, 2*item1[1]]   = item2[1,0]
        Xdummy[2*item1[0]+1, 2*item1[1]+1] = item2[1,1]

        Xdummy[2*item1[1], 2*item1[0]]     = np.conj(item2[0,0])
        Xdummy[2*item1[1]+1, 2*item1[0]]   = np.conj(item2[0,1])
        Xdummy[2*item1[1], 2*item1[0]+1]   = np.conj(item2[1,0])
        Xdummy[2*item1[1]+1, 2*item1[0]+1] = np.conj(item2[1,1])

        X.append(Xdummy)
        
        if (a==1 and DEBUG):
            print("elem1, elem2, item1, item2: ")
            print(elem1)
            print(elem2)
            print(item1)
            print(item2)
            print("")

            print("Sparse Array coordinates and values:")
            print(2*item1[0], 2*item1[1], item2[0,0])
            print(2*item1[0], 2*item1[1]+1, item2[0,1])
            print(2*item1[0]+1, 2*item1[1], item2[1,0])
            print(2*item1[0]+1, 2*item1[1]+1, item2[1,1])
            print("")
            print(2*item1[1], 2*item1[0], np.conj(item2[0,0]))
            print(2*item1[1]+1, 2*item1[0], np.conj(item2[0,1]))
            print(2*item1[1], 2*item1[0]+1, np.conj(item2[1,0]))
            print(2*item1[1]+1, 2*item1[0]+1, np.conj(item2[1,1]))
            print("")

        
    X.append(np.diag([0.353553, 0.353553, -0.353553, -0.353553,
                      0.,0.,0.,0.,0.,0.,0.,0.,0.,0.]))
    X.append(np.diag([0.204124, 0.204124, 0.204124, 0.204124,-0.408248,-0.408248,
                      0.,0.,0.,0.,0.,0.,0.,0.]))
    X.append(np.diag([0.144338, 0.144338, 0.144338, 0.144338,0.144338,0.144338,
                      -0.433013,-0.433013,0.,0.,0.,0.,0.,0.]))
    X.append(np.diag([0.111803, 0.111803, 0.111803, 0.111803,0.111803,0.111803,
                      0.111803,0.111803,-0.447214,-0.447214,0.,0.,0.,0.]))
    X.append(np.diag([0.0912871, 0.0912871, 0.0912871, 0.0912871,0.0912871,
                      0.0912871,0.0912871,0.0912871,0.0912871,0.0912871,-0.456435,
                      -0.456435,0.,0.]))
    X.append(np.diag([0.0771517, 0.0771517, 0.0771517,0.0771517,0.0771517,0.0771517,
                      0.0771517,0.0771517,0.0771517,0.0771517,0.0771517,0.0771517,
                      -0.46291,-0.46291]))

    # Check that all Xs meet normalization conditions
    if (DEBUG):
        for a in range(1, dimXa+1):
            assert (np.isclose(normalityCheck(X[a]), 0.5))

            print("X a = X",a)
            print("Pass normalization condition? ", np.isclose(normalityCheck(X[a]),0.5))
            print("Value: ",normalityCheck(X[0]))

    return X

#-- Calculate X matrix --#
def calcXs(Ngen, DEBUG=True):
    
    if(Ngen==1):
        X = calcXs_1gen(Ngen, DEBUG=DEBUG)
    elif(Ngen==3):
        X = calcXs_3gen(Ngen, DEBUG=DEBUG)
    else:
        print("Error: Invalid Ngen. Please use either Ngen=1 or Ngen=3.")
        return 0
    
    return X