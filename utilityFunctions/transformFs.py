import numpy as np

#################################################################
##
## Transform F (or Fhat) Matrices into definite new basis
##
#################################################################

#-- Transform FMatrix into new basis --#
def transformF(V, FMatrix_old, DEBUG=True):
    """
    V: Transformation matrix such that a pion (P) old basis is expressed in the new basis as P^{old}_a = V_ab P^{new}_b.
    F: The tensor we would like to transform F(a,b,c,d) into the new basis.
    
    Note that here we pre-compute the most efficient sum path for a problem with this structure. 
    This is found via the following code:
  
        n = 15 # The path is the same whether this is 15 (Ngen=1) or 91 (Ngen=3)
        FMat = np.zeros(shape=(n,n,n,n))
        V    = np.zeros(shape=(n,n))
        path, _ = np.einsum_path('ae,bf,cg,dh,efgh -> abcd',V,V,V,V,FMat)
    
    The resulting path is as follows:
        print(path)
        >> ['einsum_path', (0, 4), (0, 3), (0, 2), (0, 1)]
    which we use below.
    """   

    path = ['einsum_path', (0, 4), (0, 3), (0, 2), (0, 1)] # Most efficient sum path 
    FMatrix_new = np.einsum('ae,bf,cg,dh,abcd -> efgh',V,V,V,V,FMatrix_old, optimize=path)

    if (DEBUG):
        #! Other checks?
        print("")
        
    return FMatrix_new