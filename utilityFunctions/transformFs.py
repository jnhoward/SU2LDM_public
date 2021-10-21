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
    """   
#     FMatrix_new = np.einsum('ae,bf,cg,dh,efgh -> abcd',V,V,V,V,FMatrix_old, optimize='greedy')
    path = ['einsum_path', (0, 4), (0, 3), (0, 2), (0, 1)] # most efficient sum path
    FMatrix_new = np.einsum('ae,bf,cg,dh,efgh -> abcd',V,V,V,V,FMatrix_old, optimize=path)

    if (DEBUG):
        #! Other checks?
        print("")
        
    return FMatrix_new