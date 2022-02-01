import numpy as np

#-- Calculate factors from Feynman diagrams --#
def calcDiagramFactors(a, b, c, d, F1Mat, F2Mat):
    G1 = F1Mat[a, b, c, d]
    G2 = F1Mat[d, b, c, a]
    G3 = F1Mat[c, d, a, b]
    G4 = F1Mat[a, c, b, d]
    G5 = F1Mat[a, d, c, b] 
    G6 = F1Mat[c, b, a, d] 
    G7 = F2Mat[a, b, c, d]
    
    return G1, G2, G3, G4, G5, G6, G7

#-- Calculate components of cross section --#
def calcCrossSectionConstants(a, b, c, d, s, M2, F1Mat, F2Mat):
  
    G1, G2, G3, G4, G5, G6, G7 = calcDiagramFactors(a, b, c, d, F1Mat, F2Mat)
    
    C = 0.5*( 2*G7 + M2[a]*(G2 + G3 - G5) + M2[b]*(G3 + G4 - G6) + M2[c]*(G1 + G2 - G6) + M2[d]*(G1 + G4 - G5) )
    Ccc = C.conjugate()
    
    G13_56   = G1 + G3 - G5 - G6
    G13_56cc = G13_56.conjugate()
    G24_56   = G2 + G4 - G5 - G6
    G24_56cc = G24_56.conjugate()
  
    Cconst  = C*Ccc + 0.25*(G13_56*G13_56cc)*(s**2) - 0.5*s*(Ccc*G13_56 + C*G13_56cc)    # Factor in front of O(t^0) term
    Clin    = 0.25*s*(G13_56*G24_56cc + G13_56cc*G24_56) - 0.5*(Ccc*G24_56 + C*G24_56cc) # Factor in front of O(t^1) term
    Cquad   = 0.25*(G24_56*G24_56cc)                                                     # Factor in front of O(t^2) term
    
    return Cconst, Clin, Cquad

#-- Define lambda kinematic function --#
def lambdaFunc(s, c, d, M2):
    lambdaVal = s**2 - 2*(M2[c] + M2[d])*s + (M2[c] - M2[d])**2
    return lambdaVal

#-- Define useful notational constant for describing bounds of the t integral --#
def W1(s, a, b, c, d, M2):
    assert s != 0.
    W1 = M2[a] + M2[c] - (1/(2.*s))*(s + M2[a] - M2[b])*(s + M2[c] - M2[d])
    return W1

#-- Calculate the cross section for 2-to-2 reactions of the form \Pi^+_a \Pi^-_b -> \Pi^0_c \Pi^0_d --#
def calcCrossSection(a, b, c, d, M2, F1Mat, F2Mat, DEBUG):

    # Create Mass array
    massArr = np.sqrt(M2)

    # Calculate s for v=0 condition
    s = (massArr[a] + massArr[b])**2
    assert s != 0.

    # Calculate lambda
    lam = lambdaFunc(s, c, d, M2)
    if (DEBUG): 
        print('Lambda:',lam)

    # Calculate coeffs
    Cconst, Clin, Cquad = calcCrossSectionConstants(a, b, c, d, s, M2, F1Mat, F2Mat)
    if (DEBUG): 
        print('Cconst, Clin, Cquad: ',Cconst, Clin, Cquad)

    # Calculate W1
    W1val = W1(s, a, b, c, d, M2)
    if (DEBUG): 
        print('W1: ',W1val)

    # Calculate s-wave Cross Section
    assert massArr[a] != 0. and massArr[b] != 0.
   
    if(lam<=0.): 
        sig = 0j
    else: 
        sig = (1./(32.*np.pi*massArr[a]*massArr[b]))*(np.sqrt(lam)/s)*(Cconst + Clin*W1val + Cquad*(W1val**2))

    small_num = 1e-20 # Impose lower threshold on value of sig 
    if (np.abs(sig) < small_num):
        return 0j
    else:
        return sig
