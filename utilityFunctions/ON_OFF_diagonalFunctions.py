import numpy as np

#################################################################
##
## Functions related to the diagonalization of the mass matrix 
##
#################################################################

#-- Define necessary "OFF" (off-diagonal) functions --#
"""
${\rm OFF}_1 = - \frac{2 \sqrt{6}}{7} \frac{\Lambda_W^3 m_{\rm DM}}{f^2}$
${\rm OFF}(2) =   \frac{1}{16 \pi^2} C_W g_s^2 \Lambda_W^2$
${\rm OFF}(3) = - \frac{1}{16 \pi^2} C_W g_s^2 \Lambda_W^2$
"""

def off_1(f, lamW, mDM):
    const = (-1. * 2. * np.sqrt(6))/7.
    
    return (const*(lamW**3)*mDM)/(f**2)

def off_2(CW, gs, lamW):
    
    const = 1./(16.*(np.pi**2))
    
    return const*CW*(gs**2)*(lamW**2)

def off_3(CW, gs, lamW):
    
    return -off_2(CW, gs, lamW)

#-- Define necessary "ON" (on-diagonal) functions --#
"""
$\text{ON}_1 = \frac{2 \Lambda_W^3 m_{\rm DM}}{7 f^2}+56 \kappa \Lambda_W^2$
$\text{ON}(2) = -\frac{1}{32 \pi^2} C_A e_Q^2 \Lambda_W^2 -\frac{3}{32 \pi^2} C_G g_s^2 \Lambda_W^2
                +\frac{1}{16 \pi^2} C_W g_s^2 \Lambda_W^2+\frac{1}{48 \pi^2} C_Z e_Q^2 \Lambda_W^2
                -\frac{1}{32 \pi^2} C_Z e_Q^2 \Lambda_W^2 s^2_{Q}+\frac{1}{96 \pi^2} \frac{C_Z e_Q^2 \Lambda_W^2}{s^2_{Q}}$       
$\text{ON}(3) = 0$
$\text{ON}(4) = -\frac{1}{8 \pi^2}  C_A e_Q^2 \Lambda_W^2+\frac{1}{12 \pi^2} C_Z e_Q^2 \Lambda_W^2
                -\frac{1}{8 \pi^2}  C_Z e_Q^2 \Lambda_W^2 s^2_{Q}-\frac{1}{72 \pi^2} \frac{C_Z e_Q^2 \Lambda_W^2}{s^2_{Q}}$
$\text{ON}(5) = -\frac{1}{32 \pi^2} C_A e_Q^2 \Lambda_W^2-\frac{3}{32 \pi^2} C_G g_s^2 \Lambda_W^2
                +\frac{1}{48 \pi^2} C_Z e_Q^2 \Lambda_W^2-\frac{1}{32 \pi^2} C_Z e_Q^2 \Lambda_W^2 s^2_{Q}
                +\frac{1}{96 \pi^2} \frac{C_Z e_Q^2 \Lambda_W^2}{s^2_{Q}}$
$\text{ON}(6) = \frac{\Lambda_W^3 m_{\rm DM}}{f^2}$
$\text{ON}(7) = -\frac{1}{8 \pi^2}  C_A e_Q^2 \Lambda_W^2+\frac{1}{24 \pi^2} C_Z e_Q^2 \Lambda_W^2
                -\frac{1}{8 \pi^2}  C_Z e_Q^2 \Lambda_W^2 s^2_{Q}+\frac{\Lambda_W^3 m_{\rm DM}}{f^2}$
$\text{ON}(8) = -\frac{1}{4 \pi^2}  C_G g_s^2 \Lambda_W^2$
$\text{ON}(9) = -\frac{1}{32 \pi^2} C_A e_Q^2 \Lambda_W^2-\frac{3}{32 \pi^2} C_G g_s^2 \Lambda_W^2
                -\frac{1}{32 \pi^2} C_Z e_Q^2 \Lambda_W^2 s^2_{Q}+\frac{1}{288 \pi^2} \frac{C_Z e_Q^2 \Lambda_W^2}{s^2_{Q}}
                +\frac{\Lambda_W^3 m_{\rm DM}}{f^2}$
$\text{ON}_{10} = \frac{12 \Lambda_W^3 m_{\rm DM}}{7 f^2}$
"""
def on_1(f, lamW, mDM, kappa):
    const = 2./7.
    
    return (const*(lamW**3)*mDM)/(f**2) + 56.*kappa*(lamW**2)

def on_2(CA, CG, CW, CZ, eQ, gs, sQSq, lamW):
    
    piSq   = np.pi**2
    lamWSq = lamW**2
    eQSq   = eQ**2
    gsSq   = gs**2
    
    term1 = - (1./(32.*piSq))*CA*eQSq*lamWSq
    term2 = - (3./(32.*piSq))*CG*gsSq*lamWSq
    term3 =   (1./(16.*piSq))*CW*gsSq*lamWSq
    term4 =   (1./(48.*piSq))*CZ*eQSq*lamWSq
    term5 = - (1./(32.*piSq))*CZ*eQSq*lamWSq*sQSq
    term6 =   (1./(96.*piSq))*CZ*eQSq*lamWSq/sQSq
    
    return term1 + term2 + term3 + term4 + term5 + term6

def on_3():
    return 0.

def on_4(CA, CZ, eQ, sQSq, lamW):
    
    piSq   = np.pi**2
    lamWSq = lamW**2
    eQSq   = eQ**2
    
    term1 = - (1./(8.*piSq))*CA*eQSq*lamWSq
    term2 =   (1./(12.*piSq))*CZ*eQSq*lamWSq
    term3 = - (1./(8.*piSq))*CZ*eQSq*lamWSq*sQSq
    term4 =   (1./(72.*piSq))*CZ*eQSq*lamWSq/sQSq  
    
    return term1 + term2 + term3 + term4

def on_5(CA, CG, CZ, eQ, gs, sQSq, lamW):
    
    piSq   = np.pi**2
    lamWSq = lamW**2
    eQSq   = eQ**2
    gsSq   = gs**2
    
    term1 = - (1./(32.*piSq))*CA*eQSq*lamWSq
    term2 = - (3./(32.*piSq))*CG*gsSq*lamWSq
    term3 =   (1./(48.*piSq))*CZ*eQSq*lamWSq
    term4 = - (1./(32.*piSq))*CZ*eQSq*lamWSq*sQSq
    term5 =   (1./(96.*piSq))*CZ*eQSq*lamWSq/sQSq
    
    return term1 + term2 + term3 + term4 + term5

def on_6(f, lamW, mDM):
    
    return ((lamW**3)*mDM)/(f**2)

def on_7(CA, CZ, eQ, sQSq, lamW, f, mDM):
    
    piSq   = np.pi**2
    lamWSq = lamW**2
    eQSq   = eQ**2
    
    term1 = - (1./(8.*piSq))*CA*eQSq*lamWSq
    term2 =   (1./(24.*piSq))*CZ*eQSq*lamWSq
    term3 = - (1./(8.*piSq))*CZ*eQSq*lamWSq*sQSq
    term4 =   ((lamW**3)*mDM)/(f**2) 
    
    return term1 + term2 + term3 + term4

def on_8(CG, gs, lamW):
    
    const = - 1./(4.*(np.pi**2))
    
    return const*CG*(gs**2)*(lamW**2)

def on_9(CA, CG, CZ, eQ, gs, sQSq, lamW, f, mDM):
    
    piSq   = np.pi**2
    lamWSq = lamW**2
    eQSq   = eQ**2
    gsSq   = gs**2
    
    term1 = - (1./(32.*piSq))*CA*eQSq*lamWSq
    term2 = - (3./(32.*piSq))*CG*gsSq*lamWSq
    term4 = - (1./(32.*piSq))*CZ*eQSq*lamWSq*sQSq
    term5 =   (1./(288.*piSq))*CZ*eQSq*lamWSq/sQSq
    term6 = ((lamW**3)*mDM)/(f**2) 
    
    return term1 + term2 + term4 + term5 + term6

def on_10(f, lamW, mDM):
    const = 12./7.
    
    return (const*(lamW**3)*mDM)/(f**2)