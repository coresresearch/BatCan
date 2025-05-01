import numpy as np
names = ['Li','S8','Li2S','Li2S2','Li2S3','Li2S4','Li2S5','Li2S6','Li2S7','Li2S8']

# Start by inputting the molecule energies from DFT in hartrees
# Order: 
Li = -7.4327
S8 = -3181.9541
Li2S = -412.7447
Li2S2 = -810.5315
Li2S3 = -1208.3018
Li2S4 = -1606.0647
Li2S5 = -2003.8153
Li2S6 = -2401.5654
Li2S7 = -2799.3101
Li2S8 = -3197.0571

energies = np.array([Li,S8,Li2S,Li2S2,Li2S3,Li2S4,Li2S5,Li2S6,Li2S7,Li2S8])
# convert to eV
energies = energies*27.211384

# The stoich coeffs for the reactions are in the same order as the species in the energies vector
# Use the energies to calculate the heats of reactions for the gas phase
print("Delta E_gas  [eV]")
# Table 3 
nu_k_3 = np.array([[-1, -1/8, 0, 1/2, 0,0,0,0,0,0 ] , [-1, 0, 0, 2/3, 0,0,0,0,0,-1/6 ] , [-1, 0, 0, 3/4, 0,0,0,-1/4,0,0 ], [-1, 0, 0, 1, 0,-1/2,0,0,0,0 ]])
E_gas_3 = np.dot(nu_k_3, energies) # nu_k_3 @ energies 
print(np.round(E_gas_3,2))

# Table 4
nu_k_4 = (np.array([ [-1,-1/2,0,0,0,0,0,0,0,1/2 ] ,[-1,-1/4,0,1/4,0,0,0,1/4,0,0 ],  [-1,-1/4,0,0,1/4,0,1/4,0,0,0 ],
    [-1,-1/4,0,0,0,1/2,0,0,0,0 ] , [-1,-1/6,0,1/3,0,1/6,0,0,0,0 ] , [-1,-1/6,0,1/6,1/3,0,0,0,0,0 ] , 
    [-1,0,0,0,1/2,0,1/2,0,0,-1/2 ] , [-1,0,0,0,0,1,0,0,0,-1/2 ] , [-1,0,0,0,1/2,1/2,0,0,-1/2,0 ] ]) )
E_gas_4 = np.dot(nu_k_4, energies) # nu_k_4 @ energies 
print(np.round(E_gas_4,2))

# Table 5
nu_k_5 = (np.array([ [-1,-1/16,1/2,0,0,0,0,0,0,0] , [-1,0,4/7,0,0,0,0,0,0,-1/14] , [-1,0,7/12,0,0,0,0,0,-1/12,0] , [-1,0,3/5,0,0,0,0,-1/10,0,0]
                 , [-1,0,5/8,0,0,0,-1/8,0,0,0] , [-1,0,2/3,0,0,-1/6,0,0,0,0] , [-1,0,3/4,0,-1/4,0,0,0,0,0] , [-1,0,1,-1/2,0,0,0,0,0,0] ]) )
E_gas_5 = np.dot(nu_k_5, energies) # nu_k_5 @ energies 
print(np.round(E_gas_5,2))

# Use the heats of sublimations to calculate the heats of reactions for the solid phase
# E_solid = E_gas - E_sub
print(" Delta E_solid  [eV]")
E_sub_Li = 1.56
E_sub_S8 = 1.02
E_sub_Li2S = 4.23
E_sub_Li2S2 = E_sub_Li2S
E_sub_Li2S3 = E_sub_Li2S
E_sub_Li2S4 = E_sub_Li2S
E_sub_Li2S5 = E_sub_Li2S
E_sub_Li2S6 = E_sub_Li2S
E_sub_Li2S7 = E_sub_Li2S
E_sub_Li2S8 = E_sub_Li2S

E_subs = np.array([E_sub_Li,E_sub_S8,E_sub_Li2S,E_sub_Li2S2,E_sub_Li2S3,E_sub_Li2S4,E_sub_Li2S5,E_sub_Li2S6,E_sub_Li2S7,E_sub_Li2S8])

# Table 3
E_solid_3 = E_gas_3 - np.dot(nu_k_3, E_subs) 
print(np.round(E_solid_3,2))

# Table 4
E_solid_4 = E_gas_4 - np.dot(nu_k_4, E_subs) 
print(np.round(E_solid_4,2))

# Table 5
E_solid_5 = E_gas_5 - np.dot(nu_k_5, E_subs) 
print(np.round(E_solid_5,2))

# E_s = np.array([-2.84,-2.7,-2.58,-2.36,-3.28,-3.1,-3.24,-3.33,-3.0,-3.04,-3.2,-3.37,-3.16,-2.33,-2.19,-2.17,-2.12,-2.08,-2,-1.93,-1.82]) # values in the tables

def make_sqaure(A,b):
    '''
    inputs: 
        A - coefficient matrix 
        b - constant vector 

    Takes in a non square matrix A and removes linearly dependent rows then returns a square matrix
    it also removes the coresponding lines from b 
    '''
    while A.shape[0] > A.shape[1]:
        for i in range(A.shape[1]):
            test_A = np.delete(A, (i), axis=0)
            test_rank = np.linalg.matrix_rank(test_A)
            rank = np.linalg.matrix_rank(A)
            if test_rank == rank:
                A = test_A
                b = np.delete(b, (i), axis=0)
    return A, b

def plug_in_Li_S8(g_Li, g_S8, nu, E_s):
    '''
    inputs: 
        g_Li - gibs free energy of formation for aquious Li+
        g_S8 - gibs free energy of formation for aquious S8
        nu   - the stoich coeffs for the reactions
        E_s  - heat of reactions for the solid phase

    Plugs the values for the reference species Li and S8 into the heat of reaction equations 
    then removes thier stoich coeffs from the matrix of reaction coeffs. I am doing this so
    the gibs free energy of formation values will have a relative anchor 
    '''
    n_rows, n_cols = nu.shape
    nu_new = np.zeros((n_rows, n_cols - 2))
    E_s_new = E_s.copy()
    for ind, ele in enumerate(nu):
        E_s_new[ind] = E_s_new[ind] - ele[0]*g_Li - ele[1]*g_S8
        nu_new[ind] = np.delete(ele, [0,1])
    return nu_new, E_s_new

# the change in energy for the solid reactions in eV
E_solid = np.concatenate((E_solid_3, E_solid_4, E_solid_5), axis=0)
nu_k = np.concatenate((nu_k_3, nu_k_4, nu_k_5), axis=0)

F = 96.485
# convert to kJ/mol
E_solid = E_solid*F

g_Li_ion = -278 # kJ/mol
g_S8_elyte = 16.1 # kJ/mol

nu_k, E_solid = plug_in_Li_S8(g_Li_ion, g_S8_elyte, nu_k, E_solid)
nu_k, E_solid = make_sqaure(nu_k, E_solid)

# Solve for the enthalpies of formation using matrix inversion 
nu_k_inv = np.linalg.inv(nu_k)
g_f = np.dot(nu_k_inv,E_solid)
print("Delta G_f [kJ mol^-1]")
print(names[2:])
print(g_f)

# Check that the calculated values satisfy the reaction equations
G_f = np.concatenate((g_Li_ion,g_S8_elyte,g_f),axis=None)

E_solid = np.concatenate((E_solid_3, E_solid_4, E_solid_5), axis=0)
nu_k = np.concatenate((nu_k_3, nu_k_4, nu_k_5), axis=0)

E_react_calc = nu_k @ G_f
check = E_solid*F - E_react_calc
print("check")
print(check)