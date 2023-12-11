from pkgutil import ImpImporter
import matplotlib.pyplot as plt
import numpy as np
import math
from itertools import permutations
from scipy.optimize import curve_fit
import scipy as sp

"""
Utility functions that can be divided into 2 functionalities.
1) To calculate computational resources given a Dn vector
2) Create a Dn vector after some set of rules.
"""
###############################
#    RESOURCE FUNCTIONS       #
###############################

# Number of qubits.

def OHq(D, model = 'rotamer'):
    """
    Input: Dn vector
    Output: Number of qubits requried with OH
    """
    if model == 'rotamer' or model == 'coordinate':
        return np.sum(D)
    elif model == 'directional':
        N = len(D)
        standard_qubits = np.sum(D)
        aux_qubits = 0
        pair_qubits = 0
        if N > 4:
            for i in range(N-4):
                for j in range(i+4, N):
                    aux_qubits += np.abs(i-j)*((1+i-j)%2)
        if N > 3:
            for j in range(N-3):
                for k in range(N):
                    pair_qubits += (j-k)%2
        print(standard_qubits, aux_qubits, pair_qubits)
        return standard_qubits + aux_qubits + pair_qubits
    else:
        return np.sum(D)

def BINq(D, model = 'rotamer'):
    """
    Input: Dn vector
    Output: Number of qubits requried with BIN
    """
    if model == 'rotamer' or model == 'coordinate':
        return int(np.sum([np.ceil(np.log2(d)) for d in D]))
    
    elif model == 'directional':
        N = len(D)
        standard_qubits = int(np.sum([np.ceil(np.log2(d)) for d in D]))
        aux_qubits = 0
        pair_qubits = 0

        if N > 4:
            for i in range(N-4):
                for j in range(i+4, N):
                    aux_qubits += np.ceil(np.log2(np.abs(i-j)))*((1+i-j)%2)

        if N > 3:
            for j in range(N-3):
                for k in range(N):
                    pair_qubits += (j-k)%2
        print(standard_qubits, aux_qubits, pair_qubits)
        return standard_qubits + aux_qubits + pair_qubits
    else:
        return int(np.sum([np.ceil(np.log2(d)) for d in D]))
    
def BUGq(D, g = 3, model = 'rotamer'):

    if model == 'rotamer' or model == 'coordinate':
        return int(np.ceil(np.log2(g+1))*np.sum(np.ceil(D/g)))
    
    elif model == 'directional':
        standard_qubits = int(np.ceil(np.log2(g+1))*np.sum(np.ceil(D/g)))
        N = len(D)
        aux_qubits = 0
        pair_qubits = 0
        
        if N > 4:
            for i in range(N-4):
                for j in range(i+4, N):
                    aux_qubits += np.ceil(np.abs(i-j)/g)*np.ceil(np.log2(g+1))*((1+i-j)%2)
      
        if N > 3:
            for j in range(N-3):
                for k in range(N):
                    pair_qubits += (j-k)%2
        return standard_qubits + aux_qubits + pair_qubits
    else:
        return int(np.ceil(np.log2(g+1))*np.sum(np.ceil(D/g)))

# Solution set

def feasible_set_coord(D):
    # two first elements factorial times each other
    return np.math.factorial(int(D[0]))*np.math.factorial(int(D[1]))

def OHunfeasable(D, model):
    if model == 'directional' or model == 'rotamer':
        sum = 1
        for rot in D:
            current_feas = rot/(2**rot)
            if np.isinf(current_feas):
                return 1
            sum *= current_feas
        return 1-sum # returns unfeasible div by total sol set
    
    elif model == 'coordinate':
        feasible_set_size = feasible_set_coord(D)
        num_qubits = np.sum(D)
        solution_set_size = 2**num_qubits
        return 1 - (feasible_set_size/solution_set_size)

def BINunfeasable(D, model = 'rotamer'):
    sum = 1
    if model == 'directional' or model == 'rotamer':
        for rot in D:
            sum *= rot/(2**np.ceil(np.log2(rot)))
        return 1-sum
    
    elif model == 'coordinate':
        feasible_set_size = feasible_set_coord(D)
        num_qubits = np.sum([np.ceil(np.log2(i)) for i in D])
        solution_set_size = 2**num_qubits
        return 1 - (feasible_set_size/solution_set_size)
    
def BUGunfeasable(D, g = 3, model = 'rotamer'):
    if model == 'directional' or model == 'rotamer':
        sum = 1
        for rot in D:
            #sum *= (3*(np.ceil(rot/3)-1)+mod1(rot,3))/(2**(2*np.ceil(rot/3)))
            sum *= rot/(2**(2*np.ceil(rot/g)))
        return 1-sum
    
    elif model == 'coordinate':
        feasible_set_size = feasible_set_coord(D)
        num_qubits = np.sum([np.ceil(i/g)*np.ceil(np.log2(g+1)) for i in D])
        solution_set_size = 2**num_qubits
        return 1 - (feasible_set_size/solution_set_size)

def mod1(a, d):
    return int(a-d*np.floor((a-1)/d))

# Decomposition

def BINq_with_decomp(D, gate_dict):
    """
    Input: Dn vector
    Output: Number of qubits requried with BIN after 
            decomposing all 3- or higher order qubitgates into 2- and 1-qubit gates
    """    
    chi = BINq(D)
    qubits = chi
    for key, val in gate_dict.items():
        if key not in [1,2]:
            qubits += val*(2*key-5)
    return qubits

def BINq_with_decomp_3(D, gate_dict): 
    """
    Input: Dn vector
    Output: Number of qubits requried with BIN after 
            decomposing all 4- or higher order qubit gates into 3-qubit gates
    """       
    chi = BINq(D)
    qubits = chi
    for key, val in gate_dict.items():
        if key not in [1,2,3]:
            qubits += val*(key-3)
    return qubits

def BUGq_with_decomp(D, gate_dict, g):
    """
    Input: Dn vector
    Output: Number of qubits requried with BIN after 
            decomposing all 3- or higher order qubitgates into 2- and 1-qubit gates
    """    
    qubits = BUGq(D, g)
    for key, val in gate_dict.items():
        if key not in [1,2]:
            qubits += val*(2*key-5)
    return qubits

def BUGq_with_decomp_3(D, gate_dict, g): 
    """
    Input: Dn vector
    Output: Number of qubits requried with BIN after 
            decomposing all 4- or higher order qubit gates into 3-qubit gates
    """
    qubits = BUGq(D, g)
    for key, val in gate_dict.items():
        if key not in [1,2,3]:
            qubits += val*(key-3)
    return qubits

# Number of gates

def BINlocality(D):
    """
    Input: Dn vector
    Output: Locality with BIN
    """
    D = np.sort(D)
    if len(D)>1:
        return np.ceil(np.log2(D[-1]))+np.ceil(np.log2(D[-2]))
    else:
        return np.ceil(np.log2(max(D)))

def OHgates(D):
    """
    Input: Dn vector
    Output: Number of gates requried in OH
    """    
    n = OHq(D)
    return [n, math.comb(int(n),2)]

def double_counted_BIN_gates(D,p):
    sum = 0
    for d in D:
        d =int(np.ceil(np.log2(d)))
        #for p in range(3,d+1):
        sum+=math.comb(d,p)*max(len(D)-2,0)
    return sum

def BINgates(D):
    """
    Input: Dn vector
    Output: Number gates requried in BIN and 
            number of 1- and 2- qubit gates required in BIN 
    """    
    gate_dict = {}
    chi = BINq(D)
    gate_dict[1] = chi

    gate_dict[2] = math.comb(chi,2)
    for i in range(len(D)):
        j = i+1
        while j<len(D):
            N = int(np.ceil(np.log2(D[i]))+np.ceil(np.log2(D[j])))
            if N >= 3:
                for p in range(3,N+1):
                    
                    many_gates = math.comb(N,p)
                    
                    if gate_dict.get(p):
                        gate_dict[p] += many_gates
                    else:
                        gate_dict[p] = many_gates
            j+=1
    
    n_gates = 0
    doubled = 0
    for key, val in gate_dict.items():
        
        if key not in [1,2]:
            doubled += double_counted_BIN_gates(D,key)
            gate_dict[key] -= double_counted_BIN_gates(D,key)
            n_gates += gate_dict[key]
    n_gates += gate_dict[1] + gate_dict[2]
    return n_gates, gate_dict, doubled

def double_counted_BUG_gates(num_of_blocks, p, g):
    if np.ceil(np.log2(g+1))>=p:
        return (num_of_blocks-2)*math.comb(int(np.ceil(np.log2(g+1))), p)
    else:
        return 0

def BUGgates(D,g=3):
    """
    Input: Dn vector
    Output: Number gates requried in BUgray_g and 
            number of 1- and 2- qubit gates required in BIN 
    """    
    num_of_blocks = int(sum([np.ceil(d/g) for d in D]))
    num_of_block_combos = int(math.comb(num_of_blocks,2))
    gate_dict = {}
    qub = BUGq(D,g)
    gate_dict[1] = qub

    gate_dict[2] = math.comb(qub, 2) #num_of_block_combos*math.comb(int(2*np.ceil(np.log2(g+1))), 2)
    for g_val in range(3, int(2*np.ceil(np.log2(g+1)))+1):
    #gate_dict[3] = num_of_block_combos*math.comb(int(2*np.ceil(np.log2(g+1))), 3) - double_counted_BUG_gates(num_of_blocks, 3, g)
    #gate_dict[4] = num_of_block_combos*math.comb(int(2*np.ceil(np.log2(g+1))), 4) - double_counted_BUG_gates(num_of_blocks, 4, g)
        gate_dict[g_val] = num_of_block_combos*math.comb(int(2*np.ceil(np.log2(g+1))), g_val) - double_counted_BUG_gates(num_of_blocks, g_val, g)
    n_gates = 0
    for _, val in gate_dict.items():
        n_gates += val

    return n_gates, gate_dict


def gates_with_decomp_new(gate_dict):
    """
    Decomposes larger gates into 1- and 2- qubit gates.

    Input: gate_dict
    Output: Number of 1- and 2-qubit gates requried in BIN after 
            decomposing all 3- or higher order qubit gates into 2- and 1-qubit gates
    """
    _1_gates = gate_dict[1]
    _2_gates = gate_dict[2]

    for key, val in gate_dict.items():
        if key not in [1,2]:
            _2_gates += val*2*(key-1)
            #_1_gates += val*4*(key-2)
    n_gates =  _1_gates + _2_gates
    return _1_gates, _2_gates, n_gates

def gates_with_decomp_3_new(gate_dict):
    _1_gates = gate_dict[1]
    _2_gates = gate_dict[2]
    _3_gates = gate_dict.get(3,0)
    for key, val in gate_dict.items():
        if key not in [1,2,3]:
            _2_gates += val*2*(key-1)

    n_gates = _1_gates + _2_gates + _3_gates
    return _1_gates, _2_gates, _3_gates, n_gates

def gates_directional(encoding, N, two_D):
    # Build up a dictionary dependent on what encoding
    # gate_dict Key = locality, value = number of gates
    gate_dict = {8:0, 7:0, 6:0, 5:0, 4:0, 3:0, 2:0, 1:0}
    gate_dict = gates_back_dir(gate_dict, N, two_D, encoding)
    gate_dict = gates_overlap(gate_dict, N, two_D, encoding)
    gate_dict = gates_pair(gate_dict, N, two_D, encoding)

    return gate_dict

def gates_back_dir(gate_dict, N, two_D, encoding):
    if encoding == 'BIN' or encoding == 'BUBIN':
        # 3D
        if not two_D:
            for j in range(2, N-3):
                # x
                gate_dict[6] = gate_dict[6] + 1
                gate_dict[5] = gate_dict[5] + 2
                gate_dict[4] = gate_dict[4] + 1

                # y
                gate_dict[6] = gate_dict[6] + 1
                gate_dict[5] = gate_dict[5] + 3
                gate_dict[4] = gate_dict[4] + 4
                gate_dict[3] = gate_dict[3] + 3
                gate_dict[2] = gate_dict[2] + 1

                # z
                gate_dict[6] = gate_dict[6] + 1
                gate_dict[5] = gate_dict[5] + 2
                gate_dict[4] = gate_dict[4] + 1
        # 2D
        else:
            for j in range(2, 2*N-8):
                gate_dict[4] = gate_dict[4] + 1
                gate_dict[3] = gate_dict[3] + 4
                gate_dict[2] = gate_dict[2] + 6

    elif encoding == 'OH':
        # 3D
        if not two_D:
            for j in range(2, N-3):
                gate_dict[2] = gate_dict[2] + 12
        # 2D
        else:
            for j in range(2, 2*N-8):
                gate_dict[2] = gate_dict[2] + 8
    return gate_dict

def gates_overlap(gate_dict, N, two_D, encoding):
    num_anc_dist = 0
    if N > 4:
        for i in range(N-4):
            for j in range(i+4, N):
                num_anc_dist += np.ceil(np.log2(np.abs(i-j)))*((1+i-j)%2)
    if encoding == 'BIN' or encoding == 'BUBIN':
        # 3D
        if not two_D:
            # Dij2
            gate_dict[4*(3-1)] = gate_dict[4*(3-1)] + sp.special.comb(sp.special.comb(N,2), 2)*3
            # alpha2
            gate_dict[2] = gate_dict[2] + sp.special.comb(num_anc_dist, 2)
            # D alpha
            gate_dict[2*(3-1)+1] = gate_dict[2*(3-1)+1] + sp.special.comb(N,2)*num_anc_dist
            # alpha
            gate_dict[1] = gate_dict[1] + num_anc_dist

        # 2D
        else:
            # Dij2
            gate_dict[4*(2-1)] = gate_dict[4*(2-1)] + sp.special.comb(sp.special.comb(N,2), 2)
            # alpha2
            gate_dict[2] = gate_dict[2] + sp.special.comb(num_anc_dist, 2)
            # D alpha
            gate_dict[2*(2-1)+1] = gate_dict[2*(2-1)+1] + sp.special.comb(N,2)*num_anc_dist
            # alpha
            gate_dict[1] = gate_dict[1] + num_anc_dist
            
    elif encoding == 'OH':
        # 3D
        if not two_D:
            # Dij2
            gate_dict[4] = gate_dict[4] + sp.special.comb(sp.special.comb(N,2), 2)
            # alpha2
            gate_dict[2] = gate_dict[2] + sp.special.comb(num_anc_dist, 2)
            # D alpha
            gate_dict[3] = gate_dict[3] + sp.special.comb(N,2)*num_anc_dist
            # alpha
            gate_dict[1] = gate_dict[1] + num_anc_dist
            
        # 2D
        else:
            # Dij2
            gate_dict[4] = gate_dict[4] + sp.special.comb(sp.special.comb(N,2), 2)
            # alpha2
            gate_dict[2] = gate_dict[2] + sp.special.comb(num_anc_dist, 2)
            # D alpha
            gate_dict[3] = gate_dict[3] + sp.special.comb(N,2)*num_anc_dist
            # alpha
            gate_dict[1] = gate_dict[1] + num_anc_dist
            
    return gate_dict

def gates_pair(gate_dict, N, two_D, encoding):
    pair_qubits = 0
    if N > 3:
        for j in range(N-3):
            for k in range(N):
                pair_qubits += (j-k)%2

    if encoding == 'BIN' or encoding == 'BUBIN':
        # 3D
        if not two_D:
            gate_dict[2*(3-1)+1] = gate_dict[2*(3-1)+1] + sp.special.comb(N,2)*pair_qubits
            gate_dict[1] = gate_dict[1] + pair_qubits
            
        # 2D
        else:
            gate_dict[2*(2-1)+1] = gate_dict[2*(2-1)+1] + sp.special.comb(N,2)*pair_qubits
            gate_dict[1] = gate_dict[1] + pair_qubits
            
    elif encoding == 'OH':
        if not two_D:
            gate_dict[2+1] = gate_dict[2+1] + sp.special.comb(N,2)*pair_qubits
            gate_dict[1] = gate_dict[1] + pair_qubits
            
        # 2D
        else:
            gate_dict[2+1] = gate_dict[2+1] + sp.special.comb(N,2)*pair_qubits
            gate_dict[1] = gate_dict[1] + pair_qubits
    return gate_dict


###############################
#     CREATE DN FUNCTIONS     #
###############################
# Renamed from DN to Cardinality vector C in the manuscript.

def random_D(lower_len, upper_len, lower_rot, upper_rot):
    """
    Input:  lower_len: lower limit for number of sites
            upper_len: upper limit for number of sites
            lower_rot: lower limit for number of rotamers at a site
            upper_rot: upper limit for number of rotamers at a site

    Output: A vector Dn, with a random number of elements(sites) where each 
            element is a random number(rotamers)
    """
    D_len = np.random.randint(lower_len,upper_len)
    return np.random.randint(lower_rot, upper_rot, D_len)


def random_coord_based_D(lower_len, upper_len, upper_rot):
    """
    Input:  upper_len: upper limit for number of sites
            lower_rot: lower limit for number of rotamers at a site
            upper_rot: upper limit for number of rotamers at a site

    Output: A vector Dn, with a random number of elements(sites) where each 
            element is a random number(rotamers) greater than half the number of elements.
    """
    D_len = np.random.randint(lower_len,upper_len)
    return np.random.randint(int(D_len/2), upper_rot, D_len)

def D_dir_3d(N):
    return [np.array([6]*N)]

def D_dir_2d(N):
    return [np.array([4]*N)]

def D_coord_2d(N, percentage = 1.25):
    """
    Input: N is the number of amino acid, len(D)
    Output: A list of D vectors for a given number of amino acid
    """
    if type(N) == list():
        N = len(N)
    D_list = list()
    lower_area_bound = N
    upper_area_bound = np.ceil(percentage*N)
    L1 = np.floor(np.sqrt(N))
    L2 = np.ceil(np.sqrt(N))
    L1L2_set = list()
    if L1*L2 < lower_area_bound:
        L1 = np.ceil(np.sqrt(N))

    while L1*L2<=upper_area_bound:
        while L1*L2<=upper_area_bound:
            D_temp = list()
            for i in range(N):
                if i%2==0:
                    D_temp.append(np.ceil(L1*L2/2))
                elif i%2==1:
                    D_temp.append(np.floor(L2*L1/2))
            if (L1,L2) not in L1L2_set:
                D_list.append(np.array(D_temp))
                
                if (L2,L1) in L1L2_set:
                    print('Be a better programmer!')
                L1L2_set.append((L1,L2))
            L2+=1
        
        L1+=1
        L2 = L1
    return D_list, L1L2_set

def D_coord_2d_only_squares(N, percentage, save_dict):
    """
    Input: N is the number of amino acid, len(D)
    Output: A list of D vectors for a given number of amino acid
    """
    print(N)
    if type(N) == list():
        N = len(N)
    D_list = list()
    L1L2_set = list()
    lower_area_bound = N
    upper_area_bound = np.ceil(percentage*N)
    j = N
    while j <= upper_area_bound:
        rectangle_list, save_dict = all_squares(j, save_dict)
        for rectangle in rectangle_list:
            L1 = rectangle[0]
            L2 = rectangle[1]
            D_temp = list()
            for i in range(N):
                if i%2 == 0:
                    D_temp.append(np.ceil(L1*L2/2))
                elif i%2 == 1:
                    D_temp.append(np.floor(L2*L1/2))
            if ((L1,L2) not in L1L2_set) and (L1*L2 >= lower_area_bound):
                D_list.append(np.array(D_temp))
            L1L2_set.append((L1,L2))
        j+=1       
    return D_list, L1L2_set, save_dict

def D_coord_3d(N, percentage):
    """
    Input: N is the number of amino acid, len(D)
    Output: A list of D vectors for a given number of amino acid
    """
    if type(N) == list():
        N = len(N)
    D_list = list()
    lower_area_bound = float(N)
    upper_area_bound = np.ceil(percentage*N)
    L1 = np.floor(N**(1/3))
    L2 = np.floor(N**(1/3))
    L3 = np.floor(N**(1/3))
    L1L2L3_set = list()
    if L1*L2*L3 < lower_area_bound:
        L1 = np.ceil(N**(1/3))

    while L1*L2*L3 <= upper_area_bound:
        while L1*L2*L3 <= upper_area_bound:
            while L1*L2*L3 <= upper_area_bound:

                D_temp = list()
                for i in range(N):
                    if i%2 == 0:
                        D_temp.append(np.ceil(L1*L2*L3/2))
                    elif i%2 == 1:
                        D_temp.append(np.floor(L2*L1*L3/2))
                if (L1,L2,L3) not in L1L2L3_set:
                    D_list.append(np.array(D_temp))
                    
                    if (L3,L2,L1) in L1L2L3_set:
                        print('Be a better programmer!')
                    L1L2L3_set.append((L1,L2,L3))
                L1 += 1
            L2 += 1
            L1 = L2
        L3 += 1
        L2 = L3
    return D_list, L1L2L3_set

def D_coord_3d_only_cubes(N, percentage, save_dict):
    """
    Input: N is the number of amino acid, len(D)
    Output: A list of D vectors for a given number of amino acid
    """
    print(N)
    if type(N) == list():
        N = len(N)
    D_list = list()
    L1L2_set = list()
    lower_area_bound = N
    upper_area_bound = np.ceil(percentage*N)
    j = N
    while j <= upper_area_bound:
        cube_list, save_dict = all_cubes(j, save_dict)
        for cube in cube_list:
            L1 = cube[0]
            L2 = cube[1]
            L3 = cube[2]
            D_temp = list()
            for i in range(N):
                if i%2 == 0:
                    D_temp.append(np.ceil(L1*L2*L3/2))
                elif i%2 == 1:
                    D_temp.append(np.floor(L2*L1*L3/2))
            if ((L1,L2,L3) not in L1L2_set) and (L1*L2*L3 >= lower_area_bound) and not (ele == 1 for ele in (L1,L2,L3)):
                D_list.append(np.array(D_temp))
                L1L2_set.append((L1,L2,L3))
        j+=1        
    return D_list, L1L2_set, save_dict

def random_2power_D(lower_len, upper_len, lower_power, upper_power):
    D_len = np.random.randint(lower_len, upper_len)
    twos = np.ones(shape = (D_len,))*2
    powers = np.random.randint(lower_power, upper_power, D_len)
    return (twos**powers).astype(int)

def increase_last_rot(D, upper_lim, start_val = 2, lower_lim = 0):
    """
    Input:  D: A Dn vector
            upper_lim: upper limit maximum number of rotamers at a site
            start_val: starting value for new sites
            lower_lim: deprecated, the upper limit used to be a random number between lower_lim and upper_lim

    Output: A vector Dn where each element except the last is equal to upper_lim, the last element will be 1 greater
            than the input vector, if the last element is equal to the upper_lim a new site is added.
    """
    if len(D)==0:
        return np.array([start_val, start_val])
    else:
        comp = D[:]
        if lower_lim == 0:
            if comp[0]<upper_lim:#np.random.randint(15,19):
                D[0] = D[0] + 1
            elif comp[-1]<upper_lim:
                D[-1] = D[-1]+1
            else:
                D = D + [start_val]
        else: 
            if D[0]<np.random.randint(lower_lim,upper_lim):
                D[0] = D[0]+1
            elif D[-1]<np.random.randint(lower_lim,upper_lim):
                D[-1] = D[-1]+1
            else:
                np.append(D,start_val)
    return D

def increase_by_sum(D, upper_sum, lower_val = 2):
    """
    Input:  D: A Dn vector
            upper_sum: upper limit of sum of all rotamers
            lower_val_val: starting value for new sites

    Output: A vector Dn where the total sum is lower than upper_sum
            if the total sum of all rotamers is lower than upper_sum
            a random element is incremented by 1. Else the upper_sum is doubled.
            upper_sum: Updated upper_sum
    """
    if np.sum(D) > upper_sum:
        D.append(lower_val)

        upper_sum *= 2
    else:
        index = np.random.randint(0,len(D))
        D[index] = D[index] + 1
    return D, upper_sum

def increase_as_power_of_2(D,upper_sum, lower_val = 2):
    """
    Input:  D: A Dn vector where all elements are a power of 2.
            upper_sum: upper limit of sum of all rotamers
            lower_val_val: starting value for new sites

    Output: D: A Dn vector where one random element is doubled
            upper_sum: Updated upper_sum
    If the total sum of all rotamers is lower than upper_sum
    a random element doubled. Else the upper_sum is doubled.
            
    """
    if np.sum(D) > upper_sum:
        D = np.append(D,lower_val)

        upper_sum *= 2
    else:
        index = np.random.randint(0,len(D))
        while index == np.argmax(D):
            if len(D) == 1:
                break
            index = np.random.randint(0,len(D))
        #index = np.argmin(D)
        D[index] = D[index]*2
    return D, upper_sum

def increase_as_power_of_2_plus_1(D,upper_sum, lower_val = 2):
    """
    Input:  D: A Dn vector where all elements is a power of 2 plus 1.
            upper_sum: upper limit of sum of all rotamers
            lower_val_val: starting value for new sites

    Output: D, where one element is set to the "next" power of 2 plus 1
            upper_sum: Updated upper_sum
    If the total sum of all rotamers is lower than upper_sum
    a random element is set to the "next" power of two plus 1. Else the upper_sum is doubled.
    """
    if np.sum(D) > upper_sum:
        D = np.append(D,lower_val)

        upper_sum = 2**(int(np.log2(upper_sum))+1)
    else:
        index = np.random.randint(0,len(D))
        while index == np.argmax(D):
            index = np.random.randint(0,len(D))
        _2power = int(np.log2(D[index]))+1
        D[index] = 2**_2power+1
    return D, upper_sum
    
def good_and_bad_power_of_2(D_good, D_bad, upper_sum, lower_val = 2):
    """
    Input:  D_good: A Dn vector where all elements is a power of 2.
            D_bad: A Dn vector where all elements is a power of 2 plus 1.
            upper_sum: upper limit of sum of all rotamers
            lower_val_val: starting value for new sites

    Output: D_good where one elements is doubled
            D_Bad where one element is set to the "next" power of 2 plus 1.
            upper_sum: Updated upper_sum
    If the total sum of all rotamers is lower than upper_sum
    a random element is updated. Else the upper_sum is doubled.
           
    """
    if np.sum(D_good) > upper_sum:
        D_good = np.append(D_good,lower_val)
        D_bad = np.append(D_bad, lower_val+1)
        upper_sum = 2**(int(np.log2(upper_sum))+1)
    else:
        index = np.random.randint(0,len(D_good))
        while index == np.argmax(D_good):
            if np.max(D_good) == np.min(D_good):
                break
            index = np.random.randint(0,len(D_good))
        _2power = int(np.log2(D_good[index]))+1
        D_good[index] = 2**_2power
        D_bad[index] = 2**_2power + 1
    return D_good, D_bad, upper_sum

def prime_factors(n):
    i = 2
    factors = []
    while i * i <= n:
        if n % i:
            i += 1
        else:
            n //= i
            factors.append(i)
    if n > 1:
        factors.append(n)
    return factors

def all_cubes(V, save_dict):
    if V in save_dict:
        return save_dict[V], save_dict
    else:
        return_list = []
        for a in range(1, V+1):
            for b in range(1, V+1):
                if V % (a*b) == 0:
                    c = V / (a*b)
                    temp = [int(a), int(b), int(c)]
                    if not any(list(x) in return_list for x in list(permutations(temp))):
                        return_list.append(temp)
        save_dict[V] = return_list
        return return_list, save_dict

def all_squares(A, save_dict):
    if A in save_dict:
        return save_dict[A], save_dict
    else:
        return_list = []
        for a in range(1, A+1):
            for b in range(1, A+1):
                if a*b == A:
                    temp = [int(a), int(b)]
                    if not any(list(x) in return_list for x in list(permutations(temp))):
                        return_list.append(temp)
        save_dict[A] = return_list
        return return_list, save_dict
