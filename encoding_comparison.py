from cProfile import label
import matplotlib.pyplot as plt
import numpy as np

from comparison_util import *

def sort_after(reference, data):
    return [x for _,x in sorted(zip(reference, data))]

g = 3 # BUG constant

simulate = True

save = True

model = 'coordinate' #'coordinate', 'rotamer', 'directional'
decompose = True

all_list = []

for i in [150]:
    for j in [10, 101]:
        for k in [False, True]:
            all_list.append([i,j,k])

for variation in all_list:
    print(variation)
    print()
    percentage = variation[0]
    upper_limit = variation[1]
    two_D = variation[2]
    print('variation: ', variation)
    print(upper_limit)
    print()


    OHqlist = np.array([])
    OHq_with_decomp_list =  np.array([])
    OHq_with_3decomp_list =  np.array([])

    BINqlist =  np.array([])
    BINq_with_decomp_list =  np.array([])
    BINq_with_3decomp_list =  np.array([])

    BUGqlist = np.array([])
    BUGq_with_decomp_list =  np.array([])
    BUGq_with_3decomp_list =  np.array([])

    gate_3_OH = np.array([])
    gate_3_bin = np.array([])
    gate_3_bug = np.array([])

    OHgateslist = np.array([])
    BINgateslist = np.array([])
    BUGgateslist = np.array([])

    double_bin_gates = np.array([])   # Double counting

    OHgates_with_decomp_list_2 = np.array([])
    OHgates_with_decomp_list_1 = np.array([])
    OHgates_with_3decomp_list_3 = np.array([])
    OHgates_with_3decomp_list_2 = np.array([])
    OHgates_with_3decomp_list_1 = np.array([])

    BINgates_with_decomp_list_2 = np.array([])
    BINgates_with_decomp_list_1 = np.array([])
    BINgates_with_3decomp_list_3 = np.array([])
    BINgates_with_3decomp_list_2 = np.array([])
    BINgates_with_3decomp_list_1 = np.array([])

    BUGgates_with_decomp_list_2 = np.array([])
    BUGgates_with_decomp_list_1 = np.array([])
    BUGgates_with_3decomp_list_3 =np.array([])
    BUGgates_with_3decomp_list_2 = np.array([])
    BUGgates_with_3decomp_list_1 = np.array([])

    Siteslist = np.array([])
    D_list = []

    OHunfeasable_list = np.array([])
    BINunfeasable_list = np.array([])
    BUGunfeasable_list = np.array([])

    if simulate:
        print('simulating data')
        
        for N in range(3,upper_limit):
            if model == 'coordinate':
                if two_D:
                    possible_D, L1L2 = D_coord_2d(N, percentage/100)
                    if upper_limit < 100:
                        save_folder = 'encoding_data/coord_based/small_data2_' + str(percentage)
                    else:
                        save_folder = 'encoding_data/coord_based/big_data2_' + str(percentage)
                else:
                    possible_D, L1L2 = D_coord_3d(N, percentage/100)
                    if upper_limit < 100:
                        save_folder = 'encoding_data/coord_based/small_data3_' + str(percentage)
                    else:
                        save_folder = 'encoding_data/coord_based/big_data3_' + str(percentage)
                #print(N, L1L2)

            if model == 'directional':
                if two_D:
                    possible_D = D_dir_2d(N)
                    if upper_limit < 100:
                        save_folder = 'encoding_data/turn_based/small_data2/'
                    else:
                        save_folder = 'encoding_data/turn_based/big_data2/'
                else:
                    possible_D = D_dir_3d(N)
                    if upper_limit < 100:
                        save_folder = 'encoding_data/turn_based/small_data3/'
                    else:
                        save_folder = 'encoding_data/turn_based/big_data3/'
            #print('return ', N, possible_D, L1L2)

            for D in possible_D:
                #D = random_coord_based_D(2,10,101)
                #print('D ', D)
                print('N ', N)

                D_temp = D.copy()

                Siteslist = np.append(Siteslist, len(D))
                D_list.append(D_temp)
                #OHunfeasable_list = np.append(OHunfeasable_list, OHunfeasable(D, model))
                #BINunfeasable_list = np.append(BINunfeasable_list, BINunfeasable(D, model))
                #BUGunfeasable_list = np.append(BUGunfeasable_list, BUGunfeasable(D, g, model))

                # 
                if model == 'coordinate' or model == 'rotamer':
                    OHgateslist = np.append(OHgateslist, sum(OHgates(D)))
                    gate_dict_OH = {1:OHgates(D)[0], 2:OHgates(D)[1], 3:0, 4:0}

                    BINgates_, gate_dict_BIN, double = BINgates(D)
                    gate_3_bin = np.append(gate_3_bin, gate_dict_BIN.get(3,0))
                    BINgateslist = np.append(BINgateslist, BINgates_)

                    BUGgates_, gate_dict_BUG = BUGgates(D, g)
                    BUGgateslist = np.append(BUGgateslist, BUGgates_)
                    gate_3_bug =  np.append(gate_3_bug, gate_dict_BUG.get(3,0))

                    double_bin_gates = np.append(double_bin_gates, double)

                elif model == 'directional':
                    OH_gate_dict = gates_directional('OH', N, two_D)
                    OHgates_ = sum(OH_gate_dict.values())
                    gate_dict_OH = OH_gate_dict
                    OHgateslist = np.append(OHgateslist, OHgates_)
                    #print('OHgateslist ', OHgateslist)

                    BIN_gate_dict = gates_directional('BIN', N, two_D)
                    BINgates_ = sum(BIN_gate_dict.values())
                    gate_dict_BIN = BIN_gate_dict
                    BINgateslist = np.append(BINgateslist, BINgates_)
                    #print('BINgateslist ', BINgateslist)

                    BUG_gate_dict = gates_directional('BUBIN', N, two_D)
                    BUGgates_ = sum(BUG_gate_dict.values())
                    gate_dict_BUG = BUG_gate_dict
                    BUGgateslist = np.append(BUGgateslist, BUGgates_)
                    #print()


                OHqlist = np.append(OHqlist, OHq(D, model))
                BUGqlist = np.append(BUGqlist, BUGq(D, g, model))
                BINqlist = np.append(BINqlist, BINq(D, model))

                if decompose:
                    print(gate_dict_OH)
                    gates_with_decomp_3_new(gate_dict_OH)
                    print()
                    print(gate_dict_BIN)
                    gates_with_decomp_3_new(gate_dict_BIN)
                    print()
                    print(gate_dict_BUG)
                    gates_with_decomp_3_new(gate_dict_BUG)
                    print()
                    print()

                    decomp = gates_with_decomp_new(gate_dict_OH)
                    OHgates_with_decomp_list_2 = np.append(OHgates_with_decomp_list_2, decomp[1])
                    OHgates_with_decomp_list_1 = np.append(OHgates_with_decomp_list_1, decomp[0])
                    decomp3 = gates_with_decomp_3_new(gate_dict_OH)
                    OHgates_with_3decomp_list_3 = np.append(OHgates_with_3decomp_list_3, decomp3[2])
                    OHgates_with_3decomp_list_2 = np.append(OHgates_with_3decomp_list_2, decomp3[1])
                    OHgates_with_3decomp_list_1 = np.append(OHgates_with_3decomp_list_1,decomp3[0])

                    decomp = gates_with_decomp_new(gate_dict_BIN)
                    BINgates_with_decomp_list_2 = np.append(BINgates_with_decomp_list_2, decomp[1])
                    BINgates_with_decomp_list_1 = np.append(BINgates_with_decomp_list_1, decomp[0])
                    decomp3 = gates_with_decomp_3_new(gate_dict_BIN)
                    BINgates_with_3decomp_list_3 = np.append(BINgates_with_3decomp_list_3, decomp3[2])
                    BINgates_with_3decomp_list_2 = np.append(BINgates_with_3decomp_list_2, decomp3[1])
                    BINgates_with_3decomp_list_1 = np.append(BINgates_with_3decomp_list_1,decomp3[0])

                    decomp = gates_with_decomp_new(gate_dict_BUG)
                    BUGgates_with_decomp_list_2 = np.append(BUGgates_with_decomp_list_2, decomp[1])
                    BUGgates_with_decomp_list_1 = np.append(BUGgates_with_decomp_list_1, decomp[0])
                    decomp3 = gates_with_decomp_3_new(gate_dict_BUG)
                    BUGgates_with_3decomp_list_3 = np.append(BUGgates_with_3decomp_list_3, decomp3[2])
                    BUGgates_with_3decomp_list_2 = np.append(BUGgates_with_3decomp_list_2, decomp3[1])
                    BUGgates_with_3decomp_list_1 = np.append(BUGgates_with_3decomp_list_1,decomp3[0])
                    
                    '''
                    OHq_with_decomp_list = np.append(OHq_with_decomp_list, OHq_with_decomp(D, gate_dict_OH))
                    OHq_with_3decomp_list = np.append(OHq_with_3decomp_list, OHq_with_decomp_3(D, gate_dict_OH))

                    BINq_with_decomp_list = np.append(BINq_with_decomp_list, BINq_with_decomp(D, gate_dict_BIN))
                    BINq_with_3decomp_list = np.append(BINq_with_3decomp_list, BINq_with_decomp_3(D, gate_dict_BIN))

                    BUGq_with_decomp_list = np.append(BUGq_with_decomp_list, BUGq_with_decomp(D, gate_dict_BUG, g))
                    BUGq_with_3decomp_list = np.append(BUGq_with_3decomp_list, BUGq_with_decomp_3(D, gate_dict_BUG, g))
                    '''
                
    #print('D_list ', D_list)
    print('Finished calculating resources.')
    Siteslist_sort = sorted(Siteslist)
    if save:
        import os
        import time
        #save_folder = 'encoding_data/coordinate_based/small_data'
        dir_path = os.path.dirname(os.path.realpath(__file__))
        print(dir_path)
        print(f'saving data to {save_folder}')
        print('Be careful with save folder as it can overwrite old data')
        print('save_folder ', save_folder)
        #print('Waiting 10 seconds so you can abort if the save folder is wrong')
        #time.sleep(10)
        if not os.path.exists(os.path.join(dir_path, save_folder)):
            os.mkdir(os.path.join(dir_path, save_folder))
        np.savetxt(os.path.join(dir_path, f'{save_folder}/Sites.txt'), Siteslist)

        np.savetxt(os.path.join(dir_path, f'{save_folder}/OHqubits.txt'), OHqlist)
        np.savetxt(os.path.join(dir_path, f'{save_folder}/OHqubits_decomp.txt'), OHq_with_decomp_list)
        np.savetxt(os.path.join(dir_path, f'{save_folder}/OHqubits_3decomp.txt'), OHq_with_3decomp_list)

        np.savetxt(os.path.join(dir_path, f'{save_folder}/BINqubits.txt'), BINqlist)
        np.savetxt(os.path.join(dir_path, f'{save_folder}/BINqubits_decomp.txt'), BINq_with_decomp_list)
        np.savetxt(os.path.join(dir_path, f'{save_folder}/BINqubits_3decomp.txt'), BINq_with_3decomp_list)

        np.savetxt(os.path.join(dir_path, f'{save_folder}/BUGqubits.txt'), BUGqlist)
        np.savetxt(os.path.join(dir_path, f'{save_folder}/BUGqubits_decomp.txt'), BUGq_with_decomp_list)
        np.savetxt(os.path.join(dir_path, f'{save_folder}/BUGqubits_3decomp.txt'), BUGq_with_3decomp_list)
        #np.savetxt(os.path.join(dir_path, f'{save_folder}/D_n.txt'), D_list,fmt='%s')
        np.savetxt(os.path.join(dir_path, f'{save_folder}/OHgates.txt'), OHgateslist)
        np.savetxt(os.path.join(dir_path, f'{save_folder}/BINgates.txt'), BINgateslist)
        np.savetxt(os.path.join(dir_path, f'{save_folder}/BUGgates.txt'), BUGgateslist)

        np.savetxt(os.path.join(dir_path, f'{save_folder}/OHgate_decomp_2.txt'), OHgates_with_decomp_list_2)
        np.savetxt(os.path.join(dir_path, f'{save_folder}/OHgate_decomp_1.txt'), OHgates_with_decomp_list_1)
        np.savetxt(os.path.join(dir_path, f'{save_folder}/OHgate_3decomp_1.txt'), OHgates_with_3decomp_list_1)
        np.savetxt(os.path.join(dir_path, f'{save_folder}/OHgate_3decomp_2.txt'), OHgates_with_3decomp_list_2)
        np.savetxt(os.path.join(dir_path, f'{save_folder}/OHgate_3decomp_3.txt'), OHgates_with_3decomp_list_3)

        np.savetxt(os.path.join(dir_path, f'{save_folder}/BINgate_decomp_2.txt'), BINgates_with_decomp_list_2)
        np.savetxt(os.path.join(dir_path, f'{save_folder}/BINgate_decomp_1.txt'), BINgates_with_decomp_list_1)
        np.savetxt(os.path.join(dir_path, f'{save_folder}/BINgate_3decomp_1.txt'), BINgates_with_3decomp_list_1)
        np.savetxt(os.path.join(dir_path, f'{save_folder}/BINgate_3decomp_2.txt'), BINgates_with_3decomp_list_2)
        np.savetxt(os.path.join(dir_path, f'{save_folder}/BINgate_3decomp_3.txt'), BINgates_with_3decomp_list_3)

        np.savetxt(os.path.join(dir_path, f'{save_folder}/BUGgate_decomp_2.txt'), BUGgates_with_decomp_list_2)
        np.savetxt(os.path.join(dir_path, f'{save_folder}/BUGgate_decomp_1.txt'), BUGgates_with_decomp_list_1)
        np.savetxt(os.path.join(dir_path, f'{save_folder}/BUGgate_3decomp_1.txt'), BUGgates_with_3decomp_list_1)
        np.savetxt(os.path.join(dir_path, f'{save_folder}/BUGgate_3decomp_2.txt'), BUGgates_with_3decomp_list_2)
        np.savetxt(os.path.join(dir_path, f'{save_folder}/BUGgate_3decomp_3.txt'), BUGgates_with_3decomp_list_3)

        #np.savetxt(os.path.join(dir_path, f'{save_folder}/OH_unfeas.txt'), OHunfeasable_list)
        #np.savetxt(os.path.join(dir_path, f'{save_folder}/BIN_unfeas.txt'), BINunfeasable_list)
        #np.savetxt(os.path.join(dir_path, f'{save_folder}/BUG_unfeas.txt'), BUGunfeasable_list)

        np.savetxt(os.path.join(dir_path, f'{save_folder}/OH_3_OH.txt'), gate_3_OH)
        np.savetxt(os.path.join(dir_path, f'{save_folder}/BIN_3_bin.txt'), gate_3_bin)
        np.savetxt(os.path.join(dir_path, f'{save_folder}/BUG_3_gate.txt'), gate_3_bug)
        
print('Done')