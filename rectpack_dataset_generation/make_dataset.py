from rectpack import newPacker, PackingMode, PackingBin, SORT_NONE
from rectpack import MaxRectsBl,MaxRectsBssf,MaxRectsBaf,MaxRectsBlsf
from rectpack import SkylineBl, SkylineBlWm, SkylineMwf, SkylineMwfl, SkylineMwfWm, SkylineMwflWm

from rectpack import GuillotineBssfSas, GuillotineBssfLas, GuillotineBssfSlas, GuillotineBssfLlas, GuillotineBssfMaxas,\
                        GuillotineBssfMinas, GuillotineBlsfSas, GuillotineBlsfLas, GuillotineBlsfSlas, GuillotineBlsfLlas, GuillotineBlsfMaxas, \
                        GuillotineBlsfMinas, GuillotineBafSas, GuillotineBafLas, GuillotineBafSlas, GuillotineBafLlas, GuillotineBafMaxas, GuillotineBafMinas
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from util import calculate_packer, rendering, generate_random_box

import argparse

def get_args():
    parser = argparse.ArgumentParser(
        """2D BPP Dataset Generation""")
    parser.add_argument("--num_data", '-n', type=int, default=1000)

    args = parser.parse_args()
    return args







if __name__ == '__main__':
    args = get_args()
    DATASET_N = args.num_data



    for i in range(DATASET_N):
        print("Generating dataset", i, "/", DATASET_N)
        
        boxes = [generate_random_box() for i in range(60)]
        bins= [(150,150)]


        packing_algorithm_list = [MaxRectsBl,MaxRectsBssf,MaxRectsBaf,MaxRectsBlsf,
                                    SkylineBl, SkylineBlWm, SkylineMwf, SkylineMwfl, SkylineMwfWm, SkylineMwflWm,
                                    GuillotineBssfSas, GuillotineBssfLas, GuillotineBssfSlas, GuillotineBssfLlas, GuillotineBssfMaxas,
                                    GuillotineBssfMinas, GuillotineBlsfSas, GuillotineBlsfLas, GuillotineBlsfSlas, GuillotineBlsfLlas, GuillotineBlsfMaxas,
                                    GuillotineBlsfMinas, GuillotineBafSas, GuillotineBafLas, GuillotineBafSlas, GuillotineBafLlas, GuillotineBafMaxas, GuillotineBafMinas]

        results_list = []

        for packing_algorithm in packing_algorithm_list:
            print(packing_algorithm,end='\r')
            packer = newPacker(mode=PackingMode.Offline,bin_algo=PackingBin.BFF, sort_algo=SORT_NONE,rotation=False, pack_algo=packing_algorithm)   
            results = calculate_packer(boxes, bins, packer)
            results_list.append(results)

        # save : box, results_list

        save_dir = 'dataset/'
        save_name = 'bpp_' + str(i) + '.npy'

        np.save(save_dir + save_name, np.array([boxes, results_list],dtype=object))










# best_packing_algorithm = packing_algorithm_list[np.argmax(loading_rate_list)]
# print('best_packing_algorithm:', best_packing_algorithm)
# print(loading_rate_list)






# # bin algorithm: Bin First Fit (Pack rectangle into the first bin it fits (without closing))
# packer = newPacker(mode=PackingMode.Offline,bin_algo=PackingBin.BFF, sort_algo=SORT_NONE,rotation=False, pack_algo=best_packing_algorithm)   


# results = calculate_loading_rate(boxes, bins, packer)

# print(results['loading_rate'])
# print(results['number_of_loaded_box'])
# print(results['number_of_unloaded_box'])


# rendering(packer)

