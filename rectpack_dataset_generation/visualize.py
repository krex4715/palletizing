import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import argparse
from masking import collision_detect, find_boundary, box_boundaries_find





def get_args():
    parser = argparse.ArgumentParser(
        """Visualize Generated 2D BPP Dataset""")
    parser.add_argument("--data_idx", '-i', type=int, default=0)
    parser.add_argument("--algorhithm_name", '-a', type=str, default="SkylineBl")
    parser.add_argument("--best_algorithm_view", '-b', action='store_true',default=False)
    parser.add_argument("--masking_view", '-m', action='store_false',default=True)
    parser.add_argument("--render_speed", '-s', type=float, default=1.0)
    args = parser.parse_args()
    return args








if __name__ == '__main__':
    args = get_args()

    data_idx = args.data_idx
    

    # load
    data = np.load('dataset_v1/bpp_' + str(data_idx) + '.npy', allow_pickle=True)
    boxes = data[0]
    results_list = data[1]

    if args.best_algorithm_view:
        loading_rate_list = [result['loading_rate'] for result in results_list]
        best_algorithm_idx = loading_rate_list.index(max(loading_rate_list))
        algorithm_name = results_list[best_algorithm_idx]['algorhithm_name']

        bin_size = results_list[best_algorithm_idx]['bin_size']
        box_position_list = results_list[best_algorithm_idx]['box_position_list']
        number_of_loaded_box = results_list[best_algorithm_idx]['number_of_loaded_box']
        number_of_unloaded_box = results_list[best_algorithm_idx]['number_of_unloaded_box']
        loading_rate = results_list[best_algorithm_idx]['loading_rate']


    else:
        results_name_list = [result['algorhithm_name'] for result in results_list]
        algorithm_name = args.algorhithm_name
        algorithm_idx = results_name_list.index(algorithm_name)

        bin_size = results_list[algorithm_idx]['bin_size']
        box_position_list = results_list[algorithm_idx]['box_position_list']
        number_of_loaded_box = results_list[algorithm_idx]['number_of_loaded_box']
        number_of_unloaded_box = results_list[algorithm_idx]['number_of_unloaded_box']
        loading_rate = results_list[algorithm_idx]['loading_rate']


        # for result in results_list:
        #     if result['algorhithm_name'] == algorithm_name:
        #         bin_size = result['bin_size']
        #         box_position_list = result['box_position_list']
        #         number_of_loaded_box = result['number_of_loaded_box']
        #         number_of_unloaded_box = result['number_of_unloaded_box']
        #         loading_rate = result['loading_rate']
        #         break

    # rendering
    plt.ion()
    fig, ax = plt.subplots(1)
    bw, bh = bin_size
    box_p_list=[]
    for box_position in box_position_list:
        plt.axis([0, bw, 0, bh])
        plt.title('Algorithm: ' + algorithm_name)


        x, y, w, h = box_position
        array = collision_detect(box_p_list, bw, bh)
        array_boundary_results = box_boundaries_find(array, w, h, bw, bh)
        boundary_xy = np.array(np.where(array_boundary_results == 1)).T.tolist()

  
        if [y,x] not in boundary_xy:
            print('ERROR :::: CHECK MASKING & DATASET')

        
        ax.add_patch(
            patches.Rectangle(
                (x, y),  # (x,y)
                w,  # width
                h,  # height
                facecolor="#00ffff",
                edgecolor="black",
                linewidth=1,
                alpha=0.3
            )
        )

        plt.text(0, 140, 'Loading Rate: ' + str(loading_rate),fontsize=10, ha='left', color='white')
        plt.text(0, 130, 'Number of Loaded Box: ' + str(number_of_loaded_box),fontsize=10, ha='left', color='white')
        plt.text(0, 120, 'Number of Unloaded Box: ' + str(number_of_unloaded_box),fontsize=10, ha='left', color='white')
        
        if args.masking_view:
            boundary = plt.imshow(array_boundary_results, cmap='gray')


        plt.pause(args.render_speed)
        
        if args.masking_view:
            boundary.remove()
        box_p_list.append([x, y, w, h])


