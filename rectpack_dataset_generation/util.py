import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from masking import collision_detect, find_boundary, box_boundaries_find


def calculate_packer(boxes, bins, packer):
    # Add the boxes to packing queue
    for r in boxes:
        packer.add_rect(*r)

    # Add the bins where the boxes will be placed
    for b in bins:
        packer.add_bin(*b)

    # Start packing
    packer.pack()


    abin = packer[0]
    bw, bh  = abin.width, abin.height




    box_position_list = []
    masking_position_list = []
    for i in range(len(abin)-1):
        rect = abin[i]
        # heuristic알고리즘이 수행한 x,y좌표
        x, y, w, h = rect.x, rect.y, rect.width, rect.height
        
        array = collision_detect(box_position_list, bw, bh)
        array_boundary_results = box_boundaries_find(array, w,h, bw, bh)
        boundary_xy = np.where(array_boundary_results > 0)
        boundary_xy = np.array(boundary_xy).T.tolist()
        masking_position_list.append(boundary_xy)


        box_position_list.append([x,y,w,h])



    # 적재율
    loading_rate = sum([r.width * r.height for r in abin]) / (bw * bh)
    

    algorithm_name = packer._pack_algo.__name__
    packing_logs = {
        'algorithm_name': algorithm_name,
        'loading_rate': loading_rate,
        'bin_size' : (bw, bh),
        'masking_position': masking_position_list,
        'box_position_list': box_position_list,
        'number_of_loaded_box': len(abin),
        'number_of_unloaded_box': len(boxes) - len(abin)
    }

    return packing_logs




def rendering(packer):
    plt.ion()

    # for index, abin in enumerate(packer):
    abin = packer[0]
    bw, bh  = abin.width, abin.height
    # print('bin', bw, bh, "nr of boxes in bin", len(abin))

    # ax = fig.add_subplot(111, aspect='equal')
    fig, ax = plt.subplots(1)
    for rect in abin:
        x, y, w, h = rect.x, rect.y, rect.width, rect.height
        plt.axis([0,bw,0,bh])
        print('rectangle', x,y,w,h,end='\r')
        ax.add_patch(
            patches.Rectangle(
                (x, y),  # (x,y)
                w,          # width
                h,          # height
                facecolor="#00ffff",
                edgecolor="black",
                linewidth=3
            )
        )
        plt.pause(0.1)

    # 적재율
    print('적재율', sum([r.width * r.height for r in abin]) / (bw * bh))
    plt.show()
    plt.pause(0.1)
    plt.close(fig)



def masking_array(masking_position, bw, bh):
    array = np.zeros((bw, bh))
    for x, y in masking_position:
        array[x, y] = 1
    return array




# bin : 1200 / 1000
# 20~40, 단위는 5 딘위로 h>=w

def generate_random_box(base_size=(30,30), max_diff=2):
    return base_size[0] + np.random.randint(-max_diff, max_diff)*4, base_size[1] + np.random.randint(-max_diff, max_diff)*4





