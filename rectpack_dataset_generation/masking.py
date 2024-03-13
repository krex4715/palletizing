import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from scipy.signal import convolve2d


def collision_detect(box_position_list, bw, bh):
    array= np.zeros((bh, bw))
    for box in box_position_list:
        x, y, w, h = box
        array[y:y+h, x:x+w] += 1
    if np.max(array) > 1:
        print('COLLISSION DETECTED ::  Check your Algorithm')
        return False
    return array

def find_boundary(exp_array):
    kernel = np.array([[1, 1, 1],
                        [1, 0, 1],
                        [1, 1, 1]])
    convolved = convolve2d(exp_array, kernel, mode='same', boundary='fill', fillvalue=0)
    boundary = (convolved > 0) & (exp_array == 0)

    return boundary


def box_boundaries_find(array, new_boxsize_w, new_boxsize_h, bw, bh):
    add_one_array = np.ones((bh+20, bw+20))
    add_one_array[10:-10, 10:-10] = array


    w,h = int(new_boxsize_w), int(new_boxsize_h)
    if w%2 != 0 or h%2 != 0:
        print('ERROR :::: Box size should be even number')
        return False



    kernel = np.ones((h,w))
    conv = convolve2d(add_one_array, kernel, mode='same')
    conv_result = (conv > 0).astype(np.float32)
    # conv_result_flip = (conv_result==0).astype(np.float32)

    # box_half_kernel = np.ones((h//2, w//2))
    # conv_half = convolve2d(conv_result_flip, box_half_kernel, mode='same')
    # conv_half_result = (conv_half > 0).astype(np.float32)



    boundary = find_boundary(conv_result)

    boundaries_x, boundaries_y = np.where(boundary > 0)
    boundaries_x = boundaries_x - h//2 
    boundaries_y = boundaries_y - w//2

    array_boundary = np.zeros((170, 170))
    for x, y in zip(boundaries_x, boundaries_y):
        array_boundary[x, y] = 1

    array_boundary_results = array_boundary[10:-10, 10:-10]

    return array_boundary_results





def box_boundary_xy(array, new_boxsize_w, new_boxsize_h, bw, bh):
    add_one_array = np.ones((bh+20, bw+20))
    add_one_array[10:-10, 10:-10] = array


    w,h = int(new_boxsize_w), int(new_boxsize_h)
    if w%2 != 0 or h%2 != 0:
        print('ERROR :::: Box size should be even number')
        return False



    kernel = np.ones((h,w))
    conv = convolve2d(add_one_array, kernel, mode='same')
    conv_result = (conv > 0).astype(np.float32)
    # conv_result_flip = (conv_result==0).astype(np.float32)

    # box_half_kernel = np.ones((h//2, w//2))
    # conv_half = convolve2d(conv_result_flip, box_half_kernel, mode='same')
    # conv_half_result = (conv_half > 0).astype(np.float32)



    boundary = find_boundary(conv_result)

    boundaries_x, boundaries_y = np.where(boundary > 0)
    boundaries_x = boundaries_x - h//2 - 10
    boundaries_y = boundaries_y - w//2 - 10

    # array_boundary = np.zeros((170, 170))
    # for x, y in zip(boundaries_x, boundaries_y):
    #     array_boundary[x, y] = 1

    # array_boundary_results = array_boundary[10:-10, 10:-10]

    return boundaries_x, boundaries_y