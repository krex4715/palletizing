import numpy as np

def generate_random_box(base_size=8, max_diff=4):
    return base_size + int(np.random.uniform(-max_diff, max_diff)), base_size + int(np.random.uniform(-max_diff, max_diff))


def detecting_corner(erode):
        # 기존 경계 감지 로직
        vertical_diff = np.diff(erode, axis=0)
        vertical_edges = np.pad(vertical_diff, ((1,0),(0,0)), mode='constant', constant_values=0) # 미분으로 생긴 길이감소를 보정
        horizontal_diff = np.diff(erode, axis=1)
        horizontal_edges = np.pad(horizontal_diff, ((0,0),(1,0)), mode='constant', constant_values=0) # 미분으로 생긴 길이감소를 보정
        edges = vertical_edges + horizontal_edges
        # 여기서 ㄱ자 경계 처리는 일단 제외
        # 그 이유는 padding 으로 채운 부분이 0이라서 [1,0] [0,1] 경계가 생기기 때문에 ([1,1],[0,1]이어야 탐지)
        # 그래서 일단 ㄱ자 경계는 빼고 「, ㄴ, 」만 감지하도록 함.

        # Corner 감지 로직
        corners = np.zeros_like(edges)
        for i in range(1, edges.shape[0]-1):
            for j in range(1, edges.shape[1]-1):
                if edges[i,j] != 0:
                    if edges[i+1,j] == -1 and edges[i,j+1] == -1:
                        corners[i,j] = 1
                    elif edges[i+1,j] == 1 and edges[i,j-1] == -1:
                        corners[i,j] = 1
                    elif edges[i-1,j] == -1 and edges[i,j+1] == 1:
                        corners[i,j] = 1

                        
        
        corner_xy = np.array(np.where(corners != 0)).T
        return corner_xy,edges



def detecting_corners_batch(eroded):

    batch_size = eroded.shape[0]
    patch_1 = [[1,1],[0,1]] # ㄴ
    patch_2 = [[1,0],[1,1]] # 「
    patch_3 = [[0,1],[1,1]] # 」
    patch_4 = [[1,1],[1,0]] # ㄱ

    corner_xy_batch = []
    for i in range(batch_size):
        erode = eroded[i]
        corners = np.zeros_like(erode)
        for i in range(1, erode.shape[0]-1):
            for j in range(1, erode.shape[1]-1):
                if erode[i,j] != 0:
                    if np.all(erode[i:i+2,j:j+2] == patch_1):
                        corners[i,j] = 1
                    elif np.all(erode[i:i+2,j:j+2] == patch_2):
                        corners[i,j] = 1
                    elif np.all(erode[i:i+2,j:j+2] == patch_3):
                        corners[i,j] = 1
                    elif np.all(erode[i:i+2,j:j+2] == patch_4):
                        corners[i,j] = 1

        corner_xy = np.array(np.where(corners != 0)).T
        corner_xy_batch.append(corner_xy)



    return corner_xy_batch