import numpy as np
import cv2


class BoxPalletizingEnv:
    def __init__(self):
        self.bin_size = 25  # Bin의 가로 및 세로 길이를 픽셀 단위로 설정

    def reset(self):
        self.bin = np.zeros((self.bin_size, self.bin_size))
        self.eroded = np.zeros((self.bin_size, self.bin_size))
        self.stacked_boxes = []


        init_pseudo_box = (20, 20)
        h_new_box_size,w_new_box_size = init_pseudo_box[0], init_pseudo_box[1]
        self.eroded = self._search_empty(w_new_box_size, h_new_box_size)
        self.empty_space_xy = np.array(np.where(self.eroded == 0)).T

        init_observation = (self.eroded, init_pseudo_box)
        return init_observation



    def step(self,action,new_box):
        # action은 (x, y, rotation) 형태로, 박스의 중심 위치와 회전 여부를 포함
        h_new_box_size,w_new_box_size = new_box[0], new_box[1]

        self.eroded = self._search_empty(w_new_box_size, h_new_box_size)

        
        # 빈 공간의 좌표 찾기
        self.empty_space_xy = np.array(np.where(self.eroded == 0)).T

        if len(self.empty_space_xy) == 0:
            # 더 이상 둘 곳이 없으면 에피소드 종료
            reward = 0  
            done = True  # 에피소드 종료 신호
            info = {'message': 'No more space to place new box'}
            return reward, done, info


        # action = self.empty_space_xy[np.random.choice(len(self.empty_space_xy))]
        print('action',action)
        print('new_box',new_box)
        print('self.stacked_boxes',self.stacked_boxes)
        self.stacked_boxes.append((action[0], action[1], w_new_box_size, h_new_box_size))

        self.update_bin()

        # reward는 현재는 0으로 고정
        reward = 0
        done = False
        info = {}
        return reward, done, info



    def update_bin(self):
        for box in self.stacked_boxes:
            x, y, w, h = box
            self.bin[x-h:x+h, y-w:y+w] = 1

    def collision_detection(self):
        sudo_bin = np.zeros((self.bin_size, self.bin_size))
        for box in self.stacked_boxes:
            x, y, w, h = box
            sudo_bin[x-h:x+h, y-w:y+w] += 1

        if np.max(sudo_bin) > 1:
            collision_point = np.where(sudo_bin > 1)
            return collision_point
        else:
            return None
    
    def check_empty_is_zero(self):
        self.empty_space_xy = np.array(np.where(self.eroded == 0)).T
        return len(self.empty_space_xy) == 0



    def _search_empty(self, w_padding_size, h_padding_size):
        kernel = np.ones((2*h_padding_size+1, 2*w_padding_size+1), np.uint8) # height, width 순서로 넣어줘야함

        # inverted bin
        self.bin = 1 - self.bin
        
        # Erode 연산 적용: 패딩을 적용하여 실제로 박스를 놓을 수 있는 위치 찾기
        eroded = cv2.erode(self.bin, kernel, iterations=1)

        # 벽은 항상패딩
        eroded[:h_padding_size, :] = 0
        eroded[-h_padding_size:, :] = 0
        eroded[:, :w_padding_size] = 0
        eroded[:, -w_padding_size:] = 0

        eroded = 1 - eroded
        self.bin = 1 - self.bin

        return eroded

