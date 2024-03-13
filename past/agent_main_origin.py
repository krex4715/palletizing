import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
from env.env import BoxPalletizingEnv
from env.util import generate_random_box
from arg import get_args



class ml_agent:
    def __init__(self):
        pass

    def act(self, observation):
        eroded = observation[0]
        print('eroded shape',eroded.shape)
        new_box = observation[1]
        print('new_box',new_box)
        corner_xy,edge = self.detecting_corner(eroded)
        empty_space_xy = np.array(np.where(eroded == 0)).T
        
        print('corner_xy',corner_xy)


        # action = empty_space_xy[np.argmin(empty_space_xy[:, 0] + empty_space_xy[:, 1])]
        action = corner_xy[np.random.choice(len(corner_xy))]
        return action , corner_xy,edge
    
    def detecting_corner(self, erode):
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





def run_simulation():
    if args.render == True:
        plt.ion()  # 대화형 모드 켜기
    env = BoxPalletizingEnv()
    agent = ml_agent()
    observation = env.reset()
    done = False


    while True:
        if args.render == True:
            plt.clf()  # 이전 그림을 지우고 새로운 상태를 그림


        new_box = generate_random_box()
        env.eroded = env._search_empty(new_box[1], new_box[0])
        env.empty_space_xy = np.array(np.where(env.eroded == 0)).T
        empty_check = env.check_empty_is_zero()


        if env.check_empty_is_zero():
            print('NO MORE SPACE')
            break

        new_observation = (env.eroded, new_box)
        action,corner_xy,edge = agent.act(new_observation)
        reward, done, info = env.step(action, new_box)
        
        print(f'적재율 : {np.sum(env.bin)/(env.bin_size*env.bin_size)*100:.2f}%')
        print(f'collision: {env.collision_detection()}')
        
        if done:
            break
        
        bin = env.bin
        padding_img = env.eroded
        add_img = bin*0.5+padding_img*0.5


        if args.render == True:
            plt.subplot(1, 4, 1)
            plt.title('Boxes in the bin')
            plt.imshow(bin, cmap='gray')
            plt.xlim(0, env.bin_size)
            plt.ylim(0, env.bin_size)
            plt.grid(True)


            plt.subplot(1, 4, 2)
            plt.title('Padding + Placing Candidate')
            plt.imshow(padding_img*0.5, cmap='gray',vmax=1, vmin=0)
            plt.scatter(corner_xy[:,1], corner_xy[:,0], c='b', s=100, alpha=0.5)
            plt.xlim(0, env.bin_size)
            plt.ylim(0, env.bin_size)
            
            

            plt.subplot(1, 4, 3)
            plt.imshow(edge, cmap='gray')
            plt.xlim(0, env.bin_size)
            plt.ylim(0, env.bin_size)


            plt.subplot(1, 4, 4)
            plt.title('Boxes in the bin with Padding')
            plt.imshow(add_img, cmap='gray')
            plt.scatter(action[1], action[0], c='r', s=100, alpha=0.5)
            plt.xlim(0, env.bin_size)
            plt.ylim(0, env.bin_size)
            collision = env.collision_detection()
            if collision is not None:
                print('BOX COLLISION')
                plt.scatter(collision[1], collision[0], c='r', s=300, alpha=1.0)
                ##여기서 멈추기
                plt.pause(1000)
                break
            
            plt.grid(True)


            plt.draw()
            plt.pause(0.1)  # 그래프 업데이트를 위해 잠시 대기

        



        observation = new_observation

    # 적재율
    loading_rate = np.sum(env.bin)/(env.bin_size*env.bin_size)*100
    return loading_rate
    

    


if __name__ == "__main__":
    args = get_args()

    loading_rates = [run_simulation() for _ in range(10)]
    
    print(f'적재율 평균 : {np.mean(loading_rates):.2f}%')
    print(f'적재율 표준편차 : {np.std(loading_rates):.2f}%')

    if args.render == True:
        plt.ioff()
    
    