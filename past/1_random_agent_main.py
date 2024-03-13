import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
from env.env import BoxPalletizingEnv
from env.util import generate_random_box


class random_agent:
    def __init__(self):
        pass

    def act(self, observation):
        eroded = observation[0]
        empty_space_xy = np.array(np.where(eroded == 0)).T
        new_box = observation[1]
        action = empty_space_xy[np.random.choice(len(empty_space_xy))]
        return action






def run_simulation():
    plt.ion()  # 대화형 모드 켜기
    env = BoxPalletizingEnv()
    # agent = random_agent()
    agent = random_agent()
    observation = env.reset()
    done = False

    while True:
        new_box = generate_random_box()
        plt.clf()  # 이전 그림을 지우고 새로운 상태를 그림

        env.eroded = env._search_empty(new_box[1], new_box[0])
        env.empty_space_xy = np.array(np.where(env.eroded == 0)).T
        if env.check_empty_is_zero():
            print('NO MORE SPACE')
            break

        new_observation = (env.eroded, new_box)
        action = agent.act(new_observation)
        reward, done, info = env.step(action, new_box)
        
        print(f'적재율 : {np.sum(env.bin)/(env.bin_size*env.bin_size)*100:.2f}%')
        print(f'collision: {env.collision_detection()}')
        
        if done:
            break
        
        bin = env.bin
        padding_img = env.eroded
        add_img = bin*0.5+padding_img*0.5



        plt.subplot(1, 3, 1)
        plt.title('Boxes in the bin')
        plt.imshow(bin, cmap='gray')
        plt.xlim(0, env.bin_size)
        plt.ylim(0, env.bin_size)
        plt.grid(True)


        plt.subplot(1, 3, 2)
        plt.title('Boxes in the bin with Padding')
        plt.imshow(add_img, cmap='gray')
        plt.scatter(action[1], action[0], c='r', s=100, alpha=0.5)
        plt.xlim(0, env.bin_size)
        plt.ylim(0, env.bin_size)
        plt.grid(True)

        plt.subplot(1, 3, 3)
        plt.title('Padding')
        plt.imshow(padding_img*0.5, cmap='gray',vmax=1, vmin=0)
        plt.xlim(0, env.bin_size)
        plt.ylim(0, env.bin_size)

        plt.draw()
        plt.pause(0.1)  # 그래프 업데이트를 위해 잠시 대기

        if env.collision_detection():
            print('BOX COLLISION')
            break


        observation = new_observation

    # 적재율
    loading_rate = np.sum(env.bin)/(env.bin_size*env.bin_size)*100
    return loading_rate
    

    


if __name__ == "__main__":
    loading_rates = [run_simulation() for _ in range(10)]
    
    print(f'적재율 평균 : {np.mean(loading_rates):.2f}%')
    print(f'적재율 표준편차 : {np.std(loading_rates):.2f}%')


    plt.ioff()
    
    