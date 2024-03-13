import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
from env.env import BoxPalletizingEnv
from env.util import generate_random_box, detecting_corner
from arg import get_args
from agent import ml_agent




    





def run_simulation(args):
    if args.render == True:
        plt.ion() 
    env = BoxPalletizingEnv()
    agent = ml_agent()
    
    for epoch in range(args.epochs):
        observation = env.reset()
        done = False
        loading_rates=[]
        while True:
            new_box = generate_random_box(base_size=3 ,max_diff=2)
            env.eroded = env._search_empty(new_box[1], new_box[0])
            env.empty_space_xy = np.array(np.where(env.eroded == 0)).T
            empty_check = env.check_empty_is_zero()

            print('모델 밖 eroded',env.eroded)

            if env.check_empty_is_zero():
                print('NO MORE SPACE')
                break

            new_observation = (env.eroded, new_box)
            action,corner_xy = agent(new_observation)
            print('모델밖 action',action)
            reward, done, info = env.step(action, new_box)
            loading_rate = np.sum(env.bin)/(env.bin_size*env.bin_size)*100
            print(f'적재율 : {loading_rate:.2f}%')
            print(f'collision: {env.collision_detection()}')
            
            if done:
                loading_rates.append(loading_rate)
                break
            
            bin = env.bin
            padding_img = env.eroded
            add_img = bin*0.5+padding_img*0.5


            if args.render == True:
                plt.subplot(1, 3, 1)
                plt.title('Boxes in the bin')
                plt.imshow(bin, cmap='gray')
                plt.xlim(0, env.bin_size)
                plt.ylim(0, env.bin_size)
                plt.grid(True)


                plt.subplot(1, 3, 2)
                plt.title('Padding + Placing Candidate')
                plt.imshow(padding_img*0.5, cmap='gray',vmax=1, vmin=0)
                plt.scatter(corner_xy[:,1], corner_xy[:,0], c='b', s=100, alpha=0.5)
                plt.xlim(0, env.bin_size)
                plt.ylim(0, env.bin_size)
                


                plt.subplot(1, 3, 3)
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
            if args.render == True:
                plt.clf()
        
    return loading_rates
        

    


if __name__ == "__main__":
    args = get_args()

    loading_rates = run_simulation(args)
    print(loading_rates)

    print(f'적재율 평균 : {np.mean(loading_rates):.2f}%')
    print(f'적재율 표준편차 : {np.std(loading_rates):.2f}%')

    if args.render == True:
        plt.ioff()
