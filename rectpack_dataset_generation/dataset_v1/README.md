# Configurations of the dataset_V1



### Data Structure
- dataset_v1
    - box sequence : randomly generated 60 box sequences

    - packing_logs
        - algorithm_name : 알고리즘 명
        - loading_rate : 박스시퀀스에 대한 해당 알고리즘의 적재율 
        - bin_size : bin 사이즈
        - masking_position : Masking 포지션의 Trajectory
        - box_position_list : box의 위치와 크기의 Trajectory
        - number_of_loaded_box : 최종 적재된 박스의 개수
        - number_of_unloaded_box : 최종 적재되지 않은 박스의 개수 (60 - number_of_loaded_box)



**Data Loading**
```python
dataset_v1 = np.load('./dataset_v1/bpp_0.npy', allow_pickle=True)
```


**Box Sequence** (list)
```python
box_sequence = dataset_v1[0]
```


**Packing Logs** (list) : packing 알고리즘이 리스트안에 각각의 dictionary로 저장되어 있음
```python
packing_logs = dataset_v1[1]
print(packing_logs[0].keys()) ## v1 오타 algorhithm_name으로 되어있음 (v2부터 개선)
>>> dict_keys(['algorhithm_name', 'loading_rate', 'bin_size', 'masking_position', 'box_position_list', 'number_of_loaded_box', 'number_of_unloaded_box'])

```

### analysis
check /rectpack_dataset_generation/checking_and_analysis/analysis_v1.ipynb







# V2 Update 예정내용

- Rotation 추가 (w>h)
- bin_size 는 1200mm / 1000mm 를 반영
- box_size는 200,250,300,350,400mm 를 반영
- Masking Algorithm 속도개선 : multi-processing 적용, Convolution연산말고 다른 방법으로 개선

