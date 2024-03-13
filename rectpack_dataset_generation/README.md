# Offline BPP Dataset Generation & Visualization


## How to use

### Dataset generation
```python
# save to ./dataset/
# -n : number of dataset
python make_dataset.py -n 1000
```

### Visualization
```python
# -i : index of dataset
# -a : algorithm name
# -s : rendering sample rate
# -b : best algorithm

python visualize.py -i 16 -s 0.3 -a GuillotineBssfMinas
# -b : 모든 packing알고리즘 중 best 알고리즘 
python visualize.py -i 0 -b
```

