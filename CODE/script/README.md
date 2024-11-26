# Script Commands

## Step 1: Put the required :file_folder: folder `data` and `model`  

- The `data` :file_folder: folder structure should be as follows:
  - Download the `data` from this [link](https://drive.google.com/drive/u/1/folders/1KwXWNUQJ33vQ2ginCAOqjuUUTUwctCwL)

```python
data/
├── 0/
├── 1/
├── 2/
├── 3/
├── 4/
├── 5/
├── 6/
└── 7/
```

- The `model` :file_folder: folder structure should be as follows:
  - Download the `model` from this [link](https://drive.google.com/drive/u/1/folders/1hLCCeCH3r48IxkaK8nWWjyB-1M6I8S4T

```python
model/
├── PretrainAlexNet/
├── PretrainAlexNetPruneL1/
├── PretrainAlexNetPruneL2/
├── PretrainMobileNetV2/
├── PretrainMobileNetV2PruneL1/
└── PretrainMobileNetV2PruneL2/
```

## Step 2: Example commands to run the script

- Using a **Regular Train-Test** Split

```python
python predict.py --model_index 3 --model_dir ./model --data_dir ./data --batch_size 16 --test_size 0.2 --random_state 42
```

- Using **k-Fold Split**:

```python
python predict.py --model_index 3 --model_dir ./model --data_dir ./data --batch_size 16 --k_fold --num_folds 5 --fold_index 2 --random_state 42
```