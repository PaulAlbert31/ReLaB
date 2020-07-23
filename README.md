# Reliable Label Bootstrapping for semi-supervised learning (2020): [arXiv link]
Official implementation.

# How to run
Training on a dataset is sperated in 3 phases.

## First phase (optional)
Train a self-supervised algorithm on the dataset to learn good image descriptors, here we use AND (https://github.com/Raymond-sci/AND). You can alternatively use ImageNet weights (not studied in the paper) buy using --load imagenet on main_subset.py. Only ResNet50 supports this feature at the time and results are worse than SSL training.

```sh
$ cd AND
$ bash train_AND.sh
```

## Second phase 
Reliable sample bootstrapping: Propagate the few labels using the previously learned image descriptors and select an extended pool of reliable samples. The bash file also include the semi-supervised training with a Pseudo-Labeling algorithm (https://github.com/EricArazo/PseudoLabeling), you will need to use pretrained RotNet weights with the Pseudo-Labeling algorithm to reach good performance.

```sh
$ cd ..
$ bash train_subset.sh
```

## Final phase
Semi-supervised training on the extended reliable subset, here with ReMixMatch (https://github.com/google-research/remixmatch). The code is in Tensorflow.

```sh
$ cd remixmatch
$ bash train_RMM.sh
```

# Download pretrained weights for AND and for RotNet (Pseudo-Labeling)
https://drive.google.com/drive/folders/1FlUid993oMge6ppdrTmFCK7UhboPi4-c?usp=sharing

# Dependencies
## ReLaB, AND and Pseudo-Labeling
```sh
$ conda env create -f environment.yml
$ pip install torchcontrib
$ conda activate py3
```

## ReMixMatch (tensorflow)
```sh
$ cd remixmatch
$ conda env create -f condatfenv.yml
$ conda activate tf
```

# Some paper results, refer to arxiv for comparison numbers
Runs with this cleaned up code can slightly differ

| Dataset | Labeled Samples | SSL accuracy PL | SSL accuracy RMM |
| ------ | ------ | ------ | ------ |
|CIFAR10|40|83.25|90.65|
|CIFAR10|100|88.59|92.22|
|CIFAR100|400|42.71|51.13|
|CIFAR100|1000|49.36|57.90|
|miniImageNet|400|30.75|-|
|miniImageNet|1000|37.82|-|


# Please cite our paper if it helps your research
```

```
