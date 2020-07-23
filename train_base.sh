#!/bin/bash

# 1 sample per class with PseudoLabeling
CUDA_VISIBLE_DEVICES=0 python main_subset.py --net resnet50 --epochs 60 --dataset cifar10 --save-dir cifar10-1 --batch-size 512 --load weightsAND/res50/cifar10/weights_AND_cifar10 --lr 0.1 --diffuse --spc 1 --epochs 60  --seed 1 --boot-spc 50

CUDA_VISIBLE_DEVICES=0 python3 main_pseudolab.py --epoch 200 --dataset_type "sym_noise_warmUp" --M 100 --M 150 --DA "jitter" --experiment_name "WuP_model" --download --dataset cifar10 --labels cifar10-1/seed1/labels_seed1_1spc_cifar10.npz --labeled_samples cifar10-1/seed1/subset_seed1_1spc_50c_cifar10.npz --lr 0.1  --load weightsrot/wide282/cifar10/model_rot_cifar10 --network "WRN28_2" --freeze

CUDA_VISIBLE_DEVICES=0 python3 main_pseudolab.py --epoch 400  --M 250 --M 350 --initial_epoch 200 --DA "jitter" --experiment_name "cifar10-1-seed1" --download --dataset cifar10 --labels cifar10-1/seed1/labels_seed1_1spc_cifar10.npz --labeled_samples cifar10-1/seed1/subset_seed1_1spc_50c_cifar10.npz --labeled_batch_size 20 --network "WRN28_2" --freeze


# 10 samples per class with PseudoLabeling
CUDA_VISIBLE_DEVICES=0 python main_subset.py --net resnet50 --epochs 60 --dataset cifar100 --save-dir cifar10-10 --batch-size 512 --load weightsAND/res50/cifar100/weights_AND_cifar100 --lr 0.1 --diffuse --spc 10 --epochs 60 --seed 1 --boot-spc 40

CUDA_VISIBLE_DEVICES=0 python3 main_pseudolab.py --epoch 200 --dataset_type "sym_noise_warmUp" --M 100 --M 150 --DA "jitter" --experiment_name "WuP_model" --download --dataset cifar100 --labels cifar100-10/seed1/labels_seed1_1spc_cifar100.npz --labeled_samples cifar100-10/seed1/subset_seed1_1spc_40c_cifar100.npz --lr 0.1  --load weightsrot/wide282/cifar100/model_rot_cifar100 --network "WRN28_2" --freeze

CUDA_VISIBLE_DEVICES=0 python3 main_pseudolab.py --epoch 400  --M 250 --M 350 --initial_epoch 200 --DA "jitter" --experiment_name "cifar100-10-seed1" --download --dataset cifar100 --labels cifar100-10/seed1/labels_seed1_1spc_cifar100.npz --labeled_samples cifar100-10/seed1/subset_seed1_1spc_40c_cifar100.npz --labeled_batch_size 20 --network "WRN28_2" --freeze


# previous experiments with RMM
conda deactivate
conda activate tf # create the new env with remixmatch/condatfenv.yml

cd remixmatch

bash run_RMM.sh


