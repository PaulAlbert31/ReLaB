export ML_DATA=data
export PYTHONPATH=.
# Download datasets

CUDA_VISIBLE_DEVICES= ./scripts/create_datasets.py
# Create unlabeled datasets

CUDA_VISIBLE_DEVICES= scripts/create_unlabeled.py $ML_DATA/SSL2/cifar10 $ML_DATA/cifar10-train.tfrecord &
CUDA_VISIBLE_DEVICES= scripts/create_unlabeled.py $ML_DATA/SSL2/cifar100 $ML_DATA/cifar100-train.tfrecord &
wait

# Create semi-supervised subsets
CUDA_VISIBLE_DEVICES= scripts/create_split.py --seed=1 --size=1 $ML_DATA/SSL2/cifar10 $ML_DATA/cifar10-train.tfrecord --labels ../cifar10-1/seed1/labels_seed1_1spc_cifar10.npz --labeled ../cifar10-1/seed1/subset_seed1_1spc_50c_cifar10.npz  --name cifar10.1spc.seed1@50c &
CUDA_VISIBLE_DEVICES= scripts/create_split.py --seed=1 --size=1 $ML_DATA/SSL2/cifar100 $ML_DATA/cifar100-train.tfrecord --labels ../cifar100-10/seed1/labels_seed1_10spc_cifar100.npz --labeled ../cifar100-10/seed1/subset_seed1_10spc_40c_cifar100.npz --name cifar100.10spc.seed1@40c &
wait

# The experiments will run for 256 epochs
CUDA_VISIBLE_DEVICES=0 python cta/cta_remixmatch.py --K=8 --dataset=cifar10.1spc.seed1@50c --train_dir ./experiments/remixmatch-c10-1-seed1
CUDA_VISIBLE_DEVICES=0 python cta/cta_remixmatch.py --K=8 --dataset=cifar100.10spc.seed1@40c --train_dir ./experiments/remixmatch-c100-10-seed1

# Extract accuracy
#python scripts/extract_accuracy.py experiments/remixmatch-c10-4-seed1/cifar10.d.d.d.4spc.seed1@50c-1/CTAugment_depth2_th0.80_decay0.990/CTAReMixMatch_K8_archresnet_batch64_beta0.75_filters32_lr0.002_nclass10_redux1st_repeat4_scales3_use_dmTrue_use_xeTrue_w_kl0.5_w_match1.5_w_rot0.5_warmup_kimg1024_wd0.02/
