export ML_DATA=data
export PYTHONPATH=.
# Download datasets
CUDA_VISIBLE_DEVICES= ./scripts/create_datasets.py
# Create unlabeled datasets

CUDA_VISIBLE_DEVICES= scripts/create_unlabeled.py $ML_DATA/SSL2/cifar10 $ML_DATA/cifar10-train.tfrecord 

# Create semi-supervised subsets
#1spc
CUDA_VISIBLE_DEVICES= scripts/create_split.py --seed=1 --size=1 $ML_DATA/SSL2/cifar10 $ML_DATA/cifar10-train.tfrecord --labels ../cifar10-1/seed1/labels_seed1_1spc_cifar10.npz --labeled ../cifar10-1/seed1/subset_seed1_1spc_50c_cifar10.npz  --name cifar10.1spc.seed1@50c &
CUDA_VISIBLE_DEVICES= scripts/create_split.py --seed=2 --size=1 $ML_DATA/SSL2/cifar10 $ML_DATA/cifar10-train.tfrecord --labels ../cifar10-1/seed2/labels_seed2_1spc_cifar10.npz --labeled ../cifar10-1/seed2/subset_seed2_1spc_50c_cifar10.npz  --name cifar10.1spc.seed2@50c &
CUDA_VISIBLE_DEVICES= scripts/create_split.py --seed=3 --size=1 $ML_DATA/SSL2/cifar10 $ML_DATA/cifar10-train.tfrecord --labels ../cifar10-1/seed3/labels_seed3_1spc_cifar10.npz --labeled ../cifar10-1/seed3/subset_seed3_1spc_50c_cifar10.npz  --name cifar10.1spc.seed3@50c &
#4spc
CUDA_VISIBLE_DEVICES= scripts/create_split.py --seed=1 --size=1 $ML_DATA/SSL2/cifar10 $ML_DATA/cifar10-train.tfrecord --labels ../cifar10-1/seed1/labels_seed1_4spc_cifar10.npz --labeled ../cifar10-1/seed1/subset_seed1_4spc_50c_cifar10.npz  --name cifar10.4spc.seed1@50c &
CUDA_VISIBLE_DEVICES= scripts/create_split.py --seed=2 --size=1 $ML_DATA/SSL2/cifar10 $ML_DATA/cifar10-train.tfrecord --labels ../cifar10-1/seed2/labels_seed2_4spc_cifar10.npz --labeled ../cifar10-1/seed2/subset_seed2_4spc_50c_cifar10.npz  --name cifar10.4spc.seed2@50c &
CUDA_VISIBLE_DEVICES= scripts/create_split.py --seed=3 --size=1 $ML_DATA/SSL2/cifar10 $ML_DATA/cifar10-train.tfrecord --labels ../cifar10-1/seed3/labels_seed3_4spc_cifar10.npz --labeled ../cifar10-1/seed3/subset_seed3_4spc_50c_cifar10.npz  --name cifar10.4spc.seed3@50c &
#10spc
CUDA_VISIBLE_DEVICES= scripts/create_split.py --seed=1 --size=1 $ML_DATA/SSL2/cifar10 $ML_DATA/cifar10-train.tfrecord --labels ../cifar10-1/seed1/labels_seed1_10spc_cifar10.npz --labeled ../cifar10-1/seed1/subset_seed1_10spc_50c_cifar10.npz  --name cifar10.10spc.seed1@50c &
CUDA_VISIBLE_DEVICES= scripts/create_split.py --seed=2 --size=1 $ML_DATA/SSL2/cifar10 $ML_DATA/cifar10-train.tfrecord --labels ../cifar10-1/seed2/labels_seed2_10spc_cifar10.npz --labeled ../cifar10-1/seed2/subset_seed2_10spc_50c_cifar10.npz  --name cifar10.10spc.seed2@50c &
CUDA_VISIBLE_DEVICES= scripts/create_split.py --seed=3 --size=1 $ML_DATA/SSL2/cifar10 $ML_DATA/cifar10-train.tfrecord --labels ../cifar10-1/seed3/labels_seed3_10spc_cifar10.npz --labeled ../cifar10-1/seed3/subset_seed3_10spc_50c_cifar10.npz  --name cifar10.10spc.seed3@50c &
#25spc
CUDA_VISIBLE_DEVICES= scripts/create_split.py --seed=1 --size=1 $ML_DATA/SSL2/cifar10 $ML_DATA/cifar10-train.tfrecord --labels ../cifar10-1/seed1/labels_seed1_25spc_cifar10.npz --labeled ../cifar10-1/seed1/subset_seed1_25spc_50c_cifar10.npz  --name cifar10.25spc.seed1@50c &
CUDA_VISIBLE_DEVICES= scripts/create_split.py --seed=2 --size=1 $ML_DATA/SSL2/cifar10 $ML_DATA/cifar10-train.tfrecord --labels ../cifar10-1/seed2/labels_seed2_25spc_cifar10.npz --labeled ../cifar10-1/seed2/subset_seed2_25spc_50c_cifar10.npz  --name cifar10.25spc.seed2@50c &
CUDA_VISIBLE_DEVICES= scripts/create_split.py --seed=3 --size=1 $ML_DATA/SSL2/cifar10 $ML_DATA/cifar10-train.tfrecord --labels ../cifar10-1/seed3/labels_seed3_25spc_cifar10.npz --labeled ../cifar10-1/seed3/subset_seed3_25spc_50c_cifar10.npz  --name cifar10.25spc.seed3@50c &
wait

# The experiments will run for 256 epochs
#1spc
CUDA_VISIBLE_DEVICES=0 python cta/cta_remixmatch.py --K=8 --dataset=cifar10.1spc.seed1@50c --train_dir ./experiments/remixmatch-c10-1-seed1 &
CUDA_VISIBLE_DEVICES=1 python cta/cta_remixmatch.py --K=8 --dataset=cifar10.1spc.seed2@50c --train_dir ./experiments/remixmatch-c10-1-seed2 &
wait
CUDA_VISIBLE_DEVICES=0 python cta/cta_remixmatch.py --K=8 --dataset=cifar10.1spc.seed3@50c --train_dir ./experiments/remixmatch-c10-1-seed3 &

#4spc
CUDA_VISIBLE_DEVICES=1 python cta/cta_remixmatch.py --K=8 --dataset=cifar10.4spc.seed1@50c --train_dir ./experiments/remixmatch-c10-4-seed1 &
wait
CUDA_VISIBLE_DEVICES=0 python cta/cta_remixmatch.py --K=8 --dataset=cifar10.4spc.seed2@50c --train_dir ./experiments/remixmatch-c10-4-seed2 &
CUDA_VISIBLE_DEVICES=1 python cta/cta_remixmatch.py --K=8 --dataset=cifar10.4spc.seed3@50c --train_dir ./experiments/remixmatch-c10-4-seed3 &
wait

#10spc
CUDA_VISIBLE_DEVICES=0 python cta/cta_remixmatch.py --K=8 --dataset=cifar10.10spc.seed1@50c --train_dir ./experiments/remixmatch-c10-10-seed1 &
CUDA_VISIBLE_DEVICES=1 python cta/cta_remixmatch.py --K=8 --dataset=cifar10.10spc.seed2@50c --train_dir ./experiments/remixmatch-c10-10-seed2 &
wait
CUDA_VISIBLE_DEVICES=0 python cta/cta_remixmatch.py --K=8 --dataset=cifar10.10spc.seed3@50c --train_dir ./experiments/remixmatch-c10-10-seed3 &

#25spc
CUDA_VISIBLE_DEVICES=1 python cta/cta_remixmatch.py --K=8 --dataset=cifar10.25spc.seed1@50c --train_dir ./experiments/remixmatch-c10-25-seed1 &
wait
CUDA_VISIBLE_DEVICES=0 python cta/cta_remixmatch.py --K=8 --dataset=cifar10.25spc.seed2@50c --train_dir ./experiments/remixmatch-c10-25-seed2 &
CUDA_VISIBLE_DEVICES=1 python cta/cta_remixmatch.py --K=8 --dataset=cifar10.25spc.seed3@50c --train_dir ./experiments/remixmatch-c10-25-seed3 &
wait

# Extract accuracy
#python scripts/extract_accuracy.py experiments/remixmatch-c10-4-seed1/cifar10.d.d.d.4spc.seed1@50c-1/CTAugment_depth2_th0.80_decay0.990/CTAReMixMatch_K8_archresnet_batch64_beta0.75_filters32_lr0.002_nclass10_redux1st_repeat4_scales3_use_dmTrue_use_xeTrue_w_kl0.5_w_match1.5_w_rot0.5_warmup_kimg1024_wd0.02/
