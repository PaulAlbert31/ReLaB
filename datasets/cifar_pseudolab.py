import torch
import torchvision as tv
import numpy as np
from PIL import Image
import time
from tqdm import tqdm
import pickle


def get_dataset(args, transform_train, transform_val, dst_folder, dataset='cifar10'):
    # prepare datasets
    if dataset == 'cifar10':
        cifar10_train_val = tv.datasets.CIFAR10(args.train_root, train=True, download=args.download)
        args.num_classes = 10
    elif dataset == 'cifar100':
        cifar10_train_val = tv.datasets.CIFAR100(args.train_root, train=True, download=args.download)
        args.num_classes = 100
    else:
        raise NotImplementedError('Dataset {} not implemented'.format(dataset))

    # get train/val dataset
    train_indexes, val_indexes = train_val_split(args, cifar10_train_val.targets)
    if dataset == 'cifar10': 
        train = Cifar10(args, dst_folder, train_indexes, train=True, transform=transform_train, pslab_transform = transform_val)
        validation = Cifar10(args, dst_folder, val_indexes, train=True, transform=transform_val, pslab_transform = transform_val)
    elif dataset == 'cifar100':
        train = Cifar100(args, dst_folder, train_indexes, train=True, transform=transform_train, pslab_transform = transform_val)
        validation = Cifar100(args, dst_folder, val_indexes, train=True, transform=transform_val, pslab_transform = transform_val)

    if args.dataset_type == 'sym_noise_warmUp':
        clean_labels, noisy_labels, noisy_indexes, clean_indexes = train.symmetric_noise_warmUp_semisup()
    elif args.dataset_type == 'semiSup':
        clean_labels, noisy_labels, noisy_indexes, clean_indexes = train.symmetric_noise_for_semiSup()

    return train, clean_labels, noisy_labels, noisy_indexes, clean_indexes, validation


def train_val_split(args, train_val):

    np.random.seed(args.seed_val)
    train_val = np.array(train_val)
    train_indexes = []
    val_indexes = []
    val_num = int(args.val_samples / args.num_classes)

    for id in range(args.num_classes):
        indexes = np.where(train_val == id)[0]
        np.random.shuffle(indexes)
        val_indexes.extend(indexes[:val_num])
        train_indexes.extend(indexes[val_num:])
    np.random.shuffle(train_indexes)
    np.random.shuffle(val_indexes)

    return train_indexes, val_indexes


class Cifar10(tv.datasets.CIFAR10):
    # including hard labels & soft labels
    def __init__(self, args, dst_folder, train_indexes=None, train=True, transform=None, target_transform=None, pslab_transform=None, download=False):
        super(Cifar10, self).__init__(args.train_root, train=train, transform=transform, target_transform=target_transform, download=download)
        self.args = args
        self.soft_labels = np.zeros((len(self.targets), 10), dtype=np.float32)
        self.prediction = np.zeros((self.args.epoch_update, len(self.data), 10), dtype=np.float32)
        self.z_exp_labels = np.zeros((len(self.targets), 10), dtype=np.float32)
        self.Z_exp_labels = np.zeros((len(self.targets), 10), dtype=np.float32)
        self.z_soft_labels = np.zeros((len(self.targets), 10), dtype=np.float32)

        self._count = 0
        self.dst = dst_folder.replace('second','first') + '/labels.npz'
        self.alpha = 0.6
        self.gaus_noise = self.args.gausTF

        self.pslab_transform = pslab_transform
        
        try:
            labels = torch.from_numpy(np.load(args.labels)['arr_0'])
            self.soft_labels = labels.clone().numpy()
            labels = torch.argmax(labels, dim=1)
        except:
            labels = torch.from_numpy(np.load(args.labels)['arr_0'])
            self.soft_labels = np.zeros((len(labels), 10))
            for i, l in enumerate(labels):
                self.soft_labels[i][l] = 1
            
        self.original_labels = labels.clone().numpy()
        self.labeled_indices = torch.sort(torch.from_numpy(np.load(args.labeled_samples)['arr_0']).view(-1))[0].long()

        print((labels[self.labeled_indices] == torch.tensor(self.targets)[self.labeled_indices]).sum().float()/self.labeled_indices.shape[0])
        per_class_acc = torch.zeros(labels.max() +1)
        per_class = torch.zeros(labels.max() +1)
        for l1, l2 in zip(labels[self.labeled_indices], torch.tensor(self.targets)[self.labeled_indices]):
            per_class_acc[l1] += (l1 == l2)
            per_class[l1] += 1
        per_class[per_class == 0] = 1#no more nan
        print('Subset per class accuracy: ', per_class_acc/per_class)
        print('Mean class: ', torch.mean(per_class_acc / per_class))
        un = torch.unique(labels[self.labeled_indices], sorted=True)
        self.count = 1/torch.stack([(labels[self.labeled_indices]==x_u).sum() for x_u in un]).float()
        self.targets = labels.clone().numpy()
        self._num = int(self.original_labels.shape[0]-self.labeled_indices.shape[0])

    def symmetric_noise_for_semiSup(self):
        np.random.seed(self.args.seed)

        original_labels = np.copy(self.targets)

        noisy_indexes = torch.ones(original_labels.shape[0])
        noisy_indexes[self.labeled_indices] = 0
        noisy_indexes = torch.arange(noisy_indexes.shape[0])[noisy_indexes>0]

        return original_labels, self.targets,  noisy_indexes,  self.labeled_indices.clone().numpy()

    #return original_labels, self.targets,  np.asarray(noisy_indexes),  np.asarray(clean_indexes)


    def symmetric_noise_warmUp_semisup(self):
        np.random.seed(self.args.seed)

        original_labels = np.copy(self.targets)

        train_indexes = self.labeled_indices.clone().numpy()
        noisy_indexes = torch.ones(original_labels.shape[0])
        noisy_indexes[self.labeled_indices] = 0
        noisy_indexes = torch.arange(noisy_indexes.shape[0])[noisy_indexes>0]
        
        self.data = self.data[train_indexes]
        self.targets = np.array(self.targets)[train_indexes]
        self.soft_labels = np.array(self.soft_labels)[train_indexes]

        self.prediction = np.zeros((self.args.epoch_update, len(self.data), self.args.num_classes), dtype=np.float32)
        self.z_exp_labels = np.zeros((len(self.targets), self.args.num_classes), dtype=np.float32)
        self.Z_exp_labels = np.zeros((len(self.targets), self.args.num_classes), dtype=np.float32)
        self.z_soft_labels = np.zeros((len(self.targets), self.args.num_classes), dtype=np.float32)

        noisy_indexes = np.asarray([])

        return original_labels[train_indexes], self.targets, np.asarray(noisy_indexes), np.asarray(train_indexes)

    def update_labels_randRelab(self, result, train_noisy_indexes, rand_ratio):
        #train_noisy_indexes = np.arange(50000)
        idx = self._count % self.args.epoch_update
        self.prediction[idx,:] = result
        nb_noisy = len(train_noisy_indexes)
        nb_rand = int(nb_noisy*rand_ratio)
        idx_noisy_all = list(range(nb_noisy))
        idx_noisy_all = np.random.permutation(idx_noisy_all)

        idx_rand = idx_noisy_all[:nb_rand]
        idx_relab = idx_noisy_all[nb_rand:]

        if rand_ratio == 0.0:
            idx_relab = list(range(len(train_noisy_indexes)))
            idx_rand = []

        if self._count >= self.args.epoch_begin:

            relabel_indexes = list(train_noisy_indexes[idx_relab])
            self.soft_labels[relabel_indexes] = result[relabel_indexes]

            self.targets[relabel_indexes] = self.soft_labels[relabel_indexes].argmax(axis = 1).astype(np.int64)


            for idx_num in train_noisy_indexes[idx_rand]:
                new_soft = np.ones(self.args.num_classes)
                new_soft = new_soft*(1/self.args.num_classes)

                self.soft_labels[idx_num] = new_soft
                self.targets[idx_num] = self.soft_labels[idx_num].argmax(axis = 0).astype(np.int64)


            print("Samples relabeled with the prediction: ", str(len(idx_relab)))
            print("Samples relabeled with '{0}': ".format(self.args.relab), str(len(idx_rand)))

        self.Z_exp_labels = self.alpha * self.Z_exp_labels + (1. - self.alpha) * self.prediction[idx,:]
        self.z_exp_labels =  self.Z_exp_labels * (1. / (1. - self.alpha ** (self._count + 1)))

        self._count += 1

        # save params
        if self._count == self.args.epoch:
            np.savez(self.dst, data=self.data, hard_labels=self.targets, soft_labels=self.soft_labels)


    def gaussian(self, ins, mean, stddev):
        noise = ins.data.new(ins.size()).normal_(mean, stddev)
        return ins + noise

    def __getitem__(self, index):
        img, labels, soft_labels, z_exp_labels = self.data[index], self.targets[index], self.soft_labels[index], self.z_exp_labels[index]
        img = Image.fromarray(img)

        if self.args.DApseudolab == "False":
            img_pseudolabels = self.pslab_transform(img)
        else:
            img_pseudolabels = 0

        if self.transform is not None:
            img = self.transform(img)
            if self.gaus_noise:
                img = self.gaussian(img, 0.0, 0.15)

        if self.target_transform is not None:
            labels = self.target_transform(labels)

        return img, img_pseudolabels, labels, soft_labels, index



class Cifar100(tv.datasets.CIFAR100):
    # including hard labels & soft labels
    def __init__(self, args, dst_folder, train_indexes=None, train=True, transform=None, target_transform=None, pslab_transform=None, download=False):
        super(Cifar100, self).__init__(args.train_root, train=train, transform=transform, target_transform=target_transform, download=download)
        self.args = args
        if train_indexes is not None and False:
            self.data = self.data[train_indexes]
            self.targets = np.array(self.targets)[train_indexes]
        self.soft_labels = np.zeros((len(self.targets), 100), dtype=np.float32)
        self.prediction = np.zeros((self.args.epoch_update, len(self.data), 100), dtype=np.float32)
        self.z_exp_labels = np.zeros((len(self.targets), 100), dtype=np.float32)
        self.Z_exp_labels = np.zeros((len(self.targets), 100), dtype=np.float32)
        self.z_soft_labels = np.zeros((len(self.targets), 100), dtype=np.float32)

        self._count = 0
        self.dst = dst_folder.replace('second','first') + '/labels.npz'
        self.alpha = 0.6
        self.gaus_noise = self.args.gausTF

        self.pslab_transform = pslab_transform
        try:
            labels = torch.from_numpy(np.load(args.labels)['arr_0'])
            self.soft_labels = labels.clone().numpy()
            labels = torch.argmax(labels, dim=1)
        except:
            labels = torch.from_numpy(np.load(args.labels)['arr_0'])
                
        self.original_labels = labels.clone().numpy()
        self.labeled_indices = torch.sort(torch.from_numpy(np.load(args.labeled_samples)['arr_0']).view(-1))[0].long()

        print((labels[self.labeled_indices] == torch.tensor(self.targets)[self.labeled_indices]).sum().float()/self.labeled_indices.shape[0])
        per_class_acc = torch.zeros(labels.max() +1)
        per_class = torch.zeros(labels.max() +1)
        for l1, l2 in zip(labels[self.labeled_indices], torch.tensor(self.targets)[self.labeled_indices]):
            per_class_acc[l1] += (l1 == l2)
            per_class[l1] += 1
        per_class[per_class == 0] = 1#no more nan        
        print('Subset per class accuracy: ', per_class_acc/per_class)
        print('Mean class: ', torch.mean(per_class_acc / per_class))
        un = torch.unique(labels[self.labeled_indices], sorted=True)
        self.count = 1/torch.stack([(labels[self.labeled_indices]==x_u).sum() for x_u in un]).float()
        self.targets = labels.clone().numpy()
        self._num = int(self.original_labels.shape[0]-self.labeled_indices.shape[0])

    def symmetric_noise_for_semiSup(self):
        np.random.seed(self.args.seed)

        original_labels = np.copy(self.targets)

        noisy_indexes = torch.ones(original_labels.shape[0])
        noisy_indexes[self.labeled_indices] = 0
        noisy_indexes = torch.arange(noisy_indexes.shape[0])[noisy_indexes>0]

        return original_labels, self.targets,  noisy_indexes,  self.labeled_indices.clone().numpy()



    def symmetric_noise_warmUp_semisup(self):
        np.random.seed(self.args.seed)

        original_labels = np.copy(self.targets)

        train_indexes = self.labeled_indices.clone().numpy()
        noisy_indexes = torch.ones(original_labels.shape[0])
        noisy_indexes[self.labeled_indices] = 0
        noisy_indexes = torch.arange(noisy_indexes.shape[0])[noisy_indexes>0]
        
        self.data = self.data[train_indexes]
        self.targets = np.array(self.targets)[train_indexes]
        self.soft_labels = np.array(self.soft_labels)[train_indexes]

        self.prediction = np.zeros((self.args.epoch_update, len(self.data), self.args.num_classes), dtype=np.float32)
        self.z_exp_labels = np.zeros((len(self.targets), self.args.num_classes), dtype=np.float32)
        self.Z_exp_labels = np.zeros((len(self.targets), self.args.num_classes), dtype=np.float32)
        self.z_soft_labels = np.zeros((len(self.targets), self.args.num_classes), dtype=np.float32)

        noisy_indexes = np.asarray([])

        return original_labels[train_indexes], self.targets, np.asarray(noisy_indexes), np.asarray(train_indexes)

    def update_labels_randRelab(self, result, train_noisy_indexes, rand_ratio):
        #train_noisy_indexes = np.arange(50000)
        idx = self._count % self.args.epoch_update
        self.prediction[idx,:] = result
        nb_noisy = len(train_noisy_indexes)
        nb_rand = int(nb_noisy*rand_ratio)
        idx_noisy_all = list(range(nb_noisy))
        idx_noisy_all = np.random.permutation(idx_noisy_all)

        idx_rand = idx_noisy_all[:nb_rand]
        idx_relab = idx_noisy_all[nb_rand:]

        if rand_ratio == 0.0:
            idx_relab = list(range(len(train_noisy_indexes)))
            idx_rand = []

        if self._count >= self.args.epoch_begin:

            relabel_indexes = list(train_noisy_indexes[idx_relab])
            self.soft_labels[relabel_indexes] = result[relabel_indexes]

            self.targets[relabel_indexes] = self.soft_labels[relabel_indexes].argmax(axis = 1).astype(np.int64)


            for idx_num in train_noisy_indexes[idx_rand]:
                new_soft = np.ones(self.args.num_classes)
                new_soft = new_soft*(1/self.args.num_classes)

                self.soft_labels[idx_num] = new_soft
                self.targets[idx_num] = self.soft_labels[idx_num].argmax(axis = 0).astype(np.int64)

            weights = torch.zeros(self.targets.max()+1)
            for l in self.targets:
                weights[l] += 1
            weights = 1/weights
            weights = torch.tensor([weights[l] for l in self.targets])#sample weight
            
            print("Samples relabeled with the prediction: ", str(len(idx_relab)))
            print("Samples relabeled with '{0}': ".format(self.args.relab), str(len(idx_rand)))

        self.Z_exp_labels = self.alpha * self.Z_exp_labels + (1. - self.alpha) * self.prediction[idx,:]
        self.z_exp_labels =  self.Z_exp_labels * (1. / (1. - self.alpha ** (self._count + 1)))

        self._count += 1

        # save params
        if self._count == self.args.epoch:
            np.savez(self.dst, data=self.data, hard_labels=self.targets, soft_labels=self.soft_labels)
            
        return weights

    def gaussian(self, ins, mean, stddev):
        noise = ins.data.new(ins.size()).normal_(mean, stddev)
        return ins + noise

    def __getitem__(self, index):
        img, labels, soft_labels, z_exp_labels = self.data[index], self.targets[index], self.soft_labels[index], self.z_exp_labels[index]
        img = Image.fromarray(img)

        if self.args.DApseudolab == "False":
            img_pseudolabels = self.pslab_transform(img)
        else:
            img_pseudolabels = 0

        if self.transform is not None:
            img = self.transform(img)
            if self.gaus_noise:
                img = self.gaussian(img, 0.0, 0.15)

        if self.target_transform is not None:
            labels = self.target_transform(labels)

        return img, img_pseudolabels, labels, soft_labels, index
