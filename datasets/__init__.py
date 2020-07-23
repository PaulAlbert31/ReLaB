from datasets.cifar import CIFAR10, CIFAR100
from datasets.miniimagenet import make_dataset, MiniImagenet84
from mypath import Path
import os

def cifar10(regim='train', root=Path.db_root_dir('cifar10'), transform=None):
    return CIFAR10(root=root, train=regim=='train', download=False, transform=transform)

def cifar100(regim='train', root=Path.db_root_dir('cifar100'), transform=None):
    return CIFAR100(root=root, train=regim=='train', download=False, transform=transform)

def miniimagenet(root=Path.db_root_dir('miniimagenet'), transform=None, transform_test=None):
    train_data, train_labels, val_data, val_labels, test_data, test_labels = make_dataset(root=root)
    trainset = MiniImagenet84(train_data, train_labels, transform=transform)
    testset = MiniImagenet84(val_data, val_labels, transform=transform_test)
    return trainset, testset
