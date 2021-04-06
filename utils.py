import torch
import torchvision
import os
import datasets
import torch.nn.functional as F

def multi_class_loss(pred, target):
    pred = F.log_softmax(pred, dim=1)
    loss = - torch.sum(target*pred, dim=1)
    return loss

def make_data_loader(args, no_aug=False, **kwargs):
    
    if args.dataset == 'cifar10':
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]
    elif args.dataset == 'cifar100':
        mean = [0.5071, 0.4867, 0.4408]
        std = [0.2675, 0.2565, 0.2761]
    elif args.dataset == 'miniimagenet':
        mean = [0.4728, 0.4487, 0.4031]
        std = [0.2744, 0.2663 , 0.2806]

    size = 32
    if args.dataset == 'miniimagenet':
        size = 84

    if no_aug:
        transform_train = torchvision.transforms.Compose([
            torchvision.transforms.Resize(size),
            torchvision.transforms.CenterCrop(size),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean, std),
        ])
    else: #Some transformations to avoid fitting to the noise
        transform_train = torchvision.transforms.Compose([
            torchvision.transforms.RandomCrop(size, padding=4),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean, std),
        ])
        
    transform_test = torchvision.transforms.Compose([
        torchvision.transforms.Resize(size),
        torchvision.transforms.CenterCrop(size),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean, std)
    ])

    if args.dataset == "cifar10":
        trainset = datasets.cifar10(transform=transform_train, regim='train')
        testset = datasets.cifar10(transform=transform_test, regim='val')
    elif args.dataset == "cifar100":
        trainset = datasets.cifar100(transform=transform_train, regim='train')
        testset = datasets.cifar100(transform=transform_test, regim='val')
    elif args.dataset == "miniimagenet":
        trainset, testset = datasets.miniimagenet(transform=transform_train, transform_test=transform_test)
    else:
        raise NotImplementedError
    

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, **kwargs)    
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, **kwargs)
        
    return train_loader, test_loader

def create_save_folder(args):
    if not os.path.isdir(args.save_dir):
        os.mkdir(args.save_dir)
    if not os.path.isdir(os.path.join(args.save_dir, args.net + '_'  + args.dataset)):
        os.mkdir(os.path.join(args.save_dir, args.net + '_'  + args.dataset))
    if not os.path.isdir(os.path.join(args.save_dir, args.net + '_'  + args.dataset, args.exp_name)):
        os.mkdir(os.path.join(args.save_dir, args.net + '_'  + args.dataset, args.exp_name))

    return
       
