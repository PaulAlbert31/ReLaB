import torch
import torchvision
import os
import datasets
import torch.nn.functional as F

def multi_class_loss(pred, target):
    pred = F.log_softmax(pred, dim=1)
    loss = - torch.sum(target*pred, dim=1)
    return loss

def make_data_loader(args, no_aug=False, transform=None, **kwargs):
    
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

    if transform is not None: #Give your own transformation
        print('Custom DA')
        transform_train = torchvision.transforms.Compose([
            transform,
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean, std),
        ])
    elif no_aug:
        print('No DA')
        transform_train = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean, std),
        ])
    else: #Some transformations to avoid fitting to the noise
        print('Basic DA')
        transform_train = torchvision.transforms.Compose([
            torchvision.transforms.RandomCrop(size, padding=4),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean, std),
        ])
        
    transform_test = torchvision.transforms.Compose([
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
    
    if no_aug:
        train_loader =  torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=False, **kwargs) #Sequential loader for sample loss tracking
    else:
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, **kwargs) #Normal training
        
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, **kwargs)
        
    return train_loader, test_loader

def create_save_folder(args):
    try:
        os.mkdir(args.save_dir)
    except:
        pass
    try:
        os.mkdir(os.path.join(args.save_dir, args.net + '_'  + args.dataset))
    except:
        pass
    try:
        os.mkdir(os.path.join(args.save_dir, args.net + '_'  + args.dataset, args.exp_name))
    except:
        pass
    return
       
class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor
