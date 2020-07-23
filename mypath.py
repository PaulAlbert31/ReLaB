class Path(object):
    @staticmethod
    def db_root_dir(dataset):
        if dataset == 'cifar10':
            return 'samples/cifar10/'
        elif dataset == 'cifar100':
            return '/samples/cifar100/'
        elif dataset == 'miniimagenet':
            return 'samples/miniImagenet84/'
        else:
            print('Dataset {} not available.'.format(dataset))
            raise NotImplementedError
        
