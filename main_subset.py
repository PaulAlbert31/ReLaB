import argparse
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from utils import make_data_loader, create_save_folder, multi_class_loss
from diffusion import diffusion
import os
from torch.multiprocessing import set_sharing_strategy
set_sharing_strategy("file_system")
from sklearn.decomposition import PCA

from PIL import Image
import faiss


class Trainer(object):
    def __init__(self, args):
        self.args = args
        
        if args.net == "resnet18":
            from nets.resnet import ResNet18
            model = ResNet18(num_classes=128) #Linear projection
        elif args.net == "resnet50":
            from nets.resnet import ResNet50
            model = ResNet50(num_classes=128) #Linear projection
        elif args.net == "wideresnet282":
            from nets.wideresnet import WRN28_2
            model = WRN28_2(num_classes=self.args.num_class)
        else:
            raise NotImplementedError
        
        self.model = nn.DataParallel(model).cuda()
        
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.lr, momentum=0.9, nesterov=True, weight_decay=5e-4)
        self.criterion = nn.CrossEntropyLoss(ignore_index=-1)
        
        self.kwargs = {"num_workers": 12, "pin_memory": False}        
        self.train_loader, self.val_loader = make_data_loader(args, **self.kwargs)

        self.track_loader, _ = make_data_loader(args, no_aug=True, **self.kwargs) 

        self.best = 0
        self.best_epoch = 0
        self.acc = []

        self.toPIL = torchvision.transforms.ToPILImage()
        
    def train(self, epoch):
        running_loss = 0.0
        self.model.train()
        
        acc, total = 0, 0
        tbar = tqdm(self.train_loader)
        self.epoch = epoch
        for i, sample in enumerate(tbar):
            image, target, ids = sample["image"], sample["target"], sample["index"]
            
            if self.args.cuda:
                target, image = target.cuda(), image.cuda()

            self.optimizer.zero_grad()

            outputs = self.model(image)

            loss = multi_class_loss(outputs, target)
            loss = torch.mean(loss)
            
            preds = torch.argmax(F.log_softmax(outputs, dim=1), dim=1)
            
            acc += torch.sum(preds == torch.argmax(target, dim=1))
            total += preds.size(0)
            loss.backward()
            if i % 10 == 0:
                tbar.set_description("Training loss {0:.2f}, LR {1:.6f}".format(loss.item(), self.optimizer.param_groups[0]["lr"]))
            self.optimizer.step()
            
        print("[Epoch: {}, numImages: {}, numClasses: {}]".format(epoch, total, self.args.num_class))
        print("Training Accuracy: {0:.4f}".format(float(acc)/total))
        
        return float(acc)/total
            

    def val(self, epoch):
        self.model.eval()
        acc, total = 0, 0
        vbar = tqdm(self.val_loader)
        with torch.no_grad():
            for i, sample in enumerate(vbar):
                image, target = sample["image"], sample["target"]
                if self.args.cuda:
                    image, target = image.cuda(), target.cuda()
                outputs = self.model(image)
                loss = self.criterion(outputs, target)
                preds = torch.argmax(F.log_softmax(outputs, dim=1), dim=1)
                acc += torch.sum(preds == target.data)
                total += preds.size(0)
            
                if i % 10 == 0:
                    vbar.set_description("Validation loss: {0:.2f}".format(loss.item()))
        final_acc = float(acc)/total
        print("[Epoch: {}, numImages: {}]".format(epoch, (len(self.val_loader)-1)*self.args.batch_size + image.shape[0]))
        self.acc.append(final_acc)
        if final_acc > self.best:
            self.best = final_acc
            self.best_epoch = epoch
            
        print("Validation Accuracy: {0:.4f}, best accuracy {1:.4f} at epoch {2}".format(final_acc, self.best, self.best_epoch))
        return final_acc

                
    def load(self, unsup_weights):
        #This load function accepts iMix weights trained with our repo https://github.com/PaulAlbert31/iMix.
        state = torch.load(unsup_weights)
        self.model.load_state_dict(state, strict=True)
        
        print("Loaded unsupervised model")
        return

    def track_loss(self):
        self.model.eval()
        with torch.no_grad():
            tbar = tqdm(self.track_loader)
            tbar.set_description("Tracking loss")
            losses = torch.zeros(len(self.track_loader.dataset))
            
            for i, sample in enumerate(tbar):
                image, target, ids = sample["image"], sample["target"], sample["index"]
                if self.args.cuda:
                    target, image = target.cuda(), image.cuda()
                outputs = self.model(image)
                
                loss = multi_class_loss(outputs, target)
                losses[ids] = loss.detach().cpu()
        return losses


    def label_propagation(self, indicies, diffuse=False):
        self.model.eval()
        with torch.no_grad():
            features = torch.zeros((len(self.track_loader.dataset), 128))
            tbar = tqdm(self.track_loader)
            tbar.set_description('Evaluating features')
            for i, sample in enumerate(tbar):
                image, target, ids = sample["image"], sample["target"], sample["index"]
                if self.args.cuda:
                    target, image = target, image.cuda()
                feat = self.model(image)
                features[ids] = feat.cpu()
            torch.cuda.empty_cache()
            
        targ = torch.tensor(self.track_loader.dataset.targets)    
        features = features.numpy()

        #Normalize the features + PCA whitening
        faiss.normalize_L2(features)
        pca = PCA(whiten=True, n_components=features.shape[1])
        features = pca.fit_transform(features)
        features = np.ascontiguousarray(features)

        labels = - torch.ones(targ.shape[0])
        
        for i in indicies:
            labels[i] = targ[i] #known labels
           
        if diffuse: #Diffusion
            final_labels = torch.zeros(targ.shape[0], self.args.num_class)
            weights = torch.zeros(targ.shape[0])          
            p_labels, _, _ = diffusion(features, labels.clone(), indicies, k=200, max_iter=50, classes=self.args.num_class)
            p_labels = torch.from_numpy(p_labels).float()
        else: #KNN
            index = faiss.IndexFlatIP(features.shape[1])
            index.add(features[indicies])
            _, I = index.search(features, 1)
            p_labels = labels[indicies[I]]
            
        if diffuse:
            labels = torch.zeros(features.shape[0], self.args.num_class)
            for i, p in enumerate(torch.argmax(p_labels,1)):
                labels[i][p.item()] = 1
        else:
            labels = torch.zeros(features.shape[0], self.args.num_class)
            for i, p in enumerate(p_labels.long()):
                labels[i][p] = 1
            p_labels = labels
            
        del features
        torch.cuda.empty_cache()        
        return p_labels
    
def main():


    parser = argparse.ArgumentParser(description="Reliable Label Bootstrapping, ReLaB")
    parser.add_argument("--net", type=str, default="wideresnet282",
                        choices=["resnet18", "wideresnet282", "resnet50"],
                        help="net name, only used for loading the self-supervised weights for the label propagation (default: wideresnet282)")
    parser.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "cifar100", "miniimagenet"])
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--save-dir", type=str, default="bootstrapped_dataset")
    parser.add_argument("--load", default=None, type=str, required=True, help="Pretrained self-supervised weights")
    parser.add_argument("--diffuse", action="store_true", default=False, help="Use diffusion, default: kNN")
    parser.add_argument("--no-cuda", action="store_true", default=False, help="No cuda")
    parser.add_argument("--spc", default=1, type=int, help="Number of labeled samples per class")
    parser.add_argument("--seed", default=1, type=int, choices=[1,2,3])
    parser.add_argument("--boot-spc", default=50, type=int, help="Number of samples to bootstrap for the augmented labeled set, default: 50 (cifar10) for cifar100 and miniImageNet choose 40")

    args = parser.parse_args()
    #For reproducibility purposes
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    dict_class = {"cifar10":10, "cifar100":100, "miniimagenet":100}

    args.num_class = dict_class[args.dataset]
        
    if args.save_dir is None:
        args.save_dir = "{}_{}spc".format(args.net, args.dataset, args.spc)

    if not os.path.isdir(args.save_dir):
        os.mkdir(args.save_dir)
        
    args.save_dir = os.path.join(args.save_dir, "seed{}".format(args.seed))

    if not os.path.isdir(args.save_dir):
        os.mkdir(args.save_dir)
        
    args.cuda = not args.no_cuda

    _trainer = Trainer(args)
    
    torch.manual_seed(args.seed)    
    targ = torch.tensor(_trainer.train_loader.dataset.targets) #Original labels
    ind = torch.randperm(targ.shape[0])

    classes = [[] for _ in range(dict_class[args.dataset])]
    indicies = [[] for _ in range(dict_class[args.dataset])]
    #Random seed sample selection
    for i in ind:
        if len(classes[targ[i]]) < args.spc:
            classes[targ[i]].append(targ[i])
            indicies[targ[i]].append(i)

    #Indicies for the 1 sample per class experiments, ordered from worst to best https://github.com/google-research/fixmatch/issues/16
    one_spc_indicies = [
        [7408, 8148, 9850, 10361, 33949, 36506, 37018, 45044, 46443, 47447],
        [5022, 8193, 8902, 9601, 25226, 26223, 34089, 35186, 40595, 48024],
        [7510, 13186, 14043, 21305, 22805, 31288, 34508, 40470, 41493, 45506],
        [9915, 9978, 16631, 19915, 28008, 35314, 35801, 36149, 39215, 42557],
        [6695, 14891, 19726, 22715, 23999, 34230, 46511, 47457, 49181, 49397],
        [12830, 20293, 26835, 30517, 30898, 31061, 43693, 46501, 47310, 48517],
        [1156, 11501, 19974, 21963, 32103, 42189, 46789, 47690, 48229, 48675],
        [4255, 6446, 8580, 11759, 12598, 29349, 29433, 33759, 35345, 38639]
    ]
    
    if args.spc == 1 and args.dataset == "cifar10": #Select the comparable samples for CIFAR-10
        pos = [7, 4, 0] #first, last and one in the middle
        ind = one_spc_indicies[pos[args.seed-1]]
        indicies = [[i] for i in ind]

    indicies = torch.tensor(indicies).view(-1) #Final selected labeled samples

    bc = np.bincount(targ[indicies].numpy())
    print("Labeled samples per class: ", bc)
       
    ori_labels = torch.tensor(_trainer.train_loader.dataset.targets).clone()

    _trainer = Trainer(args)
    
    #Loading self-supervised weights
    _trainer.load(args.load)

    #Get the noisy labels using diffusion in the feature space and the original labeled samples
    p_labels = _trainer.label_propagation(indicies=indicies, diffuse=args.diffuse)

    #one hot encoding
    relabel = torch.zeros(p_labels.shape[0], _trainer.args.num_class)
    for i, p in enumerate(torch.argmax(p_labels, dim=1)):
        relabel[i][p.item()] = 1

    sums = torch.zeros(args.num_class)
    accuracies = torch.zeros(args.num_class)
    for i, l in enumerate(relabel):
        l = torch.argmax(l)
        sums[l] += 1
        if l == ori_labels[i]:
            accuracies[l] += 1

    accuracies = accuracies / sums

    #Just indicative
    print("Per class accuracy subset:", accuracies)
    print("Subset mean accuracy:", accuracies.mean())
    print("Weighted mean accuracy:", (accuracies * sums).sum() / sums.sum())
    print("Per class subset:", sums)
    print("Per class noise", accuracies.median(), accuracies.std())

    #Initialise a new net to train on the set in order to detect noisy samples
    args.net = "wideresnet282"
    _trainer = Trainer(args) #Retrain from scratch
    print("Noise ratio",1-(ori_labels == torch.argmax(relabel, dim=1)).sum().float()/relabel.shape[0])

    #Update the train labels with the diffusion estimates
    _trainer.train_loader.dataset.targets = relabel.clone()
    _trainer.track_loader.dataset.targets = relabel.clone()

    losses_t = torch.zeros(30, len(relabel)) #Average over the last 30 epochs

    for eps in range(args.epochs):
        t = _trainer.train(eps)

        #Track sample loss for each epoch for the last 30 epochs
        if eps >= args.epochs - 30:
            losses =_trainer.track_loss()
            losses_t[eps%30] = losses
            
    #Loss average over the last 30 epochs
    losses = torch.mean(losses_t, dim=0)
    
    #Final ranking from clean to noisy
    ids = torch.argsort(losses)
    ids = torch.cat((indicies, ids)).view(-1).numpy() #adding the seed samples as having the lowest possible loss
    
    #Order per class
    indexes = np.unique(ids, return_index=True)[1]
    ids = np.array([ids[index] for index in sorted(indexes)]).astype(np.int)

    #Gather args.boot_spc bootstrapped samples per class for the extended labeled set
    idnx = np.zeros(args.num_class).astype(np.int)
    ids_class = np.zeros((args.num_class,args.boot_spc))
    
    relabel_am = torch.argmax(relabel, dim=1) #diffusion labels
    for i, ii in enumerate(ids): #sorted by lowest loss, small loss trick, get bottom args.boot_spc samples per class
        c = relabel_am[ii]
        if idnx[c] < args.boot_spc:
            ids_class[c][idnx[c]] = ii
        idnx[c] += 1
            
    ids_class = ids_class.flatten().astype(np.int)

    #Accuracy of the bootstrapped reliable set
    print("Top {} per class:".format(args.boot_spc), (relabel_am[ids_class] == ori_labels[ids_class]).sum().float()/ids_class.shape[0])

    per_class_acc = torch.zeros(args.num_class)
    for i in ids_class:
        if relabel_am[i] == ori_labels[i]:
            per_class_acc[relabel_am[i]] += 1
            
    print("Per class acc {}spc : ".format(args.boot_spc), per_class_acc / args.boot_spc)
    print("Mean class acc {}spc : ".format(args.boot_spc), torch.mean(per_class_acc / args.boot_spc))
    print("STD class acc {}spc : ".format(args.boot_spc), torch.std(per_class_acc / args.boot_spc))
   
    np.savez(os.path.join(args.save_dir, "labels_seed{}_{}spc_{}".format(args.seed, args.spc, args.dataset)), relabel.numpy())
    np.savez(os.path.join(args.save_dir, "subset_seed{}_{}spc_{}c_{}".format(args.seed, args.spc, args.boot_spc, args.dataset)), ids_class)
    

if __name__ == "__main__":
   main()
