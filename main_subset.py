import argparse
import torch
import torchvision
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

from utils import make_data_loader, create_save_folder, UnNormalize, multi_class_loss
from diffusion import diffusion
import os
from torch.multiprocessing import set_sharing_strategy
set_sharing_strategy("file_system")
import umap
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from PIL import Image
import faiss
import seaborn as sns
import pandas as pd
import copy
import random


class Trainer(object):
    def __init__(self, args):
        self.args = args
        
        kwargs = {"num_classes":self.args.num_class}

        if args.net == "resnet18":
            from nets.resnet import ResNet18
            model = ResNet18(pretrained=(args.load == 'imagenet'), **kwargs)
        elif args.net == "resnet50":
            from nets.resnet import ResNet50
            model = ResNet50(pretrained=(args.load == 'imagenet'), **kwargs)
        elif args.net == "wideresnet282":
            from nets.wideresnet import WRN28_2
            model = WRN28_2(**kwargs)
        else:
            raise NotImplementedError
        
        print("Number of parameters", sum(p.numel() for p in model.parameters() if p.requires_grad))
        
        self.model = nn.DataParallel(model).cuda()
        
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.lr, momentum=0.9, nesterov=True, weight_decay=5e-4)
        self.criterion = nn.CrossEntropyLoss(ignore_index=-1)
        self.criterion_nored = nn.CrossEntropyLoss(reduction="none")
        
        self.kwargs = {"num_workers": 12, "pin_memory": False}        
        self.train_loader, self.val_loader = make_data_loader(args, **self.kwargs)


        self.best = 0
        self.best_epoch = 0
        self.acc = []
        self.train_acc = []
        self.med_clean = []
        self.med_noisy = []
        self.perc_clean = []
        self.perc_noisy = []

        self.reductor_plot = umap.UMAP(n_components=2)

        self.toPIL = torchvision.transforms.ToPILImage()

        self.unorm = UnNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

        
    def train(self, epoch):
        running_loss = 0.0
        self.model.train()
        
        acc = 0
        tbar = tqdm(self.train_loader)
        m_dists = torch.tensor([])
        l = torch.tensor([])
        self.epoch = epoch
        total_sum = 0
        for i, sample in enumerate(tbar):
            image, target, ids = sample["image"], sample["target"], sample["index"]
            
            if self.args.cuda:
                target, image = target.cuda(), image.cuda()

            self.optimizer.zero_grad()
            
            outputs, feat = self.model(image)

            loss = multi_class_loss(outputs, target)
            loss = torch.mean(loss)
            
            preds = torch.argmax(F.log_softmax(outputs, dim=1), dim=1)
            
            acc += torch.sum(preds == torch.argmax(target, dim=1))
            total_sum += preds.size(0)
            loss.backward()
            if i % 10 == 0:
                tbar.set_description("Training loss {0:.2f}, LR {1:.6f}".format(loss.item(), self.optimizer.param_groups[0]["lr"]))
            self.optimizer.step()
        print("[Epoch: {}, numImages: {}, numClasses: {}]".format(epoch, total_sum, self.args.num_class))
        print("Training Accuracy: {0:.4f}".format(float(acc)/total_sum))
        self.train_acc.append(float(acc)/total_sum)
        return float(acc)/total_sum
            

    def val(self, epoch):
        self.model.eval()
        acc = 0
        vbar = tqdm(self.val_loader)
        total = 0
        with torch.no_grad():
            for i, sample in enumerate(vbar):
                image, target = sample["image"], sample["target"]
                if self.args.cuda:
                    image, target = image.cuda(), target.cuda()
                outputs = self.model(image)[0]
                    
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

                
    def load(self, dir, load_linear=False, load_optimizer=False):
        #This load function accepts different types of checkpoint dictionaries and will remove the last layer of resnets/wideresnets/cnns by default
        dict = torch.load(dir)
        if load_optimizer:
            self.optimizer.load_state_dict(dict["optimizer"])
            self.best = dict["best"]
            self.best_epoch = dict["best_epoch"]

        if "state_dict" in dict.keys():
            dic = dict["state_dict"]
        elif "network" in dict.keys():
            dic = dict["network"]
        elif "net" in dict.keys():
            dic = dict["net"]
        else:
            dic = dict

        if "module" in list(dic.keys())[0]:
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in dic.items():
                name = k[7:] # remove `module.`
                new_state_dict[name] = v
            dic = new_state_dict

        if not load_linear:
            if self.args.net == "wideresnet282":
                del dic["output.weight"]
                del dic["output.bias"]
            elif "resnet" in self.args.net:
                del dic["linear.weight"]
                del dic["linear.bias"]
        self.model.module.load_state_dict(dic, strict=False)

        if "state_dict" in list(dict.keys()):
            print("Loaded model with top accuracy {} at epoch {}".format(self.best, dict["best_epoch"]))
            self.epoch = dict["epoch"]
            return dict["epoch"]
        
        print("Loaded model with top accuracy %3d" % (self.best))        

    def track_loss(self, relabel, plot=False, multi=False, feats=False):
        self.model.eval()
        acc = 0
        total_sum = 0
        with torch.no_grad():
            nargs = copy.deepcopy(self.args)
            loader, _ = make_data_loader(nargs, no_aug=True, **self.kwargs) 
            loader.dataset.targets = relabel #unshuffled original label guess
            
            tbar = tqdm(loader)
            tbar.set_description("Tracking loss")
            
            features = torch.tensor([])            
            losses = torch.tensor([])
            for i, sample in enumerate(tbar):
                image, target, ids = sample["image"], sample["target"], sample["index"]
                if self.args.cuda:
                    target, image = target.cuda(), image.cuda()
                outputs, feat = self.model(image)
                features = torch.cat((features, feat.cpu()))
                
                loss = multi_class_loss(outputs, target)

                losses = torch.cat((losses, loss.detach().cpu()))

            losses = losses.view(-1)

            if feats:
                return losses, features
            return losses


    def label_propagation(self, plot=False, diffuse=False):
        self.model.eval()
        with torch.no_grad():
            transform = None
            if self.args.load == "imagenet":
                transform = torchvision.transforms.Resize(224) #Was trained at a different resolution than cifar
            loader, _ = make_data_loader(self.args, no_aug=True, transform=transform, **self.kwargs)
            dim = 2048 if self.args.net == "resnet50" else 512
            dim = dim if self.args.net != "wideresnet282" else 128
            features_average = torch.zeros((len(loader.dataset), dim))

            features = torch.tensor([])
            tbar = tqdm(loader)
            for i, sample in enumerate(tbar):
                image, target, ids = sample["image"], sample["target"], sample["index"]
                if self.args.cuda:
                    target, image = target, image.cuda()
                outputs, feat = self.model(image)
                features = torch.cat((features, feat.cpu()))
            features_average += features
            torch.cuda.empty_cache()
        features_average = features_average
        targ = torch.tensor(loader.dataset.targets)    
        features = features_average.numpy()

        #Normalize the features + PCA whitening
        faiss.normalize_L2(features)
        pca = PCA(whiten=True, n_components=features.shape[1])        
        features = pca.fit_transform(features)
        features = np.ascontiguousarray(features)

        labels = - torch.ones(targ.shape[0])
        
        for i,ii in enumerate(self.indicies):
            labels[ii] = targ[ii] #known samples
           
        if diffuse: #Diffusion
            final_labels = torch.zeros(targ.shape[0], self.args.num_class)
            weights = torch.zeros(targ.shape[0])          
            p_labels, p_weights, class_weights = diffusion(features, labels.clone(), self.indicies, k=200, max_iter=50, classes=self.args.num_class)
            p_labels = torch.from_numpy(p_labels).float()
            p_weights = torch.from_numpy(p_weights).float()
        else: #KNN
            index = faiss.IndexFlatIP(features.shape[1])
            index.add(features[self.indicies])
            _, I = index.search(features, 1)
            p_labels = labels[self.indicies[I]]
            p_weights = torch.ones(features.shape[0])
            
        if plot is not None: #Optional UMap plots
            embedding = self.reductor_plot.fit_transform(features)
            emb = embedding[self.indicies]#Centroids, at least one per class
            plt.figure(7)
            plt.scatter(embedding[:, 0], embedding[:, 1], c=[sns.color_palette(n_colors=self.args.num_class)[x] for x in targ], s=0.1)
            plt.scatter(emb[:, 0], emb[:, 1], c=[sns.color_palette(n_colors=self.args.num_class)[x] for x in targ[self.indicies]], marker="*")
            plt.scatter(emb[:, 0], emb[:, 1], c="#000000", marker="*", s=1.1)
            plt.savefig("data/embedding{}.png".format(plot))
            
            df = pd.DataFrame(embedding, columns=["x", "y"])
            sns_plot = sns.jointplot(x="x", y="y", data=df, kind="kde");
            sns_plot.savefig("data/embedding_density{}.png".format(plot))
            
            plt.figure(6)
            plt.scatter(embedding[:, 0], embedding[:, 1], c=[sns.color_palette(n_colors=self.args.num_class)[x] for x in torch.argmax(p_labels, dim=1)], s=0.1)
            plt.scatter(emb[:, 0], emb[:, 1], c=[sns.color_palette(n_colors=self.args.num_class)[x] for x in torch.argmax(p_labels[self.indicies], dim=1)], marker="*")
            plt.scatter(emb[:, 0], emb[:, 1], c="#000000", marker="*", s=1.1)
            plt.savefig("data/embedding_diffusion{}.png".format(plot))
            plt.close()

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
        return p_labels, p_weights
    
def main():


    parser = argparse.ArgumentParser(description="Reliable Label Bootstrapping, ReLaB")
    parser.add_argument("--net", type=str, default="wideresnet282",
                        choices=["resnet18", "wideresnet282", "resnet50"],
                        help="net name, only used for loading the self-supervised weights for the label propagation (default: wideresnet282)")
    parser.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "cifar100", "miniimagenet"])
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=0.1, help="Multiplicative factor for lr decrease, default .1")
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--save-dir", type=str, default="bootstrapped_dataset")
    parser.add_argument("--load", default=None, type=str, required=True, help="Pretrained self-supervised weights, type imagenet for imagenet weights")
    parser.add_argument("--diffuse", action="store_true", default=False, help="Use diffusion, default: kNN")
    parser.add_argument("--no-cuda", action="store_true", default=False, help="No cuda")
    parser.add_argument("--spc", default=1, type=int, help="Number of labeled samples per class")
    parser.add_argument("--seed", default=-1, type=int)
    parser.add_argument("--boot-spc", default=50, type=int, help="Number of samples to bootstrap, default: 50 (cifar10)")

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

    nargs = copy.deepcopy(args)
    _trainer = Trainer(nargs)
    
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

    relabel_acc = []
            
    _trainer = Trainer(args)
    _trainer.indicies = indicies
    
    #Loading self-supervised weights
    if args.load is not None and args.load != "imagenet":
        _trainer.load(args.load, load_linear=False)

    #Get the noisy labels using diffusion in the feature space and the original samples
    final_labels = torch.zeros(targ.shape[0], _trainer.args.num_class)

    p_labels, _ = _trainer.label_propagation(plot=None, diffuse=args.diffuse)

    relabel = torch.zeros(p_labels.shape[0], _trainer.args.num_class)
    for i, p in enumerate(torch.argmax(p_labels,1)):
        relabel[i][p.item()] = 1

    args.relabel = relabel
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

    #Initialise a new net to train on the set
    nargs = copy.deepcopy(args)
    nargs.net = "wideresnet282"
    _trainer = Trainer(nargs) #Retrain from scratch

    _trainer.indicies = torch.tensor(indicies).view(-1) # Known samples

    print("Noise ratio",1-(ori_labels == torch.argmax(relabel, dim=1)).sum().float()/relabel.shape[0])
    relabel_acc.append((ori_labels == torch.argmax(relabel, dim=1)).sum().float()/relabel.shape[0])

    _trainer.train_loader.dataset.targets = relabel.clone()

    losses_t = torch.zeros(30, len(relabel)) #Average over the last 30 epochs

    for eps in range(args.epochs):
        t = _trainer.train(eps)

        #Track sample loss for each epoch for the last 30 epochs
        if eps >= args.epochs - 30:
            #Validation (Not used for anything)
            #v = _trainer.val(eps)

            losses, features =_trainer.track_loss(relabel, plot=True, feats=True)
            losses_t[eps%30] = losses

            relabel_a = torch.argmax(relabel, dim=1)
            
            # Loss ranking tracking
            ids = torch.argsort(losses, descending=False)
            ids_tf = ids[:int(1.*ids.shape[0]/4)] #top 25%
            ids_t = ids[:int(1.*ids.shape[0]/10)] #top 10%
            ids_to = ids[:int(1.*ids.shape[0]/100)] #top 1%

            print("Top 25%:", (relabel_a[ids_tf] == ori_labels[ids_tf]).sum().float()/ids_tf.shape[0], (losses[ids_tf] * 1000).mean())
            print("Top 10%:", (relabel_a[ids_t] == ori_labels[ids_t]).sum().float()/ids_t.shape[0], (losses[ids_t] * 1000).mean())
            print("Top 1%:", (relabel_a[ids_to] == ori_labels[ids_to]).sum().float()/ids_to.shape[0], (losses[ids_to] * 1000).mean())

    #Loss average and normalisation
    losses = torch.mean(losses_t, dim=0)
        
    relabel = relabel_a
    
    #Final ranking
    ids = torch.argsort(losses, descending=False)
    ids = torch.cat((_trainer.indicies, ids)).view(-1).numpy() #adding the seed samples as having the lowest possible loss

    indexes = np.unique(ids, return_index=True)[1]
    ids = np.array([ids[index] for index in sorted(indexes)]).astype(np.int)

    #Number of samples per class
    
    idnx = np.zeros(args.num_class).astype(np.int)
    ids_class = np.zeros((args.num_class,args.boot_spc))
    
    for i, ii in enumerate(ids): #sorted by lowest loss, small loss trick, get bottom args.boot_spc samples per class
        c = relabel[ii]
        if idnx[c] < args.boot_spc:
            ids_class[c][idnx[c]] = ii
        idnx[c] += 1
            
    ids_class = ids_class.flatten().astype(np.int)

    #Accuracy of the bootstrapped reliable set
    print("Top {} per class:".format(args.boot_spc), (relabel[ids_class] == ori_labels[ids_class]).sum().float()/ids_class.shape[0])

    #one hot
    relab = torch.zeros((len(_trainer.train_loader.dataset), args.num_class))
    for i, r in enumerate(relabel):
        relab[i][r] = 1

    per_class_acc = torch.zeros(args.num_class)
    for i in ids_class:
        if relabel[i] == ori_labels[i]:
            per_class_acc[relabel[i]] += 1
            
    print("Per class acc {}spc : ".format(args.boot_spc), per_class_acc / args.boot_spc)
    print("Mean class acc {}spc : ".format(args.boot_spc), torch.mean(per_class_acc / args.boot_spc))
    print("STD class acc {}spc : ".format(args.boot_spc), torch.std(per_class_acc / args.boot_spc))
   
    np.savez(os.path.join(args.save_dir, "labels_seed{}_{}spc_{}".format(args.seed, args.spc, args.dataset)), relab.numpy())
    np.savez(os.path.join(args.save_dir, "subset_seed{}_{}spc_{}c_{}".format(args.seed, args.spc, args.boot_spc, args.dataset)), ids_class)
    
    print("Stats: relabel accuracies", relabel_acc)
    

if __name__ == "__main__":
   main()
