from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import scipy.stats as stats
import math
import numpy as np
from matplotlib import pyplot as plt
from utils_pseudolab.utils.AverageMeter import AverageMeter
from utils_pseudolab.utils.criterion import *
import time
import warnings
warnings.filterwarnings('ignore')
from sklearn import preprocessing as preprocessing


import sys
from tqdm import tqdm

from math import pi
from math import cos

##############################################################################
############################# TRAINING LOSSSES ###############################
##############################################################################

def loss_soft_reg_ep(preds, labels, soft_labels, device, args):
    prob = F.softmax(preds, dim=1)
    prob_avg = torch.mean(prob, dim=0)
    p = torch.ones(args.num_classes).to(device) / args.num_classes

    L_c = -torch.sum(soft_labels * F.log_softmax(preds, dim=1), dim=1)   # Soft labels
    L_p = -torch.sum(torch.log(prob_avg) * p)
    L_e = -torch.mean(torch.sum(prob * F.log_softmax(preds, dim=1), dim=1))
    loss = L_c
    if args.dataset_type != 'sym_noise_warmUp':
        loss += args.alpha * L_p + args.beta * L_e
    
    return prob, loss

##############################################################################

def cyclic_lr(args, iteration, current_epoch, it_per_epoch):
    # proposed learning late function
    T_iteration = it_per_epoch*current_epoch + iteration
    T_epoch_per_cycle = args.SE_epoch_per_cycle*it_per_epoch
    T_iteration = T_iteration%T_epoch_per_cycle
    return args.lr * (cos(pi * T_iteration / T_epoch_per_cycle) + 1) / 2
##############################################################################

##############################################################################

def mixup_data(x, y, alpha=1.0, device='cuda'):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
        lam = max(lam, 1-lam)
    else:
        lam = 1

    device = x.get_device()
    batch_size = x.size()[0]
    
    index = torch.randperm(batch_size).to(device)
    
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def cut_mix(img, lab, alpha=1.0, device='cuda'):
    #CutMix https://arxiv.org/pdf/1905.04899.pdf
    
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
        lam = max(lam, 1-lam)
    else:
        lam = 1
    
    device = img.get_device()
    batch_size = img.size()[0]
    
    index = torch.randperm(batch_size).to(device)

    h = img.shape[1]
    w = img.shape[2]
    y = np.random.randint(h)
    x = np.random.randint(w)
    
    y1 = np.clip(y - int(h*lam) // 2, 0, h)
    y2 = np.clip(y + int(h*lam) // 2, 0, h)
    x1 = np.clip(x - int(w*lam) // 2, 0, w)
    x2 = np.clip(x + int(w*lam) // 2, 0, w)
    for k in range(img.shape[0]):
        img[k,y1:y2,x1:x2] = img[index][k,y1:y2,x1:x2]

    #Adjust lam
    lam = 1 - ((x2 - x1) * (y2 - y1) / (h * w))
    
    y_a, y_b = lab, lab[index]
    return img, y_a, y_b, lam


def loss_mixup_reg_ep(preds, labels, targets_a, targets_b, device, lam, args):
    prob = F.softmax(preds, dim=1)
    prob_avg = torch.mean(prob, dim=0)
    p = torch.ones(args.num_classes).to(device) / args.num_classes
    
    mixup_loss_a = -torch.sum(targets_a * F.log_softmax(preds, dim=1), dim=1)
    mixup_loss_b = -torch.sum(targets_b * F.log_softmax(preds, dim=1), dim=1)
    mixup_loss = lam * mixup_loss_a + (1 - lam) * mixup_loss_b
    L_p = -torch.sum(torch.log(prob_avg) * p)
    L_e = -torch.mean(torch.sum(prob * F.log_softmax(preds, dim=1), dim=1))
    
    loss = mixup_loss
    if args.dataset_type != 'sym_noise_warmUp':
        loss += args.alpha * L_p + args.beta * L_e
    return prob, loss


##############################################################################

def train_CrossEntropy_partialRelab(args, model, device, train_loader, optimizer, epoch, train_noisy_indexes):
    batch_time = AverageMeter()
    train_loss = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    w = torch.Tensor([0.0])

    top1_origLab = AverageMeter()

    # switch to train mode
    model.train()
    loss_per_batch = []
    acc_train_per_batch = []

    alpha_hist = []

    end = time.time()

    results = np.zeros((len(train_loader.dataset), args.num_classes), dtype=np.float32)


    if args.loss_term == "Reg_ep":
        print("Training with cross entropy and regularization for soft labels and for predicting different classes (Reg_ep)")
    elif args.loss_term == "MixUp_ep":
        print("Training with Mixup and regularization for soft labels and for predicting different classes (MixUp_ep)")
        alpha = args.Mixup_Alpha

        print("Mixup alpha value:{}".format(alpha))

    target_original = torch.from_numpy(train_loader.dataset.original_labels)

    counter = 1
    for imgs, img_pslab, labels, soft_labels, index in train_loader:
        images = imgs.to(device)
        labels = labels.to(device)
        soft_labels = soft_labels.to(device)

        if args.DApseudolab == "False":
            images_pslab = img_pslab.to(device)

        if args.loss_term == "MixUp_ep":
            if args.dropout > 0.0 and args.drop_extra_forward == "True":
                if args.network == "PreactResNet18_WNdrop":
                    tempdrop = model.drop
                    model.drop = 0.0

                elif args.network == "WRN28_2_wn":
                    for m in model.modules():
                        if isinstance(m, nn.Dropout):
                            tempdrop = m.p
                            m.p = 0.0
                else:
                    tempdrop = model.drop.p
                    model.drop.p = 0.0
                    
            optimizer.zero_grad()
            if args.DApseudolab == "False":
                output_x1, _ = model(images_pslab)
            else:
                output_x1, _ = model(images)
                
            output_x1.detach_()
            optimizer.zero_grad()

            if args.dropout > 0.0 and args.drop_extra_forward == "True":
                if args.network == "PreactResNet18_WNdrop":
                    model.drop = tempdrop

                elif args.network == "WRN28_2_wn":
                    for m in model.modules():
                        if isinstance(m, nn.Dropout):
                            m.p = tempdrop
                else:
                    model.drop.p = tempdrop

            images, targets_a, targets_b, lam = mixup_data(images, soft_labels, alpha, device)

        # compute output
        outputs, _ = model(images)

        if args.loss_term == "Reg_ep":
            prob, loss = loss_soft_reg_ep(outputs, labels, soft_labels, device, args)
            loss = torch.mean(loss)

        elif args.loss_term == "MixUp_ep":
            prob_mixup, loss = loss_mixup_reg_ep(outputs, labels, targets_a, targets_b, device, lam, args)
            loss = torch.mean(loss)

        # compute gradient and do SGD step & possible ema
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if args.DApseudolab == "False":
            with torch.no_grad():
                output_x1, _ = model(images_pslab)
                outputs = output_x1

        prob = F.softmax(output_x1, dim=1)
        results[index.detach().numpy().tolist()] = prob.cpu().detach().numpy().tolist()
        
        prec1, prec5 = accuracy_v2(outputs.data, labels.data, top=[1, 1])
        train_loss.update(loss.item(), images.size(0))
        top1.update(prec1.item(), images.size(0))
        top5.update(prec5.item(), images.size(0))
        top1_origLab_avg = 0

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if counter % 15 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, Accuracy: {:.0f}%, Learning rate: {:.6f}'.format(
                epoch, counter * len(images), len(train_loader.dataset),
                       100. * counter / len(train_loader), loss.item(),
                       prec1, optimizer.param_groups[0]['lr']))
        counter = counter + 1

    if args.swa == 'True':
        if epoch > args.swa_start and epoch%args.swa_freq == 0 :
            optimizer.update_swa()

    # update soft labels

    train_loader.dataset.update_labels_randRelab(results, train_noisy_indexes, args.label_noise)

    return train_loss.avg, top5.avg, top1_origLab_avg, top1.avg, batch_time.sum

###################################################################################


def testing(args, model, device, test_loader):
    model.eval()
    loss_per_batch = []
    acc_val_per_batch =[]
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output, _ = model(data)
            output = F.log_softmax(output, dim=1)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            loss_per_batch.append(F.nll_loss(output, target).item())
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            acc_val_per_batch.append(100. * correct / ((batch_idx+1)*args.test_batch_size))

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    loss_per_epoch = [np.average(loss_per_batch)]
    acc_val_per_epoch = [np.array(100. * correct / len(test_loader.dataset))]

    return (loss_per_epoch, acc_val_per_epoch)


def validating(args, model, device, test_loader):
    model.eval()
    loss_per_batch = []
    acc_val_per_batch =[]
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for batch_idx, (data, _, target, _, _, _) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output, _ = model(data)
            output = F.log_softmax(output, dim=1)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            loss_per_batch.append(F.nll_loss(output, target).item())
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            acc_val_per_batch.append(100. * correct / ((batch_idx+1)*args.test_batch_size))

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    loss_per_epoch = [np.average(loss_per_batch)]
    acc_val_per_epoch = [np.array(100. * correct / len(test_loader.dataset))]

    return (loss_per_epoch, acc_val_per_epoch)
