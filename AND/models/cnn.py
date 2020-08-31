import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
from lib.normalize import Normalize

class CNN(nn.Module):
    """
    CNN from Mean Teacher paper
    """
    
    def __init__(self, low_dim=10):
        super(CNN, self).__init__()

        self.activation = nn.LeakyReLU(0.1)
        self.conv1a = weight_norm(nn.Conv2d(3, 128, 3, padding=1))
        self.bn1a = nn.BatchNorm2d(128)
        self.conv1b = weight_norm(nn.Conv2d(128, 128, 3, padding=1))
        self.bn1b = nn.BatchNorm2d(128)
        self.conv1c = weight_norm(nn.Conv2d(128, 128, 3, padding=1))
        self.bn1c = nn.BatchNorm2d(128)
        self.mp1 = nn.MaxPool2d(2, stride=2, padding=0)
        self.drop1  = nn.Dropout(0.0)
        
        self.conv2a = weight_norm(nn.Conv2d(128, 256, 3, padding=1))
        self.bn2a = nn.BatchNorm2d(256)
        self.conv2b = weight_norm(nn.Conv2d(256, 256, 3, padding=1))
        self.bn2b = nn.BatchNorm2d(256)
        self.conv2c = weight_norm(nn.Conv2d(256, 256, 3, padding=1))
        self.bn2c = nn.BatchNorm2d(256)
        self.mp2 = nn.MaxPool2d(2, stride=2, padding=0)
        self.drop2  = nn.Dropout(0.0)
        
        self.conv3a = weight_norm(nn.Conv2d(256, 512, 3, padding=0))
        self.bn3a = nn.BatchNorm2d(512)
        self.conv3b = weight_norm(nn.Conv2d(512, 256, 1, padding=0))
        self.bn3b = nn.BatchNorm2d(256)
        self.conv3c = weight_norm(nn.Conv2d(256, 128, 1, padding=0))
        self.bn3c = nn.BatchNorm2d(128)
        self.ap3 = nn.AvgPool2d(6, stride=2, padding=0)
        
        self.fc1 =  weight_norm(nn.Linear(128, low_dim))#512
        self.l2 = Normalize(2)
    
    def forward(self, x, debug=False, lin=0, lout=4):

        if lout > 0:
            x = self.activation(self.bn1a(self.conv1a(x)))
            x = self.activation(self.bn1b(self.conv1b(x)))
            x = self.activation(self.bn1c(self.conv1c(x)))
            x = self.mp1(x)
            x = self.drop1(x)
            
        if lout > 1:
            x = self.activation(self.bn2a(self.conv2a(x)))
            x = self.activation(self.bn2b(self.conv2b(x)))
            x = self.activation(self.bn2c(self.conv2c(x)))
            x = self.mp2(x)
            x = self.drop2(x)
            
        if lout > 2:
            x = self.activation(self.bn3a(self.conv3a(x)))
            x = self.activation(self.bn3b(self.conv3b(x)))
            x = self.activation(self.bn3c(self.conv3c(x)))
            x = self.ap3(x)
            
        features = x.view(x.shape[0], -1)

        return self.l2(self.fc1(features))
        
class SmallCNN(nn.Module):
    
    def __init__(self, num_classes=10, p=0.5, fm1=128, fm2=256):
        super(SmallCNN, self).__init__()
        self.fm1   = fm1
        self.fm2   = fm2
        
        self.act   = nn.ReLU()
        self.drop  = nn.Dropout(p)
        
        self.conv1 = weight_norm(nn.Conv2d(3, self.fm1, 32, padding=1))
        self.conv2 = weight_norm(nn.Conv2d(self.fm1, self.fm2, 32, padding=1))
        
        self.mp    = nn.MaxPool2d(1)
        self.fc    = nn.Linear(self.fm2, num_classes)
    
    def forward(self, x):
        
        x = self.act(self.mp(self.conv1(x)))
        x = self.act(self.mp(self.conv2(x)))
        
        x = x.view(-1, self.fm2)
        
        x = self.drop(x)
        x = self.fc(x)
        
        return x
