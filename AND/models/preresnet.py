'''ResNet in PyTorch.
BasicBlock and Bottleneck module is from the original ResNet paper:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
PreActBlock and PreActBottleneck module is from the later paper:
[2] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Identity Mappings in Deep Residual Networks. arXiv:1603.05027
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from torch.nn.utils import weight_norm
from lib.normalize import Normalize



def conv3x3(in_planes, out_planes, stride=1):
    return weight_norm(nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False))


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                weight_norm(nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class PreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                weight_norm(nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False))
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out)
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = weight_norm(nn.Conv2d(in_planes, planes, kernel_size=1, bias=False))
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = weight_norm(nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False))
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = weight_norm(nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False))
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                weight_norm(nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class PreActBottleneck(nn.Module):
    '''Pre-activation version of the original Bottleneck module.'''
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = weight_norm(nn.Conv2d(in_planes, planes, kernel_size=1, bias=False))
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = weight_norm(nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False))
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = weight_norm(nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False))

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                weight_norm(nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False))
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out)
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv3(F.relu(self.bn3(out)))
        out += shortcut
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, widening=4, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 16*widening
        
        self.conv1 = conv3x3(3,16*widening)
        self.bn1 = nn.BatchNorm2d(16*widening)
        self.layer1 = self._make_layer(block, 16*widening, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 16*widening*2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 16*widening*4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 16*widening*8, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)
        self.l2norm = Normalize(2)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,                                                                                                                                              
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.                                                                                                  
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677                                                                                                                 
        for m in self.modules():
            if isinstance(m, Bottleneck):
                nn.init.constant_(m.bn3.weight, 0)
            elif isinstance(m, BasicBlock):
                nn.init.constant_(m.bn2.weight, 0)


    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = x

        out = self.conv1(out)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        out = self.l2norm(out)

        return out


def PreResNet10(pretrained=False, **kwargs):
    return ResNet(PreActBlock, [1,1,1,1], **kwargs)

def PreResNet18(low_dim=128):
    return ResNet(PreActBlock, [2,2,2,2], num_classes=low_dim)

def ResNet18(pretrained=False, **kwargs):
    return ResNet(BasicBlock, [2,2,2,2], **kwargs)

def PreResNet34(pretrained=False, **kwargs):
    return ResNet(PreActBlock, [3,4,6,3], **kwargs)

def ResNet50(pretrained=False, **kwargs):
    return ResNet(PreActBottleneck, [3,4,6,3], **kwargs)

def ResNet101(pretrained=False, **kwargs):
    return ResNet(Bottleneck, [3,4,23,3], **kwargs)

def PreResNet152(pretrained=False, **kwargs):
    return ResNet(PreActBottleneck, [3,8,36,3], **kwargs)


def test():
    net = PreResNet18()
    y = net(Variable(torch.randn(1,3,32,32)))
    print(y.size())

# test()
