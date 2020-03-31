"""SE-ResNet in PyTorch
Based on preact_resnet.py

Author: Xu Ma.
Date: Apr/15/2019
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

__all__ = ['NAResNet18', 'NAResNet34', 'NAResNet50', 'NAResNet101', 'NAResNet152']

class NALayer(nn.Module):
    def __init__(self):
        super(NALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.weight1 = Parameter(torch.zeros(1))
        self.bias1 = Parameter(torch.ones(1))
        self.weight2 = Parameter(torch.zeros(1))
        self.bias2 = Parameter(torch.ones(1))
        self.sig = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()

        ## Spatial, (parallel)
        x_spatial = torch.mean(x,1)
        x_spatial =x_spatial.view(b,h*w)
        x_spatial = x_spatial - x_spatial.mean(dim=1,keepdim=True)
        std = x_spatial.std(dim=1, keepdim=True) + 1e-5
        x_spatial = x_spatial / std
        x_spatial = x_spatial.view(b, 1, h, w)
        x_spatial = x_spatial*self.weight1 + self.bias1
        x_spatial = self.sig(x_spatial)
        out = x*x_spatial

        ## Channel, (parallel)
        x_channel = self.avg_pool(x).view(b,c)
        x_channel = x_channel - x_channel.mean(dim=1, keepdim=True)
        std = x_channel.std(dim=1, keepdim=True) + 1e-5
        x_channel = x_channel / std
        x_channel = x_channel.view(b, c, 1, 1)
        x_channel = x_channel * self.weight2 + self.bias2
        x_channel = self.sig(x_channel)
        out = out * x_channel

        return out

class SEPreActBlock(nn.Module):
    """SE pre-activation of the BasicBlock"""
    expansion = 1 # last_block_channel/first_block_channel

    def __init__(self,in_planes,planes,stride=1,reduction=16):
        super(SEPreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes,planes,kernel_size=3,stride=stride,padding=1,bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes,planes,kernel_size=3,stride=1,padding=1,bias=False)
        self.se = NALayer()
        if stride !=1 or in_planes!=self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes,self.expansion*planes,kernel_size=1,stride=stride,bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self,'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        # Add SE block
        out = self.se(out)
        out += shortcut
        return out


class SEPreActBootleneck(nn.Module):
    """Pre-activation version of the bottleneck module"""
    expansion = 4 # last_block_channel/first_block_channel

    def __init__(self,in_planes,planes,stride=1,reduction=16):
        super(SEPreActBootleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1=nn.Conv2d(in_planes,planes,kernel_size=1,bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes,planes,kernel_size=3,stride=stride,padding=1,bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes,self.expansion*planes,kernel_size=1,bias=False)
        self.se = NALayer()

        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self,'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv3(F.relu(self.bn3(out)))
        # Add SE block
        out = self.se(out)
        out +=shortcut
        return out


class SEResNet(nn.Module):
    def __init__(self,block,num_blocks,num_classes=10,reduction=16):
        super(SEResNet, self).__init__()
        self.in_planes=64
        self.conv1 = nn.Conv2d(3,64,kernel_size=3,stride=1,padding=1,bias=False)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1,reduction=reduction)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2,reduction=reduction)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2,reduction=reduction)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2,reduction=reduction)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    #block means SEPreActBlock or SEPreActBootleneck
    def _make_layer(self,block, planes, num_blocks,stride,reduction):
        strides = [stride] + [1]*(num_blocks-1) # like [1,1,1]
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes,planes,stride,reduction))
            self.in_planes = planes*block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def NAResNet18(num_classes=10):
    return SEResNet(SEPreActBlock, [2,2,2,2],num_classes)


def NAResNet34(num_classes=10):
    return SEResNet(SEPreActBlock, [3,4,6,3],num_classes)


def NAResNet50(num_classes=10):
    return SEResNet(SEPreActBootleneck, [3,4,6,3],num_classes)


def NAResNet101(num_classes=10):
    return SEResNet(SEPreActBootleneck, [3,4,23,3],num_classes)


def NAResNet152(num_classes=10):
    return SEResNet(SEPreActBootleneck, [3,8,36,3],num_classes)


def test():
    net = NAResNet18()
    y = net((torch.randn(1,3,32,32)))
    print(y.size())


# test()
