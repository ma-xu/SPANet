"""PSE-ResNet in PyTorch
Pre-activated SE-ResNet
Based on preact_resnet.py

Author: Xu Ma.
Date: Apr/17/2019
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['PSPPSEResNet18', 'PSPPSEResNet34', 'PSPPSEResNet50', 'PSPPSEResNet101', 'PSPPSEResNet152']

class PSPPSELayer(nn.Module):
    def __init__(self,in_cahnnel, channel, reduction=16):
        super(PSPPSELayer, self).__init__()
        self.avg_pool1 = nn.AdaptiveAvgPool2d(1)
        self.avg_pool2 = nn.AdaptiveAvgPool2d(2)
        self.avg_pool4 = nn.AdaptiveAvgPool2d(4)
        self.fc = nn.Sequential(
            nn.Linear(in_cahnnel*21, in_cahnnel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_cahnnel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size() # b: number; c: channel;
        y1 = self.avg_pool1(x).view(b, c)  # like resize() in numpy
        y2 = self.avg_pool2(x).view(b, 4 * c)
        y3 = self.avg_pool4(x).view(b, 16 * c)
        y = torch.cat((y1, y2, y3), 1)
        y = self.fc(y)
        b,out_channel = y.size()
        y = y.view(b, out_channel, 1, 1)
        return y


class PSPPSEPreActBlock(nn.Module):
    """SE pre-activation of the BasicBlock"""
    expansion = 1 # last_block_channel/first_block_channel

    def __init__(self,in_planes,planes,stride=1,reduction=16):
        super(PSPPSEPreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes,planes,kernel_size=3,stride=stride,padding=1,bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes,planes,kernel_size=3,stride=1,padding=1,bias=False)
        self.se = PSPPSELayer(in_cahnnel=in_planes,channel=self.expansion*planes,reduction=reduction)
        if stride !=1 or in_planes!=self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes,self.expansion*planes,kernel_size=1,stride=stride,bias=False)
            )

    def forward(self, x):
        PSE = self.se(x)
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self,'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out = out * PSE.expand_as(out)
        out += shortcut
        return out


class PSPPSEPreActBootleneck(nn.Module):
    """Pre-activation version of the bottleneck module"""
    expansion = 4 # last_block_channel/first_block_channel

    def __init__(self,in_planes,planes,stride=1,reduction=16):
        super(PSPPSEPreActBootleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1=nn.Conv2d(in_planes,planes,kernel_size=1,bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes,planes,kernel_size=3,stride=stride,padding=1,bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes,self.expansion*planes,kernel_size=1,bias=False)
        #self.se = SELayer(self.expansion*planes, reduction)
        self.se = PSPPSELayer(in_cahnnel=in_planes, channel=self.expansion * planes, reduction=reduction)
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        PSE = self.se(x)
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self,'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv3(F.relu(self.bn3(out)))
        out = out * PSE.expand_as(out)
        out +=shortcut
        return out


class PSPPSEResNet(nn.Module):
    def __init__(self,block,num_blocks,num_classes=10,reduction=16):
        super(PSPPSEResNet, self).__init__()
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


def PSPPSEResNet18(num_classes=10):
    return PSPPSEResNet(PSPPSEPreActBlock, [2,2,2,2],num_classes)


def PSPPSEResNet34(num_classes=10):
    return PSPPSEResNet(PSPPSEPreActBlock, [3,4,6,3],num_classes)


def PSPPSEResNet50(num_classes=10):
    return PSPPSEResNet(PSPPSEPreActBootleneck, [3,4,6,3],num_classes)


def PSPPSEResNet101(num_classes=10):
    return PSPPSEResNet(PSPPSEPreActBootleneck, [3,4,23,3],num_classes)


def PSPPSEResNet152(num_classes=10):
    return PSPPSEResNet(PSPPSEPreActBootleneck, [3,8,36,3],num_classes)


def test():
    net = PSPPSEResNet18()
    y = net((torch.randn(1,3,32,32)))
    print(y.size())


# test()
