'''VGG16 in Pytorch.'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
# from torchsummary import summary

__all__=['SPPSEVGG16']

class SPPSELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SPPSELayer, self).__init__()
        self.avg_pool1 = nn.AdaptiveAvgPool2d(1)
        self.avg_pool2 = nn.AdaptiveAvgPool2d(2)
        self.avg_pool4 = nn.AdaptiveAvgPool2d(4)
        self.fc = nn.Sequential(
            nn.Linear(channel*21, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y1 = self.avg_pool1(x).view(b, c)  # like resize() in numpy
        y2 = self.avg_pool2(x).view(b, 4 * c)
        y3 = self.avg_pool4(x).view(b, 16 * c)
        y = torch.cat((y1, y2, y3), 1)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class VGGBlock(nn.Module):
    def __init__(self,in_channels, channels, stride=1):
        super(VGGBlock, self).__init__()
        self.conv =  nn.Conv2d(in_channels, channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(channels)
        self.se = SPPSELayer(channels)

    def forward(self, x):
        out=self.conv(x)
        out = F.relu(self.bn(out))
        out = self.se(out)
        return out

class SPPSEVGG16(nn.Module):
    def __init__(self, num_classes=100,init_weights=True):
        super(SPPSEVGG16, self).__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.vggblock1 = self._make_layer(3,64,2)
        self.vggblock2 = self._make_layer(64, 128, 2)
        self.vggblock3 = self._make_layer(128, 256, 3)
        self.vggblock4 = self._make_layer(256, 512, 3)
        self.vggblock5 = self._make_layer(512, 512, 3)
        self.classifier = nn.Linear(512, num_classes)
        if init_weights:
            self._initialize_weights()


    def _make_layer(self, in_channels,channels, num_blocks):

        layers = []
        layers.append(VGGBlock(in_channels, channels))
        for i in range(0,num_blocks-1):
            layers.append(VGGBlock(channels, channels))
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                if hasattr(m,'bias.data'):
                    m.bias.data.zero_()

    def forward(self, x):
        out = self.vggblock1(x)
        out = self.maxpool(out)
        out = self.vggblock2(out)
        out = self.maxpool(out)
        out = self.vggblock3(out)
        out = self.maxpool(out)
        out = self.vggblock4(out)
        out = self.maxpool(out)
        out = self.vggblock5(out)
        out = self.maxpool(out)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out


def test():
    net = SPPSEVGG16(num_classes=10)
    x = torch.randn(2,3,32,32)
    y = net(x)
    print(y.size())
    # summary(net, input_size=(3, 32, 32))

# test()





