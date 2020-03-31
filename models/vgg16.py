'''VGG16 in Pytorch.'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
# from torchsummary import summary

__all__=['VGG16']
class VGGBlock(nn.Module):
    def __init__(self,in_channels, channels, stride=1):
        super(VGGBlock, self).__init__()
        self.conv =  nn.Conv2d(in_channels, channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(channels)

    def forward(self, x):
        out=self.conv(x)
        out = F.relu(self.bn(out))
        return out

class VGG16(nn.Module):
    def __init__(self, num_classes=100,init_weights=True):
        super(VGG16, self).__init__()
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
    net = VGG16(num_classes=10)
    x = torch.randn(2,3,32,32)
    y = net(x)
    print(y.size())
    # summary(net, input_size=(3, 32, 32))

# test()





