## Results on CIFAR100
The table prvoides the models and results of various models on CIFAR100. 
Learning rate =0.1 and will be divided by 10 every 70 epochs. Total 300 epochs.
Using SGD optimizer, momentum=0.9, weight_decay=5e-4.
Loss is CrossEntropyLoss.
Batch-size=512.

Model | Parameters| Flops | CIFAR-100 | 
-------|:-------:|:--------:|:--------:|
[PreActResNet18](https://drive.google.com/open?id=1w2VGpFPDuS9NzcfcGfPUXoEdXwVftFep) |- |- |74.91%
[PreActResNet50](https://drive.google.com/open?id=1Nz_JmzLxuzefGzekBRoCutDIeRgaKWMY) |- |- |77.39%
[PreActResNet101](https://drive.google.com/open?id=1gZoIQhJCzSMhN9b6OeoLL_lyxgU5vCVT) |- |- |77.74%
[SEResNet18](https://drive.google.com/open?id=17Ynt2pLrbew-n2Wu3P8coZ1vTUiV8h3I) |- |- |75.19%
[SEResNet50](https://drive.google.com/open?id=1ESIH2Vmqk5kP2VMuUd53FtXDiyhV-ZGe) |- |- |77.91%
[SEResNet101](https://drive.google.com/open?id=1ASubbeI6l3RQR9WAJakqxwOnDGo1iSl9) |- |- |78.03%
[PSEResNet18](https://drive.google.com/open?id=1ZHYAyjiVsBtpCe7pDp3Ip204UYDpe_aR) |- |- |74.97%
[PSEResNet50](https://drive.google.com/open?id=1V_-qkfvGorDDzOMEsEb9peHyj-tI2IB2) |- |- |77.45%
[PSEResNet101](https://drive.google.com/open?id=17zRZipc8Dj32b4iaDcD4J9w-8-tcEfqb) |- |- |77.88%
[CPSEResNet18](https://drive.google.com/open?id=12Hne8epBFV2YjakHP43PwYSYizdHlG0D) |- |- |75.25%
[CPSEResNet50](https://drive.google.com/open?id=1axp5bjRTkmkxRd3CGRTP_WwBOcdh74GM) |- |- |77.43%
[CPSEResNet101](https://drive.google.com/open?id=1MtfiV8vjHNfiXwB6q-AncuTe2Y1dkNxQ) |- |- |77.61%
[SPPSEResNet18](https://drive.google.com/open?id=1EYcqDd70KHLKC2v_DaZ35qW1SLVzwaqN) |- |- |75.41%
[SPPSEResNet50](https://drive.google.com/open?id=1xEMjxxOe3X3-fOvU9wdxJWtoPoFA74T_) |- | |78.21%
[SPPSEResNet101](https://drive.google.com/open?id=1H5gpBjWSnf4RbaZg2tdJ5LJRMTSLCxgP) |- |- |78.11
[PSPPSEResNet18](https://drive.google.com/open?id=1h-d4b1qaGgzxu8_yPlwrVu-BIN9ZUbNo) |- |- |75.01%
[PSPPSEResNet50](https://drive.google.com/open?id=11-4nxqOE9_cYwC6iR8DUHSln0z5nwoTD) |- |- |78.11%
[PSPPSEResNet101](https://drive.google.com/open?id=134ZG8H0TY545MArhon1A8YKEZcOrPB9y) |- |- |78.35%
[CPSPPSEResNet18](https://drive.google.com/open?id=1G1vPvLYFCTCq7nE4TQFTiwIthKFE9yso) |- |- |75.56%
[CPSPPSEResNet50](https://drive.google.com/open?id=1tVB-ml5JUnmGMqw7mToPxhmkjoaCkRE2) |- |- |77.95%
[CPSPPSEResNet101](https://drive.google.com/file/d/1M3OXCflFFZ9E8jvMxYTqq0s2TZFiAvWy/view) |- |- |79.17%


For a better understanding, we reschedule the table as follows:

Model | 18-Layer| 50-Layer | 101-Layer | 
-------|:-------:|:--------:|:--------:|
PreActResNet    |74.91% |77.39% |77.74%
SEResNet        |75.19% |77.91% |78.03%
PSEResNet       |74.97% |77.45% |77.88%
CPSEResNet      |75.25% |77.43% |77.61%
SPPSEResNet     |75.41% |`78.21%`|78.11%
PSPPSEResNet    |75.01% |78.11% |78.35%
CPSPPSEResNet   |`75.56%`|77.95% |`79.17%`
