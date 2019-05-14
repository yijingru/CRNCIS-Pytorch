import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch


__all__ = ['ResNetSSD', 'resnetssd18', 'resnetssd34', 'resnetssd50', 'resnetssd101', 'resnetssd152']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNetSSD(nn.Module):

    def __init__(self, block, layers, num_classes=2):
        self.inplanes = 64
        super(ResNetSSD, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)

        self.new_layer1 = nn.Sequential(nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0),
                                        nn.BatchNorm2d(256),
                                        nn.ReLU(inplace=True),
                                        nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
                                        nn.BatchNorm2d(512),
                                        nn.ReLU(inplace=True))

        self.num_classes = num_classes

        self.conf_c3 = nn.Conv2d(512,  4 * num_classes, kernel_size=3, padding=1)
        self.conf_c4 = nn.Conv2d(1024, 6 * num_classes, kernel_size=3, padding=1)
        self.conf_c5 = nn.Conv2d(512,  6 * num_classes, kernel_size=3, padding=1)
        # self.conf_c6 = nn.Conv2d(256,  6 * num_classes, kernel_size=3, padding=1)

        self.locs_c3 = nn.Conv2d(512,  4 * 4, kernel_size=3, padding=1)
        self.locs_c4 = nn.Conv2d(1024, 6 * 4, kernel_size=3, padding=1)
        self.locs_c5 = nn.Conv2d(512,  6 * 4, kernel_size=3, padding=1)
        # self.locs_c6 = nn.Conv2d(256,  6 * 4, kernel_size=3, padding=1)


    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def locs_forward(self, c3, c4, c5):
        c3_locs = self.locs_c3(c3).permute(0, 2, 3, 1).contiguous().view(c3.shape[0], -1, 4)
        c4_locs = self.locs_c4(c4).permute(0, 2, 3, 1).contiguous().view(c4.shape[0], -1, 4)
        c5_locs = self.locs_c5(c5).permute(0, 2, 3, 1).contiguous().view(c5.shape[0], -1, 4)
        # c6_locs = self.locs_c6(c6).permute(0, 2, 3, 1).contiguous().view(c6.shape[0], -1, 4)
        return torch.cat([c3_locs, c4_locs, c5_locs], dim=1)

    def conf_forward(self, c3, c4, c5):
        c3_conf = self.conf_c3(c3).permute(0, 2, 3, 1).contiguous().view(c3.shape[0], -1, self.num_classes)
        c4_conf = self.conf_c4(c4).permute(0, 2, 3, 1).contiguous().view(c4.shape[0], -1, self.num_classes)
        c5_conf = self.conf_c5(c5).permute(0, 2, 3, 1).contiguous().view(c5.shape[0], -1, self.num_classes)
        # c6_conf = self.conf_c6(c6).permute(0, 2, 3, 1).contiguous().view(c6.shape[0], -1, self.num_classes)
        return torch.cat([c3_conf, c4_conf, c5_conf], dim=1)

    def forward(self, x):
        """
        c0: torch.Size([1, 3, 512, 640])
        c1: torch.Size([1, 64, 256, 320])
        c2: torch.Size([1, 256, 128, 160])
        c3: torch.Size([1, 512, 64, 80])
        c4: torch.Size([1, 1024, 32, 40])
        c5: torch.Size([1, 512, 16, 20])
        """
        c0 = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        c1 = x

        x = self.maxpool(x)


        x = self.layer1(x)

        c2 = x

        x = self.layer2(x)
        c3 = x


        x = self.layer3(x)
        c4 = x

        x = self.new_layer1(x)
        c5 = x

        # print(c0.shape)
        # print(c1.shape)
        # print(c2.shape)
        # print(c3.shape)
        # print(c4.shape)
        # print(c5.shape)


        locs = self.locs_forward(c3, c4, c5)
        conf = self.conf_forward(c3, c4, c5)

        return (locs, conf, [c0, c1, c2, c3, c4])


def resnetssd18(pretrained=False, num_classes=2):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNetSSD(BasicBlock, [2, 2, 2, 2], num_classes)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18'], model_dir='.'), strict=False)
    return model


def resnetssd34(pretrained=False, num_classes=2):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNetSSD(BasicBlock, [3, 4, 6, 3], num_classes)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34'], model_dir='.'), strict=False)
    return model


def resnetssd50(pretrained=False, num_classes=2):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNetSSD(Bottleneck, [3, 4, 6, 3], num_classes)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50'], model_dir='.'), strict=False)
    return model


def resnetssd101(pretrained=False, num_classes=2):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNetSSD(Bottleneck, [3, 4, 23, 3], num_classes)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101'], model_dir='.'), strict=False)
    return model


def resnetssd152(pretrained=False, num_classes=2):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNetSSD(Bottleneck, [3, 8, 36, 3], num_classes)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152'], model_dir='.'), strict=False)
    return model
