import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import parameters
import torch.nn.functional as F
if parameters.args.shared:
    print("shared")
    import satshared as sat
else:
    print("not shared")
    import sat

__all__ = ['ResNet', 'resnet18', 'resnetW', 'resnet50', 'resnet101',
           'resnet152']




def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class SATBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        
        super(SATBlock, self).__init__()
        self.conv1 = sat.ShiftAttention(in_planes, planes, kernel_size=parameters.args.kernel_size, stride=stride, padding=parameters.args.kernel_size //2, bias=False)
        if parameters.args.shared:
            parameters.nb_params += in_planes * planes + in_planes * 4/32
        else:
            parameters.nb_params += in_planes * planes * 36/32

#        self.conv1 = nn.Conv2d(in_planes, mid_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        parameters.nb_params += sum([x.data.nelement() for x in self.bn1.parameters()])
#        self.conv2 = nn.Conv2d(mid_planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = sat.ShiftAttention(planes, planes, kernel_size=parameters.args.kernel_size, stride=1, padding=parameters.args.kernel_size//2, bias=False)
        if parameters.args.shared:
            parameters.nb_params += planes * planes + planes * 4/32
        else:
            parameters.nb_params += planes * planes * 36/32

        self.bn2 = nn.BatchNorm2d(planes)
        parameters.nb_params += sum([x.data.nelement() for x in self.bn2.parameters()])


        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            conv = nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False)
            parameters.nb_params += in_planes * planes
            bn = nn.BatchNorm2d(planes)
            parameters.nb_params += sum([x.data.nelement() for x in bn.parameters()])
            self.shortcut = nn.Sequential(
                conv,
                bn
            )

    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        
        out = self.bn2(self.conv2(out))
        
        out += self.shortcut(x)
        out = F.relu(out)
        return out

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
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False):
        super(ResNet, self).__init__()
        w=32
        self.inplanes = w
        self.conv1 = nn.Conv2d(3, w, kernel_size=3, stride=2, padding=1,
                               bias=False)
        parameters.nb_params += sum([x.data.nelement() for x in self.conv1.parameters()])
        self.bn1 = nn.BatchNorm2d(w)
        parameters.nb_params += sum([x.data.nelement() for x in self.bn1.parameters()])
        self.relu = nn.ReLU(inplace=True)
        self.layer0=self._make_layer(block, w, layers[0])
        self.layer1 = self._make_layer(block, w, layers[1],  stride=2)
        self.layer2 = self._make_layer(block, 2*w, layers[2], stride=2)
        self.layer3 = self._make_layer(block, 4*w, layers[3], stride=2)
        self.layer4 = self._make_layer(block, 8*w, layers[4], stride=2)
        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1, padding=0)#AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(8*w * block.expansion, num_classes)
        parameters.nb_params += sum([x.data.nelement() for x in self.fc.parameters()])

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    

    def forward(self, x):
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.layer0(x)
        
        x = self.layer1(x)
        
        x = self.layer2(x)
        
        x = self.layer3(x)
        
        x = self.layer4(x)
        
        x = self.avgpool(x)
        
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


def resnetW(pretrained=False, **kwargs):
    """Constructs a ResNetW model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(SATBlock, [1,3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model
