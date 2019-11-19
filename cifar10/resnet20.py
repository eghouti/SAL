"""PyTorch implementation of ResNet
ResNet modifications written by Bichen Wu and Alvin Wan, based
off of ResNet implementation by Kuang Liu.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
"""
import torch.nn as nn
import torch.nn.functional as F
import parameters
if parameters.args.shared:
    import satshared as sat
else:
    import sat

nb_activation=0

class BasicBlock(nn.Module):

     def __init__(self, in_planes, planes, stride=1, reduction=1):
         super(BasicBlock, self).__init__()
         self.expansion = 1 / float(reduction)
         self.in_planes = in_planes
         self.mid_planes = mid_planes = int(self.expansion * planes)
         self.out_planes = planes

         self.conv1 = nn.Conv2d(in_planes, mid_planes, kernel_size=3, stride=stride, padding=1, bias=False)
         self.bn1 = nn.BatchNorm2d(mid_planes)
         self.conv2 = nn.Conv2d(mid_planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
         self.bn2 = nn.BatchNorm2d(planes)
         self.stride = stride
         self.in_planes = in_planes
         self.planes = planes
         self.shortcut = nn.Sequential()
         if stride != 1 or in_planes != planes:
             self.shortcut = nn.Sequential(
                 nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                 nn.BatchNorm2d(planes)
             )

     def flops(self):
         if not hasattr(self, 'int_nchw'):
             raise UserWarning('Must run forward at least once')
         (_, _, int_h, int_w), (_, _, out_h, out_w) = self.int_nchw, self.out_nchw
         flops = int_h*int_w*9*self.mid_planes*self.in_planes + out_h*out_w*9*self.mid_planes*self.out_planes
         if len(self.shortcut) > 0:
             flops += self.in_planes*self.out_planes*out_h*out_w
         return flops

     def forward(self, x):
         global nb_activation
         out = self.conv1(x)
         nb_activation+= out.shape[1]*out.shape[2]*out.shape[3]
         self.bn1(out)
         nb_activation+= out.shape[1]*out.shape[2]*out.shape[3]
         out = F.relu(out)
         self.int_nchw = out.size()
         out = self.conv2(out)
         nb_activation+= out.shape[1]*out.shape[2]*out.shape[3]
         out = self.bn2(out)
         nb_activation+= out.shape[1]*out.shape[2]*out.shape[3]
         self.out_nchw = out.size()
         out += self.shortcut(x)
         if self.stride != 1 or self.in_planes != self.planes:
             nb_activation+= out.shape[1]*out.shape[2]*out.shape[3]
         out = F.relu(out)
         return out


class SATBlock(nn.Module):

    def __init__(self, in_planes, planes, stride=1, reduction=1):
        super(SATBlock, self).__init__()
        self.expansion = 1 / float(reduction)
        self.in_planes = in_planes
        self.mid_planes = mid_planes = int(self.expansion * planes)
        self.out_planes = planes

        self.conv1 = sat.ShiftAttention(in_planes, mid_planes, kernel_size=parameters.args.kernel_size, stride=stride, padding=parameters.args.kernel_size //2, bias=False)
        if parameters.args.shared:
            parameters.nb_params += in_planes * mid_planes + in_planes * 4/32
        else:
            parameters.nb_params += in_planes * mid_planes * 36/32

#        self.conv1 = nn.Conv2d(in_planes, mid_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_planes)
        parameters.nb_params += sum([x.data.nelement() for x in self.bn1.parameters()])
#        self.conv2 = nn.Conv2d(mid_planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = sat.ShiftAttention(mid_planes, planes, kernel_size=parameters.args.kernel_size, stride=1, padding=parameters.args.kernel_size//2, bias=False)
        if parameters.args.shared:
            parameters.nb_params += mid_planes * planes + mid_planes * 4/32
        else:
            parameters.nb_params += mid_planes * planes * 36/32

        self.bn2 = nn.BatchNorm2d(planes)
        parameters.nb_params += sum([x.data.nelement() for x in self.bn2.parameters()])
        self.stride = stride
        self.in_planes = in_planes
        self.planes = planes
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

    def flops(self):
        if not hasattr(self, 'int_nchw'):
            raise UserWarning('Must run forward at least once')
        (_, _, int_h, int_w), (_, _, out_h, out_w) = self.int_nchw, self.out_nchw
        flops = int_h*int_w*9*self.mid_planes*self.in_planes + out_h*out_w*9*self.mid_planes*self.out_planes
        if len(self.shortcut) > 0:
            flops += self.in_planes*self.out_planes*out_h*out_w
        return flops

    def forward(self, x):
        global nb_activation
        out = self.conv1(x)
        nb_activation+= out.shape[1]*out.shape[2]*out.shape[3]
        out = self.bn1(out)
        nb_activation+= out.shape[1]*out.shape[2]*out.shape[3]
        out = F.relu(out)
        self.int_nchw = out.size()
        out = self.conv2(out)
        nb_activation+= out.shape[1]*out.shape[2]*out.shape[3]
        out = self.bn2(out)
        nb_activation+= out.shape[1]*out.shape[2]*out.shape[3]
        self.out_nchw = out.size()
        out += self.shortcut(x)
        if self.stride != 1 or self.in_planes != self.planes:
            nb_activation+= out.shape[1]*out.shape[2]*out.shape[3]
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, reduction=1, num_classes=10):
        super(ResNet, self).__init__()
        self.reduction = float(reduction) ** 0.5
        self.num_classes = num_classes
        self.in_planes = int(parameters.args.feature_maps / self.reduction)
        value = self.in_planes

        self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        parameters.nb_params += sum([x.data.nelement() for x in self.conv1.parameters()])
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        parameters.nb_params += sum([x.data.nelement() for x in self.bn1.parameters()])
        self.layer1 = self._make_layer(block, self.in_planes, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, int(value*2 / self.reduction), num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, int(value*4 / self.reduction), num_blocks[2], stride=2)
        self.linear = nn.Linear(int(value*4 / self.reduction), num_classes)
        parameters.nb_params += sum([x.data.nelement() for x in self.linear.parameters()])

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        planes = int(planes)
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def flops(self):
        if not hasattr(self, 'int_nchw'):
            raise UserWarning('Must run forward at least once')
        (_, _, int_h, int_w), (out_h, out_w) = self.int_nchw, self.out_hw
        flops = 0
        for mod in (self.layer1, self.layer2, self.layer3):
            for layer in mod:
                flops += layer.flops()
        return int_h*int_w*9*self.in_planes*3 + out_w*self.num_classes + flops

    def forward(self, x):
        global nb_activation
        nb_activation= x.shape[1]*x.shape[2]*x.shape[3]
        out = self.conv1(x)
        nb_activation+= out.shape[1]*out.shape[2]*out.shape[3]
        out = self.bn1(out)
        nb_activation+= out.shape[1]*out.shape[2]*out.shape[3]
        out = F.relu(out)
        self.int_nchw = out.size()
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, 8)
        nb_activation+= out.shape[1]*out.shape[2]*out.shape[3]
        out = out.view(out.size(0), -1)
        self.out_hw = out.size()
        out = self.linear(out)
        nb_activation+= out.shape[1]
        return out,nb_activation


def ResNetWrapper(num_blocks, reduction=1, reduction_mode='net', num_classes=10):
    if reduction_mode == 'block':
        block = lambda in_planes, planes, stride: \
            BasicBlock(in_planes, planes, stride, reduction=reduction)
        return ResNet(block, num_blocks, num_classes=num_classes)
    return ResNet(BasicBlock, num_blocks, num_classes=num_classes, reduction=reduction)


def SATWrapper(num_blocks, reduction=1, reduction_mode='net', num_classes=10):
    if reduction_mode == 'block':
        block = lambda in_planes, planes, stride: \
            SATBlock(in_planes, planes, stride, reduction=reduction)
        return ResNet(block, num_blocks, num_classes=num_classes)
    return ResNet(SATBlock, num_blocks, num_classes=num_classes, reduction=reduction)


def ResNet20(reduction=1, reduction_mode='net', num_classes=10):
    return ResNetWrapper([3, 3, 3], reduction, reduction_mode, num_classes)

def ResNet56(reduction=1, reduction_mode='net', num_classes=10):
    return SATWrapper([9, 9, 9], reduction, reduction_mode, num_classes)

def SATResNet20(reduction=1, reduction_mode='net', num_classes=10):
    return SATWrapper([3, 3, 3], reduction, reduction_mode, num_classes)
