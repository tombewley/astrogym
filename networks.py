"""
NOTE: This is copied and expanded from:
https://github.com/tombewley/rlutils/blob/main/rlutils/common/networks.py

Should eventually merge this back into that repo
"""

import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
from gym.spaces.box import Box


# ===================================================================
# RESNET (https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py))


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, in_channels, num_classes=128):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.gpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(512, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        #out = F.avg_pool2d(out, 4)
        out = self.gpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        return out


def ResNet18(in_channels, num_classes=128):
    return ResNet(BasicBlock, [2,2,2,2], in_channels, num_classes=num_classes)

def ResNet34(in_channels, num_classes=128):
    return ResNet(BasicBlock, [3,4,6,3], in_channels, num_classes=num_classes)

def ResNet50(in_channels):
    return ResNet(Bottleneck, [3,4,6,3], in_channels)

def ResNet101(in_channels):
    return ResNet(Bottleneck, [3,4,23,3], in_channels)

def ResNet152(in_channels):
    return ResNet(Bottleneck, [3,8,36,3], in_channels)
    

# ===================================================================
# MULTI-HEADED NETWORK


class MultiHeadedNetwork(nn.Module):
    def __init__(self, device, common, head_codes, eval_only=False, optimiser=optim.Adam, lr=1e-3, clip_grads=False):
        super(MultiHeadedNetwork, self).__init__() 
        self.common = common
        self.heads = nn.ModuleList(nn.Sequential(*code_parser(code)) for code in head_codes)
        if eval_only: self.eval()
        elif optimiser is not None: 
            self.optimiser = optimiser(self.parameters(), lr=lr)
            self.clip_grads = clip_grads
        self.to(device)

    def optimise(self, loss, do_backward=True, retain_graph=True): 
        assert self.training, "Network is in eval_only mode."
        if do_backward: 
            self.optimiser.zero_grad()
            loss.backward()#retain_graph=retain_graph) 
        if self.clip_grads: # Optional gradient clipping.
            for param in self.parameters(): param.grad.data.clamp_(-1, 1) 
        self.optimiser.step()

    def forward(self, x, x_heads): 
        x = self.common(x)
        return tuple(head(torch.cat((x, x_head), axis=1)) for head, x_head in zip(self.heads, x_heads))

    
def code_parser(code, input_shape=None, output_size=None):
    layers = []
    for l in code:
        if type(l) in {list, tuple}:   
            i, o = l[0], l[1]
            if i is None: i = input_shape # NOTE: Only works for vectors at the moment.
            if o is None: o = output_size 
            layers.append(nn.Linear(i, o))
        elif l == "R":          layers.append(nn.ReLU())
        elif l == "T":          layers.append(nn.Tanh())
        elif l == "S":          layers.append(nn.Softmax(dim=1))
        elif l[0] == "D":       layers.append(nn.Dropout(p=l[1]))
        elif l[0] == "B":       layers.append(nn.BatchNorm2d(l[1]))
    return layers