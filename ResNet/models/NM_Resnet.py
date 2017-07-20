'''
Nesterov Momentum Resnet
cu_tt+u_t = f(u)
'''
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as functional

from torch.autograd import Variable

from msra_init import msra_init


import torchvision.transforms as transforms

from torch.autograd import Variable

import torch.optim as optim

import os

from file_control import *


global_conv_bias = False




class StartBlock(nn.Module):
    """
    First several blocks for resnet
    Only contains a single layer of conv2d and a batch norm layer
    """

    def __init__(self, out_planes, kernel_size):
        super(StartBlock, self).__init__()
        self.out_plane = out_planes
        self.kernel_size = kernel_size

        self.conv = nn.Conv2d(
            3, out_planes, kernel_size=kernel_size,
            padding=1, bias=global_conv_bias
        )
        self.bn = nn.BatchNorm2d(out_planes)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = functional.relu(out)
        return out

class BasicBlock(nn.Module):
    """
    Repeated blockes for resnet
    Contains two conv layers, two batch norm layers and a shortcut
    """

    def __init__(self, in_planes, out_planes, kernel_size, stride):
        super(BasicBlock, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_size = kernel_size
        self.stride = stride

        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(
            in_planes, out_planes, kernel_size=kernel_size,
            stride=stride, padding=1, bias=global_conv_bias
        )
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.conv2 = nn.Conv2d(
            out_planes, out_planes, kernel_size=kernel_size,
            padding=1, bias=global_conv_bias
        )
        self.shortcut = nn.Conv2d(
            in_planes, out_planes, kernel_size=1,
            stride=stride, bias=global_conv_bias
        )

    def forward(self, x):
        out = self.bn1(x)
        out = functional.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = functional.relu(out)
        out = self.conv2(out)
        if self.stride != 1 or self.in_planes != self.out_planes:
            out += self.shortcut(x)
        return out

class EndBlock(nn.Module):
    """
    Last several blocks for resnet
    Only contains a global average pooling layer and a fully
    connected layer.
    """

    def __init__(self, in_planes):
        super(EndBlock, self).__init__()
        self.fc = nn.Linear(in_planes, 10)

    def forward(self, x):
        out = torch.mean(x, dim=2)
        out = torch.mean(out, dim=3)
        out = out.view(out.size()[0], -1)
        out = self.fc(out)
        return out



class Weight_Id(nn.Module):
    def __init__(self):
        super(Weight_Id,self).__init__()
        self.weight = nn.Parameter(torch.Tensor(1))
        self.weight.data.uniform_(-0.1, 0.0)

    def forward(self):
        return self.weight


class NM_Residual_block(nn.Module):
    def __init__(self,in_planes,out_plane,kernel_size,blocks,subsample=True):
        super(NM_Residual_block, self).__init__()
        self.blocks = blocks
        self.subsample=subsample
        self.basic_block = []
        for i in range(blocks):
            self.basic_block.append(BasicBlock(in_planes,in_planes,kernel_size,1))
        if self.subsample:
            self.subsample = BasicBlock(in_planes,out_plane,kernel_size,2)
        self.para = nn.Sequential(*self.basic_block)
        self.momentum = []
        for i in range(self.blocks-1):
            self.momentum.append(Weight_Id())
        self.momentum_para = nn.Sequential(*self.momentum)

    def forward(self,x):
        out = x
        out_new = x + self.basic_block[0](x)
        for i in range(self.blocks-1):
            tmp = out_new
            out_new = (self.momentum[i]().expand_as(out_new)*out)+((1-self.momentum[i]()).expand_as(out))*out_new + self.basic_block[i](out_new)
            out = tmp
        if self.subsample:
            out = self.subsample(x)
        return out




class NM_ResNet(nn.Module):
    def __init__(self,block_num,out_classes=10):
        super(NM_ResNet,self).__init__()
        self.block_list = []
        self.block_list.append(StartBlock(16, 3))
        self.block_list.append(NM_Residual_block(16, 32, 3, block_num))
        self.block_list.append(NM_Residual_block(32, 64, 3, block_num))
        self.block_list.append(NM_Residual_block(64, 64, 3, block_num,subsample=False))
        self.block_list.append(EndBlock(64))
        self.blocks = nn.Sequential(*self.block_list)
        msra_init(self)

    def forward(self,x):
        out = self.blocks(x)
        return out

NM_ResNet20 = NM_ResNet(3)
NM_ResNet32 = NM_ResNet(5)
NM_ResNet44 = NM_ResNet(7)
NM_ResNet56 = NM_ResNet(9)
