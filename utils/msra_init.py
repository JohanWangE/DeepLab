'''
MSRA INIT
net parameter initialization.
He K, Zhang X, Ren S, et al.
Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification,
arxiv:1502.01852
'''

import os
import sys
import time
import math

import torch.nn as nn
import torch.nn.init as init


'''
useage:
class Net(nn.Module):
    def __init__(*parameters):
        super(Net, self).__init__()
        ...parameter setting

        msra_init(self)

    def forward(*para):
        ...
'''

def msra_init(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight)
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.kaiming_normal(m.weight)
            if m.bias:
                init.constant(m.bias, 0)
