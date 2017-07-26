'''
MSRA INIT
net parameter initialization.
He K, Zhang X, Ren S, et al.
Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification,
arxiv:1502.01852
Reference:http://damon.studio/2017/06/11/WeightsInitialization/
'''

import torch.nn as nn
import torch.nn.init as init

'''
usage:
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
            # Modified by lzh @ 201707251408:
            # <<< Old:
            # if m.bias:
            # >>> New:
            if m.bias is not None:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            # Modified by lzh @ 201707241734:
            # <<< Old:
            # if m.bias:
            # >>> New:
            if m.bias is not None:
            # --- End
                init.constant(m.bias, 0)

# Added by lzh @ 201707251404:
def xavier_init(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.xavier_normal(m.weight)
            if m.bias is not None:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias is not None:
                init.constant(m.bias, 0)

def gauss_init(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.normal(0.0, 0.01)
            if m.bias is not None:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias is not None:
                init.constant(m.bias, 0)
# --- End