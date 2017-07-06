import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.autograd import Variable
from torch import optim
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from random import random


from matplotlib import pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

from test2D import  f,n,TestNet,test_func,block



testnet = torch.load('2dmodel_.pkl')
testnet.eval()

def map_func(func):
    return lambda x:list(map(lambda x:func([[x[0],x[1]]]),x))

n1=32
data_test = [[(i/n,j/n) for j in range(n1)]for i in range(n1)]
data_plot = list(map(lambda x:list(map(lambda y:max(testnet.forward(Variable(torch.Tensor([[y[0],y[1]]]))).data.numpy()[0]),x)),data_test))


fig = plt.figure()
ax = Axes3D(fig)
X = np.arange(0, 1, 1/n1)
Y = np.arange(0, 1, 1/n1)
X, Y = np.meshgrid(X, Y)


# 具体函数方法可用 help(function) 查看，如：help(ax.plot_surface)
ax.plot_surface(X, Y, data_plot, rstride=1, cstride=1, cmap='rainbow')

for i in range(n):
    ax.scatter(f(i/n),i/n,test_func(i/n))
    ax.scatter( f(i / n),i / n, alpha=0.4,marker='+')

plt.show()