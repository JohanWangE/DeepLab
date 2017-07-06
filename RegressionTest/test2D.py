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
from math import sqrt

from math import  sin,pi

class block(nn.Module):
    def __init__(self,inplanes,outplanes):
        super(block,self).__init__()
        self.fc = nn.Linear(inplanes,outplanes)

    def forward(self,x):
        x = F.relu(self.fc(x))
        return x

class TestNet(nn.Module):
    def __init__(self):
        super(TestNet, self).__init__()
        layer = [block(2,50)]
        for i in range(10):
            layer.append(block(50,50))

        self.fc = nn.Linear(50,1)
        layer.append(self.fc)
        self.forward_prop=nn.Sequential(*layer)


    def forward(self, x):
        return self.forward_prop(x)

f = lambda x:sqrt(1-x**2)

test_func = lambda x:10*sin(2*pi*x)
n =  20
data_generate = [i/n for i in range(n)]

if __name__ =="__main__":
    print(data_generate)
    dataset = Variable(torch.Tensor(list(map(lambda x:[x,f(x)],data_generate))))
    data_test = Variable(torch.Tensor(list(map(test_func,data_generate))))

    Net = TestNet()
    #print(data_test)
    print(data_test)

    learning_rate = 1e-4

    criterian = nn.MSELoss()
    optimizer = optim.Adam(Net.parameters(),lr=learning_rate)

    min_los = float('inf')

    iter_time = 50000
    for i in range(iter_time):
        output = Net(dataset)
        loss = criterian(output,data_test)
        loss.backward()
        optimizer.step()

        running_loss = 0.
        running_acc = 0.
        running_loss += loss.data[0]


        running_loss /= len(dataset)
        if running_loss < min_los:
            torch.save(Net, '2dmodel_.pkl')
            min_los = running_loss

        print("[%d/%d] Loss: %.5f" % (i + 1, iter_time, running_loss))
    print(min_los)



