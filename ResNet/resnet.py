import numpy
import torch
import torchvision
import torchvision.transforms as transforms

from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as functional

import torch.optim as optim

from utils import write_file_and_close, check_control, generate_filename

import os
import errno

global_batch_size = 128
global_resnet_n = 3
global_conv_bias = False
global_data_print_freq = 20
global_epoch_num = 200
global_cuda_available = True
global_output_filename = "out.txt"
global_control_filename = "control.txt"
global_epoch_test_freq = 1

if global_cuda_available:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

transform_train = transforms.Compose([
    # transforms.RandomCrop(32, padding=4),
    # transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Here per-pixel mean isn't subtracted

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = torchvision.datasets.CIFAR10(
    root="./data", download=True, train=True, transform=transform_train
)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=global_batch_size, shuffle=True, num_workers=2
)

testset = torchvision.datasets.CIFAR10(
    root="./data", download=True, train=False, transform=transform_test
)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=global_batch_size, shuffle=False, num_workers=2
)

class StartBlock(nn.Module):
    """First several blocks for resnet

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
    """Repeated blockes for resnet

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
        out += self.shortcut(x)
        return out

class EndBlock(nn.Module):
    """Last several blocks for resnet

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

class ResNet(nn.Module):
    """ResNet-(6n + 2)"""

    def __init__(self, n):
        super(ResNet, self).__init__()
        self.block_list = []
        self.block_list.append(StartBlock(16, 3))
        for i in range(n):
            self.block_list.append(BasicBlock(16, 16, 3, 1))
        self.block_list.append(BasicBlock(16, 32, 3, 2))
        for i in range(n - 1):
            self.block_list.append(BasicBlock(32, 32, 3, 1))
        self.block_list.append(BasicBlock(32, 64, 3, 2))
        for i in range(n - 1):
            self.block_list.append(BasicBlock(64, 64, 3, 1))
        self.block_list.append(EndBlock(64))
        self.blocks = nn.Sequential(*self.block_list)

    def forward(self, x):
        out = self.blocks(x)
        return out

net = ResNet(global_resnet_n)

if global_cuda_available:
    net.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(
    net.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0001
)

def lr_adjust(it):
    if it < 32000:
        return 0.1
    elif it < 48000:
        return 0.01
    elif it < 64000:
        return 0.001
    else:
        return 0.0001

def train(data, info):
    global net, optimizer, criterion
    inputs, labels = data
    inputs, labels = Variable(inputs), Variable(labels)
    if global_cuda_available:
        inputs, labels = inputs.cuda(), labels.cuda()
    optimizer.zero_grad()
    outputs = net(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    info[0] = loss.data[0]
    info[1] = labels.size()[0]

def test(info):
    global net
    correct_sum = 0
    total_loss_sum = 0.
    total_ctr = 0
    for data in testloader:
        inputs, labels = data
        inputs, labels = Variable(inputs), Variable(labels)
        if global_cuda_available:
            inputs, labels = inputs.cuda(), labels.cuda()
        outputs = net(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total_ctr += labels.size()[0]
        correct_sum += (predicted == labels.data).sum()
        loss = criterion(outputs, labels)
        total_loss_sum += loss.data[0]
    info[0] = correct_sum
    info[1] = total_ctr
    info[2] = total_loss_sum

write_file_and_close(global_output_filename, "Cleaning...", flag = "w")
write_file_and_close(
    global_output_filename,
    "The length of trainloader and testloader is {:d} and {:d} resp."
    .format(len(trainloader), len(testloader))
)

write_file_and_close(global_output_filename, "Start training")

it = 0
for epoch in range(global_epoch_num):
    if not check_control(global_control_filename):
        write_file_and_close(gloabl_output_filename, "Control llost")
    running_loss_sum = 0.
    total_loss_sum = 0.
    ctr_sum = 0
    total_ctr = 0
    for g in optimizer.param_groups:
        g["lr"] = lr_adjust(it)
    for i, data in enumerate(trainloader):
        info = [0., 0]
        train(data, info)
        running_loss_sum += info[0]
        total_loss_sum += info[0]
        ctr_sum += 1
        total_ctr += info[1]
        if (i + 1) % global_data_print_freq == 0:
            write_file_and_close(global_output_filename,
                "epoch: {:d}, "
                "train set index: {:d}, "
                "average loss: {:.10f}"
                .format(epoch, i, running_loss_sum / ctr_sum)
            )
            running_loss_sum = 0.0
            ctr_sum = 0
        it = it + 1
    write_file_and_close(global_output_filename,
        "Epoch {:d} finished, average loss: {:.10f}"
        .format(epoch, total_loss_sum / total_ctr)
    )
    if (epoch + 1) % global_epoch_test_freq == 0:
        write_file_and_close(global_output_filename, "Starting testing")
        info = [0., 0., 0.]
        test(info)
        write_file_and_close(global_output_filename,
            "Correct: {:d}, total: {:d}, "
            "accuracy: {:.10f}, average loss: {:.10f}"
            .format(info[0], info[1], info[0] / info[1], info[2] / info[1])
        )
        write_file_and_close(global_output_filename, "Finished testing")

model_filename = generate_filename()
torch.save(net, model_filename)
