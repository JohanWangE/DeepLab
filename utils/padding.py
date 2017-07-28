from PIL import Image
from PIL._util import isStringType
import operator
import numbers
import functools
import numpy as np

import matplotlib.pyplot as plt



def expand(image, border=0):
    """
    Add border to the image(Symmetric padding)

    :param image: The image to expand.
    :param border: Border width, in pixels.
    :return: An image.
    """
    img = np.asarray(image)
    img = np.pad(img,pad_width=border,mode="reflect")
    return Image.fromarray(np.uint8(img[:,:,2:5]))

class Reflect_Pad(object):
    """Pads the given PIL.Image on all sides with the given "pad" """

    def __init__(self, padding):
        assert isinstance(padding, numbers.Number)
        self.padding = padding

    def __call__(self, img):
        return expand(img, border=self.padding)
def expand_edge(image, border=0):
    """
    Add border to the image(Symmetric padding)

    :param image: The image to expand.
    :param border: Border width, in pixels.
    :return: An image.
    """
    img = np.asarray(image)
    img = np.pad(img,pad_width=border,mode="edge")
    return Image.fromarray(np.uint8(img[:,:,2:5]))

class Edge_Pad(object):
    """Pads the given PIL.Image on all sides with the given "pad":edge pad """

    def __init__(self, padding):
        assert isinstance(padding, numbers.Number)
        self.padding = padding

    def __call__(self, img):
        return expand_edge(img, border=self.padding)


def expand_symmetric(image, border=0):
    """
    Add border to the image(Symmetric padding)

    :param image: The image to expand.
    :param border: Border width, in pixels.
    :return: An image.
    """
    img = np.asarray(image)
    img = np.pad(img,pad_width=border,mode="symmetric")
    return Image.fromarray(np.uint8(img[:,:,2:5]))

class Symmetric_Pad(object):
    """Pads the given PIL.Image on all sides with the given "pad":symmetric pad """

    def __init__(self, padding):
        assert isinstance(padding, numbers.Number)
        self.padding = padding

    def __call__(self, img):
        return expand_symmetric(img, border=self.padding)

def imshow(img):
    img = img / 2 + 0.5 # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)))
    plt.show()


if __name__ == "__main__":
    import torch
    import torchvision
    import  torchvision.transforms as transforms

    global_batch_size = 10

    transform_train = transforms.Compose([
        Reflect_Pad(2),
        #transforms.RandomCrop(32, padding=0),
        #transforms.RandomHorizontalFlip(),
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

    # show some random training images
    dataiter = iter(trainloader)
    images, labels = dataiter.next()

    # print images
    imshow(torchvision.utils.make_grid(images))
