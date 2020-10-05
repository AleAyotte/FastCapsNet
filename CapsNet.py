"""
    @file:              CapsNet.py
    @Author:            Alexandre Ayotte

    @Inspired by:       https://github.com/gram-ai/capsule-networks
                        https://github.com/higgsfield/Capsule-Network-Tutorial
    @Creation Date:     04/10/2020
    @Last modification: 04/10/2020

    An optimized implementation of the CapsNet from GRAM-AI that do not required TorchNet and Visdom
"""

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim import Adam
from torchvision import datasets, transforms
from tqdm import tqdm

BATCH_SIZE = 100
NUM_CLASSES = 10
NUM_EPOCHS = 30
NUM_ROUTING_ITERATIONS = 3


class Mnist:
    def __init__(self, batch_size):
        dataset_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        train_dataset = datasets.MNIST('../data', train=True, download=True, transform=dataset_transform)
        test_dataset = datasets.MNIST('../data', train=False, download=True, transform=dataset_transform)

        self.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)


class CapsuleLayer(nn.Module):
    def __init__(self, num_capsules, num_route_nodes, in_channels, out_channels, kernel_size=None, stride=None,
                 num_iterations=NUM_ROUTING_ITERATIONS):
        super(CapsuleLayer, self).__init__()

        self.num_route_nodes = num_route_nodes
        self.num_iterations = num_iterations
        self.num_capsules = num_capsules

        if num_route_nodes != -1:
            self.route_weights = nn.Parameter(torch.randn(num_capsules, num_route_nodes, in_channels, out_channels))

        else:
            # Only one convolution layer instead of num_capsules convolution layers.
            self.capsules = nn.Conv2d(
                in_channels,
                out_channels*num_capsules,
                kernel_size=kernel_size,
                stride=stride,
                padding=0
            )

    def squash(self, tensor, dim=-1):
        squared_norm = (tensor ** 2).sum(dim=dim, keepdim=True)
        return torch.sqrt(squared_norm) / (1 + squared_norm) * tensor

    def forward(self, x):
        if self.num_route_nodes != -1:

            # It seem that pytorch is faster when we transpose capsule dimension(8) with vector dimension(1152).
            priors = torch.matmul(
                x[None, :, :, None, :],
                self.route_weights[:, None, :, :, :]
            ).transpose(2, 4)

            logits = Variable(torch.zeros(*priors.size())).cuda()
            for i in range(self.num_iterations):
                probs = F.softmax(logits, dim=-1)
                outputs = self.squash((probs * priors).sum(dim=-1, keepdim=True), dim=2)

                if i != self.num_iterations - 1:
                    delta_logits = (priors * outputs).sum(dim=2, keepdim=True)
                    logits = logits + delta_logits
        else:
            # Reshape the output to simulate self.num_capsules convolution layers.
            outputs = self.capsules(x).view(x.size(0), self.num_capsules, -1).transpose(1, 2)
            outputs = self.squash(outputs)

        return outputs


class CapsuleNet(nn.Module):
    def __init__(self):
        super(CapsuleNet, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=256, kernel_size=9, stride=1)
        self.primary_capsules = CapsuleLayer(num_capsules=8, num_route_nodes=-1, in_channels=256, out_channels=32,
                                             kernel_size=9, stride=2)
        self.digit_capsules = CapsuleLayer(num_capsules=NUM_CLASSES, num_route_nodes=32 * 6 * 6, in_channels=8,
                                           out_channels=16)

        self.decoder = nn.Sequential(
            nn.Linear(16 * NUM_CLASSES, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 784),
            nn.Sigmoid()
        )

    def forward(self, x, y=None):
        x = F.relu(self.conv1(x), inplace=True)
        x = self.primary_capsules(x)
        x = self.digit_capsules(x).squeeze().transpose(0, 1)

        classes = (x ** 2).sum(dim=-1) ** 0.5
        classes = F.softmax(classes, dim=-1)

        if y is None:
            # In all batches, get the most active capsule.
            _, max_length_indices = classes.max(dim=1)
            y = Variable(torch.eye(NUM_CLASSES)).cuda().index_select(dim=0, index=max_length_indices.data)

        reconstructions = self.decoder((x * y[:, :, None]).reshape(x.size(0), -1))

        return classes, reconstructions


class CapsuleLoss(nn.Module):
    def __init__(self):
        super(CapsuleLoss, self).__init__()
        self.reconstruction_loss = nn.MSELoss(reduction='sum')

    def forward(self, images, labels, classes, reconstructions):
        left = F.relu(0.9 - classes, inplace=True) ** 2
        right = F.relu(classes - 0.1, inplace=True) ** 2

        margin_loss = labels * left + 0.5 * (1. - labels) * right
        margin_loss = margin_loss.sum()

        images = images.view(reconstructions.size()[0], -1)
        reconstruction_loss = self.reconstruction_loss(reconstructions, images)

        return (margin_loss + 0.0005 * reconstruction_loss) / images.size(0)



capsule_net = CapsuleNet().cuda()
cap_loss = CapsuleLoss().cuda()

optimizer = Adam(capsule_net.parameters())
mnist = Mnist(BATCH_SIZE)

torch.backends.cudnn.benchmark = True

with tqdm(total=NUM_EPOCHS) as t:
    train_loss = 0
    test_loss = 0
    test_accuracy = 0

    for epoch in range(NUM_EPOCHS):
        capsule_net.train()
        train_loss = 0
        test_loss = 0
        test_accuracy = 0
        total = 0

        for batch_id, (data, target) in enumerate(mnist.train_loader):

            target = torch.sparse.torch.eye(10).index_select(dim=0, index=target)
            data, target = Variable(data).cuda(), Variable(target).cuda()

            optimizer.zero_grad()
            output, reconstructions = capsule_net(data)
            loss = cap_loss(data, target, output, reconstructions)
            loss.backward()
            optimizer.step()

            train_loss += loss

        capsule_net.eval()
        for batch_id, (data, target) in enumerate(mnist.test_loader):

            target = torch.sparse.torch.eye(10).index_select(dim=0, index=target)
            data, target = Variable(data).cuda(), Variable(target).cuda()

            with torch.no_grad():
                output, reconstructions= capsule_net(data)
                test_loss = cap_loss(data, target, output, reconstructions)
                pred = torch.argmax(output, dim=1)
                test_accuracy += (torch.argmax(output, dim=1) == torch.argmax(target, dim=1)).sum()

            total += len(target)

        train_loss = train_loss.item() / len(mnist.train_loader)
        test_loss = test_loss.item() / len(mnist.test_loader)
        test_accuracy = test_accuracy.item() / total

        t.postfix = "train loss: {:.4f}, test loss: {:.4f}, test acc: {:.2f}%".format(
            train_loss, test_loss, test_accuracy * 100
        )

        t.update()