# encoding: utf-8

import os
import torch
import random
import argparse
import mnist_model
import data_loader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--batch-size', type = int, default = 64)
parser.add_argument('--test-batch-size', type = int, default = 1000)
parser.add_argument('--epochs', type = int, default = 10)
parser.add_argument('--lr', type = float, default = 0.01)
parser.add_argument('--momentum', type=float, default = 0.5)
parser.add_argument('--no-cuda', action = 'store_true', default = False)
parser.add_argument('--seed', type = int, default = 1)
parser.add_argument('--log-interval', type = int, default = 10, metavar = 'N')
parser.add_argument('--model', required = True)
parser.add_argument('--angle', type = int, default=60)
parser.add_argument('--grid_size', type = int, default = 3)
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

model = mnist_model.get_model(args)
if args.cuda:
    model.cuda()

optimizer = optim.SGD(model.parameters(), lr = args.lr, momentum = args.momentum)
train_loader = data_loader.get_train_loader(args)
test_loader = data_loader.get_test_loader(args)

def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))

def test(epoch):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile = True), Variable(target)
        output = model(data)
        test_loss += F.nll_loss(output, target).data[0]
        pred = output.data.max(1)[1] # get the index of the max log-probability
        correct += pred.eq(target.data).cpu().sum()

    test_loss = test_loss
    test_loss /= len(test_loader) # loss function already averages over batch size
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


if not os.path.isdir('checkpoint'):
    os.makedirs('checkpoint')
for epoch in range(1, args.epochs + 1):
    train(epoch)
    test(epoch)
    torch.save(model.cpu().state_dict(), 'checkpoint/%s_angle%d_%03d.pth' % (args.model, args.angle, epoch))
    if args.cuda:
        model.cuda()
