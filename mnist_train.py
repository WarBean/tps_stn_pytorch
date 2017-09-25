# encoding: utf-8

import os
import torch
import random
import argparse
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from mnist_model import STNClsNet, ClsNet
from torchvision import datasets, transforms

# Training settings
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
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST(
        'mnist_data',
        train = True,
        download = True,
        transform = transforms.Compose([
            transforms.Lambda(lambda image: image.rotate(random.random() * args.angle * 2 - args.angle)),
            transforms.ToTensor(),
        ]),
    ),
    batch_size = args.batch_size,
    shuffle = True,
    num_workers = 4,
    pin_memory = True if args.cuda else False,
)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST(
        'mnist_data',
        train = False,
        transform = transforms.Compose([
            transforms.ToTensor(),
        ]),
    ),
    batch_size = args.batch_size,
    shuffle = True,
    num_workers = 4,
    pin_memory = True if args.cuda else False,
)

if args.model == 'no_stn':
    print('create model without STN')
    model = ClsNet()
else:
    print('create model with STN')
    model = STNClsNet(args.model)
if args.cuda:
    model.cuda()

optimizer = optim.SGD(model.parameters(), lr = args.lr, momentum = args.momentum)

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
    torch.save(model.cpu().state_dict(), 'checkpoint/%s_%03d.pth' % (args.model, epoch))
    if args.cuda:
        model.cuda()
