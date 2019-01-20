import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import sampler
from AllCNN import *
import torchvision.datasets as dset
import torchvision.transforms as T
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torchvision
from torchvision import datasets, transforms
from utils import Visualizer
from load_data import *
vis = Visualizer()




cuda = True
train_batch_size = 32
test_batch_size = 124
best_loss = float("inf")
best_epoch = -1
best_acc = 0
dataset_path = './cifar-100/class1'
#gsync_save = True


from os import path
from wheel.pep425tags import get_abbr_impl, get_impl_ver, get_abi_tag
platform = '{}{}-{}'.format(get_abbr_impl(), get_impl_ver(), get_abi_tag())

accelerator = 'cu80' if path.exists('/opt/bin/nvidia-smi') else 'cpu'




USE_GPU = True

dtype = torch.float32 # we will be using float throughout this tutorial

if USE_GPU and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# Constant to control how frequently we print train loss
print_every = 100

print('using device:', device)



cuda = cuda and torch.cuda.is_available()


kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
transform = transforms.Compose(
    [transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = CustomData(root=dataset_path, train=True,transform=transform)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=train_batch_size,
                                          shuffle=True, **kwargs)

testset = CustomData(root=dataset_path, train=False,transform=transform)
                                       
test_loader = torch.utils.data.DataLoader(testset, batch_size=test_batch_size,
                                         shuffle=False, **kwargs)

model = AllConvNet(3)
model.init_weights()
if cuda:
    model.cuda()

#num_ftrs = model.conv9.in_channels
#model.conv9 = nn.Linear(num_ftrs, 2)

#model = model.to(device)
checkpath = './checkpoint/AllCNN_.pth'
#model.load_weight(checkpath)




# model_dict = model.state_dict()
# print(model_dict.keys())
# pretrained_dict = model_dict['conv1.weight', 'conv1.bias', 'conv2.weight', 'conv2.bias',\
# 'conv3.weight', 'conv3.bias', 'conv4.weight', 'conv4.bias', 'conv5.weight', 'conv5.bias', \
# 'conv6.weight', 'conv6.bias', 'conv7.weight', 'conv7.bias', 'conv8.weight', 'conv8.bias']





criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[200, 250, 300], gamma=0.1)


def train(epoch):
    model.train()
    train_loss = 0
    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        if cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)

        optimizer.zero_grad()
        output = model(data)
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()
        #print(output.shape, target.shape)
        loss = criterion(output, target)
        loss.backward()
        train_loss += loss.data[0]
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.data[0]))
    train_loss /= len(train_loader.dataset)
    train_acc = 100. * correct /len(train_loader.dataset)
    return train_loss, train_acc


def test(epoch, best_loss,best_acc):
    model.eval()

    test_loss = 0
    correct = 0
    for data, target in test_loader:
        if cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)

        output = model(data)
        # sum up batch loss
        test_loss += criterion(output, target).data[0]
        # get the index of the max log-probability
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()

    test_loss /= len(test_loader.dataset)
    test_acc = 100. * correct /len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset), test_acc))

    if test_loss < best_loss:
        best_loss = test_loss
        torch.save(model, "best.pt")
    if test_acc > best_acc:
        best_acc = test_acc
    return test_loss, test_acc, best_loss, best_acc


for epoch in range(300):
    scheduler.step()
    train_loss, train_acc = train(epoch)
    vis.plot('train_loss',train_loss)
    vis.plot('train_acc',train_acc)
    test_loss, test_acc, best_loss, best_acc = test(epoch, best_loss, best_acc)
    vis.plot('test_loss',test_loss)
    vis.plot('test_acc',test_acc)
    vis.plot('best_loss',best_loss)
    vis.plot('best_acc',best_acc)
