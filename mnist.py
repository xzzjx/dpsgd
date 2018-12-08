# coding: utf-8

from __future__ import division, print_function, unicode_literals, absolute_import

import torch
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F 
import torch.nn as nn 
from torchvision import datasets, transforms
import collections

def creat_nn(batch_size=200, learning_rate=0.01, epochs=10, log_interval=10):
    
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=True, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=batch_size, shuffle=True
    )

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=batch_size, shuffle=True
    )

    class Net(nn.Module):
        
        def __init__(self):
            super(Net, self).__init__()
            self.fc1 = nn.Linear(28*28, 200)
            self.fc2 = nn.Linear(200, 64)
            self.fc3 = nn.Linear(64, 10)
        
        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            # x = F.sigmoid(self.fc1(x))
            # x = F.sigmoid(self.fc2(x))
            x = self.fc3(x)
            return F.log_softmax(x)
        
    net = Net()
    print(net)
    device = torch.device("cuda" if torch.cuda.is_available else "cpu")
    net = net.to(device)

    train_op = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)

    criterion = nn.NLLLoss()

    for epoch in range(epochs):
        grad_of_params = collections.defaultdict(list)
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = Variable(data), Variable(target)
            data = data.to(device)
            target = target.to(device)
            data = data.view(-1, 28*28)
            train_op.zero_grad()
            net_out = net(data)
            loss = criterion(net_out, target)

            loss.backward()

            # noise_dist = torch.distributions.Normal(loc=0.0, scale=20.0)
            for name, param in net.named_parameters():
                # grad_of_params[name] = param.grad
                l2_norm = torch.norm(param.grad)
                grad_of_params[name].append(l2_norm)
                # if 'bias' in name:
                    # param.grad += noise_dist.sample(param.size()).to(device)
                # print(name, l2_norm)
            
            train_op.step()

            if batch_idx % log_interval == 0:
                print('Train Epoch: {} [{} / {} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx*len(data), len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader), loss.item()
                ))
        for name in grad_of_params:
            value = grad_of_params[name]
            value = torch.Tensor(value)
            print('name: {} std: {} mean: {}'.format(name, value.std(dim=0), value.mean(dim=0)))

    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data, target = Variable(data), Variable(target)
        data = data.to(device)
        target = target.to(device)
        data = data.view(-1, 28*28)
        net_out = net(data)

        test_loss += criterion(net_out, target).item()
        pred = net_out.data.max(1)[1]
        correct += pred.eq(target.data).sum()

    test_loss /= len(test_loader.dataset)
    print(
        '\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset), 100.0 * correct / len(test_loader.dataset)
        )
    )

if __name__ == '__main__':
    creat_nn()