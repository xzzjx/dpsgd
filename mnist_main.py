# coding: utf-8

import torch
import torchvision
from autoencoder import autoencoder
from torch import nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.autograd import Variable
from torch import optim


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(60, 1000)
        self.fc2 = nn.Linear(1000, 10)
    
    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

def do_PCA(dataloader):
    net_param_path = './sim_autoencoder.pth'
    ae = autoencoder().cuda()
    ae.load_state_dict(torch.load(net_param_path))
    ae.eval()
    encodeds = []
    for img, y in dataloader:
        img = Variable(img).cuda()
        # y = Variable(y).cuda()

        img = img.view(-1, 28*28)
        output = ae(img)[0]
        # print(len(output))
        encodeds.append(output)
    codes = torch.cat(encodeds, dim=0)

    return codes

class CodeDataset(torch.utils.data.Dataset):
    def __init__(self, codes, y_set):
        self.data = codes
        self.y_set = y_set
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index], self.y_set.__getitem__(index)[1]

def build_mlp(epochs=10, batch_size=128, learning_rate=0.01):
    img_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307, ), (0.3081, ))
    ])
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=True, transform=img_transforms),
        batch_size=batch_size, shuffle=False
    )
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=False,  transform=img_transforms),
        batch_size=batch_size, shuffle=False
    )
    codes = do_PCA(train_loader)
    
    code_loader = torch.utils.data.DataLoader(
        CodeDataset(codes, datasets.MNIST('./data', train=True)),
        batch_size=batch_size, shuffle=True
    )
    mlp = MLP().cuda()
    loss_fn = nn.CrossEntropyLoss()
    train_op = optim.SGD(mlp.parameters(), lr=learning_rate, momentum=0.9)

    for epoch in range(epochs):
        for batch_idx, (data, y) in enumerate(code_loader):
            data = Variable(data).cuda()
            y = Variable(y).cuda()

            output = mlp(data)
            loss = loss_fn(output, y)

            train_op.zero_grad()
            loss.backward()
            train_op.step()

            if batch_idx % 100 == 0:
                print('Train Epoch: {} [{} / {} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx*len(data), len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader), loss.item()
                ))

    test_loss = 0
    correct = 0
    codes_test = do_PCA(test_loader)
    code_test_loader = torch.utils.data.DataLoader(
        CodeDataset(codes_test, datasets.MNIST('./data', train=False)),
        batch_size=batch_size, shuffle=True
    )
    for data, target in code_test_loader:
        data, target = Variable(data), Variable(target)
        data = data.cuda()
        target = target.cuda()
        # data = data.view(-1, 28*28)
        output = mlp(data)

        test_loss += loss_fn(output, target).item()
        pred = output.data.max(1)[1]
        correct += pred.eq(target.data).sum()

    test_loss /= len(test_loader.dataset)
    print(
        '\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset), 100.0 * correct / len(test_loader.dataset)
        )
    )
if __name__ == '__main__':
    build_mlp()