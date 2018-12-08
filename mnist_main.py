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
    ae = autoencoder()
    ae.load_state_dict(torch.load(net_param_path))
    ae.eval()
    encodeds = []
    for img, y in dataloader:
        img = Variable(img).cuda()
        # y = Variable(y).cuda()

        img = img.view(-1, 28*28)
        output = ae(img)
        encodeds.append(output)
    codes = torch.stack(encodeds, dim=0)

    return codes
def build_mlp(epochs=100, batch_size=128, learning_rate=0.001):
    img_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307, ), (0.3081, ))
    ])
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=True, transform=img_transforms),
        batch_size=batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=False,  transform=img_transforms),
        batch_size=batch_size, shuffle=True
    )
    codes = do_PCA(train_loader)
    mlp = MLP().cuda()
    loss_fn = nn.CrossEntropyLoss()
    train_op = optim.SGD(mlp.parameters(), lr=learning_rate, momentum=0.9)

    for epoch in range(epochs):
        for i in range(train_loader.size()[0]):
            

if __name__ == '__main__':
    build_mlp()