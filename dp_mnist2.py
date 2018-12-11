# coding: utf-8

import torch
import torchvision
from autoencoder import autoencoder
from torch import nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.autograd import Variable
from torch import optim
import collections
import copy
import math

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(60, 1000, bias=False)
        self.fc2 = nn.Linear(1000, 10, bias=False)
    
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
        img = Variable(img)

        img = img.view(-1, 28*28)
        output = ae(img)[0]
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


def per_example_gradient(data, y, model, loss_fn, train_op):
    '''
    compute per example gradient by feeding an example each pass and recording its gradient
    '''
    loss_val = 0
    g_dict = collections.defaultdict(list)
    batch_size = data.size()[0]
    for i in range(batch_size):
        d_i = data[i].unsqueeze(0)
        y_i = y[i].unsqueeze(0)
        output = model(d_i)
        loss = loss_fn(output, y_i)
        loss_val += loss.item()
        train_op.zero_grad()
        loss.backward()
        for name, param in model.named_parameters():
            g_dict[name].append(copy.deepcopy(param.grad.data))
    return g_dict, loss_val/batch_size

def santinizer(g_dict, C, sigma, batch_size):
    '''
    g_dict: key is var name, value is list of gradient
    we need to restrict every gradient of g_dict lower than C
    '''
    noise_dist = torch.distributions.Normal(loc=0.0, scale=sigma*C/batch_size)
    for var in g_dict:
        gradients = g_dict[var]
        assert len(gradients) == batch_size
        for i in range(batch_size):
            l2_norm = torch.norm(gradients[i])
            inv_norm = max(1.0, l2_norm/C)
            gradients[i] = gradients[i] / inv_norm
        g_cat = torch.stack(gradients, dim=0)
        g_mean = torch.mean(g_cat, dim=0)
        noise = noise_dist.sample(g_mean.size())
        g_dict[var] = g_mean + noise
    return g_dict


def STH_denoise(g_dict, sigma, batch_size, C):
    ratio = math.sqrt(batch_size) / C
    for var in g_dict:
        g = g_dict[var] * ratio
        d = g.view(-1).size()[0]
        lm = sigma * math.sqrt(2*math.log(d))
        g_ = torch.sign(g) * torch.max(torch.zeros_like(g), torch.abs(g) - lm)
        g_dict[var] = g_ / ratio
    return g_dict

def JS_denoise(g_dict, sigma, batch_size, C):
    for var in g_dict:
        g = g_dict[var]
        d = g.view(-1).size()[0]
        l2_norm = torch.norm(g)
        g_ = (1-(d-2)*sigma*sigma/l2_norm)*g
        g_dict[var] = g_
    return g_dict

def WMA(old_g_dict, g_dict, step, alpha=0.9):
    '''
    weighted moving average of gradients
    old_g_dict: contain the old gradient
    g_dict: contain this step's gradient
    step: global step count
    '''
    if step == 1:
        return g_dict
    for var in g_dict:
        g_dict[var] = alpha * g_dict[var] + (1-alpha) * old_g_dict[var]
    return g_dict
    
def resign_gradient(g_dict, model):
    '''
    assign gradient from g_dict to model parameter
    '''
    for name, param in model.named_parameters():
        param.grad = g_dict[name]
    return model

def adjust_learning_rate(optimizer, epoch, init_lr=0.1, saturate_epoch=10, stop_lr=0.052):
    '''
    linearly adjust learning_rate, accoring to tensorflow's implementation
    '''
    step = (init_lr - stop_lr) / (saturate_epoch-1)
    if epoch < saturate_epoch:
        lr = init_lr - step * epoch
    else:
        lr = stop_lr
    for param_group in optimizer.param_groups:
            param_group['lr'] = lr


def test(mlp, dataloader, loss_fn):
    '''
    evaluate model's performance on test dataset
    '''
    test_loss = 0
    correct = 0
    for data, target in dataloader:
        data, target = Variable(data), Variable(target)
        data = data
        target = target
        output = mlp(data)

        test_loss += loss_fn(output, target).mean()
        pred = output.data.max(1)[1]
        correct += pred.eq(target.data).sum()

    test_loss /= len(dataloader.dataset)
    print(
        '\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(dataloader.dataset), 100.0 * correct / len(dataloader.dataset)
        )
    )
def build_mlp(epochs=100, batch_size=600, learning_rate=0.1, C=4.0, sigma=2.0):
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
        batch_size=batch_size, shuffle=True, drop_last=True
    )

    codes_test = do_PCA(test_loader)
    code_test_loader = torch.utils.data.DataLoader(
        CodeDataset(codes_test, datasets.MNIST('./data', train=False)),
        batch_size=batch_size, shuffle=True
    )

    mlp = MLP()
    loss_fn = nn.CrossEntropyLoss(reduce=False)
    train_op = optim.SGD(mlp.parameters(), lr=learning_rate)

    # train phase
    old_g_dict = collections.defaultdict(list)
    step = 0
    for epoch in range(epochs):
        adjust_learning_rate(train_op, epoch)
        for batch_idx, (data, y) in enumerate(code_loader):
            data = Variable(data)
            y = Variable(y)
            # g_dict = collections.defaultdict(list)
            step += 1
            g_dict, loss_val = per_example_gradient(data, y, mlp, loss_fn, train_op)
            g_dict = santinizer(g_dict, C, sigma, batch_size)
            train_op.zero_grad()
            # g_dict = STH_denoise(g_dict, sigma, batch_size, C)
            # g_dict = JS_denoise(g_dict, sigma, batch_size, C)
            g_dict = WMA(old_g_dict, g_dict, step)
            mlp = resign_gradient(g_dict, mlp)
            old_g_dict  = g_dict
            train_op.step()

            if batch_idx % 10 == 0:
                print('Train Epoch: {} [{} / {} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx*len(data), len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader), loss_val
                    ))
                output = mlp(data)
                pred = output.data.max(1)[1]
                correct = pred.eq(y.data).sum()
                print('train accuracy: ', correct.item()/batch_size)
        

        # test phase
        
        test(mlp, code_test_loader, loss_fn)
    
if __name__ == '__main__':
    build_mlp()
