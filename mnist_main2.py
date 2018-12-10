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


def per_example_gradient(data, y, model, loss_fn, g_dict, train_op):
    loss_val = 0
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
    # print("--------------")
    # print(loss_val / batch_size)
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
        # print(g_cat.size()) 
        g_mean = torch.mean(g_cat, dim=0)
        noise = noise_dist.sample(g_mean.size())
        g_dict[var] = g_mean + noise
        # g_dict[var] = g_mean
        # print("g_dict_var: ", g_dict[var])
    return g_dict

# def santinizer(mlp, C, sigma, batch_size):
#     noise_dist = torch.distributions.Normal(loc=0.0, scale=sigma*C)
#     for name, param in mlp.named_parameters():
#         grad = param.grad
#         grad_l2_norm = torch.norm(grad, dim=2)
        

def resign_gradient(g_dict, model):
    '''
    ratio to adjust between batch_size and lot_size, ratio = batch_size / lot_size
    '''
    for name, param in model.named_parameters():
        # print("g_dict_var: ", g_dict[name].size())
        # print("param_grad: ", param.grad.size())
        param.grad = g_dict[name]
    return model

def adjust_learning_rate(optimizer, epoch, init_lr=0.1, saturate_epoch=10, stop_lr=0.052):
    step = (init_lr - stop_lr) / (saturate_epoch-1)
    if epoch < saturate_epoch:
        lr = init_lr - step * epoch
    else:
        lr = stop_lr
    for param_group in optimizer.param_groups:
            param_group['lr'] = lr

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
    mlp = MLP()
    loss_fn = nn.CrossEntropyLoss(reduce=False)
    train_op = optim.SGD(mlp.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        adjust_learning_rate(train_op, epoch)
        for batch_idx, (data, y) in enumerate(code_loader):
            data = Variable(data)
            y = Variable(y)

            # output = mlp(data)
            # loss = loss_fn(output, y)

            # train_op.zero_grad()
            ####
            g_dict = collections.defaultdict(list)
            g_dict, loss_val = per_example_gradient(data, y, mlp, loss_fn, g_dict, train_op)
            g_dict = santinizer(g_dict, C, sigma, batch_size)
            train_op.zero_grad()
            mlp = resign_gradient(g_dict, mlp)
            # for var in g_dict:
            #     g_dict[var] = sum(g_dict[var]) / batch_size
            # for name, param in mlp.named_parameters():
            #     param.grad = g_dict[name]
            
            # del g_dict
            ####
            # loss.backward(torch.ones_like(loss.data))
            # for name, param in mlp.named_parameters():
            #     param.grad = param.grad / batch_size
            #     print(param.grad.size())
            # santinizer(mlp, C, sigma, batch_size)
            # loss.sum().backward()
            # loss.backward()
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



    test_loss = 0
    correct = 0
    codes_test = do_PCA(test_loader)
    code_test_loader = torch.utils.data.DataLoader(
        CodeDataset(codes_test, datasets.MNIST('./data', train=False)),
        batch_size=batch_size, shuffle=True
    )
    for data, target in code_test_loader:
        data, target = Variable(data), Variable(target)
        data = data
        target = target
        # data = data.view(-1, 28*28)
        output = mlp(data)

        test_loss += loss_fn(output, target).mean()
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
