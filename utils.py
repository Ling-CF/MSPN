import lpips
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from torch.utils import data
import os
os.environ['TORCH_HOME']='/home/Ling.cf/HHD/conda/miniconda3/torch-model'

class util_of_lpips():
    def __init__(self, net):
        self.loss_fn = lpips.LPIPS(net=net)


    def calc_lpips(self, predict, target):
        device = predict.device
        self.loss_fn.to(device)
        dist01 = self.loss_fn.forward(predict, target, normalize=True)
        return dist01

class GDLLoss():
    def __init__(self, sigma=1000):
        self.sigma = sigma

    def laplacian_filter(self, x):
        channels = x.size(1)
        f = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
        f = torch.as_tensor(f, device=x.device).reshape(1, 1, 3, 3)
        f = f.repeat(channels, 1, 1, 1).float()
        out = F.conv2d(x, weight=f, groups=channels, padding=0)
        return out

    def calc_loss(self, predict, true):
        gd_pred = self.laplacian_filter(predict)
        gd_true = self.laplacian_filter(true)
        diff = gd_pred - gd_true
        gdloss = torch.mean(torch.sum(0.5 * self.sigma * diff ** 2, dim=1, keepdim=True))
        return gdloss

class Loss():
    def __init__(self):
        self.cross_loss = nn.CrossEntropyLoss()
        self.GDLLoss = GDLLoss(sigma=1000)

    def d_loss(self, logits_real, logits_fake):
        # discriminator loss
        loss = torch.mean((logits_real-1)**2) +torch.mean((logits_fake+1)**2)
        return loss, logits_fake.mean().item(), logits_real.mean().item()

    def g_loss(self, logits_fake,logits_real):
        # generator loss
        loss = torch.mean((logits_fake-1)**2)
        return loss , logits_fake.mean().item(), logits_real.mean().item()

    def pix_loss(self, inputs, targets, sigma=1):
        if isinstance(inputs, list):
            loss = 0
            for index in range(len(inputs)):
                cur_input = inputs[index]
                h, w = cur_input.size()[-2:]
                cur_target = F.interpolate(targets, size=(h, w))
                loss += torch.mean(torch.sum(0.5 * sigma * (cur_input- cur_target) ** 2, dim=1))
        else:
            loss = torch.mean(torch.sum(0.5 * sigma * (inputs- targets) ** 2, dim=1))
        return loss

    def gdl_loss(self, inputs, targets):
        if isinstance(inputs, list):
            loss = 0
            for index in range(len(inputs)):
                cur_input = inputs[index]
                h, w = cur_input.size()[-2:]
                cur_target = F.interpolate(targets, size=(h, w))
                loss += self.GDLLoss.calc_loss(cur_input, cur_target)
        else:
            loss = self.GDLLoss.calc_loss(inputs, targets)
        return loss

class MyDataset(data.Dataset):
    def __init__(self, path, len_input, interval=1, begin=0):
        '''
        The images have been pre-processed into data of size (T, C, H, W), where T denotes length of total sequence.
        '''
        self.data = torch.load(path)[begin:]
        self.len_seq = self.data.size(0)
        self.interval = interval
        self.len_input = len_input
        self.nodes = []

    def __len__(self):
        len_index = 0
        for i in range(self.interval):
            len_index += ((self.len_seq + (self.interval - 1) - i) // (self.len_input * self.interval))
            self.nodes.append(len_index)
        return len_index

    def __getitem__(self, item):
        # print(len(self.nodes))
        for round in range(len(self.nodes)):
            if item < self.nodes[round]:
                if round - 1 >= 0:
                    item = item - self.nodes[round-1]
                break
        input_seq = [self.data[i:i+1, :] for i in range(round + item * self.len_input * self.interval, round + (item + 1) * self.len_input *self.interval, self.interval)]
        return torch.cat(input_seq, dim=0)

