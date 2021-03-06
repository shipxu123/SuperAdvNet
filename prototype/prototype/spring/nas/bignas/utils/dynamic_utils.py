import math
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def int2list(val, repeat_time=1):
    if isinstance(val, list) or isinstance(val, np.ndarray):
        return val
    elif isinstance(val, tuple):
        return list(val)
    else:
        return [val for _ in range(repeat_time)]


def sub_filter_start_end(kernel_size, sub_kernel_size):
    center = kernel_size // 2
    dev = sub_kernel_size // 2
    start, end = center - dev, center + dev + 1
    assert end - start == sub_kernel_size
    return start, end


def get_same_padding(kernel_size):
    if isinstance(kernel_size, tuple):
        assert len(kernel_size) == 2, 'invalid kernel size: %s' % kernel_size
        p1 = get_same_padding(kernel_size[0])
        p2 = get_same_padding(kernel_size[1])
        return p1, p2
    assert isinstance(kernel_size, int), 'kernel size should be either `int` or `tuple`'
    assert kernel_size % 2 > 0, 'kernel size should be odd number'
    return kernel_size // 2


def make_divisible(v, divisor, min_val=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_val:
    :return:
    """
    if min_val is None:
        min_val = divisor
    new_v = max(min_val, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def build_activation(act_func, inplace=True):
    if act_func == 'relu':
        return nn.ReLU(inplace=inplace)
    elif act_func == 'relu6':
        return nn.ReLU6(inplace=inplace)
    elif act_func == 'tanh':
        return nn.Tanh()
    elif act_func == 'sigmoid':
        return nn.Sigmoid()
    elif act_func == 'h_swish':
        return Hswish(inplace=inplace)
    elif act_func == 'swish':
        return swish()
    elif act_func == 'h_sigmoid':
        return Hsigmoid(inplace=inplace)
    elif act_func is None:
        return None
    else:
        raise ValueError('do not support: %s' % act_func)


class swish(nn.Module):
    def __init__(self):
        super(swish, self).__init__()

    def forward(self, x):
        x = x * torch.sigmoid(x)
        return x


class Hswish(nn.Module):

    def __init__(self, inplace=True):
        super(Hswish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return x * F.relu6(x + 3., inplace=self.inplace) / 6.


class Hsigmoid(nn.Module):

    def __init__(self, inplace=True):
        super(Hsigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return F.relu6(x + 3., inplace=self.inplace) / 6.


class SEModule(nn.Module):
    REDUCTION = 4

    def __init__(self, channel, divisor=8):
        super(SEModule, self).__init__()

        self.channel = channel
        self.reduction = SEModule.REDUCTION

        num_mid = make_divisible(self.channel // self.reduction, divisor=divisor)

        self.fc = nn.Sequential(OrderedDict([
            ('reduce', nn.Conv2d(self.channel, num_mid, 1, 1, 0, bias=True)),
            ('relu', nn.ReLU(inplace=True)),
            ('expand', nn.Conv2d(num_mid, self.channel, 1, 1, 0, bias=True)),
            ('h_sigmoid', Hsigmoid(inplace=True)),
        ]))

    def forward(self, x):
        y = x.mean(3, keepdim=True).mean(2, keepdim=True)
        y = self.fc(y)
        return x * y


# this is used in Quantization
def gradscale(x, scale):
    yOut = x
    yGrad = x * scale
    y = (yOut - yGrad).detach() + yGrad
    return y


def signpass(x):
    yOut = x.sign()
    yGrad = x
    y = (yOut - yGrad).detach() + yGrad
    return y


def clippass(x, Qn, Qp):
    yOut = F.hardtanh(x, Qn, Qp)
    yGrad = x
    y = (yOut - yGrad).detach() + yGrad
    return y


def roundpass(x):
    yOut = x.round()
    yGrad = x
    y = (yOut - yGrad).detach() + yGrad
    return y


# v is input
# s in learnable step size
# p in the number of bits
def quantize(v, s, p, isActivation):
    if isActivation:
        Qn = 0
        Qp = 2 ** p - 1
        nfeatures = v.nelement()
        gradScaleFactor = 1 / math.sqrt(nfeatures * Qp)

    else:
        Qn = -2 ** (p - 1)
        Qp = 2 ** (p - 1) - 1
        if p == 1:
            Qp = 1
        nweight = v.nelement()
        gradScaleFactor = 1 / math.sqrt(nweight * Qp)

    s = gradscale(s, gradScaleFactor)
    v = v / s.abs()
    v = F.hardtanh(v, Qn, Qp)
    if not isActivation and p == 1:
        vbar = signpass(v)
    else:
        vbar = roundpass(v)
    vhat = vbar * s.abs()
    return vhat
