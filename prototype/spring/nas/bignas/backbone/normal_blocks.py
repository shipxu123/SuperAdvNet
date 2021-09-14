from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F


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


class Identity(nn.Module):

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class SEBlock(nn.Module):

    def __init__(self, channel, reduction=4, bias=True, act_func1='relu', act_func2='h_sigmoid',
                 inplace=True, divisor=8):
        super(SEBlock, self).__init__()
        self.channel = channel
        self.reduction = reduction
        self.bias = bias
        self.act_func1 = act_func1
        self.act_func2 = act_func2
        self.inplace = inplace
        self.divisor = divisor

        num_mid = make_divisible(self.channel // self.reduction, divisor=self.divisor)

        self.fc = nn.Sequential(OrderedDict([
            ('reduce', nn.Conv2d(self.channel, num_mid, 1, 1, 0, bias=bias)),
            ('act1', build_activation(self.act_func1, inplace=inplace)),
            ('expand', nn.Conv2d(num_mid, self.channel, 1, 1, 0, bias=bias)),
            ('act2', build_activation(self.act_func2, inplace=True)),
        ]))

    def forward(self, x):
        y = F.adaptive_avg_pool2d(x, output_size=1)
        y = self.fc(y)
        return x * y


class LinearBlock(nn.Module):

    def __init__(self, in_features, out_features, bias=True, dropout_rate=0.):
        super(LinearBlock, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.dropout_rate = dropout_rate

        if self.dropout_rate > 0:
            self.dropout = nn.Dropout(self.dropout_rate, inplace=True)
        else:
            self.dropout = None
        self.linear = nn.Linear(
            in_features=self.in_features, out_features=self.out_features, bias=self.bias
        )

    def forward(self, x):
        if self.dropout is not None:
            x = self.dropout(x)
        return self.linear(x)


class ConvBlock(nn.Module):

    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, dilation=1,
                 use_bn=True, act_func='relu', NormLayer=nn.BatchNorm2d, shortcut=False):
        super(ConvBlock, self).__init__()

        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.use_bn = use_bn
        self.act_func = act_func
        self.shortcut = shortcut

        padding = get_same_padding(self.kernel_size)
        self.conv = nn.Conv2d(in_channels=self.in_channel, out_channels=self.out_channel,
                              kernel_size=self.kernel_size, padding=padding, groups=1,
                              stride=self.stride, dilation=self.dilation, bias=False)
        if self.use_bn:
            self.bn = NormLayer(self.out_channel)
        if self.act_func == '':
            self.act = Identity()
        else:
            self.act = build_activation(self.act_func, inplace=True)

    def forward(self, x):
        identity = x
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        x = self.act(x)
        if self.shortcut:
            return x + identity
        return x


class MBConvBlock(nn.Module):

    def __init__(self, in_channel, out_channel,
                 kernel_size=3, expand_ratio=6, stride=1, act_func='relu6',
                 use_se=False, act_func1='relu', act_func2='h_sigmoid',
                 divisor=8, NormLayer=nn.BatchNorm2d):
        super(MBConvBlock, self).__init__()

        self.in_channel = in_channel
        self.out_channel = out_channel

        self.kernel_size = kernel_size
        self.expand_ratio = expand_ratio

        self.stride = stride
        self.act_func = act_func
        self.use_se = use_se
        self.divisor = divisor

        # build modules
        middle_channel = make_divisible(self.in_channel * self.expand_ratio, self.divisor)
        padding = get_same_padding(self.kernel_size)
        if self.expand_ratio == 1:
            self.inverted_bottleneck = None
        else:
            self.inverted_bottleneck = nn.Sequential(OrderedDict([
                ('conv', nn.Conv2d(self.in_channel, middle_channel, kernel_size=1, groups=1, bias=False)),
                ('bn', NormLayer(middle_channel)),
                ('act', build_activation(self.act_func, inplace=True)),
            ]))

        self.depth_conv = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(middle_channel, middle_channel, self.kernel_size, stride=self.stride,
                               groups=middle_channel, padding=padding, bias=False)),
            ('bn', NormLayer(middle_channel)),
            ('act', build_activation(self.act_func, inplace=True))
        ]))
        if self.use_se:
            self.depth_conv.add_module('se', SEBlock(middle_channel, act_func1=act_func1, act_func2=act_func2))

        self.point_linear = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(middle_channel, self.out_channel, kernel_size=1, groups=1, bias=False)),
            ('bn', NormLayer(self.out_channel)),
        ]))

        self.shortcut = True if self.in_channel == self.out_channel and self.stride == 1 else False

    def forward(self, x):
        identity = x
        if self.inverted_bottleneck is not None:
            x = self.inverted_bottleneck(x)
        x = self.depth_conv(x)
        x = self.point_linear(x)

        if self.shortcut:
            return x + identity
        return x


class RegBottleneckBlock(nn.Module):

    def __init__(self, in_channel, out_channel, kernel_size=3, expand_ratio=0.25, group_width=1, stride=1,
                 act_func='relu', use_se=False, act_func1='relu', act_func2='sigmoid', divisor=8,
                 NormLayer=nn.BatchNorm2d):
        super(RegBottleneckBlock, self).__init__()

        self.stride = stride
        self.act_func = act_func
        self.divisor = divisor

        # build modules
        middle_channel = make_divisible(out_channel * expand_ratio, self.divisor)
        group_num = middle_channel // group_width
        self.point_conv1 = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(in_channel, middle_channel, kernel_size=1, groups=1, bias=False)),
            ('bn', NormLayer(middle_channel)),
            ('act', build_activation(self.act_func, inplace=True)),
        ]))

        padding = get_same_padding(kernel_size)
        self.group_conv = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(middle_channel, middle_channel, kernel_size, stride=self.stride,
                               groups=group_num, bias=False, padding=padding)),
            ('bn', NormLayer(middle_channel)),
            ('act', build_activation(self.act_func, inplace=True))
        ]))
        if use_se:
            self.group_conv.add_module('se', SEBlock(middle_channel, act_func1=act_func1, act_func2=act_func2))

        self.point_conv2 = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(middle_channel, out_channel, kernel_size=1, groups=1, bias=False)),
            ('bn', NormLayer(out_channel)),
        ]))
        self.act3 = build_activation(self.act_func, inplace=True)

        if in_channel == out_channel and self.stride == 1:
            self.shortcut = None
        else:
            self.shortcut = nn.Sequential(OrderedDict([
                ('conv', nn.Conv2d(in_channel, out_channel, kernel_size=1, groups=1,
                                   stride=stride, bias=False)),
                ('bn', NormLayer(out_channel)),
            ]))

    def forward(self, x):
        identity = x
        x = self.point_conv1(x)
        x = self.group_conv(x)
        x = self.point_conv2(x)
        if self.shortcut is not None:
            x += self.shortcut(identity)
        return self.act3(x)


class BasicBlock(nn.Module):

    def __init__(self, in_channel, out_channel,
                 kernel_size=3, expand_ratio=1, stride=1, act_func='relu', divisor=8,
                 NormLayer=nn.BatchNorm2d):
        super(BasicBlock, self).__init__()

        self.in_channel = in_channel
        self.out_channel = out_channel

        self.kernel_size = kernel_size
        self.expand_ratio = expand_ratio

        self.stride = stride
        self.act_func = act_func

        # build modules default is 1
        middle_channel = make_divisible(self.out_channel * self.expand_ratio, divisor)
        padding = get_same_padding(self.kernel_size)
        self.normal_conv1 = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(self.in_channel, middle_channel, self.kernel_size,
                               stride=self.stride, groups=1, padding=padding, bias=False)),
            ('bn', NormLayer(middle_channel)),
            ('act', build_activation(self.act_func, inplace=True))
        ]))

        self.normal_conv2 = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(middle_channel, self.out_channel, self.kernel_size, groups=1,
                               padding=padding, bias=False)),
            ('bn', NormLayer(self.out_channel)),
        ]))
        self.act2 = build_activation(self.act_func, inplace=True)

        if self.in_channel == self.out_channel and self.stride == 1:
            self.shortcut = None
        else:
            self.shortcut = nn.Sequential(OrderedDict([
                ('conv', nn.Conv2d(self.in_channel, self.out_channel, kernel_size=1, groups=1,
                                   stride=stride, bias=False)),
                ('bn', NormLayer(self.out_channel)),
            ]))

    def forward(self, x):
        identity = x

        x = self.normal_conv1(x)
        x = self.normal_conv2(x)
        if self.shortcut is None:
            x += identity
        else:
            x += self.shortcut(identity)
        return self.act2(x)


class BottleneckBlock(nn.Module):

    def __init__(self, in_channel, out_channel,
                 kernel_size=3, expand_ratio=0.25, stride=1, act_func='relu',
                 NormLayer=nn.BatchNorm2d, downsample_mode='conv'):
        super(BottleneckBlock, self).__init__()

        self.in_channel = in_channel
        self.out_channel = out_channel

        self.kernel_size = kernel_size
        self.expand_ratio = expand_ratio

        self.stride = stride
        self.act_func = act_func

        # build modules
        middle_channel = make_divisible(self.out_channel * self.expand_ratio, 8)
        padding = get_same_padding(self.kernel_size)
        self.point_conv1 = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(self.in_channel, middle_channel, kernel_size=1, groups=1, bias=False)),
            ('bn', NormLayer(middle_channel)),
            ('act', build_activation(self.act_func, inplace=True)),
        ]))

        self.normal_conv = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(middle_channel, middle_channel, self.kernel_size, stride=self.stride,
                               padding=padding, bias=False)),
            ('bn', NormLayer(middle_channel)),
            ('act', build_activation(self.act_func, inplace=True))
        ]))

        self.point_conv2 = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(middle_channel, self.out_channel, kernel_size=1, groups=1, bias=False)),
            ('bn', NormLayer(self.out_channel)),
        ]))
        self.act3 = build_activation(self.act_func, inplace=True)

        if self.in_channel == self.out_channel and self.stride == 1:
            self.shortcut = None
        else:
            if downsample_mode == 'conv':
                self.shortcut = nn.Sequential(OrderedDict([
                    ('conv', nn.Conv2d(self.in_channel, self.out_channel, kernel_size=1, groups=1,
                                       stride=stride, bias=False)),
                    ('bn', NormLayer(out_channel)),
                ]))
            elif downsample_mode == 'avgpool_conv':
                self.shortcut = nn.Sequential(OrderedDict([
                    ('avg_pool', nn.AvgPool2d(kernel_size=stride, stride=stride, padding=0, ceil_mode=True)),
                    ('conv', nn.Conv2d(in_channel, out_channel, 1, 1, 0, bias=False)),
                    ('bn', NormLayer(out_channel)),
                ]))
            else:
                raise NotImplementedError

    def forward(self, x):
        identity = x

        x = self.point_conv1(x)
        x = self.normal_conv(x)
        x = self.point_conv2(x)
        if self.shortcut is None:
            x += identity
        else:
            x += self.shortcut(identity)
        return self.act3(x)


def get_same_length(element, depth):
    if len(element) == len(depth):
        element_list = []
        for i, d in enumerate(depth):
            element_list += [element[i]] * d
    elif len(element) == sum(depth):
        element_list = element
    else:
        raise ValueError('we only need stage-wise or block wise settings')
    return element_list
