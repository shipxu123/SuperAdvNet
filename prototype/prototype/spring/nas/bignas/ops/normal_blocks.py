from collections import OrderedDict
import torch.nn as nn
import torch.nn.functional as F

from ..utils.dynamic_utils import build_activation, make_divisible, get_same_padding


class MBConvBlock(nn.Module):

    def __init__(self, in_channel, out_channel,
                 kernel_size=3, expand_ratio=6, stride=1, act_func='relu6',
                 use_se=False, act_func1='relu', act_func2='h_sigmoid',
                 divisor=8):
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
                ('conv', nn.Conv2d(self.in_channel, middle_channel, kernel_size=1, groups=1)),
                ('bn', nn.BatchNorm2d(middle_channel)),
                ('act', build_activation(self.act_func, inplace=True)),
            ]))

        self.depth_conv = nn.Sequential(OrderedDict([
            ('conv',
             nn.Conv2d(middle_channel, middle_channel, self.kernel_size, stride=self.stride,
                       groups=middle_channel, padding=padding)),
            ('bn', nn.BatchNorm2d(middle_channel)),
            ('act', build_activation(self.act_func, inplace=True))
        ]))
        if self.use_se:
            self.depth_conv.add_module('se', SEBlock(middle_channel, act_func1=act_func1, act_func2=act_func2))

        self.point_linear = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(middle_channel, self.out_channel, kernel_size=1, groups=1)),
            ('bn', nn.BatchNorm2d(self.out_channel)),
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


class MBConvBlock_Shortcut(nn.Module):

    def __init__(self, in_channel, out_channel,
                 kernel_size=3, expand_ratio=6, stride=1, act_func='swish',
                 use_se=False, act_func1='swish', act_func2='sigmoid', divisor=8):
        super(MBConvBlock_Shortcut, self).__init__()

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
                ('conv', nn.Conv2d(self.in_channel, middle_channel, kernel_size=1, groups=1)),
                ('bn', nn.BatchNorm2d(middle_channel)),
                ('act', build_activation(self.act_func, inplace=True)),
            ]))

        self.depth_conv = nn.Sequential(OrderedDict([
            ('conv',
             nn.Conv2d(middle_channel, middle_channel, self.kernel_size, stride=self.stride,
                       groups=middle_channel, padding=padding)),
            ('bn', nn.BatchNorm2d(middle_channel)),
            ('act', build_activation(self.act_func, inplace=True))
        ]))
        if self.use_se:
            self.depth_conv.add_module('se', SEBlock(middle_channel, act_func1=act_func1, act_func2=act_func2))

        self.point_linear = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(middle_channel, self.out_channel, kernel_size=1, groups=1)),
            ('bn', nn.BatchNorm2d(self.out_channel)),
        ]))

        if self.in_channel == self.out_channel and self.stride == 1:
            self.shortcut = None
        else:
            self.shortcut = nn.Conv2d(self.in_channel, self.out_channel, kernel_size=1, groups=1, stride=stride)
            self.shortcutbn = nn.BatchNorm2d(self.out_channel)

    def forward(self, x):
        identity = x

        if self.inverted_bottleneck is not None:
            x = self.inverted_bottleneck(x)
        x = self.depth_conv(x)
        x = self.point_linear(x)
        if self.shortcut is None:
            residual = identity
        else:
            residual = self.shortcutbn(self.shortcut(identity))
        return x + residual


class BottleneckBlock(nn.Module):

    def __init__(self, in_channel, out_channel,
                 kernel_size=3, expand_ratio=0.25, stride=1, act_func='relu', divisor=8):
        super(BottleneckBlock, self).__init__()

        self.in_channel = in_channel
        self.out_channel = out_channel

        self.kernel_size = kernel_size
        self.expand_ratio = expand_ratio

        self.stride = stride
        self.act_func = act_func
        self.divisor = divisor

        # build modules
        middle_channel = make_divisible(self.out_channel * self.expand_ratio, self.divisor)
        padding = get_same_padding(self.kernel_size)
        self.point_conv1 = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(self.in_channel, middle_channel, kernel_size=1, groups=1)),
            ('bn', nn.BatchNorm2d(middle_channel)),
            ('act', build_activation(self.act_func, inplace=True)),
        ]))

        self.normal_conv = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(middle_channel, middle_channel, self.kernel_size, stride=self.stride,
                               groups=1, padding=padding)),
            ('bn', nn.BatchNorm2d(middle_channel)),
            ('act', build_activation(self.act_func, inplace=True))
        ]))

        self.point_conv2 = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(middle_channel, self.out_channel, kernel_size=1, groups=1)),
            ('bn', nn.BatchNorm2d(self.out_channel)),
        ]))
        self.act3 = build_activation(self.act_func, inplace=True)

        if self.in_channel == self.out_channel and self.stride == 1:
            self.shortcut = None
        else:
            self.shortcut = nn.Conv2d(self.in_channel, self.out_channel, kernel_size=1, groups=1, stride=stride)
            self.shortcutbn = nn.BatchNorm2d(self.out_channel)

    def forward(self, x):
        identity = x

        x = self.point_conv1(x)
        x = self.normal_conv(x)
        x = self.point_conv2(x)
        if self.shortcut is None:
            x += identity
        else:
            x += self.shortcutbn(self.shortcut(identity))
        return self.act3(x)


class BasicBlock(nn.Module):

    def __init__(self, in_channel, out_channel,
                 kernel_size=3, expand_ratio=1, stride=1, act_func='relu', divisor=8):
        super(BasicBlock, self).__init__()

        self.in_channel = in_channel
        self.out_channel = out_channel

        self.kernel_size = kernel_size
        self.expand_ratio = expand_ratio

        self.stride = stride
        self.act_func = act_func
        self.divisor = divisor

        # build modules default is 1
        middle_channel = make_divisible(self.out_channel * self.expand_ratio, self.divisor)
        padding = get_same_padding(self.kernel_size)
        self.normal_conv1 = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(self.in_channel, middle_channel, self.kernel_size, stride=self.stride,
                               groups=1, padding=padding)),
            ('bn', nn.BatchNorm2d(middle_channel)),
            ('act', build_activation(self.act_func, inplace=True))
        ]))

        self.normal_conv2 = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(middle_channel, self.out_channel, self.kernel_size, groups=1, padding=padding)),
            ('bn', nn.BatchNorm2d(self.out_channel)),
        ]))
        self.act2 = build_activation(self.act_func, inplace=True)

        if self.in_channel == self.out_channel and self.stride == 1:
            self.shortcut = None
        else:
            self.shortcut = nn.Sequential(OrderedDict([
                ('conv', nn.Conv2d(self.in_channel, self.out_channel, kernel_size=1, groups=1, stride=stride)),
                ('bn', nn.BatchNorm2d(self.out_channel)),
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


class ConvBlock(nn.Module):

    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, dilation=1,
                 use_bn=True, act_func='relu'):
        super(ConvBlock, self).__init__()

        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.use_bn = use_bn
        self.act_func = act_func

        padding = get_same_padding(self.kernel_size)
        self.conv = nn.Conv2d(in_channels=self.in_channel, out_channels=self.out_channel, kernel_size=self.kernel_size,
                              padding=padding, groups=1, stride=self.stride, dilation=self.dilation)
        if self.use_bn:
            self.bn = nn.BatchNorm2d(self.out_channel)
        self.act = build_activation(self.act_func, inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        x = self.act(x)
        return x


class LinearBlock(nn.Module):

    def __init__(self, in_features, out_features, bias=True, dropout_rate=0):
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
