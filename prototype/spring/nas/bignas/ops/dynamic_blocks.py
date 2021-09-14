from collections import OrderedDict
import torch.nn as nn
import torch.nn.functional as F

from ..utils.dynamic_utils import int2list, build_activation, make_divisible
from .dynamic_ops import DynamicConv2d, DynamicLinear, DynamicBatchNorm2d, Identity


class DynamicMBConvBlock(nn.Module):

    def __init__(self, in_channel_list, out_channel_list, kernel_size_list=3, expand_ratio_list=6, stride=1,
                 act_func='relu6', use_se=False, act_func1='relu', act_func2='h_sigmoid',
                 KERNEL_TRANSFORM_MODE=False, divisor=8):
        super(DynamicMBConvBlock, self).__init__()

        self.in_channel_list = int2list(in_channel_list)
        self.out_channel_list = int2list(out_channel_list)

        self.kernel_size_list = int2list(kernel_size_list)
        self.expand_ratio_list = int2list(expand_ratio_list)

        self.stride = stride
        self.act_func = act_func
        self.use_se = use_se
        self.KERNEL_TRANSFORM_MODE = True if KERNEL_TRANSFORM_MODE and len(self.kernel_size_list) > 1 else False
        self.divisor = divisor

        # build modules
        max_middle_channel = make_divisible(max(self.in_channel_list) * max(self.expand_ratio_list), self.divisor)
        if max(self.expand_ratio_list) == 1:
            self.inverted_bottleneck = None
        else:
            self.inverted_bottleneck = nn.Sequential(OrderedDict([
                ('conv', DynamicConv2d(max(self.in_channel_list), max_middle_channel, kernel_size=1, groups=1,
                                       KERNEL_TRANSFORM_MODE=KERNEL_TRANSFORM_MODE)),
                ('bn', DynamicBatchNorm2d(max_middle_channel)),
                ('act', build_activation(self.act_func, inplace=True)),
            ]))

        self.depth_conv = nn.Sequential(OrderedDict([
            ('conv', DynamicConv2d(max_middle_channel, max_middle_channel, self.kernel_size_list, stride=self.stride,
                                   groups=max_middle_channel, KERNEL_TRANSFORM_MODE=KERNEL_TRANSFORM_MODE)),
            ('bn', DynamicBatchNorm2d(max_middle_channel)),
            ('act', build_activation(self.act_func, inplace=True))
        ]))
        if self.use_se:
            self.depth_conv.add_module('se',
                                       DynamicSEBlock(max_middle_channel, act_func1=act_func1, act_func2=act_func2))

        self.point_linear = nn.Sequential(OrderedDict([
            ('conv', DynamicConv2d(max_middle_channel, max(self.out_channel_list), kernel_size=1, groups=1,
                                   KERNEL_TRANSFORM_MODE=KERNEL_TRANSFORM_MODE)),
            ('bn', DynamicBatchNorm2d(max(self.out_channel_list))),
        ]))

        self.active_kernel_size = max(self.kernel_size_list)
        self.active_expand_ratio = max(self.expand_ratio_list)
        self.active_out_channel = max(self.out_channel_list)
        self.active_in_channel = max(self.in_channel_list)
        self.active_block = True
        self.shortcut = True if max(self.in_channel_list) == max(self.out_channel_list) and self.stride == 1 else False


    def forward(self, x):
        if not self.active_block:
            # 之前bignas由于都有max，所以不会出现没有grad的情况
            for param in l.parameters():
                param.requires_grad = False
            return x
        identity = x
        in_channel = x.size(1)
        self.active_in_channel = in_channel

        mid_channel = make_divisible(round(in_channel * self.active_expand_ratio), self.divisor)
        if self.inverted_bottleneck is not None:
            self.inverted_bottleneck.conv.active_out_channel = mid_channel

        self.depth_conv.conv.active_kernel_size = self.active_kernel_size
        self.depth_conv.conv.active_out_channel = mid_channel
        self.point_linear.conv.active_out_channel = self.active_out_channel

        if self.inverted_bottleneck is not None:
            x = self.inverted_bottleneck(x)
        x = self.depth_conv(x)
        x = self.point_linear(x)
        self.shortcut = True if in_channel == self.active_out_channel and self.stride == 1 else False
        if self.shortcut:
            return x + identity
        return x

    @property
    def module_str(self):
        if self.use_se:
            return 'SE(I_%d, O_%d, E_%d, K_%d, S_%d, Shortcut_%s, Transform_%s)' % (
                self.active_in_channel, self.active_out_channel, self.active_expand_ratio, self.active_kernel_size,
                self.stride, self.shortcut, self.KERNEL_TRANSFORM_MODE)
        else:
            return '(I_%d, O_%d, E_%d, K_%d, S_%d, Shortcut_%s, Transform_%s)' % (
                self.active_in_channel, self.active_out_channel, self.active_expand_ratio, self.active_kernel_size,
                self.stride, self.shortcut, self.KERNEL_TRANSFORM_MODE)


class DynamicMBConvBlock_Shortcut(nn.Module):

    def __init__(self, in_channel_list, out_channel_list, kernel_size_list=3, expand_ratio_list=6, stride=1,
                 act_func='swish', use_se=False, act_func1='swish', act_func2='sigmoid',
                 KERNEL_TRANSFORM_MODE=False, divisor=8):
        super(DynamicMBConvBlock_Shortcut, self).__init__()

        self.in_channel_list = int2list(in_channel_list)
        self.out_channel_list = int2list(out_channel_list)

        self.kernel_size_list = int2list(kernel_size_list)
        self.expand_ratio_list = int2list(expand_ratio_list)

        self.stride = stride
        self.act_func = act_func
        self.use_se = use_se
        self.KERNEL_TRANSFORM_MODE = True if KERNEL_TRANSFORM_MODE and len(self.kernel_size_list) > 1 else False
        self.divisor = divisor

        # build modules
        max_middle_channel = make_divisible(max(self.in_channel_list) * max(self.expand_ratio_list), self.divisor)
        if max(self.expand_ratio_list) == 1:
            self.inverted_bottleneck = None
        else:
            self.inverted_bottleneck = nn.Sequential(OrderedDict([
                ('conv', DynamicConv2d(max(self.in_channel_list), max_middle_channel, kernel_size=1, groups=1,
                                       KERNEL_TRANSFORM_MODE=KERNEL_TRANSFORM_MODE)),
                ('bn', DynamicBatchNorm2d(max_middle_channel)),
                ('act', build_activation(self.act_func, inplace=True)),
            ]))

        self.depth_conv = nn.Sequential(OrderedDict([
            ('conv', DynamicConv2d(max_middle_channel, max_middle_channel, self.kernel_size_list, stride=self.stride,
                                   groups=max_middle_channel, KERNEL_TRANSFORM_MODE=KERNEL_TRANSFORM_MODE)),
            ('bn', DynamicBatchNorm2d(max_middle_channel)),
            ('act', build_activation(self.act_func, inplace=True))
        ]))
        if self.use_se:
            self.depth_conv.add_module('se',
                                       DynamicSEBlock(max_middle_channel, act_func1=act_func1, act_func2=act_func2))

        self.point_linear = nn.Sequential(OrderedDict([
            ('conv', DynamicConv2d(max_middle_channel, max(self.out_channel_list), kernel_size=1, groups=1,
                                   KERNEL_TRANSFORM_MODE=KERNEL_TRANSFORM_MODE)),
            ('bn', DynamicBatchNorm2d(max(self.out_channel_list))),
        ]))

        self.active_kernel_size = max(self.kernel_size_list)
        self.active_expand_ratio = max(self.expand_ratio_list)
        self.active_out_channel = max(self.out_channel_list)
        self.active_in_channel = max(self.in_channel_list)
        self.active_block = True
        if max(self.in_channel_list) == max(self.out_channel_list) and self.stride == 1:
            self.shortcut = Identity()
        else:
            self.shortcut = DynamicConv2d(max(self.in_channel_list), max(self.out_channel_list), kernel_size=1,
                                          groups=1, stride=stride, KERNEL_TRANSFORM_MODE=KERNEL_TRANSFORM_MODE)
            self.shortcutbn = DynamicBatchNorm2d(max(self.out_channel_list))

    def forward(self, x):
        if not self.active_block:
            return x
        identity = x
        in_channel = x.size(1)
        self.active_in_channel = in_channel

        mid_channel = make_divisible(round(in_channel * self.active_expand_ratio), self.divisor)
        if self.inverted_bottleneck is not None:
            self.inverted_bottleneck.conv.active_out_channel = mid_channel

        self.depth_conv.conv.active_kernel_size = self.active_kernel_size
        self.depth_conv.conv.active_out_channel = mid_channel
        self.point_linear.conv.active_out_channel = self.active_out_channel

        if self.inverted_bottleneck is not None:
            x = self.inverted_bottleneck(x)
        x = self.depth_conv(x)
        x = self.point_linear(x)
        if in_channel == self.active_out_channel and self.stride == 1:
            x += identity
        else:
            self.shortcut.active_out_channel = self.active_out_channel
            x += self.shortcutbn(self.shortcut(identity))
        return x

    @property
    def module_str(self):
        if self.use_se:
            return 'SE(I_%d, O_%d, E_%d, K_%d, S_%d, Shortcut_%s, Transform_%s)' % (
                self.active_in_channel, self.active_out_channel, self.active_expand_ratio, self.active_kernel_size,
                self.stride, self.shortcut.module_str, self.KERNEL_TRANSFORM_MODE)
        else:
            return '(I_%d, O_%d, E_%d, K_%d, S_%d, Shortcut_%s, Transform_%s)' % (
                self.active_in_channel, self.active_out_channel, self.active_expand_ratio, self.active_kernel_size,
                self.stride, self.shortcut.module_str, self.KERNEL_TRANSFORM_MODE)


class DynamicBottleneckBlock(nn.Module):

    def __init__(self, in_channel_list, out_channel_list,
                 kernel_size_list=3, expand_ratio_list=0.25, stride=1, act_func='relu',
                 KERNEL_TRANSFORM_MODE=False, divisor=8, downsample_mode='conv'):
        super(DynamicBottleneckBlock, self).__init__()

        self.in_channel_list = int2list(in_channel_list)
        self.out_channel_list = int2list(out_channel_list)

        self.kernel_size_list = int2list(kernel_size_list)
        self.expand_ratio_list = int2list(expand_ratio_list)

        self.stride = stride
        self.act_func = act_func
        self.downsample_mode  = downsample_mode

        self.KERNEL_TRANSFORM_MODE = True if KERNEL_TRANSFORM_MODE and len(self.kernel_size_list) > 1 else False
        self.divisor = divisor

        # build modules
        max_middle_channel = make_divisible(max(self.out_channel_list) * max(self.expand_ratio_list), self.divisor)
        self.point_conv1 = nn.Sequential(OrderedDict([
            ('conv', DynamicConv2d(max(self.in_channel_list), max_middle_channel, kernel_size=1, groups=1,
                                   KERNEL_TRANSFORM_MODE=KERNEL_TRANSFORM_MODE)),
            ('bn', DynamicBatchNorm2d(max_middle_channel)),
            ('act', build_activation(self.act_func, inplace=True)),
        ]))

        self.normal_conv = nn.Sequential(OrderedDict([
            ('conv',
             DynamicConv2d(max_middle_channel, max_middle_channel, self.kernel_size_list, stride=self.stride, groups=1,
                           KERNEL_TRANSFORM_MODE=KERNEL_TRANSFORM_MODE)),
            ('bn', DynamicBatchNorm2d(max_middle_channel)),
            ('act', build_activation(self.act_func, inplace=True))
        ]))

        self.point_conv2 = nn.Sequential(OrderedDict([
            ('conv', DynamicConv2d(max_middle_channel, max(self.out_channel_list), kernel_size=1, groups=1,
                                   KERNEL_TRANSFORM_MODE=KERNEL_TRANSFORM_MODE)),
            ('bn', DynamicBatchNorm2d(max(self.out_channel_list))),
        ]))
        if max(self.in_channel_list) == max(self.out_channel_list) and self.stride == 1:
            self.shortcut = Identity()
        else:
            if downsample_mode == 'conv':
                self.shortcut = DynamicConv2d(max(self.in_channel_list), max(self.out_channel_list), kernel_size=1,
                                              groups=1, stride=stride, KERNEL_TRANSFORM_MODE=KERNEL_TRANSFORM_MODE)
                self.shortcutbn = DynamicBatchNorm2d(max(self.out_channel_list))
            elif downsample_mode == 'avgpool_conv':
                self.shortcut_avg = nn.AvgPool2d(kernel_size=stride, stride=stride, padding=0, ceil_mode=True)
                self.shortcut = DynamicConv2d(max(self.in_channel_list), max(self.out_channel_list), kernel_size=1,
                                              groups=1, stride=1, KERNEL_TRANSFORM_MODE=KERNEL_TRANSFORM_MODE)
                self.shortcutbn = DynamicBatchNorm2d(max(self.out_channel_list))
            else:
                raise NotImplementedError

        self.act3 = build_activation(self.act_func, inplace=True)

        self.active_kernel_size = max(self.kernel_size_list)
        self.active_expand_ratio = max(self.expand_ratio_list)
        self.active_out_channel = max(self.out_channel_list)
        self.active_in_channel = max(self.in_channel_list)
        self.active_block = True

    def forward(self, x):
        if not self.active_block:
            return x
        identity = x
        in_channel = x.size(1)
        self.active_in_channel = in_channel

        mid_channel = make_divisible(round(self.active_out_channel * self.active_expand_ratio), self.divisor)

        self.point_conv1.conv.active_out_channel = mid_channel

        self.normal_conv.conv.active_kernel_size = self.active_kernel_size
        self.normal_conv.conv.active_out_channel = mid_channel

        self.point_conv2.conv.active_out_channel = self.active_out_channel

        x = self.point_conv1(x)
        x = self.normal_conv(x)
        x = self.point_conv2(x)
        if in_channel == self.active_out_channel and self.stride == 1:
            x += identity
        else:
            self.shortcut.active_out_channel = self.active_out_channel
            if self.downsample_mode == 'conv':
                x += self.shortcutbn(self.shortcut(identity))
            elif self.downsample_mode == 'avgpool_conv':
                x += self.shortcutbn(self.shortcut(self.shortcut_avg(identity)))
        return self.act3(x)

    @property
    def module_str(self):
        if self.active_in_channel == self.active_out_channel and self.stride == 1:
            return 'Bottleneck(I_%d, O_%d, E_%.2f, K_%d, S_%d, Shortcut_%s, Transform_%s)' % (
                self.active_in_channel, self.active_out_channel, self.active_expand_ratio, self.active_kernel_size,
                self.stride, 'identity', self.KERNEL_TRANSFORM_MODE)
        else:
            return 'Bottleneck(I_%d, O_%d, E_%.2f, K_%d, S_%d, Shortcut_%s, Transform_%s)' % (
                self.active_in_channel, self.active_out_channel, self.active_expand_ratio, self.active_kernel_size,
                self.stride, self.shortcut.module_str, self.KERNEL_TRANSFORM_MODE)


class DynamicBasicBlock(nn.Module):

    def __init__(self, in_channel_list, out_channel_list,
                 kernel_size_list=3, expand_ratio_list=1, stride=1, act_func='relu',
                 KERNEL_TRANSFORM_MODE=False, divisor=8, IS_STAGE_BLOCK=False):
        super(DynamicBasicBlock, self).__init__()

        self.in_channel_list = int2list(in_channel_list)
        self.out_channel_list = int2list(out_channel_list)

        self.kernel_size_list = int2list(kernel_size_list)
        self.expand_ratio_list = int2list(expand_ratio_list)

        self.stride = stride
        self.act_func = act_func

        self.KERNEL_TRANSFORM_MODE = True if KERNEL_TRANSFORM_MODE and len(self.kernel_size_list) > 1 else False
        self.divisor = divisor

        # build modules default is 1
        max_middle_channel = make_divisible(max(self.out_channel_list) * max(self.expand_ratio_list), self.divisor)
        self.normal_conv1 = nn.Sequential(OrderedDict([
            ('conv', DynamicConv2d(max(self.in_channel_list), max_middle_channel, self.kernel_size_list,
                                   stride=self.stride, groups=1, KERNEL_TRANSFORM_MODE=KERNEL_TRANSFORM_MODE)),
            ('bn', DynamicBatchNorm2d(max_middle_channel)),
            ('act', build_activation(self.act_func, inplace=True))
        ]))

        self.normal_conv2 = nn.Sequential(OrderedDict([
            ('conv',
             DynamicConv2d(max_middle_channel, max(self.out_channel_list), self.kernel_size_list, groups=1,
                           KERNEL_TRANSFORM_MODE=KERNEL_TRANSFORM_MODE)),
            ('bn', DynamicBatchNorm2d(max(self.out_channel_list))),
        ]))

        self.active_kernel_size = max(self.kernel_size_list)
        self.active_expand_ratio = max(self.expand_ratio_list)
        self.active_out_channel = max(self.out_channel_list)
        self.active_in_channel = max(self.in_channel_list)
        self.active_block = True

        if max(self.in_channel_list) == max(self.out_channel_list) and self.stride == 1 and not IS_STAGE_BLOCK:
            self.shortcut = Identity()
        else:
            self.shortcut = DynamicConv2d(max(self.in_channel_list), max(self.out_channel_list), kernel_size=1,
                                          groups=1, stride=stride, KERNEL_TRANSFORM_MODE=KERNEL_TRANSFORM_MODE)
            self.shortcutbn = DynamicBatchNorm2d(max(self.out_channel_list))
        self.act2 = build_activation(self.act_func, inplace=True)

    def forward(self, x):
        if not self.active_block:
            return x
        identity = x
        in_channel = x.size(1)
        self.active_in_channel = in_channel

        mid_channel = make_divisible(round(self.active_out_channel * self.active_expand_ratio), self.divisor)
        self.normal_conv1.conv.active_kernel_size = self.active_kernel_size
        self.normal_conv1.conv.active_out_channel = mid_channel

        self.normal_conv2.conv.active_kernel_size = self.active_kernel_size
        self.normal_conv2.conv.active_out_channel = self.active_out_channel

        x = self.normal_conv1(x)
        x = self.normal_conv2(x)
        if in_channel == self.active_out_channel and self.stride == 1:
            x += identity
        else:
            self.shortcut.active_out_channel = self.active_out_channel
            x += self.shortcutbn(self.shortcut(identity))
        return self.act2(x)

    @property
    def module_str(self):
        if self.active_in_channel == self.active_out_channel and self.stride == 1:
            return 'Basic(I_%d, O_%d, E_%.2f, K_%d, S_%d, Shortcut_%s, Transform_%s)' % (
                self.active_in_channel, self.active_out_channel, self.active_expand_ratio, self.active_kernel_size,
                self.stride, 'identity', self.KERNEL_TRANSFORM_MODE)
        else:
            return 'Basic(I_%d, O_%d, E_%.2f, K_%d, S_%d, Shortcut_%s, Transform_%s)' % (
                self.active_in_channel, self.active_out_channel, self.active_expand_ratio, self.active_kernel_size,
                self.stride, 'DyConv', self.KERNEL_TRANSFORM_MODE)


class DynamicRegBottleneckBlock(nn.Module):

    def __init__(self, in_channel_list, out_channel_list, kernel_size_list=3, expand_ratio_list=0.25,
                 group_width_list=1, stride=1,
                 act_func='relu', use_se=False, act_func1='relu', act_func2='sigmoid',
                 KERNEL_TRANSFORM_MODE=False, divisor=8):
        super(DynamicRegBottleneckBlock, self).__init__()

        self.in_channel_list = int2list(in_channel_list)
        self.out_channel_list = int2list(out_channel_list)

        self.kernel_size_list = int2list(kernel_size_list)
        self.expand_ratio_list = int2list(expand_ratio_list)
        self.group_width_list = int2list(group_width_list)

        self.stride = stride
        self.act_func = act_func

        self.KERNEL_TRANSFORM_MODE = True if KERNEL_TRANSFORM_MODE and len(self.kernel_size_list) > 1 else False
        self.divisor = divisor

        # build modules
        max_middle_channel = make_divisible(max(self.out_channel_list) * max(self.expand_ratio_list), self.divisor)
        group_num = max_middle_channel // max(self.group_width_list)
        self.point_conv1 = nn.Sequential(OrderedDict([
            ('conv', DynamicConv2d(max(self.in_channel_list), max_middle_channel, kernel_size=1, groups=1,
                                   KERNEL_TRANSFORM_MODE=KERNEL_TRANSFORM_MODE)),
            ('bn', DynamicBatchNorm2d(max_middle_channel)),
            ('act', build_activation(self.act_func, inplace=True)),
        ]))

        self.group_conv = nn.Sequential(OrderedDict([
            ('conv',
             DynamicConv2d(max_middle_channel, max_middle_channel, self.kernel_size_list, stride=self.stride,
                           groups=group_num, KERNEL_TRANSFORM_MODE=KERNEL_TRANSFORM_MODE)),
            ('bn', DynamicBatchNorm2d(max_middle_channel)),
            ('act', build_activation(self.act_func, inplace=True))
        ]))
        if use_se:
            self.group_conv.add_module('se',
                                       DynamicSEBlock(max_middle_channel, act_func1=act_func1, act_func2=act_func2))

        self.point_conv2 = nn.Sequential(OrderedDict([
            ('conv', DynamicConv2d(max_middle_channel, max(self.out_channel_list), kernel_size=1, groups=1,
                                   KERNEL_TRANSFORM_MODE=KERNEL_TRANSFORM_MODE)),
            ('bn', DynamicBatchNorm2d(max(self.out_channel_list))),
        ]))
        self.act3 = build_activation(self.act_func, inplace=True)

        self.active_kernel_size = max(self.kernel_size_list)
        self.active_expand_ratio = max(self.expand_ratio_list)
        self.active_out_channel = max(self.out_channel_list)
        self.active_in_channel = max(self.in_channel_list)
        self.active_group_width = max(self.group_width_list)
        self.active_block = True

        if max(self.in_channel_list) == max(self.out_channel_list) and self.stride == 1:
            self.shortcut = Identity()
        else:
            self.shortcut = DynamicConv2d(max(self.in_channel_list), max(self.out_channel_list), kernel_size=1,
                                          groups=1, stride=stride, KERNEL_TRANSFORM_MODE=KERNEL_TRANSFORM_MODE)
            self.shortcutbn = DynamicBatchNorm2d(max(self.out_channel_list))

    def forward(self, x):
        if not self.active_block:
            return x
        identity = x
        in_channel = x.size(1)
        self.active_in_channel = in_channel

        mid_channel = make_divisible(round(self.active_out_channel * self.active_expand_ratio), self.divisor)

        self.point_conv1.conv.active_out_channel = mid_channel

        self.group_conv.conv.active_kernel_size = self.active_kernel_size
        self.group_conv.conv.active_out_channel = mid_channel
        self.group_conv.conv.active_groups = mid_channel // self.active_group_width

        self.point_conv2.conv.active_out_channel = self.active_out_channel

        x = self.point_conv1(x)
        x = self.group_conv(x)
        x = self.point_conv2(x)
        if in_channel == self.active_out_channel and self.stride == 1:
            x += identity
        else:
            self.shortcut.active_out_channel = self.active_out_channel
            x += self.shortcutbn(self.shortcut(identity))
        return self.act3(x)

    @property
    def module_str(self):
        if self.active_in_channel == self.active_out_channel and self.stride == 1:
            return 'RegBottleneck(I_%d, O_%d, E_%d, K_%d, G_%d, S_%d, Shortcut_%s, Transform_%s)' % (
                self.active_in_channel, self.active_out_channel, self.active_expand_ratio, self.active_kernel_size,
                self.active_group_width,
                self.stride, 'identity', self.KERNEL_TRANSFORM_MODE)
        else:
            return 'RegBottleneck(I_%d, O_%d, E_%d, K_%d, G_%d, S_%d, Shortcut_%s, Transform_%s)' % (
                self.active_in_channel, self.active_out_channel, self.active_expand_ratio, self.active_kernel_size,
                self.active_group_width,
                self.stride, 'DyConv', self.KERNEL_TRANSFORM_MODE)


class DynamicConvBlock(nn.Module):

    def __init__(self, in_channel_list, out_channel_list, kernel_size_list=3, stride=1, dilation=1,
                 use_bn=True, act_func='relu', KERNEL_TRANSFORM_MODE=False, bias=False):
        super(DynamicConvBlock, self).__init__()

        self.in_channel_list = int2list(in_channel_list)
        self.out_channel_list = int2list(out_channel_list)
        self.kernel_size_list = int2list(kernel_size_list)
        self.stride = stride
        self.dilation = dilation
        self.use_bn = use_bn
        self.act_func = act_func
        self.KERNEL_TRANSFORM_MODE = True if KERNEL_TRANSFORM_MODE and len(self.kernel_size_list) > 1 else False

        self.conv = DynamicConv2d(
            in_channels=max(self.in_channel_list), out_channels=max(self.out_channel_list),
            kernel_size=self.kernel_size_list, groups=1, stride=self.stride, dilation=self.dilation,
            KERNEL_TRANSFORM_MODE=self.KERNEL_TRANSFORM_MODE, bias=bias
        )
        if self.use_bn:
            self.bn = DynamicBatchNorm2d(max(self.out_channel_list))
        if self.act_func == '':
            self.act = Identity()
        else:
            self.act = build_activation(self.act_func, inplace=True)

        self.active_out_channel = max(self.out_channel_list)
        self.active_in_channel = max(self.in_channel_list)
        self.active_kernel_size = max(self.kernel_size_list)
        self.active_block = True

    def forward(self, x):
        if not self.active_block:
            return x
        self.conv.active_out_channel = self.active_out_channel
        self.conv.active_kernel_size = self.active_kernel_size
        self.active_in_channel = x.size(1)
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        x = self.act(x)
        return x

    @property
    def module_str(self):
        return 'DyConv(I_%d, O_%d, K_%d, S_%d, Transform_%s)' % (
            self.active_in_channel, self.active_out_channel, self.active_kernel_size, self.stride,
            self.KERNEL_TRANSFORM_MODE)


class DynamicLinearBlock(nn.Module):

    def __init__(self, in_features_list, out_features_list, bias=True, dropout_rate=0):
        super(DynamicLinearBlock, self).__init__()

        self.in_features_list = int2list(in_features_list)
        self.out_features_list = int2list(out_features_list)
        self.bias = bias
        self.dropout_rate = dropout_rate

        if self.dropout_rate > 0:
            self.dropout = nn.Dropout(self.dropout_rate, inplace=True)
        else:
            self.dropout = None
        self.linear = DynamicLinear(
            in_features=max(self.in_features_list), out_features=max(self.out_features_list), bias=self.bias
        )
        self.active_out_features = max(self.out_features_list)
        self.active_in_features = max(self.in_features_list)
        self.active_block = True

    def forward(self, x):
        self.linear.active_out_features = self.active_out_features
        self.active_in_features = x.size(1)

        if self.dropout is not None:
            x = self.dropout(x)
        x = self.linear(x)
        return x

    @property
    def module_str(self):
        return 'DyLinear(I_%d, O_%d)' % (self.active_in_features, self.active_out_features)


class DynamicSEBlock(nn.Module):

    def __init__(self, channel, reduction=4, bias=True, act_func1='relu', act_func2='h_sigmoid',
                 inplace=True, divisor=8):
        super(DynamicSEBlock, self).__init__()
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
        in_channel = x.size(1)
        num_mid = make_divisible(in_channel // self.reduction, divisor=self.divisor)

        y = F.adaptive_avg_pool2d(x, output_size=1)
        # reduce
        reduce_conv = self.fc.reduce
        reduce_filter = reduce_conv.weight[:num_mid, :in_channel, :, :].contiguous()
        reduce_bias = reduce_conv.bias[:num_mid] if reduce_conv.bias is not None else None
        y = F.conv2d(y, reduce_filter, reduce_bias, 1, 0, 1, 1)
        # relu
        y = self.fc.act1(y)
        # expand
        expand_conv = self.fc.expand
        expand_filter = expand_conv.weight[:in_channel, :num_mid, :, :].contiguous()
        expand_bias = expand_conv.bias[:in_channel] if expand_conv.bias is not None else None
        y = F.conv2d(y, expand_filter, expand_bias, 1, 0, 1, 1)
        # hard sigmoid
        y = self.fc.act2(y)

        return x * y
