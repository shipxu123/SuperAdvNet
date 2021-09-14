import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch
from ..utils.dynamic_utils import sub_filter_start_end, get_same_padding, int2list


class Identity(nn.Module):

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

    @property
    def module_str(self):
        return 'Identity'


class DynamicConv2d(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=False,
                 KERNEL_TRANSFORM_MODE=False):

        self.in_channels = in_channels
        self.out_channels = out_channels

        # if len(self.kernel_size_list) == 1 normal conv without changing kernel
        self.kernel_size_list = int2list(kernel_size)
        kernel_size = max(self.kernel_size_list)
        padding = get_same_padding(kernel_size)
        # if len(self.groups_list) == 1 normal conv without changing groups
        self.groups_list = int2list(groups)
        groups = min(self.groups_list)
        super(DynamicConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups,
                                            bias)

        # if False, the weights are center croped; if True, the transformation matrix is used to transform kernels
        self.KERNEL_TRANSFORM_MODE = KERNEL_TRANSFORM_MODE
        self.depthwise = True if groups == in_channels else False

        if len(self.kernel_size_list) > 1:
            self._ks_set = list(set(self.kernel_size_list))
            self._ks_set.sort()  # e.g., [3, 5, 7]
            if self.KERNEL_TRANSFORM_MODE:
                # register scaling parameters
                # 7to5_matrix, 5to3_matrix
                scale_params = {}
                for i in range(len(self._ks_set) - 1):
                    ks_small = self._ks_set[i]
                    ks_larger = self._ks_set[i + 1]
                    param_name = '%dto%d' % (ks_larger, ks_small)
                    scale_params['%s_matrix' % param_name] = Parameter(torch.eye(ks_small ** 2))
                for name, param in scale_params.items():
                    self.register_parameter(name, param)

        self.active_kernel_size = max(self.kernel_size_list)
        self.active_out_channel = self.out_channels
        self.active_in_channel = self.in_channels
        self.active_groups = min(self.groups_list)

    def get_active_filter(self, out_channel, in_channel, kernel_size):
        max_kernel_size = max(self.kernel_size_list)

        start, end = sub_filter_start_end(max_kernel_size, kernel_size)
        filters = self.weight[:out_channel, :in_channel, start:end, start:end]
        if self.KERNEL_TRANSFORM_MODE and kernel_size < max_kernel_size:
            start_filter = self.weight[:out_channel, :in_channel, :, :]  # start with max kernel
            for i in range(len(self._ks_set) - 1, 0, -1):
                src_ks = self._ks_set[i]
                if src_ks <= kernel_size:
                    break
                target_ks = self._ks_set[i - 1]
                start, end = sub_filter_start_end(src_ks, target_ks)
                _input_filter = start_filter[:, :, start:end, start:end]
                _input_filter = _input_filter.contiguous()
                _input_filter = _input_filter.view(_input_filter.size(0), _input_filter.size(1), -1)
                _input_filter = _input_filter.view(-1, _input_filter.size(2))
                _input_filter = F.linear(
                    _input_filter, self.__getattr__('%dto%d_matrix' % (src_ks, target_ks)),
                )
                _input_filter = _input_filter.view(filters.size(0), filters.size(1), target_ks ** 2)
                _input_filter = _input_filter.view(filters.size(0), filters.size(1), target_ks, target_ks)
                start_filter = _input_filter
            filters = start_filter
        return filters

    def forward(self, x, kernel_size=None, out_channel=None, groups=None):
        # Used in Depth Conv, Group Conv and Normal Conv
        if kernel_size is None:
            kernel_size = self.active_kernel_size
        if out_channel is None:
            out_channel = self.active_out_channel
        if groups is None:
            groups = self.active_groups

        in_channel = x.size(1)
        self.active_in_channel = in_channel
        if self.depthwise or groups > in_channel:
            groups = in_channel

        filters = self.get_active_filter(out_channel, in_channel // groups, kernel_size).contiguous()
        padding = get_same_padding(kernel_size)

        bias = self.bias[:out_channel].contiguous() if self.bias is not None else None

        y = F.conv2d(
            x, filters, bias, self.stride, padding, self.dilation, groups,
        )
        return y

    @property
    def module_str(self):
        return 'DyConv'


class DynamicLinear(nn.Linear):

    def __init__(self, in_features, out_features, bias=True):
        super(DynamicLinear, self).__init__(in_features, out_features, bias)
        self.out_features_list = int2list(out_features)
        self.active_out_features = max(self.out_features_list)

    def forward(self, x, out_features=None):
        if out_features is None:
            out_features = self.active_out_features

        in_features = x.size(1)
        weight = self.weight[:out_features, :in_features].contiguous()
        bias = self.bias[:out_features].contiguous() if self.bias is not None else None
        y = F.linear(x, weight, bias)
        return y

    @property
    def module_str(self):
        return 'DyLinear'


class DynamicBatchNorm2d(nn.BatchNorm2d):

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        super(DynamicBatchNorm2d, self).__init__(num_features, eps, momentum, affine, track_running_stats)

    def forward(self, x):
        self._check_input_dim(x)
        feature_dim = x.size(1)
        # Normal BN
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            # TODO: if statement only here to tell the jit to skip emitting this when it is None
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum
        return F.batch_norm(
            x, self.running_mean[:feature_dim], self.running_var[:feature_dim], self.weight[:feature_dim],
            self.bias[:feature_dim], self.training or not self.track_running_stats,
            exponential_average_factor, self.eps, )

    @property
    def module_str(self):
        return 'DyBN'
