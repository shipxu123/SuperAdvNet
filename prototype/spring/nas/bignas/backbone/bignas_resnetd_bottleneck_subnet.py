import torch.nn as nn

from prototype.spring.models.utils.normalize import build_norm_layer
from prototype.spring.models.utils.initializer import initialize_from_cfg

from .normal_blocks import ConvBlock, LinearBlock, BottleneckBlock, get_same_length


__all__ = ['bignas_resnetd_bottleneck']


class BigNAS_ResNetD_Bottleneck(nn.Module):

    def __init__(self,
                 out_channel=[64, 256, 512, 1024, 2048],
                 depth=[1, 3, 4, 6, 3],
                 kernel_size=[7, 3, 3, 3, 3],
                 expand_ratio=[0, 0.25, 0.25, 0.25, 0.25],
                 act_stages=['relu', 'relu', 'relu', 'relu', 'relu'],
                 stride_stages=[2, 1, 2, 2, 2],
                 dropout_rate=0.,
                 divisor=8,
                 use_maxpool=True,
                 use_residual_block=False,
                 downsample_mode='avgpool_conv',
                 # bn and initializer
                 normalize={'type': 'solo_bn'},
                 initializer={'method': 'msra'},
                 # configuration for task
                 frozen_layers=[],
                 out_layers=[],
                 out_strides=[],
                 task='classification',
                 num_classes=1000):
        r"""
        Arguments:

        - out_channel (:obj:`list` of 5 (stages+1) ints): channel list
        - depth (:obj:`list` of 5 (stages+1) ints): depth list for stages
        - kernel_size (:obj:`list` of 8 (blocks+1) ints): kernel size list for blocks
        - expand_ratio (:obj:`list` of 8 (blocks+1) ints): expand ratio list for blocks
        - act_stages(:obj:`list` of 8 (blocks+1) ints): activation list for blocks
        - stride_stages (:obj:`list` of 5 (stages+1) ints): stride list for stages
        - dropout_rate (:obj:`float`): dropout rate
        - divisor(:obj:`int`): divisor for channels
        - use_maxpool(: obj:`bool`): use max_pooling to downsample or not
        - use_residual(: obj:`bool`): use residual conv block or not
        - downsample_mode(: obj:`str`): downsample connection in shortcut
        - normalize (:obj:`dict`): configurations for normalize
        - initializer (:obj:`dict`): initializer method
        - frozen_layers (:obj:`list` of :obj:`int`): Index of frozen layers, [0,4]
        - out_layers (:obj:`list` of :obj:`int`): Index of output layers, [0,4]
        - out_strides (:obj:`list` of :obj:`int`): Stride of outputs
        - task (:obj:`str`): Type of task, 'classification' or object 'detection'
        - num_classes (:obj:`int`): number of classification classes
        """

        super(BigNAS_ResNetD_Bottleneck, self).__init__()

        global NormLayer

        NormLayer = build_norm_layer(normalize)

        self.depth = depth
        self.out_channel = out_channel
        self.kernel_size = get_same_length(kernel_size, self.depth)
        self.expand_ratio = get_same_length(expand_ratio, self.depth)

        self.act_stages = act_stages
        self.stride_stages = stride_stages
        self.dropout_rate = dropout_rate
        self.divisor = divisor
        self.use_maxpool = use_maxpool
        self.use_residual_block = use_residual_block

        self.frozen_layers = frozen_layers
        self.out_layers = out_layers
        self.out_strides = out_strides
        self.task = task
        self.num_classes = num_classes
        self.out_planes = [int(self.out_channel[i]) for i in self.out_layers]

        # first conv layer
        self.first_conv = ConvBlock(
            in_channel=3, out_channel=self.out_channel[0], kernel_size=self.kernel_size[0],
            stride=stride_stages[0], act_func=act_stages[0], NormLayer=NormLayer)
        if self.use_residual_block:
            self.residual_block = ConvBlock(
                in_channel=self.out_channel[0], out_channel=self.out_channel[1], kernel_size=self.kernel_size[1],
                stride=stride_stages[1], act_func=act_stages[1], NormLayer=NormLayer)
            _conv_index = 2
        else:
            _conv_index = 1

        self.second_conv = ConvBlock(
            in_channel=self.out_channel[_conv_index - 1], out_channel=self.out_channel[_conv_index],
            kernel_size=self.kernel_size[_conv_index], stride=stride_stages[_conv_index],
            act_func=act_stages[_conv_index], NormLayer=NormLayer)
        input_channel = self.out_channel[_conv_index]
        _conv_index += 1

        if self.use_maxpool:
            self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=True)
            self.stride_stages[_conv_index] = 1
        else:
            self.stride_stages[_conv_index] = 2

        blocks = []

        _block_index = _conv_index
        self.stage_out_idx = []
        for s, act_func, n_block, output_channel in zip(self.stride_stages[_conv_index:], self.act_stages[_conv_index:],
                                                        self.depth[_conv_index:], self.out_channel[_conv_index:]):
            for i in range(n_block):
                kernel_size = self.kernel_size[_block_index]
                expand_ratio = self.expand_ratio[_block_index]
                _block_index += 1
                if i == 0:
                    stride = s
                else:
                    stride = 1
                basic_block = BottleneckBlock(
                    in_channel=input_channel, out_channel=output_channel, kernel_size=kernel_size,
                    expand_ratio=expand_ratio, stride=stride, act_func=act_func, NormLayer=NormLayer,
                    downsample_mode=downsample_mode)
                blocks.append(basic_block)
                input_channel = output_channel
            self.stage_out_idx.append(_block_index - _conv_index - 1)

        self.blocks = nn.ModuleList(blocks)

        if self.task == 'classification':
            self.avg_pool = nn.AdaptiveAvgPool2d(output_size=1)
            self.classifier = LinearBlock(
                in_features=self.out_channel[-1], out_features=num_classes, bias=True, dropout_rate=dropout_rate)

        if initializer is not None:
            initialize_from_cfg(self, initializer)

        # It's IMPORTANT when you want to freeze part of your backbone.
        # ALWAYS remember freeze layers in __init__ to avoid passing freezed params to optimizer
        self.freeze_layer()

    def forward(self, input):
        x = input['image'] if isinstance(input, dict) else input
        outs = []
        # first conv
        x = self.first_conv(x)
        outs.append(x)
        if self.use_residual_block:
            x = self.residual_block(x)
        x = self.second_conv(x)

        if self.use_maxpool:
            x = self.max_pool(x)

        # blocks
        for idx, block in enumerate(self.blocks):
            x = block(x)
            if idx in self.stage_out_idx:
                outs.append(x)

        if self.task == 'classification':
            x = self.avg_pool(x)
            x = x.view(x.size(0), -1)
            x = self.classifier(x)
            return x

        features = [outs[i] for i in self.out_layers]
        return {'features': features, 'strides': self.get_outstrides()}

    def freeze_layer(self):
        """
        Freezing layers during training.
        """
        layers = [self.first_conv]
        start_idx = 0
        for stage_out_idx in self.stage_out_idx:
            end_idx = stage_out_idx + 1
            stage = [self.blocks[i] for i in range(start_idx, end_idx)]
            layers.append(nn.Sequential(*stage))
            start_idx = end_idx

        for layer_idx in self.frozen_layers:
            layer = layers[layer_idx]
            layer.eval()
            for param in layer.parameters():
                param.requires_grad = False

    def train(self, mode=True):
        """
        Sets the module in training mode.
        This has any effect only on modules such as Dropout or BatchNorm.

        Returns:
            - module: self
        """
        self.training = mode
        for module in self.children():
            module.train(mode)
        self.freeze_layer()
        return self

    def get_outplanes(self):
        """
        Get dimensions of the output tensors.

        Returns:
            - out (:obj:`list` of :obj:`int`)
        """
        return self.out_planes

    def get_outstrides(self):
        """
        Get strides of output tensors w.r.t inputs.

        Returns:
            - out (:obj:`list` of :obj:`int`)
        """
        return self.out_strides


def bignas_resnetd_bottleneck(**kwargs):
    return BigNAS_ResNetD_Bottleneck(**kwargs)


def bignas_resnetd_bottleneck_600M(**kwargs):
    # image_size = 160
    kwargs['out_channel'] = [24, 40, 168, 336, 664, 1640]
    kwargs['expand_ratio'] = [0, 0, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.25, 0.2, 0.2, 0.25]
    kwargs['depth'] = [1, 1, 2, 2, 5, 2]
    kwargs['kernel_size'] = [3, 3, 3, 3, 3, 3]
    kwargs['use_residual_block'] = False
    kwargs['act_stages'] = ['relu', 'relu', 'relu', 'relu', 'relu', 'relu']
    kwargs['stride_stages'] = [2, 1, 1, 2, 2, 2]
    return BigNAS_ResNetD_Bottleneck(**kwargs)


def bignas_resnetd_bottleneck_900M(**kwargs):
    # image_size = 160
    kwargs['out_channel'] = [24, 64, 168, 336, 816, 1640]
    kwargs['expand_ratio'] = [0, 0, 0.2, 0.25, 0.2, 0.25, 0.2, 0.2, 0.2, 0.25, 0.2, 0.2, 0.25, 0.25, 0.25]
    kwargs['depth'] = [1, 1, 2, 3, 5, 3]
    kwargs['kernel_size'] = [3, 3, 3, 3, 3, 3]
    kwargs['use_residual_block'] = False
    kwargs['act_stages'] = ['relu', 'relu', 'relu', 'relu', 'relu', 'relu']
    kwargs['stride_stages'] = [2, 1, 1, 2, 2, 2]
    return BigNAS_ResNetD_Bottleneck(**kwargs)


def bignas_resnetd_bottleneck_1200M(**kwargs):
    # image_size = 160
    kwargs['out_channel'] = [24, 64, 208, 408, 816, 2048]
    kwargs['expand_ratio'] = [0, 0, 0.2, 0.2, 0.2, 0.2, 0.25, 0.2, 0.2, 0.25, 0.25, 0.25, 0.2, 0.25, 0.25, 0.2]
    kwargs['depth'] = [1, 1, 3, 3, 5, 3]
    kwargs['kernel_size'] = [3, 3, 3, 3, 3, 3]
    kwargs['use_residual_block'] = False
    kwargs['act_stages'] = ['relu', 'relu', 'relu', 'relu', 'relu', 'relu']
    kwargs['stride_stages'] = [2, 1, 1, 2, 2, 2]
    return BigNAS_ResNetD_Bottleneck(**kwargs)


def bignas_resnetd_bottleneck_1800M(**kwargs):
    # image_size = 192
    kwargs['out_channel'] = [24, 64, 168, 336, 816, 2048]
    kwargs['expand_ratio'] = [0, 0, 0.25, 0.25, 0.2, 0.2, 0.25, 0.25, 0.25, 0.2, 0.2, 0.25, 0.25, 0.25, 0.25, 0.25,
                              0.25, 0.25]
    kwargs['depth'] = [1, 1, 4, 4, 5, 3]
    kwargs['kernel_size'] = [3, 3, 3, 3, 3, 3]
    kwargs['use_residual_block'] = False
    kwargs['act_stages'] = ['relu', 'relu', 'relu', 'relu', 'relu', 'relu']
    kwargs['stride_stages'] = [2, 1, 1, 2, 2, 2]
    return BigNAS_ResNetD_Bottleneck(**kwargs)


def bignas_resnetd_bottleneck_2400M(**kwargs):
    # image_size = 192
    kwargs['out_channel'] = [24, 24, 64, 168, 336, 1024, 2048]
    kwargs['expand_ratio'] = [0, 0, 0, 0.25, 0.25, 0.25, 0.25, 0.35, 0.25, 0.25, 0.2, 0.2, 0.2, 0.25, 0.25, 0.2, 0.25,
                              0.35, 0.25, 0.2]
    kwargs['depth'] = [1, 1, 1, 4, 4, 5, 4]
    kwargs['kernel_size'] = [3, 3, 3, 3, 3, 3, 3]
    kwargs['use_residual_block'] = True
    kwargs['act_stages'] = ['relu', 'relu', 'relu', 'relu', 'relu', 'relu', 'relu']
    kwargs['stride_stages'] = [2, 1, 1, 1, 2, 2, 2]
    return BigNAS_ResNetD_Bottleneck(**kwargs)


def bignas_resnetd_bottleneck_3000M(**kwargs):
    # image_size = 224
    kwargs['out_channel'] = [24, 24, 64, 168, 336, 816, 2048]
    kwargs['expand_ratio'] = [0, 0, 0, 0.25, 0.2, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.2, 0.25, 0.25, 0.25, 0.2, 0.2,
                              0.25, 0.35, 0.25, 0.25]
    kwargs['depth'] = [1, 1, 1, 4, 4, 6, 4]
    kwargs['kernel_size'] = [3, 3, 3, 3, 3, 3, 3]
    kwargs['use_residual_block'] = True
    kwargs['act_stages'] = ['relu', 'relu', 'relu', 'relu', 'relu', 'relu', 'relu']
    kwargs['stride_stages'] = [2, 1, 1, 1, 2, 2, 2]
    return BigNAS_ResNetD_Bottleneck(**kwargs)


def bignas_resnetd_bottleneck_3700M(**kwargs):
    # image_size = 224
    kwargs['out_channel'] = [24, 24, 64, 168, 408, 1024, 1640]
    kwargs['expand_ratio'] = [0, 0, 0, 0.25, 0.35, 0.25, 0.2, 0.25, 0.25, 0.25, 0.25, 0.25, 0.2, 0.35, 0.25, 0.25, 0.2,
                              0.35, 0.35, 0.25, 0.25]
    kwargs['depth'] = [1, 1, 1, 4, 4, 6, 4]
    kwargs['kernel_size'] = [3, 3, 3, 3, 3, 3, 3]
    kwargs['use_residual_block'] = True
    kwargs['act_stages'] = ['relu', 'relu', 'relu', 'relu', 'relu', 'relu', 'relu']
    kwargs['stride_stages'] = [2, 1, 1, 1, 2, 2, 2]
    return BigNAS_ResNetD_Bottleneck(**kwargs)


def bignas_resnetd_bottleneck_4100M(**kwargs):
    # image_size = 224
    kwargs['out_channel'] = [24, 24, 64, 168, 512, 1024, 2048]
    kwargs['expand_ratio'] = [0, 0, 0, 0.25, 0.25, 0.25, 0.35, 0.2, 0.25, 0.2, 0.35, 0.25, 0.25, 0.25, 0.2, 0.2, 0.2,
                              0.25, 0.25, 0.35, 0.2]
    kwargs['depth'] = [1, 1, 1, 4, 4, 6, 4]
    kwargs['kernel_size'] = [3, 3, 3, 3, 3, 3, 3]
    kwargs['use_residual_block'] = True
    kwargs['act_stages'] = ['relu', 'relu', 'relu', 'relu', 'relu', 'relu', 'relu']
    kwargs['stride_stages'] = [2, 1, 1, 1, 2, 2, 2]
    return BigNAS_ResNetD_Bottleneck(**kwargs)
