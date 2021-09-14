import torch
import torch.nn as nn

from prototype.spring.models.utils.normalize import build_norm_layer
from prototype.spring.models.utils.initializer import initialize_from_cfg

from .normal_blocks import ConvBlock, LinearBlock, RegBottleneckBlock, get_same_length


__all__ = ['bignas_regnet']


class BigNAS_RegNet(nn.Module):

    def __init__(self,
                 num_classes=1000,
                 # search settings
                 out_channel=[8, 8, 16, 48, 224],
                 depth=[1, 2, 2, 1, 2],
                 kernel_size=[7, 3, 3, 3, 3],
                 expand_ratio=[0, 1, 1, 1, 1],
                 group_width=[8, 8, 8, 8, 8],
                 # other settings
                 stride_stages=[2, 2, 2, 2, 2],
                 se_stages=[False, False, False, False, False],
                 act_stages=['relu', 'relu', 'relu', 'relu', 'relu'],
                 dropout_rate=0.1,
                 divisor=8,
                 se_act_func1='relu',
                 se_act_func2='sigmoid',
                 # bn and initializer
                 normalize={'type': 'solo_bn'},
                 initializer={'method': 'msra'},
                 # configuration for task
                 frozen_layers=[],
                 out_layers=[],
                 out_strides=[],
                 task='classification'):
        super(BigNAS_RegNet, self).__init__()

        r"""
        Arguments:
        - out_channel (:obj:`list` of 5 (stages+1) ints): channel list
        - depth (:obj:`list` of 5 (stages+1) ints): depth list for stages
        - kernel_size (:obj:`list` of 5 (stages+1) or 8 (blocks+1) ints): kernel size list for blocks
        - expand_ratio (:obj:`list` of 5 (stages+1) or 8 (blocks+1) ints): expand ratio list for blocks
        - group_width (:obj:`list` of 5 (stages+1) or 8 (blocks+1) ints): group width list for blocks
        - act_stages(:obj:`list` of 5 (stages+1) ints): activation list for blocks
        - stride_stages (:obj:`list` of 5 (stages+1) ints): stride list for stages
        - se_stages (:obj:`list` of 5 (stages+1) ints): se list for stages
        - se_act_func1(:obj:`str`: first activation function for se block)
        - se_act_func1(:obj:`str`: second activation function for se block)
        - dropout_rate (:obj:`float`): dropout rate
        - divisor(:obj:`int`): divisor for channels
        - normalize (:obj:`dict`): configurations for normalize
        - initializer (:obj:`dict`): initializer method
        - frozen_layers (:obj:`list` of :obj:`int`): Index of frozen layers, [0,4]
        - out_layers (:obj:`list` of :obj:`int`): Index of output layers, [0,4]
        - out_strides (:obj:`list` of :obj:`int`): Stride of outputs
        - task (:obj:`str`): Type of task, 'classification' or object 'detection'
        - num_classes (:obj:`int`): number of classification classes
        """

        self.frozen_layers = frozen_layers
        self.out_layers = out_layers
        self.out_strides = out_strides
        self.task = task

        global NormLayer

        NormLayer = build_norm_layer(normalize)

        self.depth = depth
        self.out_channel = out_channel
        self.kernel_size = get_same_length(kernel_size, depth)
        self.expand_ratio = get_same_length(expand_ratio, depth)
        self.group_width = get_same_length(group_width, depth)

        self.se_stages = se_stages
        self.stride_stages = stride_stages
        self.act_stages = act_stages
        self.se_act_func1 = se_act_func1
        self.se_act_func2 = se_act_func2
        self.divisor = divisor

        # first conv layer
        self.first_conv = ConvBlock(
            in_channel=3, out_channel=self.out_channel[0], kernel_size=self.kernel_size[0],
            stride=self.stride_stages[0], act_func=self.act_stages[0], NormLayer=NormLayer)

        # inverted residual blocks
        input_channel = self.out_channel[0]

        blocks = []
        stage_num = 1
        self.stage_out_idx = []
        _block_index = 1
        for s, act_func, use_se, n_block in zip(self.stride_stages[1:], self.act_stages[1:],
                                                self.se_stages[1:], self.depth[1:]):
            out_channel = self.out_channel[stage_num]
            stage_num += 1
            for i in range(n_block):
                kernel_size = self.kernel_size[_block_index]
                expand_ratio = self.expand_ratio[_block_index]
                group_width = self.group_width[stage_num]
                _block_index += 1
                if i == 0:
                    stride = s
                else:
                    stride = 1
                mobile_inverted_conv = RegBottleneckBlock(
                    input_channel, out_channel, kernel_size, expand_ratio=expand_ratio, group_width=group_width,
                    stride=stride, act_func=act_func, act_func1=self.se_act_func1,
                    act_func2=self.se_act_func2, use_se=use_se, divisor=self.divisor, NormLayer=NormLayer)
                blocks.append(mobile_inverted_conv)
                input_channel = out_channel
            self.stage_out_idx.append(_block_index - 2)

        self.out_planes = [self.out_channel[i] for i in self.out_layers]

        self.blocks = nn.ModuleList(blocks)
        if self.task == 'classification':
            self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
            self.classifier = LinearBlock(input_channel, num_classes, bias=True, dropout_rate=dropout_rate)

        if initializer is not None:
            initialize_from_cfg(self, initializer)

        # It's IMPORTANT when you want to freeze part of your backbone.
        # ALWAYS remember freeze layers in __init__ to avoid passing freezed params to optimizer
        self.freeze_layer()

    def forward(self, input):
        x = input['image'] if isinstance(input, dict) else input
        outs = []
        x = self.first_conv(x)
        outs.append(x)

        # blocks
        for idx, block in enumerate(self.blocks):
            x = block(x)
            if idx in self.stage_out_idx:
                outs.append(x)

        if self.task == 'classification':
            # classifier
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


def bignas_regnet(**kwargs):
    return BigNAS_RegNet(**kwargs)


if __name__ == '__main__':
    from prototype.spring.models import SPRING_MODELS_REGISTRY
    SPRING_MODELS_REGISTRY.register('bignas_regnet', bignas_regnet)

    cls_model = SPRING_MODELS_REGISTRY['bignas_regnet']()
    det_model = SPRING_MODELS_REGISTRY['bignas_regnet'](
        normalize={'type': 'freeze_bn'},
        frozen_layers=[0, 1],
        out_layers=[2, 3, 4],
        out_strides=[8, 16, 32],
        task='detection',
    )
    input = torch.randn(4, 3, 224, 224)
    det_output = det_model({'image': input})
    cls_output = cls_model(input)
    # function test
    det_model.train()
    print(det_model.get_outstrides())
    print(det_model.get_outplanes())
    print('detection output size: {}'.format(det_output['features'][0].size()))
    print('detection output size: {}'.format(det_output['features'][1].size()))
    print('detection output size: {}'.format(det_output['features'][2].size()))
    print('classification output size: {}'.format(cls_output.size()))
