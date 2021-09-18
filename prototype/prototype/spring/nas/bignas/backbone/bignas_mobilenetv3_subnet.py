import torch
import torch.nn as nn

from prototype.spring.models.utils.normalize import build_norm_layer
from prototype.spring.models.utils.initializer import initialize_from_cfg
from .normal_blocks import ConvBlock, LinearBlock, MBConvBlock, get_same_length


__all__ = ['bignas_mobilenetv3']


class BigNAS_MobileNetV3(nn.Module):

    def __init__(self,
                 # search settings
                 out_channel=[16, 16, 24, 40, 80, 112, 160, 960, 1280],
                 depth=[1, 1, 4, 4, 4, 4, 4, 1, 1],
                 kernel_size=[7, 3, 3, 3, 3, 3, 3, 3],
                 expand_ratio=[1, 1, 6, 6, 6, 6, 6, 1, 1],
                 # other settings
                 act_stages=['h_swish', 'relu', 'relu', 'relu', 'h_swish', 'h_swish', 'h_swish', 'h_swish', 'h_swish'],
                 se_stages=[False, False, False, True, False, True, True, False, False],
                 stride_stages=[2, 1, 2, 2, 2, 1, 2, 1, 1],
                 se_act_func1='relu', se_act_func2='sigmoid',
                 dropout_rate=0.2,
                 divisor=8,
                 # bn and initializer
                 normalize={'type': 'solo_bn'},
                 initializer={'method': 'msra'},
                 # configuration for task
                 frozen_layers=[],
                 out_layers=[],
                 out_strides=[],
                 task='classification',
                 num_classes=1000):
        super(BigNAS_MobileNetV3, self).__init__()

        r"""
        Arguments:
        - out_channel (:obj:`list` of 9 (stages+3) ints): channel list
        - depth (:obj:`list` of 9 (stages+3) ints): depth list for stages
        - kernel_size (:obj:`list` of 9 (stages+3) or 24 (blocks+3) ints): kernel size list for blocks
        - expand_ratio (:obj:`list` of 9 (stages+3) or 24 (blocks+3) ints): expand ratio list for blocks
        - act_stages(:obj:`list` of 9 (stages+3) ints): activation list for blocks
        - stride_stages (:obj:`list` of 9 (stages+3) ints): stride list for stages
        - se_stages (:obj:`list` of 9 (stages+3) ints): se list for stages
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

        global NormLayer

        NormLayer = build_norm_layer(normalize)

        self.depth = depth
        self.out_channel = out_channel
        self.kernel_size = get_same_length(kernel_size, depth)
        self.expand_ratio = get_same_length(expand_ratio, depth)

        self.act_stages = act_stages
        self.stride_stages = stride_stages
        self.se_stages = se_stages
        self.se_act_func1 = se_act_func1
        self.se_act_func2 = se_act_func2
        self.dropout_rate = dropout_rate
        self.divisor = divisor

        self.frozen_layers = frozen_layers
        self.out_layers = out_layers
        self.out_strides = out_strides
        self.task = task

        # first conv layer
        self.first_conv = ConvBlock(
            in_channel=3, out_channel=self.out_channel[0], kernel_size=self.kernel_size[0],
            stride=self.stride_stages[0], act_func=self.act_stages[0], NormLayer=NormLayer)

        # inverted residual blocks
        input_channel = self.out_channel[0]
        self.stage_out_idx = []

        blocks = []
        stage_num = 1
        _block_idx = 1
        all_stage = len(act_stages)
        for s, act_func, use_se, n_block in zip(self.stride_stages[1:all_stage - 2], self.act_stages[1:all_stage - 2],
                                                self.se_stages[1:all_stage - 2], self.depth[1:all_stage - 2]):
            output_channel = self.out_channel[stage_num]
            stage_num += 1
            for i in range(n_block):
                ks_list = self.kernel_size[_block_idx]
                expand_ratio = self.expand_ratio[_block_idx]
                if i == 0:
                    stride = s
                else:
                    stride = 1
                mobile_inverted_conv = MBConvBlock(
                    in_channel=input_channel, out_channel=output_channel, kernel_size=ks_list,
                    expand_ratio=expand_ratio, stride=stride, act_func=act_func, act_func1=self.se_act_func1,
                    act_func2=self.se_act_func2, use_se=use_se, divisor=self.divisor, NormLayer=NormLayer)
                blocks.append(mobile_inverted_conv)
                input_channel = output_channel
                _block_idx += 1
            self.stage_out_idx.append(_block_idx - 2)

        self.blocks = nn.ModuleList(blocks)
        self.final_expand_layer = ConvBlock(
            in_channel=input_channel, out_channel=self.out_channel[-2],
            kernel_size=self.kernel_size[-2], act_func=act_stages[-2], NormLayer=NormLayer)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.feature_mix_layer = ConvBlock(
            in_channel=self.out_channel[-2], out_channel=self.out_channel[-1],
            kernel_size=self.kernel_size[-1], use_bn=False, act_func=act_stages[-1], NormLayer=NormLayer)

        if self.task == 'classification':
            self.classifier = LinearBlock(
                in_features=self.out_channel[-1], out_features=num_classes, bias=True, dropout_rate=dropout_rate)

        self.out_planes = [self.out_channel[i] for i in self.out_layers]

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

        x = self.final_expand_layer(x)
        outs.append(x)
        x = self.avg_pool(x)
        x = self.feature_mix_layer(x)
        outs.append(x)

        if self.task == 'classification':
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

    def get_module_list(self, input_shape=[1, 3, 224, 224], block_types=[ConvBlock, MBConvBlock, LinearBlock]):
        module_list = []
        handles = []
        input = torch.zeros(*input_shape)
        device = next(self.parameters(), torch.tensor([])).device
        input = input.to(device)

        def make_hook(module_name):
            def hook(module, data, out):
                module_list.append((module))
            return hook

        for n, m in self.named_modules():
            if sum(map(lambda x: isinstance(m, x), block_types)):
                handle = m.register_forward_hook(make_hook(n))
                handles.append(handle)
        self.forward(input)
        for handle in handles:
            handle.remove()

        return module_list


def bignas_mobilenetv3(**kwargs):
    return BigNAS_MobileNetV3(**kwargs)


if __name__ == '__main__':
    from prototype.spring.models import SPRING_MODELS_REGISTRY
    SPRING_MODELS_REGISTRY.register('bignas_mobilenetv3', bignas_mobilenetv3)

    cls_model = SPRING_MODELS_REGISTRY['bignas_mobilenetv3']()
    det_model = SPRING_MODELS_REGISTRY['bignas_mobilenetv3'](
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
