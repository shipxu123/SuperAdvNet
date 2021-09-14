import torch.nn as nn

from ..ops.dynamic_blocks import DynamicConvBlock, DynamicLinearBlock, DynamicBasicBlock
from .base_bignas_searchspace import Bignas_SearchSpace
from ..backbone.bignas_resnetd_basic_subnet import bignas_resnetd_basic

big_resnetd_basic_kwargs = {
    'out_channel': {
        'space': {
            'min': [16, 16, 32, 32, 64, 128, 256],
            'max': [64, 64, 64, 96, 192, 384, 768],
            'stride': [16, 16, 16, 16, 32, 32, 64]},
        'sample_strategy': 'ordered_stage_wise'},
    'kernel_size': {
        'space': {
            'min': [3, 3, 3, 3, 3, 3, 3],
            'max': [3, 3, 3, 3, 3, 3, 3],
            'stride': 2},
        'sample_strategy': 'stage_wise'},
    'expand_ratio': [0, 0, 0, 1, 1, 1, 1],
    'depth': {
        'space': {
            'min': [1, 1, 1, 1, 1, 1, 1],
            'max': [1, 1, 1, 3, 3, 3, 3],
            'stride': 1},
        'sample_strategy': 'stage_wise_depth'},
}


class Big_ResNetD_Basic(Bignas_SearchSpace):

    def __init__(self,
                 # other settings
                 act_stages=['relu', 'relu', 'relu', 'relu', 'relu', 'relu', 'relu'],
                 stride_stages=[2, 1, 1, 1, 2, 2, 2],
                 use_maxpool=True,
                 dropout_rate=0,
                 divisor=8,
                 # clarify settings for detection task
                 frozen_layers=[],
                 out_layers=[],
                 out_strides=[],
                 task='classification',
                 num_classes=1000,
                 # search settings
                 kernel_transform=False,
                 zero_last_gamma=True,
                 # search space
                 **kwargs):
        super(Big_ResNetD_Basic, self).__init__(**kwargs)
        r"""
        Arguments:
        - act_stages(:obj:`list` of 5 (stages+1) ints): activation list for blocks
        - stride_stages (:obj:`list` of 5 (stages+1) ints): stride list for stages
        - use_maxpool (:obj `bool`): use maxpooling to downsample or not
        - dropout_rate (:obj:`float`): dropout rate
        - divisor(:obj:`int`): divisor for channels
        - frozen_layers (:obj:`list` of :obj:`int`): Index of frozen layers, [0,4]
        - out_layers (:obj:`list` of :obj:`int`): Index of output layers, [0,4]
        - out_strides (:obj:`list` of :obj:`int`): Stride of outputs
        - task (:obj:`str`): Type of task, 'classification' or object 'detection'
        - num_classes (:obj:`int`): number of classification classes
        - kernel_transform (:obj:`bool`): use kernel tranformation matrix or not
        - zero_last_gamma (:obj:`bool`): zero the last gamma of BatchNorm in the main path or not
        - out_channel (:obj:`list` or `dict`): stage-wise search settings for out channel
        - depth (:obj:`list` or `dict`): stage-wise search settings for depth
        - kernel_size (:obj:`list` or `dict`): stage-wise search settings for kernel size
        - expand_ratio (:obj:`list` or `dict`): stage-wise search settings for expand ratio
        """

        self.act_stages = act_stages
        self.stride_stages = stride_stages
        self.use_maxpool = use_maxpool
        self.dropout_rate = dropout_rate
        self.divisor = divisor

        self.frozen_layers = frozen_layers
        self.out_layers = out_layers
        self.out_strides = out_strides
        self.task = task
        self.num_classes = num_classes

        self.out_channel = self.normal_settings['out_channel']
        self.depth = self.normal_settings['depth']
        self.kernel_size = self.normal_settings['kernel_size']
        self.expand_ratio = self.normal_settings['expand_ratio']

        self.out_planes = [int(self.curr_subnet_settings['out_channel'][i]) for i in self.out_layers]

        # first conv layer
        self.first_conv = DynamicConvBlock(
            in_channel_list=3, out_channel_list=self.out_channel[0],
            kernel_size_list=self.kernel_size[0],
            stride=self.stride_stages[0], act_func=self.act_stages[0],
            KERNEL_TRANSFORM_MODE=kernel_transform)
        self.second_conv = DynamicConvBlock(
            in_channel_list=self.out_channel[0], out_channel_list=self.out_channel[1],
            kernel_size_list=self.kernel_size[1],
            stride=self.stride_stages[1], act_func=self.act_stages[1],
            KERNEL_TRANSFORM_MODE=kernel_transform)
        self.third_conv = DynamicConvBlock(
            in_channel_list=self.out_channel[1], out_channel_list=self.out_channel[2],
            kernel_size_list=self.kernel_size[2],
            stride=self.stride_stages[2], act_func=self.act_stages[2],
            KERNEL_TRANSFORM_MODE=kernel_transform)

        if self.use_maxpool:
            self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=True)
            self.stride_stages[3] = 1
        else:
            self.stride_stages[3] = 2

        blocks = []
        input_channel = self.out_channel[2]

        stage_num = 3
        _block_index = 3
        self.stage_out_idx = []
        for s, act_func, n_block in zip(self.stride_stages[3:], self.act_stages[3:], self.depth[3:]):
            ks_list = self.kernel_size[stage_num]
            expand_ratio = self.expand_ratio[stage_num]
            output_channel = self.out_channel[stage_num]
            stage_num += 1
            for i in range(n_block):
                if i == 0:
                    stride = s
                else:
                    stride = 1
                mobile_inverted_conv = DynamicBasicBlock(
                    in_channel_list=input_channel, out_channel_list=output_channel, kernel_size_list=ks_list,
                    expand_ratio_list=expand_ratio, stride=stride, act_func=act_func,
                    KERNEL_TRANSFORM_MODE=kernel_transform, divisor=self.divisor, IS_STAGE_BLOCK=(i == 0))
                blocks.append(mobile_inverted_conv)
                input_channel = output_channel
                _block_index += 1
            self.stage_out_idx.append(_block_index - 4)

        self.blocks = nn.ModuleList(blocks)
        if self.task == 'classification':
            self.avg_pool = nn.AdaptiveAvgPool2d(output_size=1)
            self.classifier = DynamicLinearBlock(
                in_features_list=input_channel, out_features_list=num_classes, bias=True,
                dropout_rate=dropout_rate)

        self.gen_dynamic_module_list(dynamic_types=[DynamicConvBlock, DynamicBasicBlock, DynamicLinearBlock])
        self.init_model()
        # zero_last gamma is used in default
        if zero_last_gamma:
            self.zero_last_gamma()

        # It's IMPORTANT when you want to freeze part of your backbone.
        # ALWAYS remember freeze layers in __init__ to avoid passing freezed params to optimizer
        if self.task == 'detection':
            self.freeze_layer()

    @staticmethod
    def name():
        return 'Big_ResNet_Basic'

    @property
    def module_str(self):
        _str = self.first_conv.module_str + '\n'
        _str += self.second_conv.module_str + '\n'
        _str += self.third_conv.module_str + '\n'

        if self.use_maxpool:
            _str += 'max_pool' + '\n'

        for idx, block in enumerate(self.blocks):
            if block.active_block:
                _str += block.module_str + '\n'

        if self.task == 'classification':
            _str += 'avg_pool' + '\n'
            _str += self.classifier.module_str + '\n'
        return _str

    def forward(self, input):
        x = input['image'] if isinstance(input, dict) else input
        outs = []

        # first conv
        x = self.first_conv(x)
        outs.append(x)
        x = self.second_conv(x)
        outs.append(x)
        x = self.third_conv(x)
        outs.append(x)

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
        layers = [self.first_conv, self.second_conv, self.third_conv]
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
        self.out_planes = [int(self.curr_subnet_settings['out_channel'][i]) for i in self.out_layers]
        return self.out_planes

    def get_outstrides(self):
        """
        Get strides of output tensors w.r.t inputs.

        Returns:
            - out (:obj:`list` of :obj:`int`)
        """
        return self.out_strides

    def build_active_subnet(self, subnet_settings):
        subnet = bignas_resnetd_basic(act_stages=self.act_stages, stride_stages=self.stride_stages,
                                     use_maxpool=self.use_maxpool, dropout_rate=self.dropout_rate, divisor=self.divisor,
                                     frozen_layers=self.frozen_layers, out_layers=self.out_layers,
                                     out_strides=self.out_strides,
                                     task=self.task, num_classes=self.num_classes,
                                     out_channel=subnet_settings['out_channel'],
                                     depth=subnet_settings['depth'],
                                     kernel_size=subnet_settings['kernel_size'],
                                     expand_ratio=subnet_settings['expand_ratio'])
        return subnet


def big_resnetd_basic(**kwargs):
    return Big_ResNetD_Basic(**kwargs)


if __name__ == '__main__':
    import torch
    from prototype.spring.models import SPRING_MODELS_REGISTRY

    SPRING_MODELS_REGISTRY.register('big_resnet_basic', big_resnetd_basic)

    cls_model = SPRING_MODELS_REGISTRY['big_resnetd_basic']
    det_model = SPRING_MODELS_REGISTRY['big_resnetd_basic'](
        normalize={'type': 'freeze_bn'},
        frozen_layers=[0, 1],
        out_layers=[4, 5, 6],
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
