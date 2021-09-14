# Standard Library
import torch
import torch.nn as nn

# Import from local library
from ...search_space import Bignas_SearchSpace
from ...ops.dynamic_blocks import DynamicConvBlock
from ...ops.dynamic_ops import DynamicBatchNorm2d
from ...backbone.bignas_roi_head_subnet import bignas_roi_head


kwargs = {
    'out_channel': {
        'space': {
            'min': [128],
            'max': [256],
            'stride': 16},
        'sample_strategy': 'stage_wise'},
    'kernel_size': {
        'space': {
            'min': [1],
            'max': [3],
            'stride': 2},
        'sample_strategy': 'stage_wise'},
    'depth': {
        'space': {
            'min': [2],
            'max': [4],
            'stride': 1},
        'sample_strategy': 'stage_wise_depth'},
}


class Big_RoI_Head(Bignas_SearchSpace):
    """
    Head for the first stage detection task

    .. note::

        0 is always for the background class.
    """

    def __init__(self, inplanes, num_levels, num_conv, **kwargs):
        """
        Arguments:
            - inplanes (:obj:`list` or :obj:`int`): input channel
            - num_levels (:obj:`int`): num of level
            - num_conv (:obj:`int`): num of conv
        """
        super(Big_RoI_Head, self).__init__(**kwargs)

        self.num_levels = num_levels
        self.inplanes = inplanes

        self.mlvl_heads = nn.ModuleList()

        self.num_conv = num_conv
        assert num_conv == sum(self.normal_settings['depth'])

        for lvl in range(num_levels):
            layers = []
            inplanes = self.inplanes
            feat_planes = self.normal_settings['out_channel'][0]
            kernel_size = self.normal_settings['kernel_size'][0]
            for conv_idx in range(self.num_conv):
                if lvl == 0:
                    layers.append(nn.Sequential(
                        DynamicConvBlock(inplanes, feat_planes, kernel_size_list=kernel_size, stride=1,
                                         use_bn=False, act_func=''),
                        DynamicBatchNorm2d(feat_planes),
                        nn.ReLU(inplace=True)))
                    inplanes = feat_planes
                else:
                    layers.append(nn.Sequential(
                        self.mlvl_heads[-1][conv_idx][0],
                        DynamicBatchNorm2d(feat_planes),
                        nn.ReLU(inplace=True)))
                    inplanes = feat_planes
            self.mlvl_heads.append(nn.Sequential(*layers))

        self.gen_dynamic_module_list(dynamic_types=[DynamicConvBlock])

    def forward(self, x):
        for conv_idx in range(self.num_conv):
            x = self.mlvl_heads[0][conv_idx](x)
        return x

    def get_outplanes(self):
        return self.curr_subnet_settings['out_channel'][-1]

    def get_fake_input(self, input_shape=None):
        if not input_shape:
            input_shape = [1, self.inplanes, 64, 64]
        else:
            assert len(input_shape) == 4
        input = torch.zeros(*input_shape)
        device = next(self.parameters(), torch.tensor([])).device
        input = input.to(device)
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d) and m.weight.dtype == torch.float16:
                input = input.half()
        return input

    @property
    def module_str(self):
        _str = ''
        for lvl in range(self.num_levels):
            for conv_idx in range(self.num_conv):
                _str += self.mlvl_heads[lvl][conv_idx][0].module_str + '\n'
        return _str

    def build_active_subnet(self, subnet_settings):
        subnet = bignas_roi_head(inplanes=self.inplanes, num_levels=self.num_levels, num_conv=self.num_conv,
                                 depth=subnet_settings['depth'], kernel_size=subnet_settings['kernel_size'],
                                 out_channel=subnet_settings['out_channel'])
        return subnet


def big_roi_head(*args, **kwargs):
    return Big_RoI_Head(*args, **kwargs)
