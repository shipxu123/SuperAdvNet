# Standard Library
import torch.nn as nn

# Import from third library
from prototype.spring.models.utils.normalize import build_norm_layer

# Import from local library
from .normal_blocks import ConvBlock


class BigNAS_RoI_Head(nn.Module):
    """
    Head for the first stage detection task

    .. note::

        0 is always for the background class.
    """

    def __init__(self, inplanes, num_levels=5, num_conv=4, normalize={'type': 'solo_bn'},
                 depth=[1, 1, 1, 1], kernel_size=[3, 3, 3, 3], out_channel=[256, 256, 256, 256]):
        """
        Arguments:
            - inplanes (:obj:`list` or :obj:`int`): input channel
            - num_level (:obj:`int`): number of levels
            - depth (:obj:`list`): number of conv
            - kernel_size (:obj:`list` of `int`): kernel size settings
            - out_channel (:obj:`list` of `int`): out_channel settings
        """
        super(BigNAS_RoI_Head, self).__init__()

        global NormLayer

        NormLayer = build_norm_layer(normalize)

        self.num_levels = num_levels
        self.inplanes = inplanes

        self.mlvl_heads = nn.ModuleList()

        self.num_conv = num_conv
        assert num_conv == sum(depth)
        self.out_channel = out_channel
        self.kernel_size = kernel_size

        feat_planes = self.out_channel[0]
        kernel_size = self.kernel_size[0]
        for lvl in range(num_levels):
            layers = []
            inplanes = self.inplanes
            for conv_idx in range(self.num_conv):
                if lvl == 0:
                    layers.append(nn.Sequential(
                        ConvBlock(inplanes, feat_planes, kernel_size=kernel_size, stride=1,
                                  use_bn=False, act_func=''),
                        NormLayer(feat_planes),
                        nn.ReLU(inplace=True)))
                    inplanes = feat_planes
                else:
                    layers.append(nn.Sequential(
                        self.mlvl_heads[-1][conv_idx][0],
                        NormLayer(feat_planes),
                        nn.ReLU(inplace=True)))
                    inplanes = feat_planes
            self.mlvl_heads.append(nn.Sequential(*layers))

    def forward(self, x):
        for conv_idx in range(self.num_conv):
            x = self.mlvl_heads[0][conv_idx](x)
        return x


def bignas_roi_head(**kwargs):
    return BigNAS_RoI_Head(**kwargs)
