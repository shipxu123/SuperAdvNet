# Import from pod
# Import from third library
import torch

from .bn_helper import (
    CaffeFrozenBatchNorm2d,
    FrozenBatchNorm2d,
    GroupNorm,
    GroupSyncBatchNorm
)

_norm_cfg = {
    'solo_bn': ('bn', torch.nn.BatchNorm2d),
    'freeze_bn': ('bn', FrozenBatchNorm2d),
    'caffe_freeze_bn': ('bn', CaffeFrozenBatchNorm2d),
    'sync_bn': ('bn', GroupSyncBatchNorm),
    'gn': ('gn', GroupNorm)
}


def build_norm_layer(num_features, cfg, postfix=''):
    assert isinstance(cfg, dict) and 'type' in cfg
    cfg = cfg.copy()
    layer_type = cfg.pop('type')
    kwargs = cfg.get('kwargs', {})

    if layer_type not in _norm_cfg:
        raise KeyError('Unrecognized norm type {}'.format(layer_type))
    else:
        abbr, norm_layer = _norm_cfg[layer_type]
        if norm_layer is None:
            raise NotImplementedError

    assert isinstance(postfix, (int, str))
    name = abbr + str(postfix)

    layer = norm_layer(num_features, **kwargs)
    return name, layer
