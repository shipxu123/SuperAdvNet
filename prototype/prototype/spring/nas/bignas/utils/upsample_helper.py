import torch
import torch.nn as nn


class InterpolateFunction(torch.autograd.Function):
    # hold original interpolate
    from torch.nn.functional import interpolate

    @staticmethod
    def forward(ctx, input, target, size, scale_factor, mode, align_corners):
        if target is not None:
            size = target.size()[-2:]
        return InterpolateFunction.interpolate(input, size, scale_factor, mode, align_corners)

    @staticmethod
    def symbolic(g, input, target, size, scale_factor, mode, align_corners):
        if target is not None:
            return g.op("DynamicUpsample", input, target, mode_s=mode)
        if scale_factor is not None:
            height = int(input.type().sizes()[2] * scale_factor)
            width = int(input.type().sizes()[3] * scale_factor)
        elif len(size) == 2:
            height = size[0]
            width = size[1]
        elif len(size == 1):
            height = size[0]
            width = size[0]
        return g.op("Upsample", input, mode_s=mode, height_i=height, width_i=width)


class ToOnnxUpsample(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, target, size=None, scale_factor=None, mode='nearest', align_corners=None):
        return InterpolateFunction.apply(x, target, size, scale_factor, mode, align_corners)
