from __future__ import division

# Standard Library
from collections.abc import Iterable, Mapping

# Import from third library
import numpy as np
import torch


def detach(x):
    """detach from given tensor to block the trace route"""
    if torch.is_tensor(x):
        shape = tuple(map(int, x.shape))
        return torch.zeros(shape, dtype=x.dtype, device=x.device)
    elif isinstance(x, str) or isinstance(x, bytes):  # there is a dead loop when x is a str with len(x)=1n
        return x
    elif isinstance(x, np.ndarray):  # numpy recommends building array by calling np.array
        return np.array(list(map(detach, x)))
    elif isinstance(x, Mapping):
        return type(x)((k, detach(v)) for k, v in x.items())
    elif isinstance(x, Iterable):
        try:
            output = type(x)(map(detach, x))
        except Exception as e:
            raise e
        return output

    else:
        return x


class ToCaffe(object):
    _tocaffe = False

    @classmethod
    def disable_trace(self, func):
        def wrapper(*args, **kwargs):
            output = func(*args, **kwargs)
            if not self._tocaffe:
                return output
            else:
                return detach(output)
        return wrapper

    @classmethod
    def prepare(self):
        if self._tocaffe:
            return

        self._tocaffe = True

        # workaround to avoid onnx tracing tensor.shape
        torch.Tensor._shpae = torch.Tensor.shape

        @property
        def shape(self):
            return self.detach()._shpae
        torch.Tensor.shape = shape


class Wrapper(torch.nn.Module):
    def __init__(self, detector):
        super(Wrapper, self).__init__()
        self.detector = detector

    def forward(self, image, return_meta=False):
        b, c, height, width = map(int, image.size())
        input = {
            'image_info': [[height, width, 1.0, height, width, 0]] * b,
            'image': image
        }
        print('before detector forward')
        output = self.detector(input)
        print(f'detector output:{output.keys()}')
        base_anchors = output['base_anchors']
        blob_names = []
        blob_datas = []
        output_names = sorted(output.keys())
        for idx, name in enumerate(output_names):
            if name.find('blobs') >= 0:
                blob_names.append(str(idx))
                blob_datas.append(output[name])
                print(f'blobs:{name}')
        assert len(blob_datas) > 0, 'no valid output provided, please set "tocaffe: True" in your config'
        if return_meta:
            return blob_names, base_anchors
        else:
            return blob_datas
