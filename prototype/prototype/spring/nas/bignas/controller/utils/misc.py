import torch
import numpy as np
import itertools
from ...ops.dynamic_ops import DynamicConv2d


def count_dynamic_flops_and_params(model, input):

    flops_dict = {}
    params_dict = {}

    def make_dynamic_conv2d_hook(name):

        def dynamic_conv2d_hook(m, input):
            n, c, h, w = input[0].size(0), input[0].size(
                1), input[0].size(2), input[0].size(3)
            flops = n * h * w * c * m.active_out_channel * m.active_kernel_size * m.active_kernel_size \
                / m.stride[1] / m.stride[1] / m.groups
            if name in flops_dict.keys():
                flops_dict[name] += int(flops)
            else:
                flops_dict[name] = int(flops)
            params = c * m.active_out_channel * m.active_kernel_size * m.active_kernel_size
            params_dict[name] = int(params)

        return dynamic_conv2d_hook

    def make_linear_hook(name):

        def linear_hook(m, input):
            _, c = input[0].size(0), input[0].size(1)
            flops = c * m.out_features
            flops_dict[name] = int(flops)
            params = c * m.out_features
            params_dict[name] = int(params)

        return linear_hook

    def make_conv2d_hook(name):

        def conv2d_hook(m, input):
            n, _, h, w = input[0].size(0), input[0].size(
                1), input[0].size(2), input[0].size(3)
            flops = n * h * w * m.in_channels * m.out_channels * m.kernel_size[0] * m.kernel_size[1] \
                / m.stride[1] / m.stride[1] / m.groups
            if name in flops_dict.keys():
                flops_dict[name] += int(flops)
            else:
                flops_dict[name] = int(flops)
            params = m.weight.size(0) * m.weight.size(1) * m.weight.size(2) * m.weight.size(3)
            params_dict[name] = int(params)

        return conv2d_hook

    hooks = []
    for name, m in model.named_modules():
        if isinstance(m, DynamicConv2d):
            h = m.register_forward_pre_hook(make_dynamic_conv2d_hook(name))
            hooks.append(h)
        elif isinstance(m, torch.nn.Conv2d) and not isinstance(m, DynamicConv2d):
            h = m.register_forward_pre_hook(make_conv2d_hook(name))
            hooks.append(h)
        elif isinstance(m, torch.nn.Linear):
            h = m.register_forward_pre_hook(make_linear_hook(name))
            hooks.append(h)

    model.eval()
    with torch.no_grad():
        _ = model(input)

    model.train()
    total_flops = 0
    for k, v in flops_dict.items():
        total_flops += v
    total_flops = round(total_flops / 1e6, 3)
    total_params = 0
    for k, v in params_dict.items():
        total_params += v
    total_params = round(total_params / 1e6, 3)

    for h in hooks:
        h.remove()
    return total_flops, total_params


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, length=0):
        self.length = length
        self.reset()

    def reset(self):
        if self.length > 0:
            self.history = []
        else:
            self.count = 0
            self.sum = 0.0
        self.val = 0.0
        self.avg = 0.0

    def reduce_update(self, tensor, num=1):
        self.update(tensor.item(), num=num)

    def update(self, val, num=1):
        if self.length > 0:
            # currently assert num==1 to avoid bad usage, refine when there are some explict requirements
            assert num == 1
            self.history.append(val)
            if len(self.history) > self.length:
                del self.history[0]

            self.val = self.history[-1]
            self.avg = np.mean(self.history)
        else:
            self.val = val
            self.sum += val * num
            self.count += num
            self.avg = self.sum / self.count


class DistModule(torch.nn.Module):
    def __init__(self, module, sync=False):
        super(DistModule, self).__init__()
        self.module = module
        self.broadcast_params()

        self.sync = sync
        if not sync:
            self._grad_accs = []
            self._register_hooks()

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

    def _register_hooks(self):
        for i, (name, p) in enumerate(self.named_parameters()):
            if p.requires_grad:
                p_tmp = p.expand_as(p)
                grad_acc = p_tmp.grad_fn.next_functions[0][0]
                grad_acc.register_hook(self._make_hook(name, p, i))
                self._grad_accs.append(grad_acc)

    def _make_hook(self, name, p, i):
        def hook(*ignore):
            pass
            #link.allreduce_async(name, p.grad.data)
        return hook

    def sync_gradients(self):
        """ average gradients """
        pass
        # if self.sync and link.get_world_size() > 1:
        #     for name, param in self.module.named_parameters():

        #         if param.requires_grad and param.grad is not None:
        #             link.allreduce(param.grad.data)
        # else:
        #     link.synchronize()

    def broadcast_params(self):
        """ broadcast model parameters """
        # for name, param in self.module.state_dict().items():
        #     link.broadcast(param, 0)
        pass


def get_kwargs_itertools(kwargs_cfg):
    def parse_range_attribute(attribute):
        ranges = []

        mins = attribute['space']['min']
        maxs = attribute['space']['max']
        steps = attribute['space']['stride']

        for min_item, max_item, step in zip(mins, maxs, steps):
            if isinstance(step, float):
                extra_adder = 1e-4
                ranges.append([float(i) for i in np.arange(min_item, max_item + extra_adder, step)])
            elif isinstance(step, int):
                extra_adder = 1
                ranges.append([int(i) for i in np.arange(min_item, max_item + extra_adder, step)])
            else:
                raise NotImplementedError

        if len(ranges) > 1:
            return itertools.product(*ranges)
        elif len(ranges) == 1:
            return ranges[0]
        else:
            return ranges

    parsed_cfg = {}

    for k, v in kwargs_cfg.items():
        if isinstance(v, dict) and v.search:
            parsed_cfg[k] = parse_range_attribute(v)

    return parsed_cfg
