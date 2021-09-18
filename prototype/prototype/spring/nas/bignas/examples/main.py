from prototype.spring.nas.bignas.search_space.bignas_resnet_basic import big_resnet_basic
from prototype.spring.nas.bignas.controller import ClsController
from prototype.spring.utils.log_helper import default_logger as logger
from easydict import EasyDict
import yaml
import linklink as linklink
import torch
import os


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
            linklink.allreduce_async(name, p.grad.data)
        return hook

    def sync_gradients(self):
        """ average gradients """
        if self.sync and linklink.get_world_size() > 1:
            for name, param in self.module.named_parameters():
                if param.requires_grad:
                    linklink.allreduce(param.grad.data)
        else:
            linklink.synchronize()

    def broadcast_params(self):
        """ broadcast model parameters """
        for name, param in self.module.state_dict().items():
            linklink.broadcast(param, 0)



def parse_config(config_file):
    with open(config_file) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        # config = yaml.safe_load(f)
    config = EasyDict(config)
    return config

def makedir(path):
    if linklink.get_rank() == 0 and not os.path.exists(path):
        os.makedirs(path)
    linklink.barrier()


linklink.initialize()
torch.cuda.set_device(linklink.get_local_rank())

makedir('bignas')
config_file = 'config.yaml'
config = parse_config(config_file)
model = big_resnet_basic(**config.model.kwargs)
model.cuda()
model = DistModule(model)
controller = ClsController(config.bignas)
controller.set_supernet(model)
controller.set_logger(logger)
controller.set_path('bignas')

logger.info(model)

onnx_name = controller.get_subnet_prototxt(image_size=config.bignas.subnet.image_size,
                                           subnet_settings=config.bignas.subnet.subnet_settings,
                                           onnx_only=False)
controller.get_subnet_latency(onnx_name)
controller.sample_subnet_lut()

linklink.finalize()
