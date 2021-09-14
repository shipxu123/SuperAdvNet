import copy
import json
import itertools
import os
import time
import math
import onnx

import torch
import torch.nn as nn

from .utils.misc import count_dynamic_flops_and_params, DistModule, get_kwargs_itertools
from .utils.tocaffe_helper import Wrapper, ToCaffe
from .cls_controller import ClsController
from ..search_space import Bignas_SearchSpace

from spring.nart.utils.onnx_utils import OnnxDecoder
from spring.nart.passes import DeadCodeElimination, ConvFuser, GemmFuser
from spring.nart.core import Model
import spring.nart.tools.pytorch as pytorch
import linklink as link


class PodController(ClsController):

    def __init__(self, config):
        super(PodController, self).__init__(config)

    def subnet_log(self, curr_subnet_num, input, subnet_settings):
        if curr_subnet_num == 0:
            self.subnet_str = ''
        self.subnet_str += '%d: ' % curr_subnet_num + 'R_%s_%s_' % (input.size(2), input.size(3))
        for name, settings in subnet_settings.items():
            self.subnet_str += name + ': ' + ','.join(['%s_%s' % (
                key, '%s' % val) for key, val in settings.items()]) + ' || '
        if self.distiller_weight > 0:
            loss_type = '%.1f_%s' % (self.distiller_weight, self.distiller_type) + '\t'
            self.subnet_str += loss_type

    def get_subnet_flops(self, image_size=None, subnet_settings=None):
        if image_size is None:
            image_size = self.sample_image_size()
        else:
            image_size = self.get_image_size_with_shape(image_size)
        curr_subnet_settings = self.sample_subnet_settings(sample_mode='random', subnet_settings=subnet_settings)
        flops_dict = {}
        params_dict = {}
        for name, m in self.model.named_modules():
            if isinstance(m, Bignas_SearchSpace):
                if name == 'backbone':
                    input_shape = image_size
                    new_input_shape = []
                    for stride, out_channel in zip(m.get_outstrides(), m.get_outplanes()):
                        shape = [1, out_channel, image_size[2] // stride, image_size[3] // stride]
                        new_input_shape.append(shape)
                elif name == 'neck':
                    input_shape = new_input_shape
                    new_input_shape = [1, m.get_outplanes(), math.ceil(image_size[2] / m.get_outstrides()[0]),
                                       math.ceil(image_size[3] / m.get_outstrides()[0])]
                elif 'cls_subnet' in name:
                    input_shape = new_input_shape
                elif 'box_subnet' in name:
                    input_shape = new_input_shape
                input = m.get_fake_input(input_shape=input_shape)
                if name == 'backbone':
                    flops, params = count_dynamic_flops_and_params(self.model, input)
                    flops_dict['total'] = flops
                    params_dict['total'] = params
                flops, params = count_dynamic_flops_and_params(m, input)
                flops_dict[name] = flops
                params_dict[name] = params
        self.logger.info('Subnet with settings: {}\timage_size {}\tflops {}\tparams {}'.format(
            curr_subnet_settings, image_size, flops_dict, params_dict))
        return flops_dict, params_dict, image_size, curr_subnet_settings

    def adjust_model(self, curr_step, curr_subnet_num):
        # calculate current sample mode
        if self.sample_subnet_num > 1:
            sample_mode = self.sample_strategy[curr_subnet_num % len(self.sample_strategy)]
        else:
            sample_mode = self.sample_strategy[curr_step % len(self.sample_strategy)]

        # adjust model
        if self.subnet is not None:
            subnet_settings = self.sample_subnet_settings(sample_mode='subnet',
                                                          subnet_settings=self.subnet.subnet_settings)
        else:
            subnet_settings = self.sample_subnet_settings(sample_mode=sample_mode)
        return subnet_settings, sample_mode

    def sample_subnet_settings(self, sample_mode='random', subnet_settings=None):
        curr_subnet_settings = {}
        for name, m in self.model.named_modules():
            if not isinstance(m, Bignas_SearchSpace):
                continue
            if subnet_settings is None:
                _subnet_settings = m.sample_active_subnet(sample_mode=sample_mode)
                curr_subnet_settings[name] = _subnet_settings
            else:
                m.sample_active_subnet(sample_mode='subnet', subnet_settings=subnet_settings[name])
                curr_subnet_settings[name] = subnet_settings[name]
        return curr_subnet_settings

    def reset_subnet_running_statistics(self, model, data_loader):
        bn_mean = {}
        bn_var = {}
        forward_model = copy.deepcopy(model)
        forward_model.cuda()
        forward_model = DistModule(forward_model, True)
        self.reset_model_bn_forward(forward_model, bn_mean, bn_var)

        self.logger.info('calculate bn')
        iterator = iter(data_loader)
        max_iter = data_loader.get_epoch_size()
        with torch.no_grad():
            for count in range(max_iter):
                batch = next(iterator)
                forward_model(batch)

        for name, m in model.named_modules():
            if name in bn_mean and bn_mean[name].count > 0:
                feature_dim = bn_mean[name].avg.size(0)
                assert isinstance(m, nn.BatchNorm2d)
                # 最后取得的均值
                m.running_mean.data[:feature_dim].copy_(bn_mean[name].avg)
                m.running_var.data[:feature_dim].copy_(bn_var[name].avg)
        self.logger.info('bn complete')
        return model

    def get_subnet_num(self, count_flops=False, save_subnet=False, print_log=False):
        total_count = 1
        for name, m in self.model.named_modules():
            if not isinstance(m, Bignas_SearchSpace):
                continue
            kwargs_configs = get_kwargs_itertools(m.dynamic_settings)
            count = 0
            for kwargs in itertools.product(*kwargs_configs.values()):
                kwarg_config = dict(zip(kwargs_configs.keys(), kwargs))
                if print_log:
                    self.logger.info(json.dumps(kwarg_config))
                count += 1
                if count > 5000000:
                    self.logger.info('total subnet number for {} surpass 5000000'.format(name))
                    break
            self.logger.info('total subnet number for {} is {}'.format(name, count))
            total_count = total_count * count
        self.logger.info('all subnet number is {}'.format(total_count))

    def get_subnet_prototxt(self, image_size=None, subnet_settings=None, flops=None, onnx_only=True):
        """
        Args:
            image_size(tuple): configuration input size with 4 dimension
            subnet_settings(dict): configuration for subnet settings
            flops(float): flops
            onnx_only: whether to produce prototxt and others
        """
        if image_size is None:
            image_size = self.sample_image_size()
        else:
            image_size = self.get_image_size_with_shape(image_size)
        if subnet_settings is None:
            subnet_settings = self.sample_subnet_settings(sample_mode='random')
        else:
            self.sample_subnet_settings('subnet', subnet_settings)
        if flops is None:
            flops, params, image_size, subnet_settings = self.get_subnet_flops(image_size=image_size,
                                                                               subnet_settings=subnet_settings)

        self.logger.info("convert model...")
        save_prefix = os.path.join(self.path, str(image_size[2]) + '_' + str(round(flops['total'], 3)) + 'M')

        model = copy.deepcopy(self.model)
        for module in model.modules():
            module.tocaffe = True

        # disbale trace
        ToCaffe.prepare()
        time.sleep(10)

        model = model.eval().cpu().float()
        model = Wrapper(model)
        image = torch.randn(*image_size)
        output_names, base_anchors = model(image, return_meta=True)
        anchor_json = os.path.join(self.path, 'anchors.json')
        with open(anchor_json, 'w') as f:
            json.dump(base_anchors, f, indent=2)

        onnx_name = self.tocaffe(model, image_size, save_prefix, output_names=output_names,
                                 onnx_only=onnx_only)

        return onnx_name

    def tocaffe(self, model, image_size, save_prefix, input_names=['data'], output_names=['output'],
                onnx_only=True):
        data_shape = [image_size]
        prototxt_name = save_prefix + '.prototxt'
        caffemodel_name = save_prefix + '.caffemodel'
        onnx_name = save_prefix + '.onnx'
        merged_onnx_name = save_prefix + '_merged.onnx'

        if link.get_rank() == 0:
            if onnx_only:
                with pytorch.convert_mode():
                    pytorch.export_onnx(
                        model, data_shape,
                        filename=save_prefix,
                        input_names=input_names,
                        output_names=output_names,
                        verbose=False, cloze=False
                    )
            else:
                with pytorch.convert_mode():
                    pytorch.convert_v2(
                        model, data_shape,
                        filename=save_prefix,
                        input_names=input_names,
                        output_names=output_names,
                        verbose=False, cloze=False
                    )

                from spring.nart.tools.caffe.count import countFile
                _, info = countFile(prototxt_name)
                info_dict = {}
                info_dict['name'], info_dict['inputData'], info_dict['param'], \
                info_dict['activation'], info_dict['comp'], info_dict['add'], \
                info_dict['div'], info_dict['macc'], info_dict['exp'], info_dict['memacc'] = info
                for k, v in info_dict.items():
                    if not isinstance(v, str):
                        v = v / 1e6
                        info_dict[k] = v
                self.logger.info(info_dict)
                os.system('python -m spring.nart.caffe2onnx {} {} -o {}'.format(prototxt_name, caffemodel_name,
                                                                                onnx_name))

            self.logger.info("merge model...")
            onnx_model = onnx.load(onnx_name)

            graph = OnnxDecoder().decode(onnx_model)
            graph.update_tensor_shape()
            graph.update_topology()

            ConvFuser().run(graph)
            GemmFuser().run(graph)
            DeadCodeElimination().run(graph)
            graph.update_topology()

            onnx_model = Model.make_model(graph)
            onnx_model = onnx_model.dump_to_onnx()

            onnx.save(onnx_model, merged_onnx_name)

        link.barrier()
        model.train()
        model.cuda()

        return merged_onnx_name

    def check_flops_range(self, flops):
        v1 = flops['total']
        if v1 <= min(self.flops_range) or v1 >= max(self.flops_range):
            self.logger.info('current flops {} do not match target flops {}'.format(flops, self.flops_range))
            return False
        return True

    def get_subnet_weight(self, subnet_settings=None):
        subnet = {}
        for name, m in self.model.named_modules():
            if not isinstance(m, Bignas_SearchSpace):
                continue
            else:
                subnet[name] = m.sample_active_subnet_weights(subnet_settings[name])
        return subnet
