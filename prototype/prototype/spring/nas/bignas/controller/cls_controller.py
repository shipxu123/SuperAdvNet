import copy
import time
import json
import itertools
import random
import onnx
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from prototype.spring.distiller import MimicJob, Mimicker

from .utils.misc import count_dynamic_flops_and_params, AverageMeter, DistModule, get_kwargs_itertools


class ClsController(object):

    def __init__(self, config):
        self.config = config
        self.build_bignas_settings()
        print({'name': "spring.nas.bignas"})

    def set_supernet(self, model):
        self.model = model

    def set_teacher(self, model):
        self.teacher = model

    def set_logger(self, logger):
        self.logger = logger

    def set_path(self, path):
        self.path = path

    def build_bignas_settings(self):
        # data
        self.metric1 = self.config.data.get('metric1', 'top1')
        self.metric2 = self.config.data.get('metric2', 'top5')

        self.image_size_list = self.config.data.image_size_list
        self.stride = self.config.data.get('stride', 4)
        if isinstance(self.image_size_list[0], int):
            # make sure it is in ascending order
            self.image_size_list = [i for i in range(min(self.image_size_list),
                                                     max(self.image_size_list) + 1, self.stride)]
            self.test_image_size_list = self.config.data.get('test_image_size_list',
                                                             sorted({min(self.image_size_list),
                                                                     max(self.image_size_list)}))
        elif isinstance(self.image_size_list[0], list):
            self.test_image_size_list = self.config.data.get('test_image_size_list', self.image_size_list[-1])

        self.share_interpolation = self.config.data.get('share_interpolation', False)
        self.interpolation_type = self.config.data.get('interpolation_type', 'bicubic')
        assert self.interpolation_type in ['nearest', 'linear', 'bilinear', 'bicubic', 'trilinear', 'area', None]

        self.calib_meta_file = self.config.data.get('calib_meta_file', '/mnt/lustre/xupeng2/train_4k.txt')

        # train
        self.valid_before_train = self.config.train.get('valid_before_train', False)
        self.sample_subnet_num = self.config.train.get('sample_subnet_num', 1)
        self.sample_strategy = self.config.train.get('sample_strategy', ['max', 'random', 'random', 'min'])

        # distiller
        self.distiller = self.config.distiller
        self.distiller_weight = self.config.distiller.get('weight', 0)
        self.distiller_type = self.config.distiller.get('type', 'kd')
        if self.distiller_weight > 0 and 'inplace_' in self.distiller_type:
            self.teacher = None

        # subnet
        self.subnet = self.config.get('subnet', None)

        # latency
        self.latency = self.config.get('latency', None)

    def init_distiller(self):
        if self.teacher is None:
            self.mimicker = Mimicker(teacher_model=self.model, student_model=self.model)
        else:
            self.mimicker = Mimicker(teacher_model=self.teacher, student_model=self.model)
        # create a mimic job
        mimicjob = MimicJob('job1',
                            self.distiller_type.replace('inplace_', ''),
                            self.distiller.s_name,
                            self.distiller.t_name,
                            mimic_loss_weight=self.distiller_weight, **self.distiller.kwargs)
        self.mimicker.register_job(mimicjob)
        self.mimicker.prepare()

    def subnet_log(self, curr_subnet_num, input, subnet_settings):
        if curr_subnet_num == 0:
            self.subnet_str = ''
        self.subnet_str += '%d: ' % curr_subnet_num + 'R_%s_' % input.size(2) + ','.join(['%s_%s' % (
            key, '%s' % val) for key, val in subnet_settings.items()]) + ' || '
        if self.distiller_weight > 0:
            loss_type = '%.1f_%s' % (self.distiller_weight, self.distiller_type) + '\t'
            self.subnet_str += loss_type

    def show_subnet_log(self):
        self.logger.info(self.subnet_str)

    def get_subnet_flops(self, image_size=None, subnet_settings=None):
        image_size = self.sample_image_size(image_size, sample_mode='random')
        if subnet_settings is None:
            subnet_settings = self.model.sample_active_subnet('random')
        else:
            self.model.sample_active_subnet('subnet', subnet_settings)
        input = self.model.get_fake_input(input_shape=image_size)
        flops, params = count_dynamic_flops_and_params(self.model, input)
        self.logger.info('Subnet with settings: {}\timage_size {}\tflops {}\tparams {}'.format(
            subnet_settings, image_size, flops, params))
        return flops, params, image_size, subnet_settings

    def adjust_input(self, input, curr_subnet_num, sample_mode=None):
        # if all subnets share one resolution, then interpolation is not needed.
        if self.subnet:
            return input
        if self.share_interpolation and curr_subnet_num > 0:
            return input
        elif self.share_interpolation and curr_subnet_num == 0:
            # which means the image size is random sampled
            image_size = self.sample_image_size()
        else:
            image_size = self.sample_image_size(sample_mode=sample_mode)

        # 如果只存在一个image size选项，或者选到了dataloder的那个就不用插值
        input = F.interpolate(input, size=(image_size[2], image_size[3]),
                              mode=self.interpolation_type, align_corners=False)
        return input

    def get_image_size_with_shape(self, image_size):
        """
        Args:
            image_size(int or list or tuple): image_size can be int or list or tuple
        Returns:
            input_size(tuple): process the input image size to 4 dimension input size
        """
        if isinstance(image_size, int):
            input_size = (1, 3, image_size, image_size)
        elif (isinstance(image_size, list) or isinstance(image_size, tuple)) and len(image_size) == 1:
            input_size = (1, 3, image_size[0], image_size[0])
        elif (isinstance(image_size, list) or isinstance(image_size, tuple)) and len(image_size) == 2:
            input_size = (1, 3, image_size[0], image_size[1])
        elif (isinstance(image_size, list) or isinstance(image_size, tuple)) and len(image_size) == 3:
            input_size = (1, image_size[0], image_size[1], image_size[2])
        elif (isinstance(image_size, list) or isinstance(image_size, tuple)) and len(image_size) == 4:
            input_size = image_size
        else:
            raise ValueError('image size should be lower than 5 dimension')
        return input_size

    def sample_image_size(self, image_size=None, sample_mode=None):
        """
        Args:
            image_size(int or list or tuple): if not None, return the 4 dimension shape
            sample_mode(['min', 'max', 'random', None]): input resolution sample mode,
                                                        sample_mode is 'random' in default
        Returns:
            image_size(tuple): 4 dimension input size
        """
        if image_size is not None:
            input_size = self.get_image_size_with_shape(image_size)
            return input_size
        if sample_mode is None:
            sample_mode = 'random'
        if sample_mode == 'max':
            image_size = self.image_size_list[-1]
        elif sample_mode == 'min':
            image_size = self.image_size_list[0]
        elif sample_mode == 'random':
            image_size = random.choice(self.image_size_list)
        else:
            raise ValueError('only min max random are supported')

        input_size = self.get_image_size_with_shape(image_size)
        return input_size

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
        if subnet_settings is None:
            subnet_settings = self.model.sample_active_subnet(sample_mode=sample_mode)
        else:
            subnet_settings = self.model.sample_active_subnet(sample_mode='subnet',
                                                                     subnet_settings=subnet_settings)
        return subnet_settings

    def adjust_teacher(self, input, curr_subnet_num):
        self.mimicker.prepare()
        if self.teacher is not None and curr_subnet_num == 0:
            with torch.no_grad():
                self.teacher(input)

    def get_distiller_loss(self, sample_mode):
        if self.distiller_weight == 0:
            return 0
        if self.teacher is None and sample_mode == 'max':
            self.output_t_maps = self.mimicker.output_t_maps
            for t_maps in self.output_t_maps:
                for k, v in t_maps.items():
                    t_maps[k] = v.detach()
            return 0
        self.mimicker.output_t_maps = self.output_t_maps
        mimic_loss = sum(self.mimicker.mimic())
        return mimic_loss

    def reset_model_bn_forward(self, model, bn_mean, bn_var):
        for name, m in model.named_modules():
            if isinstance(m, nn.BatchNorm2d):
                bn_mean[name] = AverageMeter()
                bn_var[name] = AverageMeter()

                def new_forward(bn, mean_est, var_est):
                    def lambda_forward(x):
                        batch_mean = x.mean(0, keepdim=True).mean(2, keepdim=True).mean(3, keepdim=True)  # 1, C, 1, 1
                        batch_var = (x - batch_mean) * (x - batch_mean)
                        batch_var = batch_var.mean(0, keepdim=True).mean(2, keepdim=True).mean(3, keepdim=True)

                        batch_mean = torch.squeeze(batch_mean).float()
                        batch_var = torch.squeeze(batch_var).float()
                        # 直接算正常的bn的mean 和 var

                        # 累计mean_est = batch_mean * batch
                        reduce_batch_mean = batch_mean.clone() 
                        reduce_batch_var = batch_var.clone()

                        # link.allreduce(reduce_batch_mean.data)
                        # link.allreduce(reduce_batch_var.data)
                        mean_est.update(reduce_batch_mean.data, x.size(0))
                        var_est.update(reduce_batch_var.data, x.size(0))

                        # bn forward using calculated mean & var
                        _feature_dim = batch_mean.size(0)
                        return F.batch_norm(
                            x, batch_mean, batch_var, bn.weight[:_feature_dim],
                            bn.bias[:_feature_dim], False,
                            0.0, bn.eps,
                        )

                    return lambda_forward

                m.forward = new_forward(m, bn_mean[name], bn_var[name])

    def reset_subnet_running_statistics(self, model, data_loader):
        bn_mean = {}
        bn_var = {}
        forward_model = copy.deepcopy(model)
        forward_model.cuda()
        forward_model = DistModule(forward_model, True)
        self.reset_model_bn_forward(forward_model, bn_mean, bn_var)
        self.logger.info('calculate bn')
        with torch.no_grad():
            if 'CIFAR' in str(data_loader.dataset.__class__):
                # temporarily for cifar
                for images, label in data_loader:
                    images = images.cuda()
                    forward_model(images)
            else:
                for batch in data_loader:
                    images = images.cuda()
                    forward_model(images)

        for name, m in model.named_modules():
            if name in bn_mean and bn_mean[name].count > 0:
                feature_dim = bn_mean[name].avg.size(0)
                assert isinstance(m, nn.BatchNorm2d)
                # 最后取得的均值
                m.running_mean.data[:feature_dim].copy_(bn_mean[name].avg)
                m.running_var.data[:feature_dim].copy_(bn_var[name].avg)
        self.logger.info('bn complete')
        return model

    def check_flops_range(self, flops):
        if min(self.flops_range) < flops < max(self.flops_range):
            return True
        else:
            self.logger.info('current flops {} do not match target flops {}'.format(flops, self.flops_range))
            return False

    def get_subnet_num(self, count_flops=False, save_subnet=False):
        count = 0
        total_count = 0
        subnet_table = {}
        kwargs = copy.deepcopy(self.model.dynamic_settings)
        self.logger.info(json.dumps(kwargs))
        for image_size in self.image_size_list:
            image_size = self.get_image_size_with_shape(image_size)
            kwargs_configs = get_kwargs_itertools(kwargs)

            for kwarg in itertools.product(*kwargs_configs.values()):
                total_count += 1
                kwarg_config = dict(zip(kwargs_configs.keys(), kwarg))
                subnet_settings = kwarg_config
                if count_flops:
                    flops, params, image_size, subnet_settings = self.get_subnet_flops(image_size=image_size,
                                                                                       subnet_settings=subnet_settings)
                    if self.flops_range:
                        if self.check_flops_range(flops):
                            subnet_table[count] = {'flops': flops, 'params': params,
                                                   'image_size': image_size, 'subnet_settings': subnet_settings}
                            count += 1
                if total_count > 5000000:
                    self.logger.info('total subnet number surpass 5 million')
                    return

        self.logger.info('total subnet number {}'.format(total_count))
        if count_flops:
            self.logger.info('total subnet number {} in flops_range {}'.format(total_count, self.flops_range))
        if save_subnet:
            path = os.path.join(self.path, 'subnet.txt')
            with open(path, 'w') as f:
                for _, v in subnet_table.items():
                    f.write(json.dumps(v))
        return subnet_table

    def sample_subnet_lut(self, test_latency=True):
        assert self.subnet is not None
        self.lut_path = self.subnet.get('lut_path', None)
        self.flops_range = self.subnet.get('flops_range', None)
        self.subnet_count = self.subnet.get('subnet_count', 500)
        self.subnet_sample_mode = self.subnet.get('subnet_sample_mode', 'random')
        for name, m in self.model.named_modules():
            if isinstance(m, nn.BatchNorm2d):
                if m.bias.data[0].item() == 0:
                    m.bias.data.fill_(1)
                    self.logger.info('rewrite batch norm bias for layer {}'.format(name))

        if self.flops_range is not None:
            self.logger.info('flops range with {}'.format(self.flops_range))
        else:
            self.logger.info('No flops range defined')
        if self.subnet_sample_mode == 'traverse':
            return self.get_subnet_num(count_flops=True, save_subnet=True)
        else:
            self.get_subnet_num(count_flops=False, save_subnet=False)
        subnet_table = {}
        self.logger.info('subnet count {}'.format(self.subnet_count))
        count = 0
        seed = torch.Tensor([int(time.time() * 10000) % 10000])
        self.logger.info('seed {}'.format(seed))
        while count < self.subnet_count:
            seed += 1
            random.seed(seed.item())
            flops, params, image_size, subnet_settings = self.get_subnet_flops()
            if self.flops_range is not None:
                if not self.check_flops_range(flops):
                    self.logger.info('do not match target fp flops')
                    continue
            subnet_table[count] = {'flops': flops, 'params': params,
                                   'image_size': image_size, 'subnet_settings': subnet_settings}
        self.logger.info(json.dumps(subnet_table))
        return subnet_table

    def get_subnet_weight(self, subnet_settings=None):
        subnet = self.model.sample_active_subnet_weights(subnet_settings)
        return subnet
