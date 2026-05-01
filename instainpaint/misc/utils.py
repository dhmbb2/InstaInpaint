# Copyright (c) Facebook, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Misc functions.

Mostly copy-paste from torchvision references or other public repos like DETR:
https://github.com/facebookresearch/detr/blob/master/util/misc.py
"""
import copy
import datetime
import math
import os

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import random
import subprocess
import sys
import tempfile
import time
from collections import defaultdict, deque, OrderedDict

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.utils.data
from pathlib import Path

from .dist_helper import get_rank, get_world_size, is_dist_avail_and_initialized
from .env_utils import _suppress_print

from .io_helper import pathmgr

print_debug_info = False


class _RepeatSampler(object):
    """Sampler that repeats forever.

    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


class RepeatedDataLoader(torch.utils.data.dataloader.DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._DataLoader__initialized = False
        self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self._DataLoader__initialized = True
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


def linear_to_srgb(l):
    # s = np.zeros_like(l)
    s = torch.zeros_like(l)
    m = l <= 0.00313066844250063
    s[m] = l[m] * 12.92
    s[~m] = 1.055 * (l[~m] ** (1.0 / 2.4)) - 0.055
    return s


def pytorch_mlp_clip_gradients(model, clip):
    grad_norm = []
    for p in model.parameters():
        if p is not None and p.grad is not None:
            grad_norm.append(p.grad.view(-1))

    if len(grad_norm) > 0:
        grad_norm = torch.concat(grad_norm).norm(2).item()
        clip_coef = clip / (grad_norm + 1e-6)
        if clip_coef < 1.0:
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.data.mul_(clip_coef)
        return grad_norm
    return None


def unitwise_norm(x):
    """Computes norms of each output unit separately, assuming (HW)IO weights."""
    if len(x.shape) <= 1:  # Scalars and vectors
        return x.norm(2)
    elif len(x.shape) in [2, 3]:  # Linear layers of shape OI
        return x.norm(2, dim=-1, keepdim=True)
    elif len(x.shape) == 4:  # Conv kernels of shape OIHW
        return x.norm(2, dim=[1,2,3], keepdim=True)
    else:
        raise ValueError(f'Got a parameter with shape not in [1, 2, 3, 4]! {x}')


def clip_gradients(model, clip, check_nan_inf=True, file_name=None, adaptive=False):
    norms = []
    for name, p in model.named_parameters():
        if p.grad is not None:
            if check_nan_inf:
                p.grad.data = torch.nan_to_num(
                    p.grad.data, nan=0.0, posinf=0.0, neginf=0.0
                )

            if adaptive:
                param_norm = unitwise_norm(p)
                grad_norm = unitwise_norm(p.grad.data)
                max_norm = param_norm * clip
                trigger = grad_norm > max_norm
                clipped_grad = p.grad.data * (max_norm / grad_norm.clamp(min=1e-6))
                p.grad.data = torch.where(trigger, clipped_grad, p.grad.data)
            else:
                grad_norm = p.grad.data.norm(2)
                norms.append(grad_norm.item())
                clip_coef = clip / (grad_norm + 1e-6)
                if clip_coef < 1:
                    p.grad.data.mul_(clip_coef)
    return norms


def cancel_gradients_last_layer(epoch, model, freeze_last_layer):
    if epoch >= freeze_last_layer:
        return
    for n, p in model.named_parameters():
        if "last_layer" in n:
            p.grad = None


def filter_weights_with_wrong_size(model, weights):
    new_weights = OrderedDict()
    missing_keys = []
    state_dict = model.state_dict()
    for name, value in weights.items():
        if name in state_dict:
            target_value = state_dict[name]
            if value.size() != target_value.size():
                missing_keys.append(name)
            else:
                new_weights[name] = value
        else:
            new_weights[name] = value

    return new_weights, missing_keys


def load_ddp_state_dict(
    model, 
    weights, 
    key=None, 
    filter_mismatch=True,
    rewrite_weights=[],
):
    if weights is None:
        msg = "No weights available for module {}".format(key)
        return msg

    weights = copy.deepcopy(weights)
    if key is not None and key == "triplane":
        all_keys = [k for k in weights.keys() if "mlp." in k]
        for k in all_keys:
            new_k = k.replace("mlp.", "mlp_rf.")
            weights[new_k] = weights[k].clone()
            del weights[k]

    # Not very efficient, but this is not often used
    try:
        if len(rewrite_weights) > 0:
            for key in weights.keys():
                for rewrite_spec in rewrite_weights:
                    if rewrite_spec["pname"] in key:
                        weights[key] = rewrite_spec["rewrite_fn"](weights[key])
                        print(" [OuO] Rewrite weight {} into {}".format(key, weights[key].shape))
                        assert model.get_parameter(key).shape == weights[key].shape, \
                            "The parameter shape mismatched after rewrite! {} != {}".format(
                                model.get_parameter(key).shape, weights[key].shape)
    except Exception as e:
        raise RuntimeError("Failed to rewrite checkpoint weights") from e

    if isinstance(model, nn.parallel.DistributedDataParallel):
        if filter_mismatch:
            weights, missing_keys = filter_weights_with_wrong_size(model, weights)
            if len(missing_keys) > 0:
                print(
                    "Keys ",
                    missing_keys,
                    " are filtered out due to parameter size mismatch.",
                )
        msg = model.load_state_dict(weights, strict=False)
    elif isinstance(model, torch.optim.Optimizer):
        msg = model.load_state_dict(weights)
    else:
        new_weights = OrderedDict()
        for k, v in weights.items():
            if k[:7] == "module.":
                name = k[7:]  # remove 'module.' of DataParallel/DistributedDataParallel
            else:
                name = k
            new_weights[name] = v
        if filter_mismatch:
            new_weights, missing_keys = filter_weights_with_wrong_size(
                model, new_weights
            )
            if len(missing_keys) > 0:
                print(
                    "Keys ",
                    missing_keys,
                    " are filtered out due to parameter size mismatch.",
                )
        msg = model.load_state_dict(new_weights, strict=False)
    return msg


def restart_from_checkpoint(
    ckp_path, 
    run_variables=None, 
    load_weights_only=False, 
    rewrite_weights=[],
    **kwargs
):
    """
    Re-start from checkpoint
    """
    if not pathmgr.isfile(ckp_path):
        raise ValueError(f"Checkpoint not found at {ckp_path}")
    print("Found checkpoint at {}".format(ckp_path))

    # open checkpoint file
    if get_world_size() == 1:
        ckp_path = pathmgr.get_local_path(ckp_path)
    with pathmgr.open(ckp_path, "rb") as fb:
        checkpoint = torch.load(fb, map_location="cpu", weights_only=False)

    # key is what to look for in the checkpoint file
    # value is the object to load
    # example: {'state_dict': model}
    for key, value in kwargs.items():
        if value is None:
            continue
        if key in checkpoint and value is not None:
            try:
                msg = load_ddp_state_dict(
                    value, checkpoint[key], key=key, 
                    rewrite_weights=rewrite_weights)
                print(
                    "=> loaded '{}' from checkpoint '{}' with msg {}".format(
                        key, ckp_path, msg
                    )
                )
            except TypeError:
                print(
                    "=> failed to load '{}' from checkpoint: '{}'".format(key, ckp_path)
                )
        else:
            print("=> key '{}' not found in checkpoint: '{}'".format(key, ckp_path))

    # re load variable important for the run
    if not load_weights_only and run_variables is not None:
        for var_name in run_variables:
            if var_name in checkpoint:
                run_variables[var_name] = checkpoint[var_name]


def cosine_scheduler(
    base_value,
    final_value,
    epochs,
    niter_per_ep,
    warmup_iters,
    start_warmup_value=1e-10,
    relu_warmup=False,
):
    print(
        f"cosin scheduler - lr: {base_value}, min_lr: {final_value}, epochs: {epochs}, it_per_epoch: {niter_per_ep}, warmup_iters: {warmup_iters}, startup_warmup: {start_warmup_value}"
    )
    warmup_schedule = np.array([])
    if warmup_iters > 0:
        if warmup_iters > epochs * niter_per_ep:
            raise RuntimeError(
                f"warm iterations number is exceeding the total number iterations. Epoch: {epochs}: Iteration/Epoch: {niter_per_ep}"
            )

        if relu_warmup:
            zero_steps = warmup_iters // 2
            warmup_schedule = np.hstack([
                np.zeros((zero_steps, )),
                np.linspace(start_warmup_value, base_value, warmup_iters-zero_steps)
            ])
        else:
            warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (
        1 + np.cos(np.pi * iters / len(iters))
    )

    schedule = np.concatenate((warmup_schedule, schedule))
    assert (
        len(schedule) == epochs * niter_per_ep
    ), f"Schedule length {len(schedule)} needs to match epoch {epochs} x iteration per epoch {niter_per_ep}"
    return schedule


def linear_scheduler(
    base_value,
    final_value,
    epochs,
    niter_per_ep,
    warmup_iters,
    start_warmup_value=1e-10,
):
    print(
        f"cosin scheduler - lr: {base_value}, min_lr: {final_value}, epochs: {epochs}, it_per_epoch: {niter_per_ep}, warmup_iters: {warmup_iters}, startup_warmup: {start_warmup_value}"
    )
    warmup_schedule = np.array([])
    if warmup_iters > 0:
        if warmup_iters > epochs * niter_per_ep:
            raise RuntimeError(
                f"warm iterations number is exceeding the total number iterations. Epoch: {epochs}: Iteration/Epoch: {niter_per_ep}"
            )

        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = np.linspace(base_value, final_value, len(iters))

    schedule = np.concatenate((warmup_schedule, schedule))
    assert (
        len(schedule) == epochs * niter_per_ep
    ), f"Schedule length {len(schedule)} needs to match epoch {epochs} x iteration per epoch {niter_per_ep}"
    return schedule


def fix_random_seeds(seed=31):
    """
    Fix random seeds.
    """
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=1000, fmt=None):
        if fmt is None:
            fmt = "{median:.8f} ({global_avg:.8f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device="cuda")
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value,
        )


class MetricLogger(object):
    def __init__(self, delimiter="\t", logger=None, tb_writer=None, epoch=0):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter
        self.logger = logger
        self.tb_writer = tb_writer
        self.epoch = epoch
        assert self.logger is not None

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def log_distr(self, distrs, global_step):
        assert self.tb_writer, "Cannot log distr without Tensorboard"
        for group_key, distrs_group in distrs.items():
            for inst_key, inst_values in distrs_group.items():
                if isinstance(inst_values, torch.Tensor) and (inst_values.numel() > 1):
                    self.tb_writer.add_histogram(
                        f"distr-{group_key}/{inst_key}",
                        inst_values,
                        global_step=global_step,
                        bins="tensorflow",
                    )
                else: # Welp, some stats of distribution is a scalar
                    self.tb_writer.add_scalar(
                        f"stats-{group_key}/{inst_key}",
                        inst_values,
                        global_step=global_step,
                        new_style=True,
                    )

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError(
            "'{}' object has no attribute '{}'".format(type(self).__name__, attr)
        )

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append("{}: {}".format(name, str(meter)))
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ""
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt="{avg:.6f}")
        data_time = SmoothedValue(fmt="{avg:.6f}")
        space_fmt = ":" + str(len(str(len(iterable)))) + "d"
        if torch.cuda.is_available():
            log_msg = self.delimiter.join(
                [
                    header,
                    "[{0" + space_fmt + "}/{1}]",
                    "eta: {eta}",
                    "{meters}",
                    "time: {time}",
                    "data: {data}",
                    "max mem: {memory:.0f}",
                    "cpu mem: {cpu_memory:.0f}"
                ]
            )
        else:
            log_msg = self.delimiter.join(
                [
                    header,
                    "[{0" + space_fmt + "}/{1}]",
                    "eta: {eta}",
                    "{meters}",
                    "time: {time}",
                    "data: {data}",
                ]
            )
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if self.tb_writer:
                    for k in self.meters:
                        global_step = self.epoch * len(iterable) + i
                        self.tb_writer.add_scalar(
                            f"train/{k}",
                            self.meters[k].global_avg,
                            global_step=global_step,
                            new_style=True,
                        )
                if torch.cuda.is_available():
                    import psutil

                    self.logger.info(
                        log_msg.format(
                            i,
                            len(iterable),
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                            memory=torch.cuda.max_memory_allocated() / MB,
                            cpu_memory=psutil.Process().memory_info().rss / (1024 ** 3), # GB
                        )
                    )
                else:
                    self.logger.info(
                        log_msg.format(
                            i,
                            len(iterable),
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                        )
                    )
                sys.stdout.flush()
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        self.logger.info(
            "{} Total time: {} ({:.6f} s / it)".format(
                header, total_time_str, total_time / len(iterable)
            ),
        )


def is_main_process():
    return get_rank() == 0


def save_on_master(ckpt, model_path, backup_ckp_epoch=-1, topk=-1, max_to_backup=2):
    if not is_main_process():
        return
    basedir = os.path.dirname(model_path)
    if not pathmgr.isdir(basedir):
        pathmgr.mkdirs(basedir)
    with pathmgr.open(model_path, "wb") as fp:
        torch.save(ckpt, fp)

    if backup_ckp_epoch > 0:
        target_path = os.path.join(basedir, f"ckpt_{backup_ckp_epoch}.pth")
        pathmgr.copy(model_path, target_path, overwrite=True)

    if max_to_backup > 0:
        backup_ckps = pathmgr.ls(basedir)
        backup_ckps = [p for p in backup_ckps if "ckpt_" in p and ".pth" in p]
        if len(backup_ckps) > max_to_backup:
            backup_ckps = sorted(
                backup_ckps, 
                key=lambda p: int(p.split("ckpt_")[1].split(".pth")[0]))
            ckps_to_remove = backup_ckps[:-max_to_backup]
            for ckp_name in ckps_to_remove:
                ckp_path = os.path.join(basedir, ckp_name)
                pathmgr.rm(ckp_path)

    return


def save_image(image, name, is_gamma=False):
    import torchvision

    if len(image.shape) == 5:
        batch_size, im_num, _, h, w = image.shape
        image = image.reshape((batch_size * im_num, -1, h, w))
        nrow = im_num
    else:
        batch_size = image.shape[0]
        nrow = batch_size

    if "mask" not in name.split("/")[-1] and "env" not in name.split("/")[-1]:
        image = 0.5 * (image + 1)

    if is_gamma:
        image = linear_to_srgb(image)

    with pathmgr.open(name, "wb") as fp:
        torchvision.utils.save_image(image, fp, nrow=nrow)

def save_single_depth(depths, name, image_ids=None):
    import cv2

    batch_size = depths.shape[0]
    for n in range(0, batch_size):
        depth = depths[n]
        if len(depth.shape) == 3:
            depth = depth[0]
        depth = 1 - (depth-depth.min())/(depth.max()-depth.min())
        depth = depth.detach().cpu().numpy()
        depth = (255 * np.clip(depth, 0, 1)).astype(np.uint8)
        depth = cv2.applyColorMap(depth, cv2.COLORMAP_INFERNO)

        if image_ids is None:
            im_name = name % n
        else:
            im_name = name % image_ids[n]
        cv2.imwrite(im_name, depth)

def save_single_png(images, name, image_ids=None, is_gamma=False):
    import cv2

    if "mask" not in Path(name).stem:
        images = 0.5 * (images + 1)
    if is_gamma:
        images = linear_to_srgb(images)
    images = images.detach().cpu().numpy()
    images = (255 * np.clip(images, 0, 1)).astype(np.uint8)

    batch_size = images.shape[0]
    for n in range(0, batch_size):
        im = images[n]
        if len(im.shape) == 2:
            im = np.stack([im, im, im], axis=-1)
        else:
            im = im.transpose(1, 2, 0)
            if im.shape[-1] == 3:
                im = np.ascontiguousarray(im[:, :, ::-1])
            else:
                im = np.concatenate([im, im, im], axis=-1)

        buffer = cv2.imencode(".png", im)[1]
        buffer = np.array(buffer).tobytes()
        if image_ids is None:
            im_name = name % n
        else:
            im_name = name % image_ids[n]
        with pathmgr.open(im_name, "wb") as fp:
            fp.write(buffer)

def save_as_mp4(images, name, fps=10, image_ids=None, is_gamma=False, is_depth=False):
    """
    将一批图像保存为MP4视频
    
    Args:
        images: 输入图像张量 [B, C, H, W]
        name: 输出MP4文件名
        fps: 帧率（默认电影级24帧）
        image_ids: 可选的图像ID列表
        is_gamma: 是否进行gamma校正
    """
    import imageio
    import matplotlib.cm as cm

    images = images.detach().cpu().numpy()
    if not is_depth:
        if "mask" not in Path(name).stem:
            images = 0.5 * (images + 1)
        if is_gamma:
            images = linear_to_srgb(images)
        images = (255 * np.clip(images, 0, 1)).astype(np.uint8)
        images = images.transpose(0, 2, 3, 1) 
    else:
        colored = cm.jet(images)[..., :3]
        images = (colored * 255).astype(np.uint8)

    # 准备视频帧列表
    frames = []
    batch_size = images.shape[0]
    for n in range(batch_size):
        im = images[n]
        
        # 确保RGB格式（处理单通道情况）
        if im.shape[-1] == 1:
            im = np.concatenate([im]*3, axis=-1)
        elif im.shape[-1] == 4:
            im = im[..., :3]  # 去除alpha通道
        
        frames.append(im)

    # 保存为MP4（需要安装FFmpeg）
    # with pathmgr.open(name, "wb") as fp:
    imageio.mimsave(
        uri=name,
        ims=frames,
        format='FFMPEG',
        fps=fps,
        codec='libx264',
        output_params=['-pix_fmt', 'yuv420p']  # 兼容性更好的像素格式
    )


def save_as_gif(images, name, fps=10, image_ids=None, is_gamma=False):
    """
    将一批图像保存为GIF动画
    
    Args:
        images: 输入图像张量 [B, C, H, W]
        name: 输出GIF文件名
        fps: 每秒帧数
        image_ids: 可选的图像ID列表
        is_gamma: 是否进行gamma校正
    """
    import imageio

    if "mask" not in name:
        images = 0.5 * (images + 1)
    if is_gamma:
        images = linear_to_srgb(images)
    images = images.detach().cpu().numpy()
    images = (255 * np.clip(images, 0, 1)).astype(np.uint8)

    # 准备GIF帧列表
    frames = []
    batch_size = images.shape[0]
    for n in range(0, batch_size):
        im = images[n].transpose(1, 2, 0)
        # if im.shape[-1] == 3:
        #     im = np.ascontiguousarray(im[:, :, ::-1])  # BGR to RGB
        # else:
        #     im = np.concatenate([im, im, im], axis=-1)
        frames.append(im)

    # 保存为GIF
    with pathmgr.open(name, "wb") as fp:
        imageio.mimsave(fp, frames, format='GIF', fps=fps, loop=0)

def save_single_exr(images, name, image_ids=None, is_gamma=False):
    import cv2

    images = images.detach().cpu().float().numpy()
    images = images.astype(np.float32)

    batch_size = images.shape[0]
    for n in range(0, batch_size):
        im = images[n].transpose(1, 2, 0)
        if im.shape[-1] == 3:
            im = np.ascontiguousarray(im[:, :, ::-1])
        else:
            im = np.concatenate([im, im, im], axis=-1)
        buffer = cv2.imencode(".exr", im)[1]
        buffer = np.array(buffer).tobytes()
        if image_ids is None:
            im_name = name % n
        else:
            im_name = name % image_ids[n]
        with pathmgr.open(im_name, "wb") as fp:
            fp.write(buffer)


def save_depth(depth, depth_mask, name, depth_min=1.5, depth_max=2.5):
    import matplotlib.cm as cm
    import torchvision

    batch_size, im_num, _, h, w = depth.shape
    depth = depth * depth_mask
    depth = np.clip(depth, depth_min, depth_max)

    cmap = cm.get_cmap("jet")
    depth = (depth.reshape(-1) - depth_min) / (depth_max - depth_min)
    depth = depth.detach().cpu().numpy()
    colors = cmap(depth.flatten())[:, :3]
    colors = colors.reshape(batch_size * im_num, h, w, 3)
    colors = colors.transpose(0, 3, 1, 2)
    colors = torch.from_numpy(colors)
    with pathmgr.open(name, "wb") as fp:
        torchvision.utils.save_image(colors, fp, nrow=im_num)


def save_ply(path, xyz, rgb=None, opacity=None, scale=None, rotation=None, mode="splat"):
    from plyfile import PlyData, PlyElement

    assert mode in {"splat", "meshlab"}

    def construct_list_of_attributes():
        l = ["x", "y", "z"]
        # All channels except the 3 DC
        if rgb is not None:
            if mode == "meshlab":
                l = l + ["red", "green", "blue"]
            else:
                l = l + ["r", "g", "b"]
        if opacity is not None:
            l.append("opacity")
        if scale is not None:
            for i in range(3):
                l.append("scale_{}".format(i))
        if rotation is not None:
            for i in range(4):
                l.append("rot_{}".format(i))
        return l

    data = []
    xyz = xyz.to(dtype=torch.float32)
    xyz = xyz.detach().cpu().numpy().astype(np.float32)
    data.append(xyz)

    if rgb is not None:
        rgb = rgb.to(dtype=torch.float32)
        if mode == "meshlab":
            rgb = (rgb * 255).byte().detach().cpu().numpy().astype(np.uint8)
        else:
            rgb = rgb.detach().cpu().numpy().astype(np.float32)
        data.append(rgb)

    if opacity is not None:
        opacity = opacity.to(dtype=torch.float32)
        opacity = opacity.detach().cpu().numpy().astype(np.float32)
        data.append(opacity)

    if scale is not None:
        scale = scale.to(dtype=torch.float32)
        scale = scale.detach().cpu().numpy().astype(np.float32)
        data.append(scale)

    if rotation is not None:
        rotation = rotation.to(dtype=torch.float32)
        rotation = rotation.detach().cpu().numpy().astype(np.float32)
        data.append(rotation)

    if mode == "meshlab":
        rgb_keys = {"red", "green", "blue"}
        dtype_full = [
            (attribute, "f4") if attribute not in rgb_keys else (attribute, "u1")
                for attribute in construct_list_of_attributes()]
    else:
        dtype_full = [(attribute, "f4") for attribute in construct_list_of_attributes()]

    elements = np.empty(xyz.shape[0], dtype=dtype_full)
    attributes = np.concatenate(data, axis=1)
    elements[:] = list(map(tuple, attributes))
    el = PlyElement.describe(elements, "vertex")
    with tempfile.TemporaryDirectory() as temp_dir:
        local_point_path = os.path.join(temp_dir, path.split("/")[-1])
        PlyData([el]).write(local_point_path)
        pathmgr.copy_from_local(local_point_path, path, overwrite=True)


def save_mesh(path, values, N, threshold, radius, save_volume=False):
    import trimesh
    from skimage import measure

    values = values.detach().cpu().numpy()
    values = values.reshape(N, N, N).astype(np.float32)

    if save_volume:
        with pathmgr.open(path + ".npy", "wb") as fp:
            np.save(fp, values)

    try:
        vertices, triangles, normals, _ = measure.marching_cubes(values, threshold)
        print(
            "vertices num %d triangles num %d threshold %.3f"
            % (vertices.shape[0], triangles.shape[0], threshold)
        )

        vertices = vertices / (N - 1.0) * 2 * radius - radius
        mesh = trimesh.Trimesh(
            vertices=vertices, faces=triangles, vertex_normals=normals
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            local_mesh_path = os.path.join(temp_dir, os.path.basename(path))
            mesh.export(local_mesh_path)
            pathmgr.copy_from_local(local_mesh_path, path, overwrite=True)
    except:
        print("Failed to extract mesh.")


def save_o3d_mesh(path, mesh):
    import open3d as o3d

    with tempfile.TemporaryDirectory() as temp_dir:
        local_mesh_path = os.path.join(temp_dir, os.path.basename(path))
        print(local_mesh_path, path)
        o3d.io.write_triangle_mesh(local_mesh_path, mesh)
        pathmgr.copy_from_local(local_mesh_path, path, overwrite=True)


def save_o3d_pcd(path, pcd):
    import open3d as o3d

    with tempfile.TemporaryDirectory() as temp_dir:
        local_pcd_path = os.path.join(temp_dir, os.path.basename(path))
        print(local_pcd_path, path)
        o3d.io.write_point_cloud(local_pcd_path, pcd)
        pathmgr.copy_from_local(local_pcd_path, path, overwrite=True)


def sample_uniform_cameras(
    fov, res, dist=3, theta_range=(45, 135), theta_num=3, phi_num=4
):
    theta_arr = np.linspace(theta_range[0], theta_range[1], theta_num)
    theta_arr = theta_arr / 180.0 * np.pi
    phi_arr = np.linspace(0.0, phi_num - 1.0, phi_num) / phi_num
    phi_arr = np.pi * 2 * phi_arr
    phi_gap = phi_arr[1] - phi_arr[0]

    x_axis = np.array([1, 0, 0], dtype=np.float32)
    y_axis = np.array([0, 1, 0], dtype=np.float32)
    z_axis = np.array([0, 0, 1], dtype=np.float32)

    fov = fov / 180.0 * np.pi
    pixel_x = (np.linspace(0, res - 1, res) + 0.5) / res
    pixel_x = (2 * pixel_x - 1) * np.tan(fov / 2.0)
    pixel_y = (np.linspace(0, res - 1, res) + 0.5) / res
    pixel_y = -(2 * pixel_y - 1) * np.tan(fov / 2.0)
    pixel_x, pixel_y = np.meshgrid(pixel_x, pixel_y)
    pixel_z = -np.ones((res, res), dtype=np.float32)

    k_arr = np.array([fov, fov, 0.5, 0.5], dtype=np.float32)

    cams_arr, rays_o_arr, rays_d_arr = [], [], []
    for n in range(0, theta_num):
        phi_arr += phi_gap / theta_num * n
        for m in range(0, phi_num):
            theta = theta_arr[n]
            phi = phi_arr[m]

            origin = (
                np.sin(theta) * np.cos(phi) * x_axis
                + np.sin(theta) * np.sin(phi) * y_axis
                + np.cos(theta) * z_axis
            )
            origin = origin * dist

            cam_z_axis = origin / np.linalg.norm(origin)
            cam_y_axis = z_axis - np.sum(z_axis * cam_z_axis) * cam_z_axis
            cam_y_axis = cam_y_axis / np.linalg.norm(cam_z_axis)
            cam_x_axis = np.cross(cam_y_axis, cam_z_axis)

            cam = np.eye(4)
            cam[0:3, 0] = cam_x_axis
            cam[0:3, 1] = cam_y_axis
            cam[0:3, 2] = cam_z_axis
            cam[0:3, 3] = origin
            cam = cam.reshape(16)
            cam_line = np.concatenate([cam, k_arr])
            cams_arr.append(cam_line)

            rays_o = np.ones([res, res, 1]) * origin.reshape(1, 1, 3)
            rays_d = (
                cam_x_axis.reshape(1, 1, 3) * pixel_x[:, :, None]
                + cam_y_axis.reshape(1, 1, 3) * pixel_y[:, :, None]
                + cam_z_axis.reshape(1, 1, 3) * pixel_z[:, :, None]
            )
            rays_d = rays_d / np.sqrt(np.sum(rays_d * rays_d, axis=-1, keepdims=True))
            rays_o_arr.append(rays_o)
            rays_d_arr.append(rays_d)

    cams_arr = np.stack(cams_arr, axis=0)
    rays_o_arr = np.stack(rays_o_arr, axis=0)
    rays_d_arr = np.stack(rays_d_arr, axis=0)

    return cams_arr, rays_o_arr, rays_d_arr


def sample_oriented_points(preds, threshold=0.4):
    points = preds["points"]
    normals = preds["normal"]
    opacity = preds["mask"]

    opacity = opacity.reshape(-1)
    points = points.reshape(-1, 3)[opacity > threshold, :]
    normals = normals.reshape(-1, 3)[opacity > threshold, :]

    return points, normals


def filter_points_using_input_mask(points, cams, masks, fov):
    batch_size, _, height, width = masks.shape
    point_num = points.shape[0]

    fov = fov / 180.0 * np.pi

    # Enlarge mask a bit
    masks = nn.functional.adaptive_avg_pool2d(masks, (height // 8, width // 8))
    masks = nn.functional.interpolate(masks, (height, width), mode="bilinear")

    index_prod = torch.ones(point_num, dtype=torch.uint8, device=points.device).bool()
    for n in range(0, batch_size):
        cam = cams[n, :][:16].reshape(4, 4)
        diff = points - cam[:3, 3].reshape(1, 3)
        z = torch.sum(diff * cam[:3, 2].reshape(1, 3), dim=-1)
        x = torch.sum(diff * cam[:3, 0].reshape(1, 3), dim=-1)
        y = torch.sum(diff * cam[:3, 1].reshape(1, 3), dim=-1)
        x_ = (x / -z) / np.tan(fov / 2.0)
        y_ = (y / z) / np.tan(fov / 2.0)

        grid = torch.stack([x_, y_], dim=-1).reshape(1, 1, point_num, 2)
        index = nn.functional.grid_sample(masks, grid)
        index = index.reshape(-1) > 0
        index_prod = torch.logical_and(index, index_prod)

    return index_prod


def load_ply(path, device: str = "cpu"):
    from plyfile import PlyData

    
    if path.endswith(".pth"):
        gaus = torch.load(path)
        return {
            "xyz": gaus["xyz"].to(device),
            "rgb": gaus["color"].to(device),
            "opacity": gaus["opacity"].to(device),
            "scale": gaus["scaling"].to(device),
            "rotation": gaus["rot"].to(device),
            "sh_degree": None,
        }


    plydata = PlyData.read(path)

    xyz = np.stack(
        (
            np.asarray(plydata.elements[0]["x"]),
            np.asarray(plydata.elements[0]["y"]),
            np.asarray(plydata.elements[0]["z"]),
        ),
        axis=1,
    )
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    properties = [x.name for x in plydata.elements[0].properties]
    # this is to unify gaussian results from two source: lrm's and official gaussian splatting's
    mode = None
    if "r" in properties:
        mode = "lrm"
        rgb = np.stack(
            (
                np.asarray(plydata.elements[0]["r"]),
                np.asarray(plydata.elements[0]["g"]),
                np.asarray(plydata.elements[0]["b"]),
            ),
            axis=1,
        )
        sh_degree = None
    else:
        mode = "gau"
        # For simplicity, we only use the bash color
        f_dc = np.stack(
            (
                np.asarray(plydata.elements[0]["f_dc_0"]),
                np.asarray(plydata.elements[0]["f_dc_1"]),
                np.asarray(plydata.elements[0]["f_dc_2"]),
            ),
            axis=1,
        )
        # f_rest = []
        # for property in properties:
        #     if property.startswith("f_rest"):
        #         f_rest.append(np.asarray(plydata.elements[0][property]))
        # f_rest = np.stack(f_rest, axis=1).reshape(f_dc.shape[0], 3, -1).transpose(0, 2, 1)
        # rgb = np.concatenate((f_dc, f_rest), axis=1)
        # sh_degree = int(np.sqrt(f_rest.shape[1] + 1) - 1)
        rgb = np.maximum(SH2RGB(f_dc), 0.0)
        sh_degree = None

    opacity = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]
    scale_names = [
        p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")
    ]
    scale_names = sorted(scale_names, key=lambda x: int(x.split("_")[-1]))
    scale = np.zeros((xyz.shape[0], len(scale_names)))
    for idx, attr_name in enumerate(scale_names):
        scale[:, idx] = np.asarray(plydata.elements[0][attr_name])

    rot_names = [
        p.name for p in plydata.elements[0].properties if p.name.startswith("rot")
    ]
    rot_names = sorted(rot_names, key=lambda x: int(x.split("_")[-1]))
    rotation = np.zeros((xyz.shape[0], len(rot_names)))
    for idx, attr_name in enumerate(rot_names):
        rotation[:, idx] = np.asarray(plydata.elements[0][attr_name])

    xyz = torch.from_numpy(xyz.astype(np.float32)).to(device)
    rgb = torch.from_numpy(rgb.astype(np.float32)).to(device)

    if mode == "lrm":

        opacity = torch.from_numpy(opacity.astype(np.float32)).to(device)
        scale = torch.from_numpy(scale.astype(np.float32)).to(device)
        rotation = torch.from_numpy(rotation.astype(np.float32)).to(device)
    else:
        opacity = torch.sigmoid(torch.from_numpy(opacity.astype(np.float32)).to(device))
        scale = torch.exp(torch.from_numpy(scale.astype(np.float32)).to(device))
        rotation = torch.nn.functional.normalize(
            torch.from_numpy(rotation.astype(np.float32)).to(device)
        )

    return {
        "xyz": xyz,
        "rgb": rgb,
        "opacity": opacity,
        "scale": scale,
        "rotation": rotation,
        "sh_degree": sh_degree,
    }


def get_params_group_single_model(
    args, model, freeze_backbone=False, freeze_transformer=False
):
    upsampler = []
    regularized = []
    not_regularized = []
    deformation_group = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if (args.deform_lr_multiplier > 0.0):
            is_keyword_match = False
            for keyword in args.deform_param_group_keywords.split(","):
                if keyword in name:
                    is_keyword_match = True
                    break
            if is_keyword_match:
                deformation_group.append(param)
                continue
                    
        # we do not regularize biases nor Norm parameters
        if "upsampler" in name:
            upsampler.append(param)
        else:
            if name.endswith(".bias") or len(param.shape) == 1:
                not_regularized.append(param)
            elif len(name.split(".")) > 3 and "norm" in name.split(".")[2]:
                not_regularized.append(param)
            else:
                regularized.append(param)

    if freeze_backbone:
        lr_t = 0
        lr_c = 0
    else:
        if freeze_transformer:
            lr_t = 0
            lr_c = args.lr
        else:
            lr_t = args.lr
            lr_c = args.lr

    return [
        {"params": regularized, "weight_decay": args.weight_decay, "lr": lr_t},
        {"params": not_regularized, "weight_decay": 0.0, "lr": lr_t},
        {"params": upsampler, "weight_decay": args.weight_decay, "lr": lr_c},
        {"params": deformation_group, "weight_decay": args.weight_decay, "lr": lr_c * args.deform_lr_multiplier},
    ]


def get_params_groups(args, **kwargs):
    params_groups = []
    for key, value in kwargs.items():
        if value is None:
            continue
        if "mvencoder" in key or "tridecoder" in key:
            params_groups += get_params_group_single_model(
                args,
                value,
                freeze_backbone=args.freeze_backbone,
                freeze_transformer=args.freeze_transformer,
            )
        else:
            params_groups += get_params_group_single_model(args, value)
    return params_groups


def has_batchnorms(model):
    bn_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)
    for name, module in model.named_modules():
        if isinstance(module, bn_types):
            return True
    return False


def create_video_cameras(radius, frame_num, res, elevation=20, fov=60, init_cam=None):
    from ..data_loader.utils import compute_rays

    fov = fov / 180.0 * np.pi
    dist = radius / np.sin(fov / 2.0) * 1.2
    theta = elevation / 180.0 * np.pi
    x_axis = np.array([1.0, 0, 0], dtype=np.float32)
    y_axis = np.array([0, 1.0, 0], dtype=np.float32)
    z_axis = np.array([0, 0, 1.0], dtype=np.float32)

    if init_cam is not None:
        init_cam = init_cam.detach().cpu().numpy()
        init_cam = init_cam.transpose(1, 0)
        inv = np.eye(4, dtype=np.float32)
        inv[0:3, 0:3] = init_cam

    camera_arr, rays_o_arr, rays_d_arr, rays_d_un_arr = [], [], [], []
    for n in range(0, frame_num):
        phi = float(n) / frame_num * np.pi * 2
        origin = (
            np.cos(theta) * np.cos(phi) * x_axis
            + np.cos(theta) * np.sin(phi) * y_axis
            + np.sin(theta) * z_axis
        )
        origin = origin * dist

        target = np.array([0, 0, 0], dtype=np.float32)
        up = np.array([0, 0, 1], dtype=np.float32)
        cam_z_axis = (origin - target) / np.linalg.norm(origin - target)
        cam_y_axis = up - np.sum(cam_z_axis * up) * cam_z_axis
        cam_y_axis = cam_y_axis / np.linalg.norm(cam_y_axis)
        cam_x_axis = np.cross(cam_y_axis, cam_z_axis)

        extrinsic = np.eye(4, dtype=np.float32)
        extrinsic[:3, 0] = cam_x_axis
        extrinsic[:3, 1] = cam_y_axis
        extrinsic[:3, 2] = cam_z_axis
        extrinsic[:3, 3] = origin
        if init_cam is not None:
            extrinsic = np.matmul(inv, extrinsic)

        camera = np.zeros(20, dtype=np.float32)
        camera[0:16] = extrinsic.reshape(-1)
        camera[16:20] = np.array([fov, fov, 0.5, 0.5])

        rays_o, rays_d, rays_d_un = compute_rays(fov, extrinsic, res)

        camera_arr.append(camera)
        rays_o_arr.append(rays_o)
        rays_d_arr.append(rays_d)
        rays_d_un_arr.append(rays_d_un)

    camera_arr = np.stack(camera_arr, axis=0)[None, :, :].astype(np.float32)
    rays_o_arr = np.stack(rays_o_arr, axis=0)[None, :, :, :].astype(np.float32)
    rays_d_arr = np.stack(rays_d_arr, axis=0)[None, :, :, :].astype(np.float32)
    rays_d_un_arr = np.stack(rays_d_un_arr, axis=0)[None, :, :, :].astype(np.float32)

    return camera_arr, rays_o_arr, rays_d_arr, rays_d_un_arr


def parse_tuple_args(s):
    if s is None: return None
    toks = s.split(",")
    if len(toks) == 1:
        return float(s) if "." in s else int(s)
    else:
        return tuple(
            float(tok) if "." in tok else int(tok) 
                for tok in toks)
    
def SH2RGB(sh):
    C0 = 0.28209479177387814
    return sh * C0 + 0.5

def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))


def replace_outliers(data, threshold=3.0, method='max'):
    """
    检测并替换3D数组中的离群值
    
    参数:
        data: 输入数组，形状为(N, H, W)
        threshold: 离群值检测的阈值（默认3.0，即3σ）
        method: 替换方法，可选'max'（用非离群最大值替换）或'mean'（用非离群均值替换）
    
    返回:
        处理后的数组
    """
    # 展平数组以便计算统计量
    flattened = data.reshape(-1)
    
    # 计算统计指标
    mean = np.mean(flattened)
    std = np.std(flattened)
    
    # 确定离群值边界
    lower_bound = mean - threshold * std
    upper_bound = mean + threshold * std
    
    # 创建离群值掩码
    outlier_mask = (data < lower_bound) | (data > upper_bound)
    
    # 获取非离群值
    inliers = data[~outlier_mask]
    
    if len(inliers) == 0:
        return data  # 如果没有非离群值，返回原数组
    
    # 确定替换值
    if method == 'max':
        replacement = np.max(inliers)
    elif method == 'mean':
        replacement = np.mean(inliers)
    else:
        raise ValueError("替换方法必须是'max'或'mean'")
    
    # 替换离群值
    result = data.copy()
    result[outlier_mask] = replacement
    
    return result
