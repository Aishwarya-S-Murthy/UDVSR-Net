import importlib
import torch
from collections import OrderedDict
from copy import deepcopy
from os import path as osp
from tqdm import tqdm

from basicsr.models.archs import define_network
from basicsr.models.base_model import BaseModel
from basicsr.utils import get_root_logger, imwrite, tensor2img

loss_module = importlib.import_module('basicsr.models.losses')
metric_module = importlib.import_module('basicsr.metrics')

import os
import random
import numpy as np
import cv2
import torch.nn.functional as F
from functools import partial


class Mixing_Augment:
    def __init__(self, mixup_beta, use_identity, device):
        self.dist = torch.distributions.beta.Beta(torch.tensor([mixup_beta]), torch.tensor([mixup_beta]))
        self.device = device
        self.use_identity = use_identity
        self.augments = [self.mixup]

    def mixup(self, target, input_):
        """
        Mixup augmentation for video sequences
        target: (B, T, H, W, C) or (B, T, C, H, W)
        input_: (B, T, H, W, C) or (B, T, C, H, W)
        """
        lam = self.dist.rsample((1, 1)).item()
        
        r_index = torch.randperm(target.size(0)).to(self.device)
        
        target = lam * target + (1 - lam) * target[r_index, :]
        input_ = lam * input_ + (1 - lam) * input_[r_index, :]
        
        return target, input_

    def __call__(self, target, input_):
        if self.use_identity:
            augment = random.randint(0, len(self.augments))
            if augment < len(self.augments):
                target, input_ = self.augments[augment](target, input_)
        else:
            augment = random.randint(0, len(self.augments) - 1)
            target, input_ = self.augments[augment](target, input_)
        return target, input_


class VideoSRModel(BaseModel):
    """Video Super-Resolution model for processing video sequences."""

    def __init__(self, opt):
        super(VideoSRModel, self).__init__(opt)

        # define network
        # Modified initialization to handle test mode safely
        self.mixing_flag = False
        if 'train' in self.opt and 'mixing_augs' in self.opt['train']:
                self.mixing_flag = self.opt['train']['mixing_augs'].get('mixup', False)
                if self.mixing_flag:
                    mixup_beta = self.opt['train']['mixing_augs'].get('mixup_beta', 1.2)
                    use_identity = self.opt['train']['mixing_augs'].get('use_identity', False)
                    self.mixing_augmentation = Mixing_Augment(mixup_beta, use_identity, self.device)

        self.net_g = define_network(deepcopy(opt['network_g']))
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            self.load_network(self.net_g, load_path,
                              self.opt['path'].get('strict_load_g', True), 
                              param_key=self.opt['path'].get('param_key', 'params'))

        if self.is_train:
            self.init_training_settings()

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']

        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(
                f'Use Exponential Moving Average with decay: {self.ema_decay}')
            # define network net_g with Exponential Moving Average (EMA)
            self.net_g_ema = define_network(self.opt['network_g']).to(self.device)
            # load pretrained model
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path,
                                  self.opt['path'].get('strict_load_g', True), 
                                  'params_ema')
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()

        # define losses
        if train_opt.get('pixel_opt'):
            pixel_type = train_opt['pixel_opt'].pop('type')
            cri_pix_cls = getattr(loss_module, pixel_type)
            self.cri_pix = cri_pix_cls(**train_opt['pixel_opt']).to(self.device)
        else:
            raise ValueError('pixel loss are None.')
        
        # NEW: Initialize FFT loss if it exists in the YAML
        if train_opt.get('fft_opt'):
            fft_type = train_opt['fft_opt'].pop('type')
            cri_fft_cls = getattr(loss_module, fft_type)
            self.cri_fft = cri_fft_cls(**train_opt['fft_opt']).to(self.device)
        else:
            self.cri_fft = None

        # =======================================================
        # NEW: Initialize SSIM loss if it exists in the YAML
        # =======================================================
        if train_opt.get('ssim_opt'):
            ssim_type = train_opt['ssim_opt'].pop('type')
            cri_ssim_cls = getattr(loss_module, ssim_type)
            self.cri_ssim = cri_ssim_cls(**train_opt['ssim_opt']).to(self.device)
        else:
            self.cri_ssim = None

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []

        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')

        optim_type = train_opt['optim_g'].pop('type')
        if optim_type == 'Adam':
            self.optimizer_g = torch.optim.Adam(optim_params, **train_opt['optim_g'])
        elif optim_type == 'AdamW':
            self.optimizer_g = torch.optim.AdamW(optim_params, **train_opt['optim_g'])
        else:
            raise NotImplementedError(
                f'optimizer {optim_type} is not supperted yet.')
        self.optimizers.append(self.optimizer_g)

    def feed_train_data(self, data):
        """
        Feed training data for video SR
        Expected data format:
        - lq: (B, T, H, W, C) or (B, T, C, H, W) - low quality video
        - gt: (B, T, H_hr, W_hr, C) or (B, T, C, H_hr, W_hr) - high quality video
        """
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)

        if self.mixing_flag:
            self.gt, self.lq = self.mixing_augmentation(self.gt, self.lq)

    def feed_data(self, data):
        """Feed data during validation/testing"""
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        preds = self.net_g(self.lq) # Returns (B, T, C, H, W)
    
        if not isinstance(preds, list):
            preds = [preds]

        self.output = preds[-1]
        loss_dict = OrderedDict()
        l_pix = 0.
    
        for pred in preds:
            # --------------------------------------------------------------------
            # FIXED: Removed .permute() because model now outputs (B, T, C, H, W)
            # Both pred and self.gt are now (B, T, C, H, W), so we compare directly.
            # --------------------------------------------------------------------
            # Calculate Spatial L1 Loss
            loss_spatial = self.cri_pix(pred, self.gt)
            l_pix += loss_spatial
            
            # NEW: Calculate Frequency FFT Loss and add it to the total loss
            if self.cri_fft is not None:
                loss_freq = self.cri_fft(pred, self.gt)
                l_pix += loss_freq
                # Log the individual fft loss value (optional but helpful for TensorBoard)
                loss_dict['l_fft'] = loss_freq 
            
            # =======================================================
            # NEW: Calculate SSIM Loss
            # =======================================================
            if self.cri_ssim is not None:
                loss_ssim = self.cri_ssim(pred, self.gt)
                l_pix += loss_ssim
                loss_dict['l_ssim'] = loss_ssim

        loss_dict['l_pix'] = l_pix # This now represents the TOTAL loss (Spatial + FFT)
        l_pix.backward()
        if self.opt['train']['use_grad_clip']:
            torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), 0.01)
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def pad_test(self, window_size):
        """Pad the video frames for testing"""
        scale = self.opt.get('scale', 1)
        mod_pad_h, mod_pad_w = 0, 0
        
        # lq shape: (B, T, H, W, C) or (B, T, C, H, W)
        if len(self.lq.size()) == 5:
            if self.lq.size(-1) == 3 or self.lq.size(-1) == 1:  # (B, T, H, W, C)
                _, _, h, w, _ = self.lq.size()
                # Convert to (B, T, C, H, W) for padding
                img = self.lq.permute(0, 1, 4, 2, 3)
            else:  # (B, T, C, H, W)
                _, _, _, h, w = self.lq.size()
                img = self.lq
        else:
            raise ValueError(f"Unexpected input shape: {self.lq.size()}")
        
        if h % window_size != 0:
            mod_pad_h = window_size - h % window_size
        if w % window_size != 0:
            mod_pad_w = window_size - w % window_size
        
        # Pad each frame in the sequence
        b, t, c, h, w = img.size()
        img = img.reshape(b * t, c, h, w)
        img = F.pad(img, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        img = img.reshape(b, t, c, h + mod_pad_h, w + mod_pad_w)
        
        # Convert back to (B, T, H, W, C) if needed
        if self.lq.size(-1) == 3 or self.lq.size(-1) == 1:
            img = img.permute(0, 1, 3, 4, 2)
        
        self.nonpad_test(img)
        
        # Remove padding from output
        if self.output.size(-1) == 3 or self.output.size(-1) == 1:  # (B, T, H, W, C)
            _, _, h_out, w_out, _ = self.output.size()
            self.output = self.output[:, :, 0:h_out - mod_pad_h * scale, 0:w_out - mod_pad_w * scale, :]
        else:  # (B, T, C, H, W)
            _, _, _, h_out, w_out = self.output.size()
            self.output = self.output[:, :, :, 0:h_out - mod_pad_h * scale, 0:w_out - mod_pad_w * scale]

    def nonpad_test(self, img=None):
        """Test without padding"""
        if img is None:
            img = self.lq
            
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                pred = self.net_g_ema(img)
            if isinstance(pred, list):
                pred = pred[-1]
            self.output = pred
        else:
            self.net_g.eval()
            with torch.no_grad():
                pred = self.net_g(img)
            if isinstance(pred, list):
                pred = pred[-1]
            self.output = pred
            self.net_g.train()

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img, rgb2bgr, use_image):
        if os.environ.get('LOCAL_RANK', '0') == '0':
            return self.nondist_validation(dataloader, current_iter, tb_logger, save_img, rgb2bgr, use_image)
        else:
            return 0.

    def nondist_validation(self, dataloader, current_iter, tb_logger,
                           save_img, rgb2bgr, use_image):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        if with_metrics:
            self.metric_results = {
                metric: 0
                for metric in self.opt['val']['metrics'].keys()
            }

        window_size = self.opt['val'].get('window_size', 0)

        if window_size:
            test = partial(self.pad_test, window_size)
        else:
            test = self.nonpad_test

        cnt = 0

        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            video_folder = val_data['folder'][0]
            self.feed_data(val_data)
            test()

            visuals = self.get_current_visuals()
            
            # Process video output - save middle frame or all frames
            # Output shape: (B, T, H, W, C) or (B, T, C, H, W)
            # Process video output
            output_video = visuals['result']
            num_frames = output_video.size(1)
            
            # Extract the actual starting index from the dataset (e.g., "0/100" -> 0)
            start_idx = int(val_data['idx'][0].split('/')[0])

            if save_img:
                video_save_dir = osp.join(self.opt['path']['visualization'], video_folder)
                os.makedirs(video_save_dir, exist_ok=True)

            if with_metrics:
                opt_metric = deepcopy(self.opt['val']['metrics'])

            # Loop through ALL frames in the 8-frame chunk
            for t in range(num_frames):
                abs_frame_idx = start_idx + t 
                
                # Convert specific frame to numpy image
                sr_img_t = tensor2img([output_video[:, t]], rgb2bgr=rgb2bgr)
                if 'gt' in visuals:
                    gt_img_t = tensor2img([visuals['gt'][:, t]], rgb2bgr=rgb2bgr)

                # 1. Save Image (Naming it sequentially: frame_000, frame_001, etc.)
                if save_img:
                    # Matches standard formats for easy ffmpeg compilation
                    frame_path = osp.join(video_save_dir, f'frame_{abs_frame_idx:04d}_pred.png')
                    imwrite(sr_img_t, frame_path)

                # 2. Accumulate Metrics for EVERY frame
                if with_metrics and 'gt' in visuals:
                    if use_image:
                        for name, opt_ in opt_metric.items():
                            metric_type = opt_.get('type') # Use get() instead of pop() so we don't destroy the dict in the loop
                            # Create a clean kwargs dict without 'type'
                            kwargs = {k: v for k, v in opt_.items() if k != 'type'}
                            self.metric_results[name] += getattr(
                                metric_module, metric_type)(sr_img_t, gt_img_t, **kwargs)
                    else:
                        for name, opt_ in opt_metric.items():
                            metric_type = opt_.get('type')
                            kwargs = {k: v for k, v in opt_.items() if k != 'type'}
                            self.metric_results[name] += getattr(
                                metric_module, metric_type)(
                                    output_video[:, t], 
                                    visuals['gt'][:, t], 
                                    **kwargs
                                )
                
                # Increment counter for every single frame evaluated
                cnt += 1

            # Clean up GPU memory
            if 'gt' in visuals:
                del self.gt
            del self.lq
            del self.output
            torch.cuda.empty_cache()

        current_metric = 0.
        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= cnt
                current_metric = self.metric_results[metric]

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)
        
        return current_metric

    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        log_str = f'Validation {dataset_name},\t'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}'
        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{metric}', value, current_iter)

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
        return out_dict

    def save(self, epoch, current_iter):
        if self.ema_decay > 0:
            self.save_network([self.net_g, self.net_g_ema],
                              'net_g',
                              current_iter,
                              param_key=['params', 'params_ema'])
        else:
            self.save_network(self.net_g, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter)