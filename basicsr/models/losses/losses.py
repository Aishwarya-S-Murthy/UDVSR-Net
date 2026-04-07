import torch
from torch import nn as nn
from torch.nn import functional as F
import numpy as np
import math

from basicsr.models.losses.loss_util import weighted_loss

_reduction_modes = ['none', 'mean', 'sum']


@weighted_loss
def l1_loss(pred, target):
    return F.l1_loss(pred, target, reduction='none')


@weighted_loss
def mse_loss(pred, target):
    return F.mse_loss(pred, target, reduction='none')


# @weighted_loss
# def charbonnier_loss(pred, target, eps=1e-12):
#     return torch.sqrt((pred - target)**2 + eps)


class L1Loss(nn.Module):
    """L1 (mean absolute error, MAE) loss.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(L1Loss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * l1_loss(
            pred, target, weight, reduction=self.reduction)

class MSELoss(nn.Module):
    """MSE (L2) loss.

    Args:
        loss_weight (float): Loss weight for MSE loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(MSELoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * mse_loss(
            pred, target, weight, reduction=self.reduction)

class PSNRLoss(nn.Module):

    def __init__(self, loss_weight=1.0, reduction='mean', toY=False):
        super(PSNRLoss, self).__init__()
        assert reduction == 'mean'
        self.loss_weight = loss_weight
        self.scale = 10 / np.log(10)
        self.toY = toY
        self.coef = torch.tensor([65.481, 128.553, 24.966]).reshape(1, 3, 1, 1)
        self.first = True

    def forward(self, pred, target):
        assert len(pred.size()) == 4
        if self.toY:
            if self.first:
                self.coef = self.coef.to(pred.device)
                self.first = False

            pred = (pred * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.
            target = (target * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.

            pred, target = pred / 255., target / 255.
            pass
        assert len(pred.size()) == 4

        return self.loss_weight * self.scale * torch.log(((pred - target) ** 2).mean(dim=(1, 2, 3)) + 1e-8).mean()

class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, loss_weight=1.0, reduction='mean', eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        # loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps*self.eps)))
        return loss
    
class FFTLoss(nn.Module):
    """FFT (Frequency Domain) Loss.
    
    Transforms predictions and targets into the frequency domain and calculates
    the L1 distance between their amplitude spectrums.
    
    Args:
        loss_weight (float): Loss weight for FFT loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """
    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(FFTLoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W) or (N, T, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W) or (N, T, C, H, W). Ground truth tensor.
            weight (Tensor, optional): Element-wise weights. Default: None.
        """
        # If input is a video tensor (B, T, C, H, W), flatten T into B for the 2D FFT
        if pred.dim() == 5:
            b, t, c, h, w = pred.shape
            pred_fft_input = pred.view(b * t, c, h, w)
            target_fft_input = target.view(b * t, c, h, w)
        else:
            pred_fft_input = pred
            target_fft_input = target

        # Transform to frequency domain (use norm='ortho' to match your architecture)
        pred_fft = torch.fft.rfft2(pred_fft_input, norm='ortho')
        target_fft = torch.fft.rfft2(target_fft_input, norm='ortho')
        
        # We compute the loss on the Amplitude (Magnitude) of the spectrum
        pred_amp = torch.abs(pred_fft)
        target_amp = torch.abs(target_fft)
        
        # Calculate L1 loss between amplitudes
        loss = l1_loss(pred_amp, target_amp, weight, reduction=self.reduction)
        
        return self.loss_weight * loss
# ==========================================================
# NEW: SSIM Loss Implementation
# ==========================================================
def create_window(window_size, channel=1):
    def gaussian(window_size, sigma):
        gauss = torch.Tensor([math.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
        return gauss/gauss.sum()
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

def calc_ssim_loss(pred, target, window, window_size, reduction='mean'):
    padding = window_size // 2
    channel = pred.size(1)
    
    mu1 = F.conv2d(pred, window, padding=padding, groups=channel)
    mu2 = F.conv2d(target, window, padding=padding, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(pred * pred, window, padding=padding, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(target * target, window, padding=padding, groups=channel) - mu2_sq
    sigma12 = F.conv2d(pred * target, window, padding=padding, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2.0 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    # 1.0 - SSIM to act as a loss to minimize
    loss_map = 1.0 - ssim_map
    
    if reduction == 'mean':
        return loss_map.mean()
    elif reduction == 'sum':
        return loss_map.sum()
    else:
        return loss_map

class SSIMLoss(nn.Module):
    """SSIM loss.
    
    Args:
        loss_weight (float): Loss weight for SSIM loss. Default: 1.0.
        window_size (int): Window size for SSIM calculation. Default: 11.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """
    def __init__(self, loss_weight=1.0, window_size=11, reduction='mean'):
        super(SSIMLoss, self).__init__()
        if reduction not in _reduction_modes:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')
        self.loss_weight = loss_weight
        self.window_size = window_size
        self.reduction = reduction
        self.window = None

    def forward(self, pred, target, weight=None, **kwargs):
        # Handle Video 5D Tensor (Flatten Time dimension into Batch)
        if pred.dim() == 5:
            b, t, c, h, w = pred.shape
            pred_input = pred.view(b * t, c, h, w)
            target_input = target.view(b * t, c, h, w)
        else:
            pred_input = pred
            target_input = target

        _, channel, _, _ = pred_input.size()
        
        # Initialize or update Gaussian window dynamically
        if self.window is None or self.window.size(0) != channel:
            self.window = create_window(self.window_size, channel).to(pred_input.device)
            
        loss = calc_ssim_loss(pred_input, target_input, self.window, self.window_size, self.reduction)
        
        return self.loss_weight * loss