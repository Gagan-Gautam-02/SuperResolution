import torch
import torch.nn.functional as F
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import cv2
from typing import Dict, Tuple

class ImageQualityMetrics:
    """Comprehensive image quality assessment metrics"""
    
    @staticmethod
    def mse(img1: np.ndarray, img2: np.ndarray) -> float:
        """Mean Squared Error"""
        return np.mean((img1 - img2) ** 2)
    
    @staticmethod
    def rmse(img1: np.ndarray, img2: np.ndarray) -> float:
        """Root Mean Squared Error"""
        return np.sqrt(ImageQualityMetrics.mse(img1, img2))
    
    @staticmethod
    def psnr(img1: np.ndarray, img2: np.ndarray, data_range: float = 1.0) -> float:
        """Peak Signal-to-Noise Ratio"""
        return psnr(img1, img2, data_range=data_range)
    
    @staticmethod
    def ssim(img1: np.ndarray, img2: np.ndarray, data_range: float = 1.0) -> float:
        """Structural Similarity Index"""
        if len(img1.shape) == 3:
            return ssim(img1, img2, data_range=data_range, multichannel=True, channel_axis=2)
        else:
            return ssim(img1, img2, data_range=data_range)
    
    @staticmethod
    def calculate_all_metrics(pred: np.ndarray, target: np.ndarray) -> Dict[str, float]:
        """Calculate all full-reference metrics"""
        metrics = {}
        
        # Ensure images are in the same range
        pred = np.clip(pred, 0, 1)
        target = np.clip(target, 0, 1)
        
        metrics['mse'] = ImageQualityMetrics.mse(pred, target)
        metrics['rmse'] = ImageQualityMetrics.rmse(pred, target)
        metrics['psnr'] = ImageQualityMetrics.psnr(pred, target)
        metrics['ssim'] = ImageQualityMetrics.ssim(pred, target)
        
        return metrics

class LossFunction:
    """Loss functions for training super-resolution models"""
    
    @staticmethod
    def l1_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """L1 Loss"""
        return F.l1_loss(pred, target)
    
    @staticmethod
    def l2_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """L2 Loss (MSE)"""
        return F.mse_loss(pred, target)
    
    @staticmethod
    def ssim_loss(pred: torch.Tensor, target: torch.Tensor, window_size: int = 11) -> torch.Tensor:
        """SSIM Loss"""
        def gaussian_window(size, sigma=1.5):
            coords = torch.arange(size, dtype=torch.float32)
            coords -= size // 2
            g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
            g /= g.sum()
            return g.unsqueeze(0).unsqueeze(0) * g.unsqueeze(0).unsqueeze(1)
        
        def ssim_map(img1, img2, window, window_size, channel):
            mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
            mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)
            
            mu1_sq = mu1.pow(2)
            mu2_sq = mu2.pow(2)
            mu1_mu2 = mu1 * mu2
            
            sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size//2, groups=channel) - mu1_sq
            sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size//2, groups=channel) - mu2_sq
            sigma12 = F.conv2d(img1 * img2, window, padding=window_size//2, groups=channel) - mu1_mu2
            
            C1 = 0.01 ** 2
            C2 = 0.03 ** 2
            
            ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
            
            return ssim_map
        
        channel = pred.size(1)
        window = gaussian_window(window_size).expand(channel, 1, window_size, window_size).to(pred.device)
        
        ssim_val = ssim_map(pred, target, window, window_size, channel)
        return 1 - ssim_val.mean()
    
    @staticmethod
    def combined_loss(pred: torch.Tensor, target: torch.Tensor, 
                     l1_weight: float = 1.0, ssim_weight: float = 0.1) -> torch.Tensor:
        """Combined L1 and SSIM loss"""
        l1 = LossFunction.l1_loss(pred, target)
        ssim = LossFunction.ssim_loss(pred, target)
        
        return l1_weight * l1 + ssim_weight * ssim

class PerceptualLoss:
    """Perceptual loss using pre-trained VGG network"""
    
    def __init__(self, device='cpu'):
        import torchvision.models as models
        
        vgg = models.vgg19(pretrained=True).features
        self.feature_extractor = vgg[:36].to(device).eval()
        
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
    
    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_features = self.feature_extractor(pred)
        target_features = self.feature_extractor(target)
        
        return F.mse_loss(pred_features, target_features)
