import cv2
import numpy as np
from scipy import ndimage
from skimage import transform, registration
import torch
import torch.nn.functional as F
from typing import Tuple

class ImageRegistration:
    """Image registration for aligning dual low-resolution images"""
    
    @staticmethod
    def estimate_shift(img1: np.ndarray, img2: np.ndarray) -> Tuple[float, float]:
        """Estimate sub-pixel shift between two images"""
        # Convert to grayscale if needed
        if len(img1.shape) == 3:
            img1_gray = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
            img2_gray = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
        else:
            img1_gray, img2_gray = img1, img2
        
        # Phase correlation for sub-pixel registration
        shift, error, diffphase = registration.phase_cross_correlation(
            img1_gray, img2_gray, upsample_factor=100
        )
        
        return shift
    
    @staticmethod
    def register_images(img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
        """Register img2 to img1"""
        shift = ImageRegistration.estimate_shift(img1, img2)
        
        # Apply shift correction
        registered_img2 = ndimage.shift(img2, shift, order=1)
        
        return registered_img2

class DegradationModel:
    """Model degradation functions for classical super-resolution"""
    
    def __init__(self, blur_kernel_size=3, noise_level=0.01):
        self.blur_kernel_size = blur_kernel_size
        self.noise_level = noise_level
    
    def apply_degradation(self, hr_image: np.ndarray) -> np.ndarray:
        """Apply degradation to simulate low-resolution image"""
        # Blur
        blurred = cv2.GaussianBlur(
            hr_image, 
            (self.blur_kernel_size, self.blur_kernel_size), 
            0
        )
        
        # Downsample
        h, w = blurred.shape[:2]
        downsampled = cv2.resize(blurred, (w//2, h//2))
        
        # Add noise
        noise = np.random.normal(0, self.noise_level, downsampled.shape)
        noisy = np.clip(downsampled + noise, 0, 1)
        
        return noisy.astype(np.float32)
    
    def estimate_kernel(self, lr_img1: np.ndarray, lr_img2: np.ndarray) -> np.ndarray:
        """Estimate blur kernel from dual LR images"""
        # Simple kernel estimation using gradient analysis
        grad1_x = cv2.Sobel(lr_img1, cv2.CV_64F, 1, 0, ksize=3)
        grad1_y = cv2.Sobel(lr_img1, cv2.CV_64F, 0, 1, ksize=3)
        
        grad2_x = cv2.Sobel(lr_img2, cv2.CV_64F, 1, 0, ksize=3)
        grad2_y = cv2.Sobel(lr_img2, cv2.CV_64F, 0, 1, ksize=3)
        
        # Estimate kernel based on gradient differences
        kernel = np.ones((3, 3)) / 9  # Simple average kernel as baseline
        
        return kernel

class DataAugmentation:
    """Data augmentation for satellite images"""
    
    @staticmethod
    def random_flip(img1: torch.Tensor, img2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Random horizontal/vertical flip"""
        if torch.rand(1) > 0.5:
            img1 = torch.flip(img1, [2])  # Horizontal flip
            img2 = torch.flip(img2, [2])
        
        if torch.rand(1) > 0.5:
            img1 = torch.flip(img1, [1])  # Vertical flip
            img2 = torch.flip(img2, [1])
        
        return img1, img2
    
    @staticmethod
    def random_rotation(img1: torch.Tensor, img2: torch.Tensor, angle_range=10) -> Tuple[torch.Tensor, torch.Tensor]:
        """Random rotation within angle range"""
        angle = torch.randint(-angle_range, angle_range + 1, (1,)).item()
        
        # Convert to PIL for rotation
        from torchvision.transforms.functional import rotate
        img1 = rotate(img1, angle)
        img2 = rotate(img2, angle)
        
        return img1, img2
