import cv2
import numpy as np
import rasterio
from pathlib import Path
from typing import Union
from PIL import Image

def load_satellite_image(image_path: Union[str, Path]) -> np.ndarray:
    """Load satellite image from various formats"""
    image_path = Path(image_path)
    
    if image_path.suffix.lower() in ['.tif', '.tiff']:
        try:
            # Load using rasterio for GeoTIFF files
            with rasterio.open(image_path) as src:
                image = src.read().transpose(1, 2, 0)
                
                # Handle different band counts
                if image.shape[2] == 1:
                    image = np.repeat(image, 3, axis=2)
                elif image.shape[2] > 3:
                    image = image[:, :, :3]  # Take first 3 bands
        except:
            # Fallback to PIL
            image = np.array(Image.open(image_path).convert('RGB'))
    else:
        # Load using PIL for common formats
        image = np.array(Image.open(image_path).convert('RGB'))
    
    return image

def save_image(image: np.ndarray, output_path: Union[str, Path]) -> None:
    """Save image to file"""
    output_path = Path(output_path)
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert to uint8 if needed
    if image.dtype == np.float32 or image.dtype == np.float64:
        image = np.clip(image, 0, 1)
        image = (image * 255).astype(np.uint8)
    
    # Save using PIL
    Image.fromarray(image).save(output_path)

def normalize_image(image: np.ndarray, method: str = 'minmax') -> np.ndarray:
    """Normalize image to [0, 1] range"""
    if method == 'minmax':
        return (image - image.min()) / (image.max() - image.min() + 1e-8)
    elif method == 'zscore':
        return (image - image.mean()) / (image.std() + 1e-8)
    else:
        raise ValueError(f"Unknown normalization method: {method}")

def calculate_image_statistics(image: np.ndarray) -> dict:
    """Calculate basic image statistics"""
    stats = {
        'mean': np.mean(image),
        'std': np.std(image),
        'min': np.min(image),
        'max': np.max(image),
        'shape': image.shape
    }
    return stats

def resize_image(image: np.ndarray, target_size: tuple, 
                interpolation: str = 'bicubic') -> np.ndarray:
    """Resize image with specified interpolation"""
    
    interpolation_methods = {
        'nearest': cv2.INTER_NEAREST,
        'linear': cv2.INTER_LINEAR,
        'cubic': cv2.INTER_CUBIC,
        'bicubic': cv2.INTER_CUBIC,
        'lanczos': cv2.INTER_LANCZOS4
    }
    
    method = interpolation_methods.get(interpolation, cv2.INTER_CUBIC)
    
    if len(image.shape) == 3:
        resized = cv2.resize(image, target_size, interpolation=method)
    else:
        resized = cv2.resize(image, target_size, interpolation=method)
    
    return resized

def crop_center(image: np.ndarray, crop_size: tuple) -> np.ndarray:
    """Crop image from center"""
    h, w = image.shape[:2]
    crop_h, crop_w = crop_size
    
    start_h = (h - crop_h) // 2
    start_w = (w - crop_w) // 2
    
    return image[start_h:start_h+crop_h, start_w:start_w+crop_w]

def pad_image(image: np.ndarray, target_size: tuple, mode: str = 'reflect') -> np.ndarray:
    """Pad image to target size"""
    h, w = image.shape[:2]
    target_h, target_w = target_size
    
    pad_h = max(0, target_h - h)
    pad_w = max(0, target_w - w)
    
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    
    if len(image.shape) == 3:
        padded = np.pad(image, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), mode=mode)
    else:
        padded = np.pad(image, ((pad_top, pad_bottom), (pad_left, pad_right)), mode=mode)
    
    return padded
