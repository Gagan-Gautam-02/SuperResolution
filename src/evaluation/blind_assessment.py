import torch
import torch.nn as nn
import numpy as np
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
import cv2
from typing import Dict, List, Tuple

class BlindImageQualityAssessment:
    """Blind (no-reference) image quality assessment"""
    
    def __init__(self):
        self.feature_extractor = FeatureExtractor()
        self.quality_predictor = None
        self.scaler = StandardScaler()
    
    def extract_features(self, image: np.ndarray) -> np.ndarray:
        """Extract handcrafted features for quality assessment"""
        return self.feature_extractor.extract_all_features(image)
    
    def train_quality_predictor(self, images: List[np.ndarray], 
                               quality_scores: List[float], 
                               method: str = 'rf') -> None:
        """Train quality predictor using extracted features"""
        
        # Extract features from all images
        features = []
        for img in images:
            feat = self.extract_features(img)
            features.append(feat)
        
        features = np.array(features)
        quality_scores = np.array(quality_scores)
        
        # Normalize features
        features_scaled = self.scaler.fit_transform(features)
        
        # Train predictor
        if method == 'rf':
            self.quality_predictor = RandomForestRegressor(n_estimators=100, random_state=42)
        elif method == 'svr':
            self.quality_predictor = SVR(kernel='rbf', C=1.0, gamma='scale')
        else:
            raise ValueError(f"Unknown method: {method}")
        
        self.quality_predictor.fit(features_scaled, quality_scores)
    
    def predict_quality(self, image: np.ndarray) -> float:
        """Predict quality score for a single image"""
        if self.quality_predictor is None:
            raise ValueError("Quality predictor not trained yet")
        
        features = self.extract_features(image).reshape(1, -1)
        features_scaled = self.scaler.transform(features)
        
        return self.quality_predictor.predict(features_scaled)[0]
    
    def evaluate_correlation(self, predicted_scores: List[float], 
                           ground_truth_scores: List[float]) -> Dict[str, float]:
        """Evaluate correlation between predicted and ground truth scores"""
        
        predicted = np.array(predicted_scores)
        ground_truth = np.array(ground_truth_scores)
        
        # Pearson correlation
        pearson_corr, pearson_p = stats.pearsonr(predicted, ground_truth)
        
        # Spearman rank correlation
        spearman_corr, spearman_p = stats.spearmanr(predicted, ground_truth)
        
        # Kendall's tau
        kendall_corr, kendall_p = stats.kendalltau(predicted, ground_truth)
        
        return {
            'pearson_correlation': pearson_corr,
            'pearson_p_value': pearson_p,
            'spearman_correlation': spearman_corr,
            'spearman_p_value': spearman_p,
            'kendall_correlation': kendall_corr,
            'kendall_p_value': kendall_p
        }

class FeatureExtractor:
    """Extract handcrafted features for blind quality assessment"""
    
    def extract_all_features(self, image: np.ndarray) -> np.ndarray:
        """Extract all features and concatenate them"""
        features = []
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Statistical features
        features.extend(self.extract_statistical_features(gray))
        
        # Gradient features
        features.extend(self.extract_gradient_features(gray))
        
        # Texture features
        features.extend(self.extract_texture_features(gray))
        
        # Frequency domain features
        features.extend(self.extract_frequency_features(gray))
        
        return np.array(features)
    
    def extract_statistical_features(self, image: np.ndarray) -> List[float]:
        """Extract statistical features"""
        features = []
        
        # Basic statistics
        features.append(np.mean(image))
        features.append(np.std(image))
        features.append(np.var(image))
        features.append(stats.skew(image.flatten()))
        features.append(stats.kurtosis(image.flatten()))
        
        # Histogram features
        hist, _ = np.histogram(image, bins=256, range=(0, 1))
        hist = hist / np.sum(hist)  # Normalize
        
        features.append(np.sum(hist * np.log(hist + 1e-10)))  # Entropy
        features.extend(hist[:10])  # First 10 histogram bins
        
        return features
    
    def extract_gradient_features(self, image: np.ndarray) -> List[float]:
        """Extract gradient-based features"""
        features = []
        
        # Sobel gradients
        grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        features.append(np.mean(gradient_magnitude))
        features.append(np.std(gradient_magnitude))
        features.append(np.max(gradient_magnitude))
        
        # Laplacian
        laplacian = cv2.Laplacian(image, cv2.CV_64F)
        features.append(np.var(laplacian))
        
        return features
    
    def extract_texture_features(self, image: np.ndarray) -> List[float]:
        """Extract texture features using Local Binary Patterns"""
        features = []
        
        # Simple LBP implementation
        def lbp(img, radius=1, n_points=8):
            h, w = img.shape
            lbp_img = np.zeros((h, w), dtype=np.uint8)
            
            for i in range(radius, h - radius):
                for j in range(radius, w - radius):
                    center = img[i, j]
                    code = 0
                    for k in range(n_points):
                        angle = 2 * np.pi * k / n_points
                        x = int(i + radius * np.cos(angle))
                        y = int(j + radius * np.sin(angle))
                        if img[x, y] >= center:
                            code |= (1 << k)
                    lbp_img[i, j] = code
            
            return lbp_img
        
        # Convert to uint8 for LBP
        img_uint8 = (image * 255).astype(np.uint8)
        lbp_img = lbp(img_uint8)
        
        # LBP histogram
        hist, _ = np.histogram(lbp_img, bins=256, range=(0, 256))
        hist = hist / np.sum(hist)
        
        features.extend(hist[:10])  # First 10 LBP histogram bins
        
        return features
    
    def extract_frequency_features(self, image: np.ndarray) -> List[float]:
        """Extract frequency domain features"""
        features = []
        
        # FFT
        fft = np.fft.fft2(image)
        fft_magnitude = np.abs(fft)
        
        # Power spectral density features
        features.append(np.mean(fft_magnitude))
        features.append(np.std(fft_magnitude))
        
        # High frequency content
        h, w = fft_magnitude.shape
        center_h, center_w = h // 2, w // 2
        high_freq_mask = np.ones((h, w))
        high_freq_mask[center_h-h//4:center_h+h//4, center_w-w//4:center_w+w//4] = 0
        
        high_freq_energy = np.sum(fft_magnitude * high_freq_mask)
        total_energy = np.sum(fft_magnitude)
        
        features.append(high_freq_energy / total_energy)
        
        return features

class DeepFeatureExtractor(nn.Module):
    """Deep learning-based feature extractor for quality assessment"""
    
    def __init__(self, pretrained_model='resnet50'):
        super(DeepFeatureExtractor, self).__init__()
        
        if pretrained_model == 'resnet50':
            import torchvision.models as models
            self.backbone = models.resnet50(pretrained=True)
            self.backbone.fc = nn.Identity()  # Remove final classification layer
            self.feature_dim = 2048
        
        # Quality prediction head
        self.quality_head = nn.Sequential(
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1)
        )
    
    def forward(self, x):
        features = self.backbone(x)
        quality_score = self.quality_head(features)
        return quality_score, features

class ReducedReferenceQualityAssessment:
    """Reduced reference quality assessment using statistical features"""
    
    def __init__(self):
        self.reference_features = None
    
    def extract_rr_features(self, image: np.ndarray) -> Dict[str, float]:
        """Extract reduced reference features"""
        features = {}
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Statistical moments
        features['mean'] = np.mean(gray)
        features['std'] = np.std(gray)
        features['skewness'] = stats.skew(gray.flatten())
        features['kurtosis'] = stats.kurtosis(gray.flatten())
        
        # Gradient statistics
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        features['gradient_mean'] = np.mean(gradient_magnitude)
        features['gradient_std'] = np.std(gradient_magnitude)
        
        return features
    
    def set_reference(self, reference_image: np.ndarray) -> None:
        """Set reference image features"""
        self.reference_features = self.extract_rr_features(reference_image)
    
    def compute_rr_quality(self, test_image: np.ndarray) -> float:
        """Compute reduced reference quality score"""
        if self.reference_features is None:
            raise ValueError("Reference features not set")
        
        test_features = self.extract_rr_features(test_image)
        
        # Compute feature differences
        total_diff = 0
        for key in self.reference_features:
            diff = abs(test_features[key] - self.reference_features[key])
            total_diff += diff
        
        # Convert to quality score (higher is better)
        quality_score = 1.0 / (1.0 + total_diff)
        
        return quality_score
