"""
Colorization Model for Interactive User-Guided Image Colorization

This module contains the PyTorch-based deep learning model that performs
colorization based on user-selected regions and color inputs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Tuple, Optional, Dict, Any, List


class ColorizationModel(nn.Module):
    """
    Deep learning model for user-guided image colorization.
    
    This model takes a grayscale image and user color hints to generate
    realistic colorized output while preserving the original structure.
    """
    
    def __init__(self, input_channels: int = 3, output_channels: int = 2):
        super(ColorizationModel, self).__init__()
        
        self.input_channels = input_channels
        self.output_channels = output_channels
        
        # Encoder layers
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        
        # Decoder layers
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, output_channels, kernel_size=3, padding=1),
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        # Constrain to a reasonable chroma range [-1, 1]
        return torch.tanh(decoded)
    
    def colorize(self, 
                 grayscale: torch.Tensor, 
                 ab_hints: torch.Tensor,
                 hint_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if hint_mask is not None:
            # Ensure mask shape matches hints: [B,2,H,W]
            if hint_mask.dim() == 4 and hint_mask.size(1) == 1:
                hint_mask = hint_mask.repeat(1, ab_hints.size(1), 1, 1)
            elif hint_mask.dim() == 3:
                hint_mask = hint_mask.unsqueeze(1).repeat(1, ab_hints.size(1), 1, 1)
            ab_hints = ab_hints * hint_mask
        combined_input = torch.cat([grayscale.to(ab_hints.dtype), ab_hints], dim=1)
        ab_channels = self.forward(combined_input)
        return ab_channels


class UserGuidedColorizer:
    def __init__(self, model_path: Optional[str] = None, device: str = 'cpu'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = ColorizationModel().to(self.device)
        if model_path:
            self.load_model(model_path)
        self.model.eval()
    
    def load_model(self, model_path: str):
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✓ Model loaded from {model_path}")
        except Exception as e:
            print(f"⚠️  Could not load model from {model_path}: {e}")
            print("Using randomly initialized model")
    
    def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        if len(image.shape) == 3 and image.shape[2] == 3:
            gray = 0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]
        else:
            gray = image
        gray = gray.astype(np.float32) / 255.0
        gray_tensor = torch.from_numpy(gray).unsqueeze(0).unsqueeze(0)
        return gray_tensor.to(self.device)
    
    def create_color_hints(self, 
                          image_shape: Tuple[int, int],
                          selected_regions: List[Dict],
                          colors: List[Tuple[int, int, int]]) -> Tuple[torch.Tensor, torch.Tensor]:
        height, width = image_shape
        # We keep two channels for ab hints
        ab_hints = np.zeros((2, height, width), dtype=np.float32)
        hint_mask = np.zeros((1, height, width), dtype=np.float32)
        
        for region, color in zip(selected_regions, colors):
            rgb = np.array([[list(color)]], dtype=np.uint8)
            lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)[0, 0]  # L, a, b
            a_val = lab[1] - 128.0
            b_val = lab[2] - 128.0
            
            if 'mask' in region:
                mask = region['mask'].astype(np.float32)
                if mask.shape != (height, width):
                    mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
                ab_hints[0] = np.where(mask > 0, a_val, ab_hints[0])
                ab_hints[1] = np.where(mask > 0, b_val, ab_hints[1])
                hint_mask[0] = np.maximum(hint_mask[0], (mask > 0).astype(np.float32))
            elif 'bbox' in region:
                x1, y1, x2, y2 = region['bbox']
                ab_hints[0, y1:y2, x1:x2] = a_val
                ab_hints[1, y1:y2, x1:x2] = b_val
                hint_mask[0, y1:y2, x1:x2] = 1.0
        
        ab_hints_t = torch.from_numpy(ab_hints).unsqueeze(0).to(torch.float32)
        # Make hint_mask [B,1,H,W]
        hint_mask_t = torch.from_numpy(hint_mask).unsqueeze(0).to(torch.float32)
        return ab_hints_t.to(self.device), hint_mask_t.to(self.device)
    
    def colorize_image(self, 
                      image: np.ndarray,
                      selected_regions: List[Dict],
                      colors: List[Tuple[int, int, int]],
                      saturation_boost: float = 1.4,
                      blend_with_gray: float = 0.15,
                      guidance_strength: float = 1.0) -> np.ndarray:
        with torch.no_grad():
            gray_tensor = self.preprocess_image(image)
            ab_hints, hint_mask = self.create_color_hints(
                image.shape[:2], selected_regions, colors
            )
            # Apply guidance strength to hints
            g = float(np.clip(guidance_strength, 0.0, 1.0))
            ab_hints = ab_hints * g

            # Model prediction
            ab_pred = self.model.colorize(gray_tensor, ab_hints, hint_mask)

            # Strongly enforce user hints in selected areas and diffuse slightly
            try:
                ab_pred_np = ab_pred.detach().cpu().numpy()[0]  # [2,H,W]
                hints_np = ab_hints.detach().cpu().numpy()[0]
                mask_np = hint_mask.detach().cpu().numpy()[0, 0]  # [H,W]

                # Diffuse hints to avoid hard edges
                if mask_np.sum() > 0:
                    ksize = max(5, int(round(min(mask_np.shape) * 0.01)) | 1)  # odd
                    a_diffused = cv2.GaussianBlur(hints_np[0] * mask_np, (ksize, ksize), 0)
                    b_diffused = cv2.GaussianBlur(hints_np[1] * mask_np, (ksize, ksize), 0)
                    diffused = np.stack([a_diffused, b_diffused], axis=0)

                    # Blend diffused hints into prediction within mask
                    mask3 = np.stack([mask_np, mask_np], axis=0)
                    ab_blend = ab_pred_np * (1.0 - g * mask3) + diffused * (g * mask3)
                    ab_pred = torch.from_numpy(ab_blend).unsqueeze(0).to(ab_pred.device).to(ab_pred.dtype)
            except Exception:
                # Fallback: keep model output
                pass

            colorized = self._postprocess_output(gray_tensor, ab_pred, saturation_boost, blend_with_gray)
            return colorized
    
    def _postprocess_output(self, 
                           gray_tensor: torch.Tensor, 
                           ab_channels: torch.Tensor,
                           saturation_boost: float = 1.4,
                           blend_with_gray: float = 0.15) -> np.ndarray:
        gray_np = gray_tensor.cpu().numpy()[0, 0]
        ab_np = ab_channels.cpu().numpy()[0]
        
        h, w = gray_np.shape
        a = ab_np[0]
        b = ab_np[1]
        if a.shape != (h, w):
            a = cv2.resize(a, (w, h), interpolation=cv2.INTER_LINEAR)
        if b.shape != (h, w):
            b = cv2.resize(b, (w, h), interpolation=cv2.INTER_LINEAR)
        
        # Build LAB
        L = (gray_np * 100.0).astype(np.float32)
        # Boost saturation by scaling chroma
        a_scaled = np.clip(a, -1.0, 1.0).astype(np.float32) * (110.0 * float(np.clip(saturation_boost, 0.5, 2.5)))
        b_scaled = np.clip(b, -1.0, 1.0).astype(np.float32) * (110.0 * float(np.clip(saturation_boost, 0.5, 2.5)))
        lab = np.stack([L, a_scaled, b_scaled], axis=2).astype(np.float32)
        
        rgb = cv2.cvtColor(lab, cv2.COLOR_Lab2RGB)
        rgb = np.clip(rgb, 0.0, 1.0)
        
        # Optional blend with original grayscale to control intensity
        alpha = float(np.clip(blend_with_gray, 0.0, 1.0))
        if alpha > 0:
            gray_rgb = np.stack([gray_np, gray_np, gray_np], axis=2)
            rgb = (1.0 - alpha) * rgb + alpha * gray_rgb
            rgb = np.clip(rgb, 0.0, 1.0)
        
        return rgb
