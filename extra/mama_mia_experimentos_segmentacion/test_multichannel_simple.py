import numpy as np
import nibabel as nib
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import pandas as pd
import json
from datetime import datetime

def find_image_file(base_path, case_id):
    """Buscar archivo de imagen usando diferentes patrones de nombres"""
    base_path = Path(base_path)
    patterns = [
        f"{case_id}.nii.gz",
        f"{case_id.lower()}.nii.gz",
        f"{case_id.lower()}_0000.nii.gz",
    ]
    
    if case_id.startswith('DUKE_'):
        duke_num = case_id.split('_')[1]
        patterns.extend([
            f"duke_{duke_num.zfill(3)}_0000.nii.gz",
            f"duke_{duke_num}_0000.nii.gz",
        ])
    
    for pattern in patterns:
        candidate = base_path / pattern
        if candidate.exists():
            return candidate
    return None

class MultiChannelDCEProcessor:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.setup_paths()
        
    def setup_paths(self):
        self.paths = {
            'images': Path(r"C:\Users\usuario\Documents\Mama_Mia\datos\images"),
            'ground_truth': Path(r"C:\Users\usuario\Documents\Mama_Mia\datos\segmentations\expert"),
            'models': Path("./models"),
        }
        for path in self.paths.values():
            path.mkdir(parents=True, exist_ok=True)
    
    def normalize_image(self, image):
        p5, p95 = np.percentile(image, [5, 95])
        if p95 > p5:
            return np.clip((image - p5) / (p95 - p5), 0, 1)
        return np.zeros_like(image)
    
    def create_multichannel_image(self, image_path):
        try:
            nii = nib.load(image_path)
            img_data = nii.get_fdata()
            
            if len(img_data.shape) == 4:
                img_data = img_data[:, :, :, 0]
            
            img_norm = self.normalize_image(img_data)
            
            from scipy.ndimage import sobel
            channel1 = img_norm
            channel2 = self.normalize_image(np.abs(sobel(img_norm, axis=0)))
            channel3 = self.normalize_image(np.abs(sobel(img_norm, axis=1)))
            channel4 = self.normalize_image(np.abs(sobel(img_norm, axis=2)))
            
            multichannel = np.stack([channel1, channel2, channel3, channel4], axis=0)
            return multichannel, nii.affine, nii.header
        except Exception as e:
            print(f"Error: {e}")
            return None, None, None

def test_multichannel():
    print("TEST MULTICHANNEL SIMPLE")
    processor = MultiChannelDCEProcessor()
    
    case_id = "DUKE_019"
    image_path = find_image_file(processor.paths['images'], case_id)
    
    if image_path:
        print(f"Imagen encontrada: {image_path.name}")
        multichannel, _, _ = processor.create_multichannel_image(image_path)
        if multichannel is not None:
            print(f"Multi-canal creado: {multichannel.shape}")
            print("TEST EXITOSO")
        else:
            print("ERROR creando multi-canal")
    else:
        print("ERROR: No se encontro imagen")

if __name__ == "__main__":
    test_multichannel()
