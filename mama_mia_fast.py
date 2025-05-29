#!/usr/bin/env python3
"""
MAMA-MIA Challenge - FAST Preprocessing (No N4 Bias Field)
Quick version for testing - should process 1 case in ~5 seconds

Author: Alba & Claude
Date: May 2025
"""

import os
import numpy as np
import nibabel as nib
import SimpleITK as sitk
from pathlib import Path
import json
import shutil
from typing import Tuple, List
import warnings
warnings.filterwarnings("ignore")

class MAMAMIAFastPreprocessor:
    def __init__(self, base_dir: str, output_dir: str = "nnUNet_fast"):
        self.base_dir = Path(base_dir)
        self.output_dir = Path(output_dir)
        self.images_dir = self.base_dir / "images"
        self.segmentations_dir = self.base_dir / "segmentations"
        
        # Create nnU-Net structure
        self.nnunet_dir = self.output_dir / "Dataset001_MAMA_MIA"
        (self.nnunet_dir / "imagesTr").mkdir(parents=True, exist_ok=True)
        (self.nnunet_dir / "labelsTr").mkdir(parents=True, exist_ok=True)
        
        print(f"Fast nnU-Net structure created at: {self.nnunet_dir}")
    
    def fast_preprocessing(self, pre_path: str, post_path: str) -> sitk.Image:
        """
        Fast preprocessing WITHOUT N4 Bias Field (much faster)
        Only: Convert to float32 + Spatial normalization + Z-score + Subtraction
        """
        print("  Loading images...")
        pre_sitk = sitk.ReadImage(str(pre_path))
        post_sitk = sitk.ReadImage(str(post_path))
        
        print("  Converting to float32...")
        # Convert directly to float32 (skip N4)
        pre_float = sitk.Cast(pre_sitk, sitk.sitkFloat32)
        post_float = sitk.Cast(post_sitk, sitk.sitkFloat32)
        
        print("  Spatial normalization...")
        # Resample to 1x1x1 mm
        target_spacing = (1.0, 1.0, 1.0)
        
        def quick_resample(image):
            original_spacing = image.GetSpacing()
            original_size = image.GetSize()
            
            target_size = [
                int(round(original_size[i] * original_spacing[i] / target_spacing[i]))
                for i in range(3)
            ]
            
            resampler = sitk.ResampleImageFilter()
            resampler.SetInterpolator(sitk.sitkLinear)  # Linear faster than BSpline
            resampler.SetOutputSpacing(target_spacing)
            resampler.SetSize(target_size)
            resampler.SetOutputDirection(image.GetDirection())
            resampler.SetOutputOrigin(image.GetOrigin())
            
            return resampler.Execute(image)
        
        pre_resampled = quick_resample(pre_float)
        post_resampled = quick_resample(post_float)
        
        print("  Creating subtraction and normalization...")
        # Convert to numpy
        pre_array = sitk.GetArrayFromImage(pre_resampled)
        post_array = sitk.GetArrayFromImage(post_resampled)
        
        # Subtraction
        subtraction_array = post_array - pre_array
        
        # Z-score normalization (fast version)
        mask = subtraction_array > 0
        if np.sum(mask) > 0:
            mean_val = np.mean(subtraction_array[mask])
            std_val = np.std(subtraction_array[mask])
            if std_val > 0:
                subtraction_array = (subtraction_array - mean_val) / std_val
        
        # Convert back to SimpleITK
        result_sitk = sitk.GetImageFromArray(subtraction_array)
        result_sitk.CopyInformation(post_resampled)
        
        return result_sitk
    
    def process_one_patient(self, patient_id: str) -> bool:
        """Process single patient - FAST VERSION"""
        try:
            pre_path = self.images_dir / f"{patient_id}_0000.nii.gz"
            post_path = self.images_dir / f"{patient_id}_0001.nii.gz"
            
            print(f"Processing {patient_id}...")
            print(f"   Pre:  {pre_path.name} {'OK' if pre_path.exists() else 'MISSING'}")
            print(f"   Post: {post_path.name} {'OK' if post_path.exists() else 'MISSING'}")
            
            if not pre_path.exists() or not post_path.exists():
                print(f"Missing files for {patient_id}")
                return False
            
            # Fast preprocessing (no N4)
            result_sitk = self.fast_preprocessing(pre_path, post_path)
            
            print(f"  Result: {result_sitk.GetSize()}, {result_sitk.GetPixelIDTypeAsString()}")
            
            # Save
            output_path = self.nnunet_dir / "imagesTr" / f"{patient_id}_0000.nii.gz"
            sitk.WriteImage(result_sitk, str(output_path))
            print(f"  Saved: {output_path}")
            
            # Copy segmentation
            seg_path = self.segmentations_dir / "expert" / f"{patient_id}.nii.gz"
            if seg_path.exists():
                output_seg_path = self.nnunet_dir / "labelsTr" / f"{patient_id}.nii.gz"
                shutil.copy2(seg_path, output_seg_path)
                print(f"  Segmentation: {output_seg_path}")
            else:
                print(f"  No segmentation found for {patient_id}")
            
            print(f"SUCCESS: {patient_id} processed in fast mode")
            return True
            
        except Exception as e:
            print(f"ERROR: {patient_id} failed - {str(e)}")
            return False
    
    def run_fast_test(self):
        """Run fast test with first patient"""
        print("MAMA-MIA FAST Preprocessing Test")
        print("Configuration: NO N4 Bias Field (much faster)")
        print("Processing: Spatial + Z-score + Subtraction only")
        print()
        
        # Get first patient
        files_0000 = list(self.images_dir.glob("*_0000.nii.gz"))
        
        if not files_0000:
            print("No patients found")
            return
        
        first_file = files_0000[0]
        patient_id = first_file.name.replace("_0000.nii.gz", "")
        print(f"Testing with patient: {patient_id}")
        print()
        
        import time
        start_time = time.time()
        
        success = self.process_one_patient(patient_id)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        print(f"\n{'='*50}")
        print(f"FAST TEST RESULTS:")
        print(f"{'='*50}")
        print(f"Patient: {patient_id}")
        print(f"Success: {'YES' if success else 'NO'}")
        print(f"Time: {processing_time:.1f} seconds")
        print(f"Speed: {'FAST' if processing_time < 10 else 'SLOW'}")
        
        if success:
            # Create simple dataset.json
            dataset_json = {
                "channel_names": {"0": "Subtraction_DCE_MRI_Fast"},
                "labels": {"background": 0, "tumor": 1},
                "numTraining": 1,
                "file_ending": ".nii.gz",
                "dataset_name": "MAMA_MIA_Fast_Test",
                "description": "Fast preprocessing without N4 Bias Field"
            }
            
            json_path = self.nnunet_dir / "dataset.json"
            with open(json_path, 'w') as f:
                json.dump(dataset_json, f, indent=2)
            
            print(f"Dataset JSON created: {json_path}")
            print(f"Ready for nnU-Net!")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="MAMA-MIA Fast Preprocessing")
    parser.add_argument("--base_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="nnUNet_fast")
    
    args = parser.parse_args()
    
    preprocessor = MAMAMIAFastPreprocessor(args.base_dir, args.output_dir)
    preprocessor.run_fast_test()


if __name__ == "__main__":
    main()