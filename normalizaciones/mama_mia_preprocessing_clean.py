#!/usr/bin/env python3
"""
MAMA-MIA Challenge - Breast Cancer DCE-MRI Preprocessing Pipeline
Professional version for deep learning model training

Based on Schwarzhans et al. (2025) optimal preprocessing:
- N4 Bias Field Correction
- Spatial Normalization (1x1x1 mm isotropic)
- Z-score Intensity Normalization
- Subtraction Image Generation (Post1 - Pre)

Author: Alba 
Date: May 2025
"""

import os
import numpy as np
import nibabel as nib
import SimpleITK as sitk
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import argparse
import shutil
import json
from typing import Tuple, Optional, List
import warnings
warnings.filterwarnings("ignore")

class MAMAMIAPreprocessor:
    def __init__(self, base_dir: str, output_dir: str = "nnUNet_preprocessed"):
        """
        Initialize MAMA-MIA preprocessor
        
        Args:
            base_dir: Base directory containing 'images' and 'segmentations' folders
            output_dir: Output directory for nnU-Net structure
        """
        self.base_dir = Path(base_dir)
        self.output_dir = Path(output_dir)
        self.images_dir = self.base_dir / "images"
        self.segmentations_dir = self.base_dir / "segmentations"
        
        # Create nnU-Net directory structure
        self.setup_nnunet_structure()
        
    def setup_nnunet_structure(self):
        """Create nnU-Net directory structure"""
        self.nnunet_dir = self.output_dir / "Dataset001_MAMA_MIA"
        
        # Main directories
        (self.nnunet_dir / "imagesTr").mkdir(parents=True, exist_ok=True)
        (self.nnunet_dir / "labelsTr").mkdir(parents=True, exist_ok=True)
        (self.nnunet_dir / "imagesTs").mkdir(parents=True, exist_ok=True)
        (self.nnunet_dir / "labelsTs").mkdir(parents=True, exist_ok=True)
        
        print(f"nnU-Net structure created at: {self.nnunet_dir}")
    
    def n4_bias_correction(self, image_sitk: sitk.Image) -> sitk.Image:
        """
        Apply N4 Bias Field Correction with proper type conversion
        
        Args:
            image_sitk: SimpleITK image
            
        Returns:
            Bias-corrected image in float32 format
        """
        try:
            print(f"    Original pixel type: {image_sitk.GetPixelIDTypeAsString()}")
            
            # Convert to float32 (required for N4)
            if image_sitk.GetPixelID() != sitk.sitkFloat32:
                image_float = sitk.Cast(image_sitk, sitk.sitkFloat32)
                print(f"    Converted to: 32-bit float")
            else:
                image_float = image_sitk
                print(f"    Already float32")
            
            # Apply N4 Bias Field Correction
            corrector = sitk.N4BiasFieldCorrectionImageFilter()
            
            # Optimized parameters for DCE-MRI
            corrector.SetMaximumNumberOfIterations([50, 50, 50, 50])
            corrector.SetConvergenceThreshold(0.001)
            
            corrected_image = corrector.Execute(image_float)
            print(f"    N4 Bias Field correction applied successfully")
            
            return corrected_image
            
        except Exception as e:
            print(f"    N4 correction failed: {str(e)[:80]}...")
            # Fallback: convert to float32 without N4
            try:
                image_float = sitk.Cast(image_sitk, sitk.sitkFloat32)
                print(f"    Using original image converted to float32")
                return image_float
            except Exception as e2:
                print(f"    Critical error: {str(e2)[:50]}...")
                return image_sitk
    
    def spatial_normalization(self, image_sitk: sitk.Image, 
                            target_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)) -> sitk.Image:
        """
        Spatial normalization: resample to isotropic voxel spacing
        
        Args:
            image_sitk: SimpleITK image
            target_spacing: Target voxel spacing (default: 1x1x1 mm)
            
        Returns:
            Resampled image
        """
        original_spacing = image_sitk.GetSpacing()
        original_size = image_sitk.GetSize()
        
        # Calculate new size
        target_size = [
            int(round(original_size[i] * original_spacing[i] / target_spacing[i]))
            for i in range(3)
        ]
        
        # Configure resampling
        resampler = sitk.ResampleImageFilter()
        resampler.SetInterpolator(sitk.sitkBSpline)
        resampler.SetOutputSpacing(target_spacing)
        resampler.SetSize(target_size)
        resampler.SetOutputDirection(image_sitk.GetDirection())
        resampler.SetOutputOrigin(image_sitk.GetOrigin())
        
        resampled_image = resampler.Execute(image_sitk)
        return resampled_image
    
    def zscore_normalization(self, image_array: np.ndarray) -> np.ndarray:
        """
        Z-score intensity normalization
        
        Args:
            image_array: Image array
            
        Returns:
            Normalized image array
        """
        # Create mask excluding background (voxels = 0)
        mask = image_array > 0
        
        if np.sum(mask) == 0:
            print("    Warning: Empty mask, using entire image")
            mask = np.ones_like(image_array, dtype=bool)
        
        # Calculate statistics only in region of interest
        mean_val = np.mean(image_array[mask])
        std_val = np.std(image_array[mask])
        
        # Avoid division by zero
        if std_val == 0:
            std_val = 1.0
            print("    Warning: Standard deviation = 0, using std = 1")
        
        # Apply z-score normalization
        normalized_image = (image_array - mean_val) / std_val
        return normalized_image
    
    def create_subtraction_image(self, pre_path: str, post_path: str) -> sitk.Image:
        """
        Create subtraction image (Post1 - Pre) with full preprocessing pipeline
        
        Args:
            pre_path: Path to pre-contrast image
            post_path: Path to post-contrast image
            
        Returns:
            Preprocessed subtraction image
        """
        # Load images
        pre_sitk = sitk.ReadImage(str(pre_path))
        post_sitk = sitk.ReadImage(str(post_path))
        
        print("  Applying N4 Bias Field Correction...")
        
        # Apply N4 to both images
        pre_corrected = self.n4_bias_correction(pre_sitk)
        post_corrected = self.n4_bias_correction(post_sitk)
        
        print("  Applying spatial normalization...")
        
        # Spatial normalization
        pre_resampled = self.spatial_normalization(pre_corrected)
        post_resampled = self.spatial_normalization(post_corrected)
        
        # Convert to numpy arrays
        pre_array = sitk.GetArrayFromImage(pre_resampled)
        post_array = sitk.GetArrayFromImage(post_resampled)
        
        # Create subtraction image
        subtraction_array = post_array - pre_array
        
        print("  Applying Z-score normalization...")
        
        # Z-score normalization
        subtraction_normalized = self.zscore_normalization(subtraction_array)
        
        # Convert back to SimpleITK
        subtraction_sitk = sitk.GetImageFromArray(subtraction_normalized)
        subtraction_sitk.CopyInformation(post_resampled)
        
        return subtraction_sitk
    
    def get_patient_files(self) -> List[str]:
        """
        Get list of available patients
        
        Returns:
            Sorted list of patient IDs
        """
        patients = set()
        
        files_0000 = list(self.images_dir.glob("*_0000.nii.gz"))
        files_0001 = list(self.images_dir.glob("*_0001.nii.gz"))
        
        for file in files_0000:
            patient_id = file.name.replace("_0000.nii.gz", "")
            post_file = self.images_dir / f"{patient_id}_0001.nii.gz"
            
            if post_file.exists():
                patients.add(patient_id)
        
        return sorted(list(patients))
    
    def process_single_patient(self, patient_id: str, verbose: bool = True) -> bool:
        """
        Process a single patient
        
        Args:
            patient_id: Patient identifier
            verbose: Enable detailed logging
            
        Returns:
            True if processing succeeded
        """
        try:
            # File paths
            pre_path = self.images_dir / f"{patient_id}_0000.nii.gz"
            post_path = self.images_dir / f"{patient_id}_0001.nii.gz"
            
            if verbose:
                print(f"Verifying files:")
                print(f"   Pre:  {pre_path.name} {'OK' if pre_path.exists() else 'MISSING'}")
                print(f"   Post: {post_path.name} {'OK' if post_path.exists() else 'MISSING'}")
            
            if not pre_path.exists() or not post_path.exists():
                print(f"Missing files for {patient_id}")
                return False
            
            if verbose:
                print(f"Processing {patient_id}...")
            
            # Create subtraction image with full preprocessing
            subtraction_sitk = self.create_subtraction_image(pre_path, post_path)
            
            if verbose:
                print(f"Processed image info:")
                print(f"   Size: {subtraction_sitk.GetSize()}")
                print(f"   Spacing: {subtraction_sitk.GetSpacing()}")
                print(f"   Type: {subtraction_sitk.GetPixelIDTypeAsString()}")
            
            # Save processed image
            output_path = self.nnunet_dir / "imagesTr" / f"{patient_id}_0000.nii.gz"
            if verbose:
                print(f"Saving to: {output_path}")
            sitk.WriteImage(subtraction_sitk, str(output_path))
            
            # Copy expert segmentation if available
            seg_path = self.segmentations_dir / "expert" / f"{patient_id}.nii.gz"
            if verbose:
                print(f"Looking for segmentation: {seg_path.name}")
            
            if seg_path.exists():
                output_seg_path = self.nnunet_dir / "labelsTr" / f"{patient_id}.nii.gz"
                shutil.copy2(seg_path, output_seg_path)
                if verbose:
                    print(f"Segmentation copied: {output_seg_path}")
            else:
                if verbose:
                    print(f"Expert segmentation not found for {patient_id}")
            
            if verbose:
                print(f"{patient_id} processed successfully")
            return True
            
        except Exception as e:
            print(f"Error processing {patient_id}:")
            print(f"   Type: {type(e).__name__}")
            print(f"   Message: {str(e)}")
            return False
    
    def create_dataset_json(self, patient_list: List[str]):
        """
        Create dataset.json file for nnU-Net
        
        Args:
            patient_list: List of processed patients
        """
        dataset_json = {
            "channel_names": {
                "0": "Subtraction_DCE_MRI"
            },
            "labels": {
                "background": 0,
                "tumor": 1
            },
            "numTraining": len(patient_list),
            "file_ending": ".nii.gz",
            "dataset_name": "MAMA_MIA_Challenge",
            "description": "MAMA-MIA breast cancer DCE-MRI segmentation with optimal preprocessing",
            "reference": "Schwarzhans et al. (2025) - Bias Field + Spatial + Z-score normalization",
            "release": "2.0_professional"
        }
        
        json_path = self.nnunet_dir / "dataset.json"
        with open(json_path, 'w') as f:
            json.dump(dataset_json, f, indent=2)
        
        print(f"Dataset JSON created: {json_path}")
    
    def run_preprocessing(self, max_cases: int = 1, verbose: bool = True):
        """
        Execute complete preprocessing pipeline
        
        Args:
            max_cases: Maximum number of cases to process (None = all)
            verbose: Enable detailed logging
            
        Returns:
            Number of successfully processed cases
        """
        print("Initiating MAMA-MIA preprocessing pipeline...")
        print("Configuration:")
        print("   - Bias Field Correction: N4ITK (Float32)")
        print("   - Spatial Normalization: 1x1x1 mm isotropic")
        print("   - Intensity Normalization: Z-score")
        print("   - Input: Subtraction Image (Post1 - Pre)")
        if max_cases == 1:
            print("   - MODE: Single case test")
        print()
        
        # Get patient list
        patients = self.get_patient_files()
        
        if max_cases:
            patients = patients[:max_cases]
            print(f"Processing {max_cases} case(s) for testing")
        
        print(f"Patients to process: {len(patients)}")
        if patients and verbose:
            print(f"First patient: {patients[0]}")
        
        if len(patients) == 0:
            print("No valid patients found")
            return 0
        
        # Estimate processing time
        time_per_case = 15 if verbose else 8  # seconds
        total_time_min = (len(patients) * time_per_case) / 60
        print(f"Estimated time: {total_time_min:.1f} minutes")
        print()
        
        # Process patients
        processed_patients = []
        failed_patients = []
        
        for i, patient_id in enumerate(patients, 1):
            if verbose and len(patients) == 1:
                print(f"{'='*50}")
                print(f"PROCESSING CASE {i}/{len(patients)}: {patient_id}")
                print(f"{'='*50}")
            
            if self.process_single_patient(patient_id, verbose=verbose):
                processed_patients.append(patient_id)
                if verbose:
                    print(f"SUCCESS: {patient_id} processed correctly")
            else:
                failed_patients.append(patient_id)
                if verbose:
                    print(f"FAILED: {patient_id} could not be processed")
        
        # Results summary
        if verbose:
            print(f"\n{'='*60}")
            print(f"FINAL RESULTS:")
            print(f"{'='*60}")
        
        print(f"Successfully processed: {len(processed_patients)}")
        print(f"Failed: {len(failed_patients)}")
        print(f"Success rate: {len(processed_patients)/len(patients)*100:.1f}%")
        
        if processed_patients and verbose:
            print(f"Successful cases: {processed_patients}")
        if failed_patients and verbose:
            print(f"Failed cases: {failed_patients}")
        
        # Create dataset.json
        if processed_patients:
            self.create_dataset_json(processed_patients)
            print(f"\nPreprocessing completed successfully!")
            print(f"Files created in: {self.nnunet_dir}")
            print(f"Dataset JSON: Created")
            print(f"Ready for nnU-Net training")
            
            if verbose:
                print(f"\nNext steps:")
                print(f"1. Set environment variables:")
                print(f"   $env:nnUNet_raw='{self.output_dir.absolute()}'")
                print(f"   $env:nnUNet_preprocessed='{self.output_dir.absolute()}\\nnUNet_preprocessed'")
                print(f"   $env:nnUNet_results='{self.output_dir.absolute()}\\nnUNet_results'")
                print(f"2. Run nnU-Net:")
                print(f"   nnUNetv2_plan_and_preprocess -d 001")
                print(f"   nnUNetv2_train 001 3d_fullres 0")
        
        return len(processed_patients)


def main():
    parser = argparse.ArgumentParser(description="MAMA-MIA Preprocessing Pipeline")
    parser.add_argument("--base_dir", type=str, required=True, 
                       help="Base directory with 'images' and 'segmentations' folders")
    parser.add_argument("--output_dir", type=str, default="nnUNet_models",
                       help="Output directory for nnU-Net structure")
    parser.add_argument("--max_cases", type=int, default=1,
                       help="Maximum number of cases to process (default: 1 for testing)")
    parser.add_argument("--verbose", action="store_true", default=True,
                       help="Enable detailed logging")
    
    args = parser.parse_args()
    
    # Create and execute preprocessor
    preprocessor = MAMAMIAPreprocessor(args.base_dir, args.output_dir)
    num_processed = preprocessor.run_preprocessing(args.max_cases, args.verbose)
    
    if num_processed > 0:
        print(f"\nPreprocessing pipeline completed successfully!")
        print(f"Cases processed: {num_processed}")
        print(f"Quality: Optimal preprocessing according to scientific literature")
    else:
        print(f"\nNo cases were processed. Please check errors above.")


if __name__ == "__main__":
    main()