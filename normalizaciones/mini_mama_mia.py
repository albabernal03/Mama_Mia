#!/usr/bin/env python3
"""
Create Mini MAMA-MIA Dataset
Takes real MAMA-MIA cases and downsamples them for local testing
"""

import numpy as np
import SimpleITK as sitk
from pathlib import Path
import json
import shutil
from scipy import ndimage

def create_mini_mama_mia(base_dir, num_cases=5, target_size=(64, 64, 32)):
    """
    Create mini version of MAMA-MIA with real cases
    
    Args:
        base_dir: Directory with original MAMA-MIA data
        num_cases: Number of cases to process  
        target_size: Target size (Z, Y, X)
    """
    
    base_path = Path(base_dir)
    images_dir = base_path / "images"
    seg_dir = base_path / "segmentations" / "expert"
    
    # Output directory
    output_dir = Path("mini_mama_mia")
    (output_dir / "imagesTr").mkdir(parents=True, exist_ok=True)
    (output_dir / "labelsTr").mkdir(parents=True, exist_ok=True)
    
    print(f"Creating Mini MAMA-MIA: {target_size}, {num_cases} real cases")
    
    # Get available patients
    files_0000 = sorted(list(images_dir.glob("*_0000.nii.gz")))[:num_cases]
    
    processed = 0
    for i, pre_file in enumerate(files_0000):
        patient_id = pre_file.name.replace("_0000.nii.gz", "")
        post_file = images_dir / f"{patient_id}_0001.nii.gz"
        seg_file = seg_dir / f"{patient_id}.nii.gz"
        
        if not post_file.exists() or not seg_file.exists():
            print(f"Skipping {patient_id} - missing files")
            continue
        
        print(f"Processing mini case {processed+1}/{num_cases}: {patient_id}")
        
        try:
            # Load images
            print(f"  Loading images...")
            pre_sitk = sitk.ReadImage(str(pre_file))
            post_sitk = sitk.ReadImage(str(post_file))
            seg_sitk = sitk.ReadImage(str(seg_file))
            
            # Convert to float32 and create subtraction
            print(f"  Creating subtraction...")
            pre_float = sitk.Cast(pre_sitk, sitk.sitkFloat32)
            post_float = sitk.Cast(post_sitk, sitk.sitkFloat32)
            
            pre_array = sitk.GetArrayFromImage(pre_float)
            post_array = sitk.GetArrayFromImage(post_float)
            sub_array = post_array - pre_array
            
            print(f"  Original size: {sub_array.shape}")
            
            # Downsample to target size
            original_shape = sub_array.shape
            
            # Calculate zoom factors
            zoom_factors = [target_size[i] / original_shape[i] for i in range(3)]
            print(f"  Zoom factors: {zoom_factors}")
            
            # Resize image
            sub_small = ndimage.zoom(sub_array, zoom_factors, order=1)
            print(f"  Resized to: {sub_small.shape}")
            
            # Normalize
            print(f"  Normalizing...")
            mask = sub_small > 0
            if np.sum(mask) > 0:
                mean_val = np.mean(sub_small[mask])
                std_val = np.std(sub_small[mask])
                if std_val > 0:
                    sub_small = (sub_small - mean_val) / std_val
            
            # Process segmentation
            print(f"  Processing segmentation...")
            seg_array = sitk.GetArrayFromImage(seg_sitk)
            seg_small = ndimage.zoom(seg_array.astype(float), zoom_factors, order=0)  # Nearest neighbor
            seg_small = (seg_small > 0.5).astype(np.uint8)
            
            print(f"  Segmentation: {seg_small.shape}, unique values: {np.unique(seg_small)}")
            
            # Convert back to SimpleITK
            sub_sitk = sitk.GetImageFromArray(sub_small.astype(np.float32))
            seg_sitk_small = sitk.GetImageFromArray(seg_small.astype(np.uint8))
            
            # Set spacing
            sub_sitk.SetSpacing((1.0, 1.0, 1.0))
            seg_sitk_small.SetSpacing((1.0, 1.0, 1.0))
            
            # Save with nnU-Net naming
            case_name = f"mini_{processed:03d}"
            
            image_path = output_dir / "imagesTr" / f"{case_name}_0000.nii.gz"
            seg_path = output_dir / "labelsTr" / f"{case_name}.nii.gz"
            
            sitk.WriteImage(sub_sitk, str(image_path))
            sitk.WriteImage(seg_sitk_small, str(seg_path))
            
            print(f"  Saved: {image_path}")
            print(f"  Saved: {seg_path}")
            
            processed += 1
            print(f"  SUCCESS: {patient_id} -> mini_{processed-1:03d}")
            print()
            
        except Exception as e:
            print(f"  ERROR: {str(e)}")
            continue
    
    if processed == 0:
        print("No cases were processed successfully!")
        return None
    
    # Create dataset.json
    dataset_json = {
        "channel_names": {
            "0": "Mini_MAMA_MIA_Subtraction"
        },
        "labels": {
            "background": 0,
            "tumor": 1
        },
        "numTraining": processed,
        "file_ending": ".nii.gz",
        "dataset_name": "Mini_MAMA_MIA_Real",
        "description": f"Downsampled real MAMA-MIA cases to {target_size} for local testing"
    }
    
    with open(output_dir / "dataset.json", 'w') as f:
        json.dump(dataset_json, f, indent=2)
    
    print(f"Mini MAMA-MIA dataset created!")
    print(f"Location: {output_dir}")
    print(f"Cases processed: {processed}")
    print(f"Target size: {target_size}")
    print(f"Dataset JSON: Created")
    
    return output_dir, processed


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Create Mini MAMA-MIA Dataset")
    parser.add_argument("--base_dir", type=str, required=True, 
                       help="Base directory with MAMA-MIA data")
    parser.add_argument("--num_cases", type=int, default=5,
                       help="Number of cases to process")
    parser.add_argument("--size_z", type=int, default=32, help="Target Z size")
    parser.add_argument("--size_y", type=int, default=64, help="Target Y size")  
    parser.add_argument("--size_x", type=int, default=64, help="Target X size")
    
    args = parser.parse_args()
    
    target_size = (args.size_z, args.size_y, args.size_x)
    
    result = create_mini_mama_mia(args.base_dir, args.num_cases, target_size)
    
    if result:
        output_dir, processed = result
        print(f"\nReady for nnU-Net!")
        print(f"Next steps:")
        print(f"1. Move dataset: Move-Item {output_dir} Dataset002_MiniMAMA")
        print(f"2. Train: nnUNetv2_plan_and_preprocess -d 002")
        print(f"3. Train: nnUNetv2_train 002 3d_fullres 0")
    

if __name__ == "__main__":
    main()