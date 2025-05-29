#!/usr/bin/env python3
"""
Simple Toy Dataset Creator for nnU-Net Convergence Testing
Creates synthetic 3D volumes with spheres/cubes as "tumors"
"""

import numpy as np
import SimpleITK as sitk
from pathlib import Path
import json

def create_toy_dataset(num_cases=10, size=64):
    """Create simple toy dataset with geometric shapes"""
    
    # Create directories
    output_dir = Path("toy_dataset")
    (output_dir / "imagesTr").mkdir(parents=True, exist_ok=True)
    (output_dir / "labelsTr").mkdir(parents=True, exist_ok=True)
    
    print(f"Creating {num_cases} toy cases of size {size}x{size}x{size}")
    
    for i in range(num_cases):
        print(f"Creating case {i+1}/{num_cases}")
        
        # Create empty volume
        volume = np.zeros((size, size, size), dtype=np.float32)
        mask = np.zeros((size, size, size), dtype=np.uint8)
        
        # Add 2-3 random spheres as "tumors"
        num_spheres = np.random.randint(2, 4)
        
        for s in range(num_spheres):
            # Random center and radius
            center_z = np.random.randint(10, size-10)
            center_y = np.random.randint(10, size-10) 
            center_x = np.random.randint(10, size-10)
            radius = np.random.randint(5, 10)
            
            # Create sphere
            for z in range(size):
                for y in range(size):
                    for x in range(size):
                        dist = np.sqrt((z-center_z)**2 + (y-center_y)**2 + (x-center_x)**2)
                        if dist <= radius:
                            volume[z, y, x] = 1.0
                            mask[z, y, x] = 1
        
        # Add some noise
        volume += np.random.normal(0, 0.1, volume.shape)
        
        # Normalize volume
        if volume.std() > 0:
            volume = (volume - volume.mean()) / volume.std()
        
        # Convert to SimpleITK and save
        volume_sitk = sitk.GetImageFromArray(volume)
        mask_sitk = sitk.GetImageFromArray(mask)
        
        # Set spacing to 1x1x1 mm
        volume_sitk.SetSpacing((1.0, 1.0, 1.0))
        mask_sitk.SetSpacing((1.0, 1.0, 1.0))
        
        # Save files
        sitk.WriteImage(volume_sitk, str(output_dir / "imagesTr" / f"case_{i:03d}_0000.nii.gz"))
        sitk.WriteImage(mask_sitk, str(output_dir / "labelsTr" / f"case_{i:03d}.nii.gz"))
    
    # Create dataset.json for nnU-Net
    dataset_json = {
        "channel_names": {
            "0": "synthetic_volume"
        },
        "labels": {
            "background": 0,
            "tumor": 1
        },
        "numTraining": num_cases,
        "file_ending": ".nii.gz",
        "dataset_name": "ToyTest",
        "description": "Synthetic spheres for convergence testing"
    }
    
    with open(output_dir / "dataset.json", 'w') as f:
        json.dump(dataset_json, f, indent=2)
    
    print(f"Toy dataset created in: {output_dir}")
    print(f"Files created:")
    print(f"  - {num_cases} images in imagesTr/")
    print(f"  - {num_cases} masks in labelsTr/")
    print(f"  - dataset.json")
    
    return output_dir

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_cases", type=int, default=10)
    parser.add_argument("--size", type=int, default=64)
    
    args = parser.parse_args()
    
    create_toy_dataset(args.num_cases, args.size)