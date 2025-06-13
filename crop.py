import os
import pandas as pd
import nibabel as nib
import numpy as np
from pathlib import Path
from multiprocessing import Pool, cpu_count
from functools import partial


def crop_around_mask_data(splits_file, images_dir, segmentations_dir, output_dir, margin_percent=15):
    """Crop images and masks around GT mask area with margin using multiprocessing"""
    splits_df = pd.read_csv(splits_file)

    output_images = Path(output_dir) / "images"
    output_segs = Path(output_dir) / "segmentations"
    output_images.mkdir(parents=True, exist_ok=True)
    output_segs.mkdir(parents=True, exist_ok=True)

    # Prepare patient tasks
    tasks = []
    for _, row in splits_df.iterrows():
        for split_type in ['train_split', 'test_split']:
            patient_id = row[split_type]
            if pd.isna(patient_id) or str(patient_id).strip() in ['', 'nan', 'None']:
                continue
            patient_id = str(patient_id).strip()
            tasks.append((patient_id, split_type, images_dir, segmentations_dir,
                          output_images, output_segs, margin_percent))

    # Use multiprocessing
    num_cores = min(cpu_count(), 12)
    print(f"Processing {len(tasks)} patients using {num_cores} cores...")
    print(f"Margin: {margin_percent}% on each side")

    with Pool(num_cores) as pool:
        results = pool.map(process_patient_wrapper, tasks)

    successful = sum(results)
    print(f"Processing complete! {successful}/{len(tasks)} patients processed successfully.")


def process_patient_wrapper(args):
    """Wrapper function for multiprocessing"""
    patient_id, split_type, images_dir, segmentations_dir, output_images, output_segs, margin_percent = args
    return process_patient(patient_id, images_dir, segmentations_dir, output_images,
                           output_segs, split_type, margin_percent)


def get_mask_bounding_box(mask_data, margin_percent=15):
    """
    Get bounding box around non-zero mask regions with margin

    Args:
        mask_data: 3D numpy array of the segmentation mask
        margin_percent: Percentage margin to add on each side

    Returns:
        Tuple of (x_min, x_max, y_min, y_max, z_min, z_max) with margin
    """
    # Find non-zero regions
    nonzero_coords = np.where(mask_data > 0)

    if len(nonzero_coords[0]) == 0:
        # No segmentation found, return None
        return None

    # Get bounding box coordinates
    x_min, x_max = nonzero_coords[0].min(), nonzero_coords[0].max()
    y_min, y_max = nonzero_coords[1].min(), nonzero_coords[1].max()
    z_min, z_max = nonzero_coords[2].min(), nonzero_coords[2].max()

    # Calculate margins
    x_margin = int((x_max - x_min) * margin_percent / 100)
    y_margin = int((y_max - y_min) * margin_percent / 100)
    z_margin = int((z_max - z_min) * margin_percent / 100)

    # Add margins and ensure within bounds
    shape = mask_data.shape
    x_min_margin = max(0, x_min - x_margin)
    x_max_margin = min(shape[0] - 1, x_max + x_margin)
    y_min_margin = max(0, y_min - y_margin)
    y_max_margin = min(shape[1] - 1, y_max + y_margin)
    z_min_margin = max(0, z_min - z_margin)
    z_max_margin = min(shape[2] - 1, z_max + z_margin)

    return (x_min_margin, x_max_margin, y_min_margin, y_max_margin, z_min_margin, z_max_margin)


def process_patient(patient_id, images_dir, segmentations_dir, output_images,
                    output_segs, split_type, margin_percent):
    """Process single patient data"""
    try:
        # First, load the segmentation to get bounding box
        seg_path = Path(segmentations_dir) / "expert" / f"{patient_id.lower()}.nii.gz"

        if not seg_path.exists():
            print(f"✗ {patient_id} ({split_type}): Segmentation file not found")
            return False

        # Load segmentation
        seg_nii = nib.load(seg_path)
        seg_data = seg_nii.get_fdata()

        # Get bounding box around mask
        bbox = get_mask_bounding_box(seg_data, margin_percent)

        if bbox is None:
            print(f"✗ {patient_id} ({split_type}): No segmentation found in mask")
            return False

        x_min, x_max, y_min, y_max, z_min, z_max = bbox

        # Crop segmentation
        cropped_seg = seg_data[x_min:x_max + 1, y_min:y_max + 1, z_min:z_max + 1]

        # Save cropped segmentation
        cropped_seg_nii = nib.Nifti1Image(cropped_seg.astype(seg_data.dtype), seg_nii.affine, seg_nii.header)

        output_seg_dir = output_segs / split_type / patient_id
        output_seg_dir.mkdir(parents=True, exist_ok=True)
        nib.save(cropped_seg_nii, output_seg_dir / f"{patient_id.lower()}_seg_cropped.nii.gz")

        # MODIFICADO: Process ALL images with the same bounding box
        # Buscar imágenes directamente en images_dir, no en subcarpetas
        images_path = Path(images_dir)  # CAMBIO: no agregar / patient_id
        patient_name_lower = patient_id.lower()

        # Find ALL image files for this patient DIRECTAMENTE en la carpeta images
        img_files = list(images_path.glob(f"{patient_name_lower}_*.nii.gz"))

        if not img_files:
            print(f"✗ {patient_id} ({split_type}): No image files found matching {patient_name_lower}_*.nii.gz")
            return False

        processed_images = 0
        failed_images = 0

        # Process each image file
        for img_path in img_files:
            try:
                # Load and crop image using the same bounding box
                img_nii = nib.load(img_path)
                img_data = img_nii.get_fdata()

                # Check if image and segmentation have same shape
                if img_data.shape != seg_data.shape:
                    print(
                        f"  ⚠ {patient_id} ({split_type}): Skipping {img_path.name} - shape mismatch: {img_data.shape} vs {seg_data.shape}")
                    failed_images += 1
                    continue

                # Crop image with same bounding box
                cropped_img = img_data[x_min:x_max + 1, y_min:y_max + 1, z_min:z_max + 1]

                # Save cropped image
                cropped_img_nii = nib.Nifti1Image(cropped_img.astype(img_data.dtype), img_nii.affine, img_nii.header)

                output_img_dir = output_images / split_type / patient_id
                output_img_dir.mkdir(parents=True, exist_ok=True)

                original_name = img_path.stem.replace('.nii', '')
                nib.save(cropped_img_nii, output_img_dir / f"{original_name}_cropped.nii.gz")

                processed_images += 1

            except Exception as e:
                print(f"  ⚠ {patient_id} ({split_type}): Failed to process {img_path.name}: {str(e)}")
                failed_images += 1

        if processed_images == 0:
            print(f"✗ {patient_id} ({split_type}): No images could be processed")
            return False

        # Calculate crop info for logging
        original_shape = seg_data.shape
        cropped_shape = cropped_seg.shape
        reduction = [f"{(1 - c / o) * 100:.1f}%" for o, c in zip(original_shape, cropped_shape)]

        print(f"✓ {patient_id} ({split_type}): {processed_images} images processed, {failed_images} failed. "
              f"Shape: {original_shape} -> {cropped_shape} "
              f"(reduction: {reduction[0]}, {reduction[1]}, {reduction[2]})")

        return True

    except Exception as e:
        print(f"✗ Error processing {patient_id} ({split_type}): {str(e)}")
        return False

if __name__ == "__main__":
    crop_around_mask_data(
        splits_file=r"C:\Users\FX507\Documents\GitHub\Mama_Mia\datos\train_test_splits.csv",
        images_dir=r"C:\Users\FX507\Documents\GitHub\Mama_Mia\datos\images",
        segmentations_dir=r"C:\Users\FX507\Documents\GitHub\Mama_Mia\datos\segmentations",
        output_dir=r"datos\mask_cropped_data",
        margin_percent=15
    )