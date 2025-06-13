import os
import json
import pandas as pd
import nibabel as nib
import numpy as np
from pathlib import Path
from multiprocessing import Pool, cpu_count
from functools import partial


def crop_and_save_data(splits_file, images_dir, segmentations_dir, json_dir, output_dir):
    """Crop breast regions from images and segmentations using multiprocessing"""
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
            if pd.isna(patient_id):
                continue
            tasks.append((patient_id, split_type, images_dir, segmentations_dir,
                          json_dir, output_images, output_segs))

    # Use multiprocessing
    num_cores = min(cpu_count(), 12)  # Use up to 12 cores for i9
    print(f"Processing {len(tasks)} patients using {num_cores} cores...")

    with Pool(num_cores) as pool:
        pool.map(process_patient_wrapper, tasks)

    print("All patients processed!")


def process_patient_wrapper(args):
    """Wrapper function for multiprocessing"""
    patient_id, split_type, images_dir, segmentations_dir, json_dir, output_images, output_segs = args
    return process_patient(patient_id, images_dir, segmentations_dir, json_dir,
                           output_images, output_segs, split_type)


def get_crop_coordinates(coords, img_shape):
    """Convert JSON coordinates to image crop coordinates with axis remapping"""
    x_min, x_max = coords['x_min'], coords['x_max']
    y_min, y_max = coords['y_min'], coords['y_max']
    z_min, z_max = coords['z_min'], coords['z_max']

    # JSON coordinates (x,y,z) map to image (z,y,x)
    crop_x_min = max(0, min(z_min, img_shape[0] - 1))
    crop_x_max = max(0, min(z_max, img_shape[0] - 1))
    crop_y_min = max(0, min(y_min, img_shape[1] - 1))
    crop_y_max = max(0, min(y_max, img_shape[1] - 1))
    crop_z_min = max(0, min(x_min, img_shape[2] - 1))
    crop_z_max = max(0, min(x_max, img_shape[2] - 1))

    return crop_x_min, crop_x_max, crop_y_min, crop_y_max, crop_z_min, crop_z_max


def process_patient(patient_id, images_dir, segmentations_dir, json_dir,
                    output_images, output_segs, split_type):
    """Process single patient data"""
    try:
        # Load coordinates
        json_file = Path(json_dir) / f"{patient_id.lower()}.json"
        with open(json_file, 'r') as f:
            data = json.load(f)
        coords = data['primary_lesion']['breast_coordinates']

        # Process images
        patient_img_dir = Path(images_dir)
        # Convert patient ID to lowercase and replace underscore with underscore (NACT_01 -> nact_01)
        patient_name_lower = patient_id.lower()
        # Find all images for this patient (e.g., nact_01_0000.nii.gz, nact_01_0001.nii.gz)
        img_files = list(patient_img_dir.glob(f"{patient_name_lower}_*.nii.gz"))

        processed_imgs = 0
        for img_path in img_files:
            img_nii = nib.load(img_path)
            img_data = img_nii.get_fdata()

            # Get crop coordinates
            x_min, x_max, y_min, y_max, z_min, z_max = get_crop_coordinates(coords, img_data.shape)

            # Crop image
            cropped_img = img_data[x_min:x_max + 1, y_min:y_max + 1, z_min:z_max + 1]

            if cropped_img.size == 0:
                continue

            # Save cropped image
            cropped_img_nii = nib.Nifti1Image(cropped_img.astype(img_data.dtype), img_nii.affine, img_nii.header)

            output_img_dir = output_images / split_type / patient_id
            output_img_dir.mkdir(parents=True, exist_ok=True)

            original_name = img_path.stem.replace('.nii', '')
            nib.save(cropped_img_nii, output_img_dir / f"{original_name}_cropped.nii.gz")
            processed_imgs += 1

        # Process segmentation
        seg_processed = 0
        seg_path = Path(segmentations_dir) / "expert" / f"{patient_id.lower()}.nii.gz"
        if seg_path.exists():
            seg_nii = nib.load(seg_path)
            seg_data = seg_nii.get_fdata()

            # Get crop coordinates for segmentation
            x_min, x_max, y_min, y_max, z_min, z_max = get_crop_coordinates(coords, seg_data.shape)

            # Crop segmentation
            cropped_seg = seg_data[x_min:x_max + 1, y_min:y_max + 1, z_min:z_max + 1]

            if cropped_seg.size > 0:
                cropped_seg_nii = nib.Nifti1Image(cropped_seg.astype(seg_data.dtype), seg_nii.affine, seg_nii.header)

                output_seg_dir = output_segs / split_type / patient_id
                output_seg_dir.mkdir(parents=True, exist_ok=True)
                nib.save(cropped_seg_nii, output_seg_dir / f"{patient_id.lower()}_seg_cropped.nii.gz")
                seg_processed = 1

        print(f"✓ {patient_id} ({split_type}): {processed_imgs} images, {seg_processed} segmentation")
        return True

    except Exception as e:
        print(f"✗ Error processing {patient_id} ({split_type}): {str(e)}")
        return False


if __name__ == "__main__":
    # CAMBIAR ESTAS RUTAS POR LAS TUYAS
    crop_and_save_data(
        splits_file=r"C:\Users\usuario\Documents\Mama_Mia\datos\train_test_splits.csv",
        images_dir=r"C:\Users\usuario\Documents\Mama_Mia\datos\images",
        segmentations_dir=r"C:\Users\usuario\Documents\Mama_Mia\datos\segmentations",
        json_dir=r"C:\Users\usuario\Documents\Mama_Mia\datos\patient_info_files",
        output_dir=r"C:\Users\usuario\Documents\Mama_Mia\cropped_data"
    )