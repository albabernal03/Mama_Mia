import os
import json
import pandas as pd
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict
import pickle
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from functools import partial


def analyze_cropping_results(splits_file, original_images_dir, original_segs_dir,
                             cropped_data_dir, output_dir="analysis_results"):
    """
    Comprehensive analysis comparing original vs cropped data with multiprocessing and checkpoints
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    splits_df = pd.read_csv(splits_file)

    # Initialize storage
    analysis_data = {
        'size_comparison': {},
        'original_projections': [],
        'cropped_projections': [],
        'original_mid_slices': [],
        'cropped_mid_slices': [],
        'statistics': {}
    }

    print("Starting comprehensive analysis with multiprocessing and checkpoints...")

    # Prepare patient list
    patients = []
    for _, row in splits_df.iterrows():
        for split_type in ['train_split', 'test_split']:
            patient_id = row[split_type]
            if pd.isna(patient_id):
                continue
            patients.append((patient_id, split_type))

    print(f"Processing {len(patients)} patients...")

    # 1. Size and dimension analysis
    size_checkpoint = output_path / "size_analysis_checkpoint.pkl"
    if size_checkpoint.exists():
        print("1. Loading existing size analysis...")
        with open(size_checkpoint, 'rb') as f:
            analysis_data['size_comparison'] = pickle.load(f)
    else:
        print("1. Analyzing sizes and dimensions...")
        size_data = analyze_sizes_and_dimensions_mp(patients, original_images_dir, original_segs_dir, cropped_data_dir)
        analysis_data['size_comparison'] = size_data
        with open(size_checkpoint, 'wb') as f:
            pickle.dump(size_data, f)
        print("   Size analysis saved to checkpoint")

    # 2. Tumor projection analysis
    proj_checkpoint = output_path / "projections_checkpoint.pkl"
    if proj_checkpoint.exists():
        print("2. Loading existing projection analysis...")
        with open(proj_checkpoint, 'rb') as f:
            proj_data = pickle.load(f)
        analysis_data.update(proj_data)
    else:
        print("2. Analyzing tumor projections...")
        proj_data = analyze_tumor_projections_mp(patients, original_segs_dir, cropped_data_dir)
        analysis_data.update(proj_data)
        with open(proj_checkpoint, 'wb') as f:
            pickle.dump(proj_data, f)
        print("   Projection analysis saved to checkpoint")

    # 3. Average mid-slice analysis
    slice_checkpoint = output_path / "slices_checkpoint.pkl"
    if slice_checkpoint.exists():
        print("3. Loading existing mid-slice analysis...")
        with open(slice_checkpoint, 'rb') as f:
            slice_data = pickle.load(f)
        analysis_data.update(slice_data)
    else:
        print("3. Analyzing mid-slice averages...")
        slice_data = analyze_mid_slices_mp(patients, original_images_dir, cropped_data_dir)
        analysis_data.update(slice_data)
        with open(slice_checkpoint, 'wb') as f:
            pickle.dump(slice_data, f)
        print("   Mid-slice analysis saved to checkpoint")

    # 4. Additional analysis
    stats_checkpoint = output_path / "statistics_checkpoint.pkl"
    if stats_checkpoint.exists():
        print("4. Loading existing statistical analysis...")
        with open(stats_checkpoint, 'rb') as f:
            stats_data = pickle.load(f)
        analysis_data['statistics'].update(stats_data)
    else:
        print("4. Additional statistical analysis...")
        stats_data = additional_analysis_mp(patients, original_images_dir, original_segs_dir, cropped_data_dir)
        analysis_data['statistics'].update(stats_data)
        with open(stats_checkpoint, 'wb') as f:
            pickle.dump(stats_data, f)
        print("   Statistical analysis saved to checkpoint")

    # 5. Generate all plots
    plots_exist = all([
        (output_path / "size_analysis.png").exists(),
        (output_path / "tumor_projections.png").exists(),
        (output_path / "mid_slice_averages.png").exists()
    ])

    if plots_exist:
        print("5. Plots already exist, skipping visualization generation...")
    else:
        print("5. Generating visualizations...")
        create_all_visualizations(analysis_data, output_path)

    # 6. Save complete analysis data
    with open(output_path / "analysis_data_complete.pkl", 'wb') as f:
        pickle.dump(analysis_data, f)

    # 7. Generate summary report
    if not (output_path / "summary_report.txt").exists():
        generate_summary_report(analysis_data, output_path)
    else:
        print("7. Summary report already exists")

    print(f"Analysis complete! Results saved in {output_path}")


def process_patient_sizes(args):
    """Process single patient for size analysis"""
    patient_id, split_type, original_images_dir, original_segs_dir, cropped_data_dir = args

    result = {
        'original_img_sizes': [],
        'cropped_img_sizes': [],
        'original_seg_sizes': [],
        'cropped_seg_sizes': [],
        'original_dimensions': [],
        'cropped_dimensions': []
    }

    try:
        # Original images
        orig_img_dir = Path(original_images_dir) / patient_id
        orig_img_files = list(orig_img_dir.glob("000.nii.gz"))

        # Cropped images
        crop_img_dir = Path(cropped_data_dir) / "images" / split_type / patient_id
        crop_img_files = list(crop_img_dir.glob("*cropped.nii.gz"))

        for orig_file in orig_img_files:
            if orig_file.exists():
                # File sizes
                orig_size = orig_file.stat().st_size / (1024 * 1024)  # MB
                result['original_img_sizes'].append(orig_size)

                # Dimensions
                orig_nii = nib.load(orig_file)
                orig_shape = orig_nii.get_fdata().shape
                result['original_dimensions'].append(orig_shape)

        for crop_file in crop_img_files:
            if crop_file.exists():
                # File sizes
                crop_size = crop_file.stat().st_size / (1024 * 1024)  # MB
                result['cropped_img_sizes'].append(crop_size)

                # Dimensions
                crop_nii = nib.load(crop_file)
                crop_shape = crop_nii.get_fdata().shape
                result['cropped_dimensions'].append(crop_shape)

        # Segmentation sizes
        orig_seg = Path(original_segs_dir) / "expert" / f"{patient_id.lower()}.nii.gz"
        crop_seg = Path(
            cropped_data_dir) / "segmentations" / split_type / patient_id / f"{patient_id.lower()}_seg_cropped.nii.gz"

        if orig_seg.exists():
            result['original_seg_sizes'].append(orig_seg.stat().st_size / (1024 * 1024))
        if crop_seg.exists():
            result['cropped_seg_sizes'].append(crop_seg.stat().st_size / (1024 * 1024))

    except Exception as e:
        print(f"Error processing {patient_id}: {e}")

    return result


def analyze_sizes_and_dimensions_mp(patients, original_images_dir, original_segs_dir, cropped_data_dir):
    """Analyze file sizes and dimensions with multiprocessing"""

    # Prepare arguments
    args = [(patient_id, split_type, original_images_dir, original_segs_dir, cropped_data_dir)
            for patient_id, split_type in patients]

    # Use multiprocessing
    num_cores = min(cpu_count(), 8)
    with Pool(num_cores) as pool:
        results = list(tqdm(pool.imap(process_patient_sizes, args),
                            total=len(args), desc="Processing sizes"))

    # Combine results
    size_data = {
        'original_img_sizes': [],
        'cropped_img_sizes': [],
        'original_seg_sizes': [],
        'cropped_seg_sizes': [],
        'original_dimensions': [],
        'cropped_dimensions': [],
        'reduction_ratios': []
    }

    for result in results:
        for key in size_data.keys():
            if key in result:
                size_data[key].extend(result[key])

    # Calculate reduction ratios
    if size_data['original_img_sizes'] and size_data['cropped_img_sizes']:
        orig_avg = np.mean(size_data['original_img_sizes'])
        crop_avg = np.mean(size_data['cropped_img_sizes'])
        size_data['reduction_ratios'].append(crop_avg / orig_avg)

    return size_data


def process_patient_projections(args):
    """Process single patient for tumor projections"""
    patient_id, split_type, original_segs_dir, cropped_data_dir = args

    result = {
        'original_projection': None,
        'cropped_projection': None
    }

    try:
        # Original segmentation
        orig_seg_path = Path(original_segs_dir) / "expert" / f"{patient_id.lower()}.nii.gz"
        if orig_seg_path.exists():
            orig_seg = nib.load(orig_seg_path).get_fdata()
            # Project along z-axis (sum positive pixels per x,y position)
            result['original_projection'] = np.sum(orig_seg > 0, axis=2)

        # Cropped segmentation
        crop_seg_path = Path(
            cropped_data_dir) / "segmentations" / split_type / patient_id / f"{patient_id.lower()}_seg_cropped.nii.gz"
        if crop_seg_path.exists():
            crop_seg = nib.load(crop_seg_path).get_fdata()
            result['cropped_projection'] = np.sum(crop_seg > 0, axis=2)

    except Exception as e:
        print(f"Error processing projections for {patient_id}: {e}")

    return result


def analyze_tumor_projections_mp(patients, original_segs_dir, cropped_data_dir):
    """Analyze tumor projections with multiprocessing"""

    args = [(patient_id, split_type, original_segs_dir, cropped_data_dir)
            for patient_id, split_type in patients]

    num_cores = min(cpu_count(), 8)
    with Pool(num_cores) as pool:
        results = list(tqdm(pool.imap(process_patient_projections, args),
                            total=len(args), desc="Processing projections"))

    original_projections = []
    cropped_projections = []

    for result in results:
        if result['original_projection'] is not None:
            original_projections.append(result['original_projection'])
        if result['cropped_projection'] is not None:
            cropped_projections.append(result['cropped_projection'])

    return {
        'original_projections': original_projections,
        'cropped_projections': cropped_projections
    }


def process_patient_mid_slices(args):
    """Process single patient for mid-slice analysis"""
    patient_id, split_type, original_images_dir, cropped_data_dir = args

    result = {
        'original_mid_slices': [],
        'cropped_mid_slices': []
    }

    try:
        # Original images
        orig_img_dir = Path(original_images_dir) / patient_id
        orig_img_files = list(orig_img_dir.glob("*0000.nii.gz"))  # Just first timepoint

        for orig_file in orig_img_files:
            if orig_file.exists():
                orig_img = nib.load(orig_file).get_fdata()
                mid_z = orig_img.shape[2] // 2
                result['original_mid_slices'].append(orig_img[:, :, mid_z])

        # Cropped images
        crop_img_dir = Path(cropped_data_dir) / "images" / split_type / patient_id
        crop_img_files = list(crop_img_dir.glob("*0000*cropped.nii.gz"))

        for crop_file in crop_img_files:
            if crop_file.exists():
                crop_img = nib.load(crop_file).get_fdata()
                mid_z = crop_img.shape[2] // 2
                result['cropped_mid_slices'].append(crop_img[:, :, mid_z])

    except Exception as e:
        print(f"Error processing mid-slices for {patient_id}: {e}")

    return result


def analyze_mid_slices_mp(patients, original_images_dir, cropped_data_dir):
    """Analyze mid-slice averages with multiprocessing"""

    args = [(patient_id, split_type, original_images_dir, cropped_data_dir)
            for patient_id, split_type in patients]

    num_cores = min(cpu_count(), 8)
    with Pool(num_cores) as pool:
        results = list(tqdm(pool.imap(process_patient_mid_slices, args),
                            total=len(args), desc="Processing mid-slices"))

    original_mid_slices = []
    cropped_mid_slices = []

    for result in results:
        original_mid_slices.extend(result['original_mid_slices'])
        cropped_mid_slices.extend(result['cropped_mid_slices'])

    return {
        'original_mid_slices': original_mid_slices,
        'cropped_mid_slices': cropped_mid_slices
    }


def process_patient_stats(args):
    """Process single patient for additional statistics"""
    patient_id, split_type, original_images_dir, original_segs_dir, cropped_data_dir = args

    result = {
        'original_intensities': [],
        'cropped_intensities': [],
        'tumor_coverage': None,
        'volume_reduction': None
    }

    try:
        # Intensity analysis
        orig_img_dir = Path(original_images_dir) / patient_id
        orig_img_files = list(orig_img_dir.glob("*0000.nii.gz"))

        crop_img_dir = Path(cropped_data_dir) / "images" / split_type / patient_id
        crop_img_files = list(crop_img_dir.glob("*0000*cropped.nii.gz"))

        if orig_img_files and crop_img_files:
            orig_img = nib.load(orig_img_files[0]).get_fdata()
            crop_img = nib.load(crop_img_files[0]).get_fdata()

            # Sample intensities (don't store all to save memory)
            sample_size = min(10000, orig_img.size)
            orig_sample = np.random.choice(orig_img.flatten(), sample_size, replace=False)
            crop_sample = np.random.choice(crop_img.flatten(), sample_size, replace=False)

            result['original_intensities'] = orig_sample.tolist()
            result['cropped_intensities'] = crop_sample.tolist()

            # Volume reduction
            orig_voxels = np.prod(orig_img.shape)
            crop_voxels = np.prod(crop_img.shape)
            result['volume_reduction'] = crop_voxels / orig_voxels

        # Tumor coverage analysis
        orig_seg_path = Path(original_segs_dir) / "expert" / f"{patient_id.lower()}.nii.gz"
        crop_seg_path = Path(
            cropped_data_dir) / "segmentations" / split_type / patient_id / f"{patient_id.lower()}_seg_cropped.nii.gz"

        if orig_seg_path.exists() and crop_seg_path.exists():
            orig_seg = nib.load(orig_seg_path).get_fdata()
            crop_seg = nib.load(crop_seg_path).get_fdata()

            orig_tumor_volume = np.sum(orig_seg > 0)
            crop_tumor_volume = np.sum(crop_seg > 0)

            if orig_tumor_volume > 0:
                result['tumor_coverage'] = crop_tumor_volume / orig_tumor_volume

    except Exception as e:
        print(f"Error processing stats for {patient_id}: {e}")

    return result


def additional_analysis_mp(patients, original_images_dir, original_segs_dir, cropped_data_dir):
    """Additional statistical analysis with multiprocessing"""

    args = [(patient_id, split_type, original_images_dir, original_segs_dir, cropped_data_dir)
            for patient_id, split_type in patients]

    num_cores = min(cpu_count(), 8)
    with Pool(num_cores) as pool:
        results = list(tqdm(pool.imap(process_patient_stats, args),
                            total=len(args), desc="Processing statistics"))

    original_intensities = []
    cropped_intensities = []
    tumor_coverages = []
    volume_reductions = []

    for result in results:
        original_intensities.extend(result['original_intensities'])
        cropped_intensities.extend(result['cropped_intensities'])
        if result['tumor_coverage'] is not None:
            tumor_coverages.append(result['tumor_coverage'])
        if result['volume_reduction'] is not None:
            volume_reductions.append(result['volume_reduction'])

    stats = {
        'intensity_stats': {
            'original_mean': np.mean(original_intensities),
            'cropped_mean': np.mean(cropped_intensities),
            'original_std': np.std(original_intensities),
            'cropped_std': np.std(cropped_intensities)
        },
        'tumor_coverage': {
            'mean_coverage': np.mean(tumor_coverages),
            'std_coverage': np.std(tumor_coverages),
            'coverages': tumor_coverages
        },
        'volume_reduction': {
            'mean_reduction': np.mean(volume_reductions),
            'std_reduction': np.std(volume_reductions),
            'reductions': volume_reductions
        }
    }

    return stats


def create_all_visualizations(analysis_data, output_path):
    """Create all visualization plots with proper error handling"""

    # 1. Size comparison plots
    plt.figure(figsize=(15, 10))

    plt.subplot(2, 3, 1)
    plt.boxplot([analysis_data['size_comparison']['original_img_sizes'],
                 analysis_data['size_comparison']['cropped_img_sizes']],
                tick_labels=['Original', 'Cropped'])  # Fixed deprecation warning
    plt.title('Image File Sizes (MB)')
    plt.ylabel('Size (MB)')

    plt.subplot(2, 3, 2)
    orig_dims = np.array(analysis_data['size_comparison']['original_dimensions'])
    crop_dims = np.array(analysis_data['size_comparison']['cropped_dimensions'])

    if len(orig_dims) > 0 and len(crop_dims) > 0:
        x_pos = np.arange(3)
        plt.bar(x_pos - 0.2, np.mean(orig_dims, axis=0), 0.4, alpha=0.7, label='Original')
        plt.bar(x_pos + 0.2, np.mean(crop_dims, axis=0), 0.4, alpha=0.7, label='Cropped')
        plt.xticks(x_pos, ['X', 'Y', 'Z'])
        plt.title('Average Dimensions')
        plt.ylabel('Voxels')
        plt.legend()

    plt.subplot(2, 3, 3)
    if analysis_data['statistics']['volume_reduction']['reductions']:
        plt.hist(analysis_data['statistics']['volume_reduction']['reductions'], bins=20, alpha=0.7)
        plt.title('Volume Reduction Ratio')
        plt.xlabel('Cropped/Original Volume')
        plt.ylabel('Frequency')

    plt.subplot(2, 3, 4)
    if analysis_data['statistics']['tumor_coverage']['coverages']:
        plt.hist(analysis_data['statistics']['tumor_coverage']['coverages'], bins=20, alpha=0.7)
        plt.title('Tumor Coverage (Cropped/Original)')
        plt.xlabel('Coverage Ratio')
        plt.ylabel('Frequency')

    plt.subplot(2, 3, 5)
    plt.boxplot([analysis_data['size_comparison']['original_seg_sizes'],
                 analysis_data['size_comparison']['cropped_seg_sizes']],
                tick_labels=['Original', 'Cropped'])  # Fixed deprecation warning
    plt.title('Segmentation File Sizes (MB)')
    plt.ylabel('Size (MB)')

    plt.tight_layout()
    plt.savefig(output_path / "size_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Tumor projection heatmaps (handle different shapes)
    if analysis_data['original_projections'] and analysis_data['cropped_projections']:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        try:
            # Find common shape for averaging (use minimum dimensions)
            orig_shapes = [proj.shape for proj in analysis_data['original_projections']]
            crop_shapes = [proj.shape for proj in analysis_data['cropped_projections']]

            if orig_shapes:
                min_orig_shape = (min(s[0] for s in orig_shapes), min(s[1] for s in orig_shapes))
                # Resize all original projections to common shape
                orig_resized = []
                for proj in analysis_data['original_projections']:
                    orig_resized.append(proj[:min_orig_shape[0], :min_orig_shape[1]])
                orig_avg = np.mean(orig_resized, axis=0)

                im1 = axes[0, 0].imshow(orig_avg, cmap='hot')
                axes[0, 0].set_title('Average Original Tumor Projection')
                plt.colorbar(im1, ax=axes[0, 0])

                # Example individual projection
                im3 = axes[1, 0].imshow(orig_resized[0], cmap='hot')
                axes[1, 0].set_title('Example Original Projection')
                plt.colorbar(im3, ax=axes[1, 0])

            if crop_shapes:
                min_crop_shape = (min(s[0] for s in crop_shapes), min(s[1] for s in crop_shapes))
                # Resize all cropped projections to common shape
                crop_resized = []
                for proj in analysis_data['cropped_projections']:
                    crop_resized.append(proj[:min_crop_shape[0], :min_crop_shape[1]])
                crop_avg = np.mean(crop_resized, axis=0)

                im2 = axes[0, 1].imshow(crop_avg, cmap='hot')
                axes[0, 1].set_title('Average Cropped Tumor Projection')
                plt.colorbar(im2, ax=axes[0, 1])

                # Example individual projection
                im4 = axes[1, 1].imshow(crop_resized[0], cmap='hot')
                axes[1, 1].set_title('Example Cropped Projection')
                plt.colorbar(im4, ax=axes[1, 1])

        except Exception as e:
            print(f"Warning: Could not create projection plots: {e}")
            # Clear the figure if error
            for ax in axes.flat:
                ax.text(0.5, 0.5, 'Projection analysis\nfailed due to\nshape mismatch',
                        ha='center', va='center', transform=ax.transAxes)

        plt.tight_layout()
        plt.savefig(output_path / "tumor_projections.png", dpi=300, bbox_inches='tight')
        plt.close()

    # 3. Mid-slice averages (handle different shapes)
    if analysis_data['original_mid_slices'] and analysis_data['cropped_mid_slices']:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        try:
            # Handle different shapes for mid-slices too
            orig_slice_shapes = [slice.shape for slice in analysis_data['original_mid_slices']]
            crop_slice_shapes = [slice.shape for slice in analysis_data['cropped_mid_slices']]

            if orig_slice_shapes:
                min_orig_slice_shape = (min(s[0] for s in orig_slice_shapes), min(s[1] for s in orig_slice_shapes))
                orig_resized = []
                for slice in analysis_data['original_mid_slices']:
                    orig_resized.append(slice[:min_orig_slice_shape[0], :min_orig_slice_shape[1]])
                orig_mid_avg = np.mean(orig_resized, axis=0)

                im1 = axes[0].imshow(orig_mid_avg, cmap='gray')
                axes[0].set_title('Average Original Mid-Slice')
                plt.colorbar(im1, ax=axes[0])

            if crop_slice_shapes:
                min_crop_slice_shape = (min(s[0] for s in crop_slice_shapes), min(s[1] for s in crop_slice_shapes))
                crop_resized = []
                for slice in analysis_data['cropped_mid_slices']:
                    crop_resized.append(slice[:min_crop_slice_shape[0], :min_crop_slice_shape[1]])
                crop_mid_avg = np.mean(crop_resized, axis=0)

                im2 = axes[1].imshow(crop_mid_avg, cmap='gray')
                axes[1].set_title('Average Cropped Mid-Slice')
                plt.colorbar(im2, ax=axes[1])

        except Exception as e:
            print(f"Warning: Could not create mid-slice plots: {e}")
            for ax in axes:
                ax.text(0.5, 0.5, 'Mid-slice analysis\nfailed due to\nshape mismatch',
                        ha='center', va='center', transform=ax.transAxes)

        plt.tight_layout()
        plt.savefig(output_path / "mid_slice_averages.png", dpi=300, bbox_inches='tight')
        plt.close()


def generate_summary_report(analysis_data, output_path):
    """Generate a summary report"""
    report = []
    report.append("CROPPING ANALYSIS SUMMARY REPORT")
    report.append("=" * 50)
    report.append("")

    # Size reduction
    if analysis_data['size_comparison']['original_img_sizes'] and analysis_data['size_comparison']['cropped_img_sizes']:
        orig_avg_size = np.mean(analysis_data['size_comparison']['original_img_sizes'])
        crop_avg_size = np.mean(analysis_data['size_comparison']['cropped_img_sizes'])
        size_reduction = (1 - crop_avg_size / orig_avg_size) * 100

        report.append(f"FILE SIZE REDUCTION:")
        report.append(f"  Original average: {orig_avg_size:.2f} MB")
        report.append(f"  Cropped average: {crop_avg_size:.2f} MB")
        report.append(f"  Size reduction: {size_reduction:.1f}%")
        report.append("")

    # Volume reduction
    if analysis_data['statistics']['volume_reduction']['reductions']:
        vol_reduction = np.mean(analysis_data['statistics']['volume_reduction']['reductions'])
        vol_reduction_pct = (1 - vol_reduction) * 100

        report.append(f"VOLUME REDUCTION:")
        report.append(f"  Average volume retention: {vol_reduction:.3f}")
        report.append(f"  Average volume reduction: {vol_reduction_pct:.1f}%")
        report.append("")

    # Tumor coverage
    if analysis_data['statistics']['tumor_coverage']['coverages']:
        coverage = analysis_data['statistics']['tumor_coverage']['mean_coverage']
        coverage_std = analysis_data['statistics']['tumor_coverage']['std_coverage']

        report.append(f"TUMOR COVERAGE:")
        report.append(f"  Mean tumor retention: {coverage:.3f} ± {coverage_std:.3f}")
        report.append(f"  Tumor retention: {coverage * 100:.1f}%")
        report.append("")

    # Dimensions
    if analysis_data['size_comparison']['original_dimensions'] and analysis_data['size_comparison'][
        'cropped_dimensions']:
        orig_dims = np.mean(analysis_data['size_comparison']['original_dimensions'], axis=0)
        crop_dims = np.mean(analysis_data['size_comparison']['cropped_dimensions'], axis=0)

        report.append(f"AVERAGE DIMENSIONS:")
        report.append(f"  Original: {orig_dims[0]:.0f} x {orig_dims[1]:.0f} x {orig_dims[2]:.0f}")
        report.append(f"  Cropped:  {crop_dims[0]:.0f} x {crop_dims[1]:.0f} x {crop_dims[2]:.0f}")
        report.append("")

    report.append(f"Analysis completed successfully!")
    report.append(f"Results saved in: {output_path}")

    with open(output_path / "summary_report.txt", 'w') as f:
        f.write("\n".join(report))

    print("\n".join(report))


if __name__ == "__main__":
    analyze_cropping_results(
        splits_file= r"C:\Users\usuario\Documents\Mama_Mia\datos\train_test_splits.csv",
        original_images_dir=r"C:\Users\usuario\Documents\Mama_Mia\datos\images",
        original_segs_dir=r"C:\Users\usuario\Documents\Mama_Mia\datos\segmentations",
        cropped_data_dir=r"C:\Users\usuario\Documents\Mama_Mia\cropped_data",
        output_dir=r"C:\Users\usuario\Documents\Mama_Mia\analisis\analysis_results"
    )