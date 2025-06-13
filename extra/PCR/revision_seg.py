import os
import numpy as np
import nibabel as nib
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

def analyze_segmentation_quality(cropped_data_dir):
    cropped_data_dir = Path(cropped_data_dir)
    seg_dir = cropped_data_dir / "segmentations"
    
    results = []

    for split in ['train_split', 'test_split']:
        split_dir = seg_dir / split
        patients = sorted([p.name for p in split_dir.iterdir() if p.is_dir()])

        print(f"\n🔍 Analyzing {split} ({len(patients)} patients)...")

        for patient_id in tqdm(patients):
            seg_path = split_dir / patient_id / f"{patient_id.lower()}_seg_cropped.nii.gz"
            if not seg_path.exists():
                print(f"❌ Segmentation not found for {patient_id}")
                continue

            mask = nib.load(seg_path).get_fdata()
            tumor_voxels = np.sum(mask > 0)

            results.append({
                'patient_id': patient_id,
                'split': split,
                'tumor_voxels': tumor_voxels
            })

    results_df = pd.DataFrame(results)

    # --- Estadísticas globales
    print("\n📊 Tumor Size Statistics:")
    print(results_df.groupby('split')['tumor_voxels'].describe())

    # --- Pacientes con problemas
    empty_masks = results_df[results_df['tumor_voxels'] == 0]
    small_tumors = results_df[(results_df['tumor_voxels'] > 0) & (results_df['tumor_voxels'] < 10)]

    print(f"\n❌ Patients with empty masks (no tumor): {len(empty_masks)}")
    print(empty_masks[['patient_id', 'split']])

    print(f"\n⚠️ Patients with very small tumors (<10 voxels): {len(small_tumors)}")
    print(small_tumors[['patient_id', 'split']])

    # --- Histograma
    plt.figure(figsize=(10, 5))
    plt.hist(results_df['tumor_voxels'], bins=30, color='steelblue', edgecolor='black')
    plt.title('Tumor Size Distribution (voxels)')
    plt.xlabel('Number of Tumor Voxels')
    plt.ylabel('Number of Patients')
    plt.grid(True)
    plt.show()

    # --- Guardar resultados
    results_df.to_csv("tumor_size_analysis.csv", index=False)
    print("\n✅ Analysis complete! Results saved to 'tumor_size_analysis.csv'.")

if __name__ == "__main__":
    # CAMBIA esto a tu carpeta de datos
    cropped_data_dir = r"C:\Users\usuario\Documents\Mama_Mia\cropped_data"
    analyze_segmentation_quality(cropped_data_dir)
