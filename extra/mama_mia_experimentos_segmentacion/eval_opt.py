import numpy as np
import nibabel as nib
import pandas as pd
from medpy.metric import binary
from pathlib import Path
from tqdm import tqdm

def load_nifti(file_path):
    nii = nib.load(file_path)
    return nii.get_fdata()

def dice_score(pred, gt):
    pred_bin = pred > 0.5
    gt_bin = gt > 0.5
    intersection = np.sum(pred_bin * gt_bin)
    total = np.sum(pred_bin) + np.sum(gt_bin)
    if total == 0:
        return 1.0
    return (2.0 * intersection) / total

def hausdorff_95(pred, gt):
    pred_bin = pred > 0.5
    gt_bin = gt > 0.5
    if np.sum(pred_bin) == 0 or np.sum(gt_bin) == 0:
        return np.nan
    return binary.hd95(pred_bin, gt_bin)

def evaluate_case(case_id):
    base_path = Path(r"C:\Users\usuario\Documents\Mama_Mia")
    baseline_results = base_path / "replicacion_definitiva" / "results_output"
    improved_results = Path("./results_adaptive_postprocessing_advanced")
    ground_truth_path = base_path / "datos" / "segmentations" / "expert"
    
    baseline_file = baseline_results / f"{case_id}.nii.gz"
    improved_file = improved_results / f"{case_id}.nii.gz"
    gt_file = ground_truth_path / f"{case_id.lower()}.nii.gz"
    
    if not (baseline_file.exists() and improved_file.exists() and gt_file.exists()):
        print(f"❌ Faltan archivos para {case_id}")
        return None
    
    # Cargar datos
    baseline_seg = load_nifti(baseline_file)
    improved_seg = load_nifti(improved_file)
    gt_seg = load_nifti(gt_file)

    # Calcular métricas
    baseline_dice = dice_score(baseline_seg, gt_seg)
    improved_dice = dice_score(improved_seg, gt_seg)
    baseline_hd95 = hausdorff_95(baseline_seg, gt_seg)
    improved_hd95 = hausdorff_95(improved_seg, gt_seg)
    
    return {
        'case_id': case_id,
        'baseline_dice': round(baseline_dice, 4),
        'improved_dice': round(improved_dice, 4),
        'baseline_hd95': round(baseline_hd95, 2) if not np.isnan(baseline_hd95) else np.nan,
        'improved_hd95': round(improved_hd95, 2) if not np.isnan(improved_hd95) else np.nan
    }

def evaluate_all(test_cases_csv):
    # Cargar lista de casos
    df = pd.read_csv(test_cases_csv)
    test_cases = df['test_split'].dropna().tolist()
    
    results = []
    
    for case_id in tqdm(test_cases, desc="Evaluando casos"):
        result = evaluate_case(case_id)
        if result:
            results.append(result)
    
    # Guardar resultados
    results_df = pd.DataFrame(results)
    results_df.to_csv("results_comparison.csv", index=False)
    print("\n✅ Evaluación completada. Resultados guardados en 'results_comparison.csv'")
    return results_df

# =========================================================
# 🚀 USO
# =========================================================
if __name__ == "__main__":
    test_cases_csv = r"C:\Users\usuario\Documents\Mama_Mia\datos\train_test_splits.csv"
    evaluate_all(test_cases_csv)
