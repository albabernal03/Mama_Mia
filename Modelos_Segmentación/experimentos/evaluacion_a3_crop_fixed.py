import pandas as pd
import SimpleITK as sitk
import numpy as np
from pathlib import Path

def dice_coefficient(pred, gt, smooth=1e-6):
    intersection = np.sum(pred * gt)
    union = np.sum(pred) + np.sum(gt)
    return (2. * intersection + smooth) / (union + smooth)

# === CONFIGURA AQUÍ EL CASE ID (case_XXX)
case_id = "case_004"

# === RUTAS BASE
base_pred = Path(r"C:\Users\usuario\Documents\Mama_Mia\experimentos\complete_a3_vs_crop_tumor\crop_tumor_predictions")
base_gt = Path(r"C:\Users\usuario\Documents\Mama_Mia\cropped_data\segmentations\test_split")
csv_file = Path(r"C:\Users\usuario\Documents\Mama_Mia\datos\train_test_splits.csv")

# === Leer CSV para mapear case_004 → DUKE_XXX
df = pd.read_csv(csv_file)
test_cases = df['test_split'].dropna().tolist()

# Obtener índice del case
num = int(case_id.replace("case_", ""))
if num >= len(test_cases):
    raise ValueError(f"❌ case index {num} fuera de rango (solo hay {len(test_cases)} en test_split)")

duke_case = test_cases[num]
print(f"🔄 {case_id} → {duke_case}")

# === Rutas de archivos
pred_file = base_pred / f"{case_id}.nii.gz"
gt_dir = base_gt / duke_case
gt_candidates = list(gt_dir.glob("*.nii.gz"))

# === Validaciones
if not pred_file.exists():
    raise FileNotFoundError(f"❌ Predicción no encontrada: {pred_file}")
if not gt_candidates:
    raise FileNotFoundError(f"❌ No se encontró ground truth en: {gt_dir}")
elif len(gt_candidates) > 1:
    print(f"⚠️ Múltiples GT en {gt_dir}, usando: {gt_candidates[0].name}")
gt_file = gt_candidates[0]

# === Cargar imágenes
pred = sitk.GetArrayFromImage(sitk.ReadImage(str(pred_file)))
gt = sitk.GetArrayFromImage(sitk.ReadImage(str(gt_file)))

pred_bin = (pred > 0.5).astype(np.uint8)
gt_bin = (gt > 0).astype(np.uint8)

# === Validar formas
if pred_bin.shape != gt_bin.shape:
    raise ValueError(f"❌ Shape mismatch: pred {pred_bin.shape} vs GT {gt_bin.shape}")

# === Calcular DICE
dice = dice_coefficient(pred_bin, gt_bin)
print(f"✅ Dice para {case_id} / {duke_case}: {dice:.4f}")
