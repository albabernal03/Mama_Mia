import nibabel as nib
import numpy as np
import os
from tqdm import tqdm

def dice_coefficient(pred, gt):
    pred = pred > 0
    gt = gt > 0
    intersection = np.logical_and(pred, gt).sum()
    return 2 * intersection / (pred.sum() + gt.sum() + 1e-8)

def get_bbox_with_margin(mask, margin_percent=15):
    coords = np.array(np.where(mask > 0))
    zmin, ymin, xmin = coords.min(axis=1)
    zmax, ymax, xmax = coords.max(axis=1) + 1

    D, H, W = mask.shape
    dz = int((zmax - zmin) * margin_percent / 100)
    dy = int((ymax - ymin) * margin_percent / 100)
    dx = int((xmax - xmin) * margin_percent / 100)

    z1, z2 = max(zmin - dz, 0), min(zmax + dz, D)
    y1, y2 = max(ymin - dy, 0), min(ymax + dy, H)
    x1, x2 = max(xmin - dx, 0), min(xmax + dx, W)

    return z1, z2, y1, y2, x1, x2

# Paths
gt_dir = r"C:\nnUNet_raw\Dataset112_A2_Baseline_AllPhases\labelsTs"
pred_dir = r"C:\Users\usuario\Documents\Mama_Mia\experimentos\complete_a3_vs_crop_tumor\crop_tumor_predictions"

# Obtener lista de predicciones
pred_files = sorted([f for f in os.listdir(pred_dir) if f.endswith(".nii.gz")])

dice_scores = []

for fname in tqdm(pred_files):
    case_id = fname.replace(".nii.gz", "")
    gt_path = os.path.join(gt_dir, f"{case_id}.nii.gz")
    pred_path = os.path.join(pred_dir, fname)

    if not os.path.exists(gt_path):
        print(f"❌ No encontrado GT: {gt_path}")
        continue

    try:
        gt_img = nib.load(gt_path)
        gt_data = gt_img.get_fdata().astype(np.uint8)

        pred_img = nib.load(pred_path)
        pred_data = pred_img.get_fdata().astype(np.uint8)

        # Bounding box desde GT
        z1, z2, y1, y2, x1, x2 = get_bbox_with_margin(gt_data, margin_percent=15)

        gt_shape = gt_data.shape
        crop_shape = (z2-z1, y2-y1, x2-x1)

        if pred_data.shape != crop_shape:
            print(f"⚠️ Shape mismatch en {case_id}: pred={pred_data.shape}, esperado={crop_shape}")
            continue

        # Crear máscara vacía y colocar predicción
        pred_full = np.zeros_like(gt_data)
        pred_full[z1:z2, y1:y2, x1:x2] = pred_data > 0.5

        dice = dice_coefficient(pred_full, gt_data)
        dice_scores.append((case_id, dice))
        print(f"✅ {case_id}: Dice = {dice:.4f}")

    except Exception as e:
        print(f"❌ Error en {case_id}: {e}")

# Reporte final
if dice_scores:
    dice_vals = [d[1] for d in dice_scores]
    mean_dice = np.mean(dice_vals)
    std_dice = np.std(dice_vals)
    print(f"\n📈 Dice promedio: {mean_dice:.4f} ± {std_dice:.4f}")
else:
    print("⚠️ No se pudo evaluar ningún caso.")


