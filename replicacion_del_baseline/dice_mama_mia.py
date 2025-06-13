import numpy as np
import nibabel as nib
import os

# Carpeta de predicciones numeradas
predictions_folder = r"D:\output_inference"

# Carpeta base ground truth
ground_truth_base_folder = r"D:\preprocessed_mama_mia_style"

# Archivo de mapeo
mapping_file = r"D:\input_for_inference\map_test_patients.txt"

# Cargar mapeo num->paciente
num_to_patient = {}
with open(mapping_file, "r") as f:
    for line in f:
        num, patient_id = line.strip().split()
        num_to_patient[num] = patient_id

dice_scores = []

for filename in os.listdir(predictions_folder):
    if not filename.endswith(".nii.gz"):
        continue
    num = filename.replace(".nii.gz", "")
    if num not in num_to_patient:
        print(f"⚠️ No mapping for prediction {filename}")
        continue
    patient_id = num_to_patient[num]
    pred_path = os.path.join(predictions_folder, filename)
    gt_path = os.path.join(ground_truth_base_folder, patient_id, f"{patient_id}_segmentation.nii.gz")

    if not os.path.exists(gt_path):
        print(f"⚠️ Ground truth not found for {patient_id}")
        continue

    pred_img = nib.load(pred_path).get_fdata()
    gt_img = nib.load(gt_path).get_fdata()

    pred_mask = (pred_img > 0).astype(np.uint8)
    gt_mask = (gt_img > 0).astype(np.uint8)

    intersection = np.logical_and(pred_mask, gt_mask)
    dice = 2 * intersection.sum() / (pred_mask.sum() + gt_mask.sum()) if (pred_mask.sum() + gt_mask.sum()) > 0 else 1.0
    dice_scores.append(dice)

if dice_scores:
    import numpy as np
    mean_dice = np.mean(dice_scores)
    std_dice = np.std(dice_scores)
    print(f"Dice = {mean_dice:.4f} ± {std_dice:.4f}")
else:
    print("No valid dice scores computed.")


