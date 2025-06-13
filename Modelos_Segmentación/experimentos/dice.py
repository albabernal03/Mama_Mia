import os
import numpy as np
import SimpleITK as sitk
from pathlib import Path

def dice_coefficient(pred, gt, smooth=1e-6):
    """Calcular coeficiente Dice entre predicción y ground truth"""
    intersection = np.sum(pred * gt)
    union = np.sum(pred) + np.sum(gt)
    dice = (2 * intersection + smooth) / (union + smooth)
    return dice

def evaluate_folder(pred_folder, gt_folder):
    """Evaluar todas las predicciones vs ground truth"""
    pred_files = list(Path(pred_folder).glob("*.nii.gz"))
    results = []
    
    print(f"Evaluando {len(pred_files)} archivos...")
    
    for pred_file in pred_files:
        # Buscar archivo correspondiente en ground truth
        gt_file = Path(gt_folder) / pred_file.name
        
        if not gt_file.exists():
            print(f"ADVERTENCIA: No se encontró GT para {pred_file.name}")
            continue
            
        try:
            # Cargar imágenes
            pred_img = sitk.ReadImage(str(pred_file))
            gt_img = sitk.ReadImage(str(gt_file))
            
            # Convertir a arrays numpy
            pred_array = sitk.GetArrayFromImage(pred_img)
            gt_array = sitk.GetArrayFromImage(gt_img)
            
            # Binarizar (asumir que 1 es la clase de interés)
            pred_binary = (pred_array > 0).astype(np.uint8)
            gt_binary = (gt_array > 0).astype(np.uint8)
            
            # Calcular Dice
            dice = dice_coefficient(pred_binary, gt_binary)
            
            results.append({
                'file': pred_file.name,
                'dice': dice
            })
            
            print(f"{pred_file.name}: Dice = {dice:.4f}")
            
        except Exception as e:
            print(f"ERROR procesando {pred_file.name}: {e}")
    
    # Estadísticas finales
    if results:
        dice_scores = [r['dice'] for r in results]
        print(f"\n=== RESULTADOS FINALES ===")
        print(f"Número de casos: {len(dice_scores)}")
        print(f"Dice promedio: {np.mean(dice_scores):.4f}")
        print(f"Dice mediana: {np.median(dice_scores):.4f}")
        print(f"Dice std: {np.std(dice_scores):.4f}")
        print(f"Dice min: {np.min(dice_scores):.4f}")
        print(f"Dice max: {np.max(dice_scores):.4f}")
    
    return results

# Usar el script
if __name__ == "__main__":
    pred_folder = "./predictions_test_114"
    gt_folder = "C:/nnUNet_raw/Dataset114_A2_PrePost1_Crops/labelsTs"
    
    results = evaluate_folder(pred_folder, gt_folder)
    import pandas as pd

    # Guardar resultados por paciente
    df = pd.DataFrame(results)
    df.to_csv("dice_por_paciente.csv", index=False)
    print("\n✅ Resultados guardados en 'dice_por_paciente.csv'")
