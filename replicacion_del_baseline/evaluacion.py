import os
import nibabel as nib
import numpy as np
import pandas as pd

# Rutas
results_folder = r".\results_output"
expert_folder = r"C:\Users\usuario\Documents\Mama_Mia\datos\segmentations\expert"

def dice_coefficient(pred, gt):
    """Calcula Dice coefficient"""
    pred_bin = pred > 0.5
    gt_bin = gt > 0.5
    
    intersection = np.sum(pred_bin * gt_bin)
    total = np.sum(pred_bin) + np.sum(gt_bin)
    
    if total == 0:
        return 1.0 if np.sum(pred_bin) == 0 else 0.0
    
    return (2.0 * intersection) / total

def hausdorff_distance_95(pred, gt):
    """Calcula Hausdorff Distance 95th percentile (simplificado)"""
    try:
        from scipy.spatial.distance import directed_hausdorff
        
        pred_bin = pred > 0.5
        gt_bin = gt > 0.5
        
        # Obtener coordenadas de puntos de superficie
        pred_coords = np.argwhere(pred_bin)
        gt_coords = np.argwhere(gt_bin)
        
        if len(pred_coords) == 0 or len(gt_coords) == 0:
            return float('inf')
        
        # Hausdorff bidireccional
        d1 = directed_hausdorff(pred_coords, gt_coords)[0]
        d2 = directed_hausdorff(gt_coords, pred_coords)[0]
        
        return max(d1, d2)
    except:
        return -1  # Error en cálculo

print("EVALUACION BASELINE MAMA-MIA")
print("=" * 50)

# Verificar carpetas
if not os.path.exists(results_folder):
    print(f"❌ No existe carpeta de resultados: {results_folder}")
    exit(1)

if not os.path.exists(expert_folder):
    print(f"❌ No existe carpeta de segmentaciones expertas: {expert_folder}")
    exit(1)

# Obtener archivos de predicciones
pred_files = [f for f in os.listdir(results_folder) if f.endswith('.nii.gz')]

if not pred_files:
    print("❌ No hay archivos .nii.gz en results_output")
    exit(1)

print(f"📁 Encontrados {len(pred_files)} archivos de predicción")

# Evaluar cada caso
results = []

for pred_file in pred_files:
    case_id = pred_file.replace('.nii.gz', '')
    
    print(f"\n🔍 Evaluando: {case_id}")
    
    # Cargar predicción
    pred_path = os.path.join(results_folder, pred_file)
    try:
        pred = nib.load(pred_path).get_fdata()
        print(f"   ✅ Predicción cargada: {pred.shape}")
    except Exception as e:
        print(f"   ❌ Error cargando predicción: {e}")
        continue
    
    # Buscar ground truth
    gt_file = f"{case_id.lower()}.nii.gz"
    gt_path = os.path.join(expert_folder, gt_file)
    
    if not os.path.exists(gt_path):
        print(f"   ❌ Ground truth no encontrado: {gt_file}")
        continue
    
    try:
        gt = nib.load(gt_path).get_fdata()
        print(f"   ✅ Ground truth cargado: {gt.shape}")
    except Exception as e:
        print(f"   ❌ Error cargando ground truth: {e}")
        continue
    
    # Verificar dimensiones
    if pred.shape != gt.shape:
        print(f"   ⚠️ Dimensiones diferentes: pred {pred.shape} vs gt {gt.shape}")
        # Intentar redimensionar si es necesario
        if pred.ndim == gt.ndim:
            from scipy.ndimage import zoom
            scale_factors = [gt.shape[i] / pred.shape[i] for i in range(pred.ndim)]
            pred = zoom(pred, scale_factors, order=1)
            print(f"   🔧 Predicción redimensionada a: {pred.shape}")
    
    # Calcular métricas
    dice = dice_coefficient(pred, gt)
    hd95 = hausdorff_distance_95(pred, gt)
    
    # Estadísticas adicionales
    pred_volume = np.sum(pred > 0.5)
    gt_volume = np.sum(gt > 0.5)
    volume_diff = abs(pred_volume - gt_volume) / max(gt_volume, 1)
    
    print(f"   📊 DICE: {dice:.3f}")
    if hd95 > 0:
        print(f"   📊 Hausdorff 95: {hd95:.2f}")
    print(f"   📊 Volumen pred: {pred_volume} voxels")
    print(f"   📊 Volumen GT: {gt_volume} voxels")
    print(f"   📊 Diff volumen: {volume_diff:.1%}")
    
    # Guardar resultados
    results.append({
        'case_id': case_id,
        'dice': dice,
        'hausdorff_95': hd95,
        'pred_volume': pred_volume,
        'gt_volume': gt_volume,
        'volume_diff': volume_diff
    })

# Resumen de resultados
if results:
    print("\n" + "="*50)
    print("📊 RESUMEN BASELINE MAMA-MIA")
    print("="*50)
    
    df = pd.DataFrame(results)
    
    # Estadísticas principales
    dice_mean = df['dice'].mean()
    dice_std = df['dice'].std()
    dice_median = df['dice'].median()
    dice_min = df['dice'].min()
    dice_max = df['dice'].max()
    
    print(f"Casos evaluados: {len(results)}")
    print(f"DICE Score: {dice_mean:.3f} ± {dice_std:.3f}")
    print(f"DICE Mediana: {dice_median:.3f}")
    print(f"DICE Rango: {dice_min:.3f} - {dice_max:.3f}")
    
    # Comparación con paper
    paper_dice = 0.762
    paper_std = 0.211
    
    print(f"\nPaper esperado: {paper_dice:.3f} ± {paper_std:.3f}")
    print(f"Diferencia: {dice_mean - paper_dice:+.3f}")
    
    # Evaluación del baseline
    if abs(dice_mean - paper_dice) < 0.03:
        print("🎯 BASELINE VALIDADO PERFECTAMENTE!")
        status = "PERFECTO"
    elif dice_mean > 0.7:
        print("✅ BASELINE VALIDADO!")
        status = "VALIDADO"
    elif dice_mean > 0.5:
        print("⚠️ Baseline funciona pero bajo")
        status = "FUNCIONAL"
    else:
        print("❌ Problema con baseline")
        status = "PROBLEMA"
    
    # Hausdorff Distance
    valid_hd = df[df['hausdorff_95'] > 0]['hausdorff_95']
    if len(valid_hd) > 0:
        hd_mean = valid_hd.mean()
        print(f"Hausdorff 95: {hd_mean:.2f} mm")
    
    # Tabla detallada
    print(f"\n📋 RESULTADOS POR CASO:")
    print("-" * 50)
    for _, row in df.iterrows():
        dice_emoji = "🎯" if row['dice'] > 0.8 else "✅" if row['dice'] > 0.7 else "⚠️" if row['dice'] > 0.5 else "❌"
        print(f"{dice_emoji} {row['case_id']}: Dice = {row['dice']:.3f}")
    
    # Guardar resultados
    df.to_csv('baseline_evaluation_results.csv', index=False)
    print(f"\n💾 Resultados guardados en: baseline_evaluation_results.csv")
    
    # Conclusión final
    print("\n" + "="*50)
    print("🏁 CONCLUSION")
    print("="*50)
    
    if status == "PERFECTO":
        print("🚀 BASELINE MAMA-MIA REPLICADO PERFECTAMENTE")
        print("🎯 Listo para implementar mejoras")
        print("📈 Objetivo: Superar Dice 0.762")
    elif status == "VALIDADO":
        print("✅ BASELINE MAMA-MIA FUNCIONANDO")
        print("🎯 Listo para mejoras")
    else:
        print(f"Status: {status}")
        print("💡 Revisar configuración si es necesario")
    
    print(f"⚡ Con 4x RTX A6000: Excelente setup para mejoras")
    
else:
    print("\n❌ No se pudieron evaluar casos")
    print("💡 Verifica que:")
    print("   - Hay archivos .nii.gz en results_output")  
    print("   - Existen ground truth correspondientes")
    print("   - Los nombres coinciden (ej: DUKE_019.nii.gz)")

print("\n✅ Evaluación completada")