import os
import shutil
import nibabel as nib
import numpy as np
import torch
import time
from nnunet.inference.predict import predict_from_folder
from config_a6000 import *

# Setup inicial
setup_torch_a6000()

def dice_coefficient(pred, gt):
    """Calcula Dice coefficient"""
    pred_bin = pred > 0.5
    gt_bin = gt > 0.5
    
    intersection = np.sum(pred_bin * gt_bin)
    total = np.sum(pred_bin) + np.sum(gt_bin)
    
    if total == 0:
        return 1.0 if np.sum(pred_bin) == 0 else 0.0
    
    return (2.0 * intersection) / total

def prepare_test_cases(n_cases=None):
    """Prepara casos de test con la estructura real"""
    
    print("📂 Preparando casos de test...")
    
    # Leer CSV
    with open(SPLITS_CSV, 'r') as f:
        lines = f.readlines()[1:]  # Skip header
    
    test_cases = []
    for line in lines[:n_cases]:
        parts = line.strip().split(',')
        if len(parts) >= 2:
            test_cases.append(parts[1])  # test_split column
    
    print(f"📋 Casos del CSV: {test_cases[:3]}... (total: {len(test_cases)})")
    
    # Crear carpetas
    os.makedirs(TEMP_INPUT, exist_ok=True)
    os.makedirs(RESULTS_OUTPUT, exist_ok=True)
    
    prepared_cases = []
    
    for case_id in test_cases:
        print(f"🔍 Procesando caso: {case_id}")
        
        # Buscar archivo de imagen usando función helper
        img_file = get_post_contrast_file(case_id)
        
        if img_file and os.path.exists(img_file):
            # Copiar archivo con nomenclatura nnU-Net
            dest_img = os.path.join(TEMP_INPUT, f"{case_id}_0000.nii.gz")
            shutil.copy2(img_file, dest_img)
            prepared_cases.append(case_id)
            print(f"✅ Preparado: {case_id} -> {os.path.basename(img_file)}")
        else:
            print(f"❌ No encontrado: {case_id}")
            # Debug: mostrar archivos encontrados
            case_files = find_case_files(case_id)
            if case_files:
                print(f"   📁 Archivos encontrados: {case_files[:3]}...")
            else:
                print(f"   📁 No hay archivos para {case_id}")
            
    print(f"✅ Total preparados: {len(prepared_cases)} casos")
    return prepared_cases

def run_inference():
    """Ejecuta inferencia con 4x A6000"""
    
    print("🚀 Ejecutando inferencia con 4x RTX A6000...")
    
    # Verificar que hay archivos para procesar
    input_files = [f for f in os.listdir(TEMP_INPUT) if f.endswith('.nii.gz')]
    if not input_files:
        print("❌ No hay archivos en TEMP_INPUT para procesar")
        return
    
    print(f"📁 Archivos a procesar: {len(input_files)}")
    
    start_time = time.time()
    
    try:
        predict_from_folder(
            model=WEIGHTS_PATH,
            input_folder=TEMP_INPUT,
            output_folder=RESULTS_OUTPUT,
            folds=(0, 1, 2, 3, 4),
            save_npz=False,
            num_threads_preprocessing=16,  # Reducido para Windows
            num_threads_nifti_save=8,     # Reducido para Windows
            tta=True,
            mixed_precision=True,
            overwrite_existing=True
        )
        
        end_time = time.time()
        total_time = end_time - start_time
        
        print(f"⚡ Inferencia completada:")
        print(f"   • Tiempo total: {total_time:.1f} segundos")
        print(f"   • Casos: {len(input_files)}")
        print(f"   • Tiempo por caso: {total_time/len(input_files):.2f}s")
        
    except Exception as e:
        print(f"❌ Error en inferencia: {e}")
        print("💡 Verificando configuración de nnU-Net...")
        
        # Debug nnU-Net
        print(f"nnUNet_RESULTS_FOLDER: {os.environ.get('nnUNet_RESULTS_FOLDER', 'NO DEFINIDO')}")
        print(f"Pesos path existe: {os.path.exists(WEIGHTS_PATH)}")
        if os.path.exists(WEIGHTS_PATH):
            print(f"Contenido pesos: {os.listdir(WEIGHTS_PATH)}")

def evaluate_results(test_cases):
    """Evalúa resultados vs ground truth"""
    
    print("📊 Evaluando resultados...")
    
    dice_scores = []
    successful_cases = []
    
    for case_id in test_cases:
        pred_file = os.path.join(RESULTS_OUTPUT, f"{case_id}.nii.gz")
        
        # Buscar segmentación experta usando función helper
        gt_file = find_expert_segmentation(case_id)
        
        if not os.path.exists(pred_file):
            print(f"❌ Predicción faltante: {case_id}")
            continue
            
        if not gt_file or not os.path.exists(gt_file):
            print(f"❌ Ground truth faltante: {case_id}")
            continue
            
        try:
            pred = nib.load(pred_file).get_fdata()
            gt = nib.load(gt_file).get_fdata()
            
            dice = dice_coefficient(pred, gt)
            dice_scores.append(dice)
            successful_cases.append(case_id)
            
            print(f"✅ {case_id}: Dice = {dice:.3f}")
            
        except Exception as e:
            print(f"❌ Error evaluando {case_id}: {e}")
    
    if dice_scores:
        mean_dice = np.mean(dice_scores)
        std_dice = np.std(dice_scores)
        
        print(f"\n🏆 RESULTADOS ({len(dice_scores)} casos):")
        print(f"   • Dice Score: {mean_dice:.3f} ± {std_dice:.3f}")
        print(f"   • Esperado del paper: ~0.762 ± 0.211")
        print(f"   • Casos exitosos: {len(successful_cases)}/{len(test_cases)}")
        
        if abs(mean_dice - 0.762) < 0.05:
            print("🎯 ¡BASELINE VALIDADO PERFECTAMENTE!")
            return True, dice_scores, successful_cases
        elif mean_dice > 0.700:
            print("✅ Resultado aceptable - continuamos con mejoras")
            return True, dice_scores, successful_cases
        else:
            print("⚠️ Resultado bajo - revisar setup")
            return False, dice_scores, successful_cases
    
    return False, [], []

def debug_case_structure():
    """Debug de la estructura de casos"""
    
    print("\n🔍 DEBUG: Analizando estructura de casos...")
    
    # Tomar primer caso del CSV
    with open(SPLITS_CSV, 'r') as f:
        lines = f.readlines()[1:]
    
    if lines:
        first_case = lines[0].strip().split(',')[1]
        print(f"📋 Caso ejemplo: {first_case}")
        
        # Buscar archivos de imagen
        case_files = find_case_files(first_case)
        print(f"📁 Archivos imagen encontrados: {case_files}")
        
        # Buscar segmentación
        seg_file = find_expert_segmentation(first_case)
        print(f"📁 Segmentación encontrada: {seg_file}")
        
        # Mostrar estructura de carpeta de segmentaciones
        if os.path.exists(EXPERT_SEGS_PATH):
            seg_files = os.listdir(EXPERT_SEGS_PATH)[:10]
            print(f"📁 Ejemplos segmentaciones: {seg_files}")

if __name__ == "__main__":
    print("🚀 MAMA-MIA BASELINE TEST - 4x RTX A6000 (FIXED)")
    print("=" * 60)
    
    # Pipeline completo
    try:
        # 1. Verificar configuración
        print("🔍 Verificando configuración...")
        print(f"nnUNet_RESULTS_FOLDER: {os.environ.get('nnUNet_RESULTS_FOLDER')}")
        print(f"Datos: {MAMA_MIA_ROOT}")
        print(f"Pesos: {WEIGHTS_PATH}")
        
        # 2. Debug estructura
        debug_case_structure()
        
        # 3. Preparar casos
        test_cases = prepare_test_cases(n_cases=None)  # Empezar con 10 casos
        
        if not test_cases:
            print("❌ No se pudieron preparar casos")
            exit(1)
        
        # 4. Ejecutar inferencia
        run_inference()
        
        # 5. Evaluar
        success, dice_scores, successful_cases = evaluate_results(test_cases)
        
        if success:
            print("\n✅ ¡BASELINE FUNCIONANDO!")
            print("🚀 Listo para implementar mejoras")
            
            # Guardar casos exitosos para mejoras
            with open('successful_cases.txt', 'w') as f:
                for case in successful_cases:
                    f.write(f"{case}\n")
            print(f"💾 Casos exitosos guardados: {len(successful_cases)}")
        else:
            print("\n❌ Revisar configuración")
            
    except Exception as e:
        print(f"❌ Error general: {e}")
        import traceback
        traceback.print_exc()