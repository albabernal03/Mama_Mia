"""
EXPERIMENTO A2: BASELINE MAMA-MIA CORRECTO + CROPS EXISTENTES
Usar TODAS las fases DCE-MRI + sustracción como augmentation (método oficial)

BASELINE REAL MAMA-MIA:
- Input: Todas las fases DCE-MRI (Pre + Post1 + Post2 + ... + Sustracción)
- Normalization: Z-score usando media/std de TODAS las fases
- Training: Todas las fases + sustracción como data augmentation
- Evaluation: Solo primera fase post-contraste

ESTRUCTURA ESPERADA DE ARCHIVOS:
cropped_data/
├── images/
│   ├── test_split/
│   │   └── DUKE_019/
│   │       ├── duke_019_0000_cropped.nii.gz  (PRE)
│   │       ├── duke_019_0001_cropped.nii.gz  (POST 1)
│   │       ├── duke_019_0002_cropped.nii.gz  (POST 2, si existe)
│   │       └── ... (más fases si existen)
│   └── train_split/
└── segmentations/
    ├── test_split/
    │   └── DUKE_019/
    │       └── duke_019_seg_cropped.nii.gz
    └── train_split/
"""
import numpy as np
import SimpleITK as sitk
import pandas as pd
from pathlib import Path
import json
import shutil

print("=== EXPERIMENTO A2: BASELINE MAMA-MIA CORRECTO + CROPS ===")

# Configuración de rutas
cropped_base = Path(r"C:\Users\usuario\Documents\Mama_Mia\cropped_data")
cropped_images = cropped_base / "images"
cropped_segs = cropped_base / "segmentations"
splits_file = Path(r"C:\Users\usuario\Documents\Mama_Mia\datos\train_test_splits.csv")
output_path = Path("../Dataset112_A2_Baseline_AllPhases")

# Crear directorios de salida
(output_path / "imagesTr").mkdir(parents=True, exist_ok=True)
(output_path / "labelsTr").mkdir(parents=True, exist_ok=True)
(output_path / "imagesTs").mkdir(parents=True, exist_ok=True)
(output_path / "labelsTs").mkdir(parents=True, exist_ok=True)

# Leer casos directamente de los directorios cropped (más confiable)
print("Detectando casos disponibles en directorios cropped...")

train_split_dir = cropped_images / "train_split"
test_split_dir = cropped_images / "test_split"

# Obtener casos de entrenamiento
train_cases = []
if train_split_dir.exists():
    train_cases = [d.name for d in train_split_dir.iterdir() if d.is_dir()]
    train_cases.sort()

# Obtener casos de test  
test_cases = []
if test_split_dir.exists():
    test_cases = [d.name for d in test_split_dir.iterdir() if d.is_dir()]
    test_cases.sort()

print(f"Casos train detectados: {len(train_cases)}")
print(f"Casos test detectados: {len(test_cases)}")
print(f"Ejemplos train: {train_cases[:5]}")
print(f"Ejemplos test: {test_cases[:5]}")

def find_all_phases(case_id, search_dir):
    """
    Encuentra todas las fases DCE-MRI para un caso
    Patrón: {case_id}_{phase}_cropped.nii.gz
    """
    case_dir = search_dir / case_id
    if not case_dir.exists():
        return []
    
    case_id_lower = case_id.lower()
    phase_files = []
    
    # Buscar fases numéricas (0000, 0001, 0002, etc.)
    for phase_num in range(10):  # Buscar hasta 10 fases
        phase_file = case_dir / f"{case_id_lower}_{phase_num:04d}_cropped.nii.gz"
        if phase_file.exists():
            phase_files.append((phase_num, phase_file))
    
    # Ordenar por número de fase
    phase_files.sort(key=lambda x: x[0])
    return phase_files

def find_segmentation_file(case_id, seg_dir):
    """
    Busca archivo de segmentación
    """
    case_dir = seg_dir / case_id
    if not case_dir.exists():
        return None
    
    case_id_lower = case_id.lower()
    seg_file = case_dir / f"{case_id_lower}_seg_cropped.nii.gz"
    
    if seg_file.exists():
        return seg_file
    
    # Buscar cualquier archivo seg en el directorio del caso
    seg_files = list(case_dir.glob("*seg*cropped.nii.gz"))
    if seg_files:
        return seg_files[0]
    
    return None

def process_cropped_cases_baseline(case_list, seg_subdir, output_images_dir, output_labels_dir, split_name):
    """
    Procesa casos usando el método baseline MAMA-MIA CORRECTO:
    - Todas las fases DCE-MRI como canales
    - Sustracción como canal adicional
    - Z-score usando media/std de TODAS las fases
    """
    processed = 0
    errors = []
    
    images_split_dir = cropped_images / seg_subdir
    seg_split_dir = cropped_segs / seg_subdir
    
    for i, case_id in enumerate(case_list):
        try:
            if (i + 1) % 20 == 0:
                print(f"Procesando {split_name}: {case_id} ({i+1}/{len(case_list)})")
            
            # ============ BUSCAR TODAS LAS FASES ============
            phase_files = find_all_phases(case_id, images_split_dir)
            if len(phase_files) < 2:
                errors.append(f"{case_id}: Menos de 2 fases encontradas")
                continue
            
            # Buscar segmentación
            seg_file = find_segmentation_file(case_id, seg_split_dir)
            if seg_file is None:
                errors.append(f"{case_id}: Segmentación no encontrada")
                continue
            
            print(f"  {case_id}: {len(phase_files)} fases encontradas")
            
            # ============ CARGAR TODAS LAS FASES ============
            phase_arrays = []
            phase_sitk_images = []
            
            for phase_num, phase_file in phase_files:
                phase_sitk = sitk.Cast(sitk.ReadImage(str(phase_file)), sitk.sitkFloat32)
                phase_array = sitk.GetArrayFromImage(phase_sitk)
                phase_arrays.append(phase_array)
                phase_sitk_images.append(phase_sitk)
                print(f"    Fase {phase_num}: {phase_file.name}")
            
            # ============ BASELINE MAMA-MIA: CREAR SUSTRACCIÓN ============
            # Sustracción = Post1 - Pre (fase 1 - fase 0)
            if len(phase_arrays) >= 2:
                pre_array = phase_arrays[0]   # Fase 0000 (PRE)
                post_array = phase_arrays[1]  # Fase 0001 (POST1)
                subtraction_array = post_array - pre_array
                phase_arrays.append(subtraction_array)
                print(f"    Sustracción: POST1 - PRE añadida")
            
            # ============ BASELINE MAMA-MIA: Z-SCORE NORMALIZATION ============
            # Normalizar usando media/std de TODAS las fases (como en baseline)
            all_voxels = np.concatenate([arr.flatten() for arr in phase_arrays])
            global_mean = np.mean(all_voxels)
            global_std = np.std(all_voxels)
            
            if global_std > 0:
                normalized_arrays = []
                for arr in phase_arrays:
                    normalized_arr = (arr - global_mean) / global_std
                    normalized_arrays.append(normalized_arr)
            else:
                normalized_arrays = [arr - global_mean for arr in phase_arrays]
            
            # ============ CREAR IMAGEN MULTI-CANAL ============
            # Apilar todas las fases como canales (incluyendo sustracción)
            multi_channel_array = np.stack(normalized_arrays, axis=0)  # (C, Z, Y, X)
            
            # nnU-Net espera formato (Z, Y, X, C) - mover canales al final
            multi_channel_array = np.moveaxis(multi_channel_array, 0, -1)  # (Z, Y, X, C)
            
            # Crear archivo nnU-Net para cada canal
            reference_sitk = phase_sitk_images[0]  # Usar primera fase como referencia
            
            for channel_idx in range(multi_channel_array.shape[-1]):
                # Extraer canal individual
                channel_array = multi_channel_array[..., channel_idx]
                
                # Crear imagen SimpleITK
                channel_sitk = sitk.GetImageFromArray(channel_array.astype(np.float32))
                channel_sitk.CopyInformation(reference_sitk)
                
                # Guardar con formato nnU-Net: case_XXX_CHANNEL.nii.gz
                output_img_file = output_images_dir / f"case_{processed:03d}_{channel_idx:04d}.nii.gz"
                sitk.WriteImage(channel_sitk, str(output_img_file))
            
            # ============ GUARDAR SEGMENTACIÓN ============
            output_seg_file = output_labels_dir / f"case_{processed:03d}.nii.gz"
            shutil.copy2(seg_file, output_seg_file)
            
            processed += 1
            
        except Exception as e:
            errors.append(f"{case_id}: Error - {e}")
            continue
    
    return processed, errors

# ============ PROCESAR CASOS ============
print("\nProcesando casos de ENTRENAMIENTO...")
train_processed, train_errors = process_cropped_cases_baseline(
    train_cases,
    "train_split",
    output_path / "imagesTr",
    output_path / "labelsTr",
    "TRAIN"
)

print("\nProcesando casos de TEST...")
test_processed, test_errors = process_cropped_cases_baseline(
    test_cases,
    "test_split",
    output_path / "imagesTs",
    output_path / "labelsTs",
    "TEST"
)

# ============ MOSTRAR ERRORES ============
if train_errors:
    print(f"\nERRORES EN TRAIN ({len(train_errors)}):")
    for error in train_errors[:10]:
        print(f"  {error}")
    if len(train_errors) > 10:
        print(f"  ... y {len(train_errors) - 10} errores más")

if test_errors:
    print(f"\nERRORES EN TEST ({len(test_errors)}):")
    for error in test_errors[:10]:
        print(f"  {error}")
    if len(test_errors) > 10:
        print(f"  ... y {len(test_errors) - 10} errores más")

# Determinar número de canales procesados
# (esto requiere inspeccionar al menos un caso exitoso)
num_channels = 3  # Por defecto: PRE + POST + SUBTRACTION
if train_processed > 0 or test_processed > 0:
    # Contar archivos de imagen del primer caso procesado para determinar canales
    first_case_files = list((output_path / "imagesTr").glob("case_000_*.nii.gz"))
    if not first_case_files and test_processed > 0:
        first_case_files = list((output_path / "imagesTs").glob("case_000_*.nii.gz"))
    
    if first_case_files:
        num_channels = len(first_case_files)

# ============ CREAR DATASET.JSON ============
# Crear mapeo de canales dinámico
channel_names = {}
for i in range(num_channels):
    if i == 0:
        channel_names[str(i)] = "Pre_Contrast"
    elif i == num_channels - 1:  # Último canal es sustracción
        channel_names[str(i)] = "Subtraction_Post1_Pre"
    else:
        channel_names[str(i)] = f"Post_Contrast_{i}"

dataset_json = {
    "channel_names": channel_names,
    "labels": {"background": 0, "tumor": 1},
    "numTraining": train_processed,
    "numTest": test_processed,
    "file_ending": ".nii.gz",
    "dataset_name": "A2_MAMA_MIA_Baseline_Crops",
    "description": "A2: MAMA-MIA baseline method CORRECTO (all DCE phases + subtraction) applied to cropped regions",
    "reference": "MAMA-MIA Challenge Baseline Method with Pre-computed Crops",
    "tensorImageSize": "3D",
    "modality": channel_names,
    "experiment_details": {
        "baseline_method": "MAMA-MIA oficial: All DCE-MRI phases + subtraction as augmentation",
        "normalization": "Z-score using global mean/std of ALL phases",
        "data_source": "Pre-computed cropped images",
        "channels": f"{num_channels} channels total",
        "target_dice": "> 0.7620 (MAMA-MIA baseline oficial)"
    },
    "split_info": {
        "train_cases": len(train_cases),
        "test_cases": len(test_cases),
        "train_processed": train_processed,
        "test_processed": test_processed,
        "train_success_rate": f"{train_processed/len(train_cases)*100:.1f}%",
        "test_success_rate": f"{test_processed/len(test_cases)*100:.1f}%"
    }
}

with open(output_path / "dataset.json", 'w') as f:
    json.dump(dataset_json, f, indent=2)

# ============ CREAR MAPEO DE CASOS ============
mapping_data = {
    "train_mapping": {f"case_{i:03d}": case_id for i, case_id in enumerate(train_cases[:train_processed])},
    "test_mapping": {f"case_{i:03d}": case_id for i, case_id in enumerate(test_cases[:test_processed])},
    "creation_date": pd.Timestamp.now().isoformat(),
    "source_data": "Pre-computed crops from cropped_data directory",
    "baseline_method": "MAMA-MIA oficial (all phases + subtraction + global Z-score)",
    "enhancement": "Applied to tumor-focused cropped regions",
    "num_channels": num_channels
}

with open(output_path / "case_mapping.json", 'w') as f:
    json.dump(mapping_data, f, indent=2)

# ============ RESUMEN FINAL ============
print(f"\n{'='*70}")
print(f"EXPERIMENTO A2 COMPLETADO - BASELINE MAMA-MIA CORRECTO + CROPS")
print(f"{'='*70}")
print(f"Dataset guardado en: {output_path}")
print(f"Método: MAMA-MIA baseline OFICIAL aplicado a crops")
print(f"Canales: {num_channels} ({list(channel_names.values())})")
print(f"Normalization: Z-score global de TODAS las fases")
print(f"Casos train procesados: {train_processed}/{len(train_cases)}")
print(f"Casos test procesados: {test_processed}/{len(test_cases)}")
print(f"Tasa éxito train: {train_processed/len(train_cases)*100:.1f}%")
print(f"Tasa éxito test: {test_processed/len(test_cases)*100:.1f}%")
print(f"{'='*70}")
print(f"COMANDOS SIGUIENTES:")
print(f"   nnUNetv2_plan_and_preprocess -d 112")
print(f"   nnUNetv2_train 112 3d_fullres 0 --npz")
print(f"{'='*70}")
print(f"VENTAJAS vs BASELINE ORIGINAL:")
print(f"   ✓ MISMO método oficial MAMA-MIA")
print(f"   ✓ Aplicado a crops enfocados en tumor")
print(f"   ✓ Todas las fases DCE + sustracción")
print(f"   ✓ Z-score normalization global")
print(f"   ✓ Menos ruido de fondo por crops")
print(f"   ✓ Objetivo: Dice > 0.7620")
print(f"{'='*70}")

print(f"\n🎯 LISTO PARA SUPERAR EL BASELINE MAMA-MIA OFICIAL!")