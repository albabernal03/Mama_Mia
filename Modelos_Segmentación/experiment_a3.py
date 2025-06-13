"""
EXPERIMENTO A3 OFICIAL: MULTI-CANAL [Pre, Post1, Subtraction]
CORREGIDO: Usa train_test_splits.csv oficial
Tres canales de entrada
"""
import numpy as np
import SimpleITK as sitk
import pandas as pd
from pathlib import Path
import json
import shutil

print("=== EXPERIMENTO A3 OFICIAL: MULTI-CANAL + SPLIT OFICIAL ===")

# Configuración de rutas
base_path = Path("../datos")
images_dir = base_path / "images"
seg_dir = base_path / "segmentations" / "expert"
splits_file = Path(r"C:\Users\usuario\Documents\Mama_Mia\datos\train_test_splits.csv")
output_path = Path("../Dataset113_A3_Official")

# Crear directorios de salida
(output_path / "imagesTr").mkdir(parents=True, exist_ok=True)
(output_path / "labelsTr").mkdir(parents=True, exist_ok=True)
(output_path / "imagesTs").mkdir(parents=True, exist_ok=True)
(output_path / "labelsTs").mkdir(parents=True, exist_ok=True)

# Leer split oficial
print("Cargando train_test_splits.csv oficial...")
splits_df = pd.read_csv(splits_file)
print(f"Split oficial cargado: {len(splits_df)} filas")

# Extraer listas de casos
train_cases = splits_df['train_split'].tolist()
test_cases = splits_df['test_split'].tolist()

print(f"Casos de entrenamiento: {len(train_cases)}")
print(f"Casos de test: {len(test_cases)}")
print(f"Distribución por centro (train): {pd.Series(train_cases).str.split('_').str[0].value_counts().to_dict()}")
print(f"Distribución por centro (test): {pd.Series(test_cases).str.split('_').str[0].value_counts().to_dict()}")

def normalize_channel(array):
    """Normaliza un canal individual"""
    mask = array > 0
    if np.sum(mask) > 0:
        mean_val = np.mean(array[mask])
        std_val = np.std(array[mask])
        if std_val > 0:
            return (array - mean_val) / std_val
    return array

def process_cases(case_list, output_images_dir, output_labels_dir, split_name):
    """Procesa una lista de casos y los guarda en el directorio especificado"""
    processed = 0
    errors = []

    for i, case_id in enumerate(case_list):
        try:
            pre_file = images_dir / f"{case_id}_0000.nii.gz"
            post_file = images_dir / f"{case_id}_0001.nii.gz"
            seg_file = seg_dir / f"{case_id}.nii.gz"

            if not pre_file.exists():
                errors.append(f"{case_id}: Falta PRE ({pre_file})")
                continue
            if not post_file.exists():
                errors.append(f"{case_id}: Falta POST ({post_file})")
                continue
            if not seg_file.exists():
                errors.append(f"{case_id}: Falta SEG ({seg_file})")
                continue

            if (i + 1) % 50 == 0:
                print(f"Procesando {split_name}: {case_id} ({i+1}/{len(case_list)})")

            pre_sitk = sitk.Cast(sitk.ReadImage(str(pre_file)), sitk.sitkFloat32)
            post_sitk = sitk.Cast(sitk.ReadImage(str(post_file)), sitk.sitkFloat32)

            pre_array = sitk.GetArrayFromImage(pre_sitk)
            post_array = sitk.GetArrayFromImage(post_sitk)
            sub_array = post_array - pre_array

            pre_norm = normalize_channel(pre_array)
            post_norm = normalize_channel(post_array)
            sub_norm = normalize_channel(sub_array)

            for j, (array, suffix) in enumerate([
                (pre_norm, "0000"), 
                (post_norm, "0001"), 
                (sub_norm, "0002")
            ]):
                result_sitk = sitk.GetImageFromArray(array.astype(np.float32))
                result_sitk.CopyInformation(post_sitk)
                output_file = output_images_dir / f"case_{processed:03d}_{suffix}.nii.gz"
                sitk.WriteImage(result_sitk, str(output_file))

            output_seg_file = output_labels_dir / f"case_{processed:03d}.nii.gz"
            shutil.copy2(seg_file, output_seg_file)
            processed += 1

        except Exception as e:
            errors.append(f"{case_id}: Error de procesamiento - {e}")
            continue

    return processed, errors

print("\nProcesando casos de ENTRENAMIENTO...")
train_processed, train_errors = process_cases(
    train_cases, 
    output_path / "imagesTr", 
    output_path / "labelsTr", 
    "TRAIN"
)

print("\nProcesando casos de TEST...")
test_processed, test_errors = process_cases(
    test_cases, 
    output_path / "imagesTs", 
    output_path / "labelsTs", 
    "TEST"
)

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

dataset_json = {
    "channel_names": {
        "0": "Pre_Contrast_Official",
        "1": "Post_Contrast_Official", 
        "2": "Subtraction_Official"
    },
    "labels": {"background": 0, "tumor": 1},
    "numTraining": train_processed,
    "numTest": test_processed,
    "file_ending": ".nii.gz",
    "dataset_name": "A3_Official_MultiChannel",
    "description": "A3 Official: Pre + Post + Subtraction with official train_test_splits.csv",
    "reference": "MAMA-MIA Challenge Official Split",
    "tensorImageSize": "3D",
    "modality": {
        "0": "DCE-MRI_PreContrast",
        "1": "DCE-MRI_PostContrast",
        "2": "DCE-MRI_Subtraction"
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

mapping_data = {
    "train_mapping": {f"case_{i:03d}": case_id for i, case_id in enumerate(train_cases[:train_processed])},
    "test_mapping": {f"case_{i:03d}": case_id for i, case_id in enumerate(test_cases[:test_processed])},
    "creation_date": pd.Timestamp.now().isoformat(),
    "original_split_file": "train_test_splits.csv"
}

with open(output_path / "case_mapping.json", 'w') as f:
    json.dump(mapping_data, f, indent=2)

print(f"\n{'='*60}")
print(f"EXPERIMENTO A3 OFICIAL COMPLETADO")
print(f"{'='*60}")
print(f"Dataset guardado en: {output_path}")
print(f"Casos de entrenamiento procesados: {train_processed}/{len(train_cases)}")
print(f"Casos de test procesados: {test_processed}/{len(test_cases)}")
print(f"Tasa de éxito train: {train_processed/len(train_cases)*100:.1f}%")
print(f"Tasa de éxito test: {test_processed/len(test_cases)*100:.1f}%")
print(f"{'='*60}")
print(f"COMANDOS SIGUIENTES:")
print(f"   nnUNetv2_plan_and_preprocess -d 113")
print(f"   nnUNetv2_train 113 3d_fullres 0 --npz")
print(f"{'='*60}")

print(f"\nArchivos creados:")
print(f"   - dataset.json: Configuración nnU-Net")
print(f"   - case_mapping.json: Mapeo caso original - nnU-Net")
print(f"   - imagesTr/: {train_processed*3} imágenes entrenamiento (3 canales)")
print(f"   - labelsTr/: {train_processed} segmentaciones entrenamiento")
print(f"   - imagesTs/: {test_processed*3} imágenes test (3 canales)")
print(f"   - labelsTs/: {test_processed} segmentaciones test")

print(f"\nLISTO PARA ENTRENAR EN SUPERCOMPUTADOR 4x RTX A6000")
