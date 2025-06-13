"""
EXPERIMENTO A2 OFICIAL: POST-CONTRASTE ONLY
CORREGIDO: Usa train_test_splits.csv oficial
Solo imagen post-contraste, sin resta
"""
import numpy as np
import SimpleITK as sitk
import pandas as pd
from pathlib import Path
import json
import shutil

<<<<<<< HEAD
print("=== EXPERIMENTO A2 OFICIAL: POST-CONTRASTE ONLY + SPLIT OFICIAL ===")

# Configuración de rutas
base_path = Path("../datos")
images_dir = base_path / "images"
seg_dir = base_path / "segmentations" / "expert"
splits_file = Path(r"C:\Users\usuario\Documents\Mama_Mia\datos\train_test_splits.csv")
output_path = Path("../Dataset112_A2_Official")

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

def process_cases(case_list, output_images_dir, output_labels_dir, split_name):
    """Procesa una lista de casos y los guarda en el directorio especificado"""
    processed = 0
    errors = []
    
    for i, case_id in enumerate(case_list):
        try:
            # Buscar archivos correspondientes
            post_file = images_dir / f"{case_id}_0001.nii.gz"
            seg_file = seg_dir / f"{case_id}.nii.gz"
            
            # Verificar que existen todos los archivos
            if not post_file.exists():
                errors.append(f"{case_id}: Falta POST ({post_file})")
                continue
            if not seg_file.exists():
                errors.append(f"{case_id}: Falta SEG ({seg_file})")
                continue
            
            if (i + 1) % 50 == 0:
                print(f"Procesando {split_name}: {case_id} ({i+1}/{len(case_list)})")
            
            # Cargar solo imagen post-contraste
            post_sitk = sitk.Cast(sitk.ReadImage(str(post_file)), sitk.sitkFloat32)
            post_array = sitk.GetArrayFromImage(post_sitk)
            
            # Z-score normalization
            mask = post_array > 0
            if np.sum(mask) > 0:
                mean_val = np.mean(post_array[mask])
                std_val = np.std(post_array[mask])
                if std_val > 0:
                    post_array = (post_array - mean_val) / std_val
            
            # Crear imagen SimpleITK resultado
            result_sitk = sitk.GetImageFromArray(post_array.astype(np.float32))
            result_sitk.CopyInformation(post_sitk)
            
            # Guardar imagen procesada
            output_img_file = output_images_dir / f"case_{processed:03d}_0000.nii.gz"
            sitk.WriteImage(result_sitk, str(output_img_file))
            
            # Copiar segmentación
            output_seg_file = output_labels_dir / f"case_{processed:03d}.nii.gz"
            shutil.copy2(seg_file, output_seg_file)
            
            processed += 1
            
        except Exception as e:
            errors.append(f"{case_id}: Error de procesamiento - {e}")
            continue
    
    return processed, errors

# Procesar casos de entrenamiento
print("\nProcesando casos de ENTRENAMIENTO...")
train_processed, train_errors = process_cases(
    train_cases, 
    output_path / "imagesTr", 
    output_path / "labelsTr", 
    "TRAIN"
)

# Procesar casos de test
print("\nProcesando casos de TEST...")
test_processed, test_errors = process_cases(
    test_cases, 
    output_path / "imagesTs", 
    output_path / "labelsTs", 
    "TEST"
)

# Mostrar errores si los hay
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

# Crear dataset.json
dataset_json = {
    "channel_names": {"0": "Post_Contrast_Official"},
    "labels": {"background": 0, "tumor": 1},
    "numTraining": train_processed,
    "numTest": test_processed,
    "file_ending": ".nii.gz",
    "dataset_name": "A2_Official_PostContrast",
    "description": "A2 Official: Post-contrast only with official train_test_splits.csv",
    "reference": "MAMA-MIA Challenge Official Split",
    "tensorImageSize": "3D",
    "modality": {"0": "DCE-MRI_PostContrast"},
    "split_info": {
        "train_cases": len(train_cases),
        "test_cases": len(test_cases),
        "train_processed": train_processed,
        "test_processed": test_processed,
        "train_success_rate": f"{train_processed/len(train_cases)*100:.1f}%",
        "test_success_rate": f"{test_processed/len(test_cases)*100:.1f}%"
    }
=======
print("=== EXPERIMENTO A2: POST-CONTRASTE ONLY ===")

base_path = Path("../datos")
images_dir = base_path / "images"
seg_dir = base_path / "segmentations" / "expert"

output_path = Path("../Dataset102_A2_PostOnly")
(output_path / "imagesTr").mkdir(parents=True, exist_ok=True)
(output_path / "labelsTr").mkdir(parents=True, exist_ok=True)

files_0000 = sorted(list(images_dir.glob("*_0000.nii.gz")))[:1506]
print(f"Encontrados {len(files_0000)} archivos para procesar")

processed = 0
for i, file in enumerate(files_0000):
    patient_id = file.name.replace("_0000.nii.gz", "")
    post_file = images_dir / f"{patient_id}_0001.nii.gz"
    seg_file = seg_dir / f"{patient_id}.nii.gz"
    
    if not post_file.exists() or not seg_file.exists():
        continue
        
    try:
        if (i + 1) % 100 == 0:
            print(f"Procesando A2: {patient_id} ({i+1}/{len(files_0000)})")
        
        # Cargar solo post-contraste
        post_sitk = sitk.Cast(sitk.ReadImage(str(post_file)), sitk.sitkFloat32)
        post_array = sitk.GetArrayFromImage(post_sitk)
        
        # Z-score normalization
        mask = post_array > 0
        if np.sum(mask) > 0:
            mean_val = np.mean(post_array[mask])
            std_val = np.std(post_array[mask])
            if std_val > 0:
                post_array = (post_array - mean_val) / std_val
        
        # Guardar
        result_sitk = sitk.GetImageFromArray(post_array.astype(np.float32))
        result_sitk.CopyInformation(post_sitk)
        
        sitk.WriteImage(result_sitk, str(output_path / "imagesTr" / f"case_{processed:03d}_0000.nii.gz"))
        shutil.copy2(seg_file, output_path / "labelsTr" / f"case_{processed:03d}.nii.gz")
        
        processed += 1
        
    except Exception as e:
        print(f"Error A2 {patient_id}: {e}")
        continue

# Dataset JSON
dataset_json = {
    "channel_names": {"0": "Post_Contrast_Only"},
    "labels": {"background": 0, "tumor": 1},
    "numTraining": processed,
    "file_ending": ".nii.gz",
    "dataset_name": "A2_Post_Contrast_Only",
    "description": "Post-contrast image only with Z-score normalization"
>>>>>>> bfce48e9abdee76749a4d317a07a3c61092fd8cd
}

with open(output_path / "dataset.json", 'w') as f:
    json.dump(dataset_json, f, indent=2)

<<<<<<< HEAD
# Crear archivo de mapeo para referencia
mapping_data = {
    "train_mapping": {f"case_{i:03d}": case_id for i, case_id in enumerate(train_cases[:train_processed])},
    "test_mapping": {f"case_{i:03d}": case_id for i, case_id in enumerate(test_cases[:test_processed])},
    "creation_date": pd.Timestamp.now().isoformat(),
    "original_split_file": "train_test_splits.csv"
}

with open(output_path / "case_mapping.json", 'w') as f:
    json.dump(mapping_data, f, indent=2)

# Resumen final
print(f"\n{'='*60}")
print(f"EXPERIMENTO A2 OFICIAL COMPLETADO")
print(f"{'='*60}")
print(f"Dataset guardado en: {output_path}")
print(f"Casos de entrenamiento procesados: {train_processed}/{len(train_cases)}")
print(f"Casos de test procesados: {test_processed}/{len(test_cases)}")
print(f"Tasa de éxito train: {train_processed/len(train_cases)*100:.1f}%")
print(f"Tasa de éxito test: {test_processed/len(test_cases)*100:.1f}%")
print(f"{'='*60}")
print(f"COMANDOS SIGUIENTES:")
print(f"   nnUNetv2_plan_and_preprocess -d 112")
print(f"   nnUNetv2_train 112 3d_fullres 0 --npz")
print(f"{'='*60}")

# Verificación final
print(f"\nArchivos creados:")
print(f"   - dataset.json: Configuración nnU-Net")
print(f"   - case_mapping.json: Mapeo caso original - nnU-Net")
print(f"   - imagesTr/: {train_processed} imágenes entrenamiento")
print(f"   - labelsTr/: {train_processed} segmentaciones entrenamiento")
print(f"   - imagesTs/: {test_processed} imágenes test")
print(f"   - labelsTs/: {test_processed} segmentaciones test")

print(f"\nLISTO PARA ENTRENAR EN SUPERCOMPUTADOR 4x RTX A6000")
=======
print(f"A2 COMPLETADO: {processed} casos en {output_path}")
print("LISTO PARA: nnUNetv2_plan_and_preprocess -d 102")
print("ENTRENAR: nnUNetv2_train 102 3d_fullres 0")
>>>>>>> bfce48e9abdee76749a4d317a07a3c61092fd8cd
