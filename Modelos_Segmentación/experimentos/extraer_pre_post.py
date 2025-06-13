"""
EXTRAER SOLO PRE + POST1 (2 CANALES)
Crea dataset nnU-Net con exactamente 2 canales por caso
"""
import shutil
from pathlib import Path
import json

print("=== EXTRAER PRE + POST1 (2 CANALES) ===")

# Configuración
source_path = Path("C:/nnUNet_raw/Dataset112_A2_Baseline_AllPhases")
output_path = Path("C:/nnUNet_raw/Dataset113_A2_PrePost1")

# Crear directorios de salida
(output_path / "imagesTr").mkdir(parents=True, exist_ok=True)
(output_path / "labelsTr").mkdir(parents=True, exist_ok=True)
(output_path / "imagesTs").mkdir(parents=True, exist_ok=True)
(output_path / "labelsTs").mkdir(parents=True, exist_ok=True)

def extract_pre_post1(source_images_dir, output_images_dir, split_name):
    """Extrae solo PRE (_0000) y POST1 (_0001) de cada caso"""
    
    if not source_images_dir.exists():
        print(f"⚠️ {source_images_dir} no existe")
        return 0, []
    
    # Buscar todos los casos únicos
    all_files = list(source_images_dir.glob("case_*_*.nii.gz"))
    case_numbers = set()
    
    for file in all_files:
        # Extraer número de caso: case_123_0000.nii.gz → 123
        parts = file.stem.split('_')
        if len(parts) >= 3:
            case_numbers.add(parts[1])  # case_number
    
    case_numbers = sorted(case_numbers, key=lambda x: int(x))
    
    processed = 0
    errors = []
    
    print(f"\nProcesando {split_name}: {len(case_numbers)} casos encontrados")
    
    for i, case_num in enumerate(case_numbers):
        try:
            # Buscar archivos PRE y POST1
            pre_file = source_images_dir / f"case_{case_num}_0000.nii.gz"
            post1_file = source_images_dir / f"case_{case_num}_0001.nii.gz"
            
            if not pre_file.exists():
                errors.append(f"case_{case_num}: PRE no encontrado")
                continue
                
            if not post1_file.exists():
                errors.append(f"case_{case_num}: POST1 no encontrado")
                continue
            
            # Copiar archivos con nomenclatura nnU-Net
            output_pre = output_images_dir / f"case_{case_num}_0000.nii.gz"
            output_post1 = output_images_dir / f"case_{case_num}_0001.nii.gz"
            
            shutil.copy2(pre_file, output_pre)
            shutil.copy2(post1_file, output_post1)
            
            processed += 1
            
            if processed % 100 == 0 or processed <= 10:
                print(f"  ✓ case_{case_num}: PRE + POST1 copiados ({processed}/{len(case_numbers)})")
                
        except Exception as e:
            errors.append(f"case_{case_num}: Error - {e}")
    
    return processed, errors

def copy_segmentations(source_labels_dir, output_labels_dir, split_name):
    """Copia todas las segmentaciones"""
    
    if not source_labels_dir.exists():
        print(f"⚠️ {source_labels_dir} no existe")
        return 0
    
    seg_files = list(source_labels_dir.glob("case_*.nii.gz"))
    
    for seg_file in seg_files:
        output_seg = output_labels_dir / seg_file.name
        shutil.copy2(seg_file, output_seg)
    
    print(f"  ✓ {len(seg_files)} segmentaciones copiadas para {split_name}")
    return len(seg_files)

# ============ PROCESAR TRAIN ============
print("\n=== PROCESANDO TRAIN ===")
train_processed, train_errors = extract_pre_post1(
    source_path / "imagesTr",
    output_path / "imagesTr", 
    "TRAIN"
)

train_segs = copy_segmentations(
    source_path / "labelsTr",
    output_path / "labelsTr",
    "TRAIN"
)

# ============ PROCESAR TEST ============
print("\n=== PROCESANDO TEST ===")
test_processed, test_errors = extract_pre_post1(
    source_path / "imagesTs",
    output_path / "imagesTs",
    "TEST"  
)

test_segs = copy_segmentations(
    source_path / "labelsTs", 
    output_path / "labelsTs",
    "TEST"
)

# ============ MOSTRAR ERRORES ============
if train_errors:
    print(f"\nErrores TRAIN ({len(train_errors)}):")
    for error in train_errors[:10]:
        print(f"  {error}")

if test_errors:
    print(f"\nErrores TEST ({len(test_errors)}):")
    for error in test_errors[:10]:
        print(f"  {error}")

# ============ CREAR DATASET.JSON ============
dataset_json = {
    "channel_names": {
        "0": "T1_Pre_Contrast",
        "1": "T1_Post_Contrast"
    },
    "labels": {
        "background": 0,
        "tumor": 1
    },
    "numTraining": train_segs,
    "numTest": test_segs,
    "file_ending": ".nii.gz",
    "dataset_name": "A2_MAMA_MIA_PrePost1_Crops",
    "description": "A2: PRE + POST1 contrast DCE-MRI with tumor-focused crops",
    "reference": "MAMA-MIA Challenge Enhanced - PRE/POST1 strategy",
    "tensorImageSize": "3D",
    "modality": {
        "0": "T1_Pre_Contrast",
        "1": "T1_Post_Contrast"
    }
}

# Guardar dataset.json
with open(output_path / "dataset.json", 'w') as f:
    json.dump(dataset_json, f, indent=2)

# ============ VERIFICACIÓN FINAL ============
print(f"\n{'='*60}")
print(f"EXTRACCIÓN COMPLETADA")
print(f"{'='*60}")
print(f"Dataset guardado en: {output_path}")
print(f"Train: {train_processed} casos procesados, {train_segs} segmentaciones")
print(f"Test: {test_processed} casos procesados, {test_segs} segmentaciones")
print(f"Errores train: {len(train_errors)}")
print(f"Errores test: {len(test_errors)}")

# Verificar consistencia
train_images = len(list((output_path / "imagesTr").glob("case_*_0000.nii.gz")))
test_images = len(list((output_path / "imagesTs").glob("case_*_0000.nii.gz")))

print(f"\nVerificación:")
print(f"  Train: {train_images} casos con 2 canales cada uno")
print(f"  Test: {test_images} casos con 2 canales cada uno")
print(f"  Total imágenes train: {train_images * 2}")
print(f"  Total imágenes test: {test_images * 2}")

if train_images == train_segs and test_images == test_segs:
    print(f"\n✅ DATASET CONSISTENTE - LISTO PARA nnU-Net!")
    print(f"\nComandos:")
    print(f"   nnUNetv2_plan_and_preprocess -d 113 -c 3d_fullres")
    print(f"   nnUNetv2_train 113 3d_fullres 0 --npz")
    
    print(f"\nEstrategia:")
    print(f"   📊 Dataset: PRE + POST1 DCE-MRI crops")
    print(f"   🎯 Ventaja: Enfoque dirigido en tumor vs baseline")
    print(f"   🏆 Objetivo: Superar Dice 0.7620 del baseline MAMA-MIA")
else:
    print(f"\n⚠️ Inconsistencias detectadas - revisar errores")

print(f"{'='*60}")

# Verificar algunos casos específicos
print(f"\nVerificación de archivos (primeros 3 casos):")
for i in range(min(3, train_images)):
    case_num = f"{i:03d}"
    pre_exists = (output_path / "imagesTr" / f"case_{case_num}_0000.nii.gz").exists()
    post_exists = (output_path / "imagesTr" / f"case_{case_num}_0001.nii.gz").exists()
    seg_exists = (output_path / "labelsTr" / f"case_{case_num}.nii.gz").exists()
    
    status = "✅" if (pre_exists and post_exists and seg_exists) else "❌"
    print(f"  case_{case_num}: {status} PRE={pre_exists} POST1={post_exists} SEG={seg_exists}")

print(f"\n🚀 LISTO PARA ENTRENAR MODELO MEJORADO!")