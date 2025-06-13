# verificar_datos.py
import os
from pathlib import Path

BASE_PATH = r"C:\Users\usuario\Documents\Mama_Mia\cropped_data"

def verificar_estructura():
    print("🔍 Verificando tu estructura de datos...")
    print(f"Base path: {BASE_PATH}")
    print("=" * 50)
    
    # Verificar directorios principales
    base = Path(BASE_PATH)
    
    train_images = base / "images" / "train_split"
    test_images = base / "images" / "test_split"
    train_masks = base / "segmentations" / "train_split"
    test_masks = base / "segmentations" / "test_split"
    
    print(f"✓ Train images: {train_images} - Existe: {train_images.exists()}")
    print(f"✓ Test images: {test_images} - Existe: {test_images.exists()}")
    print(f"✓ Train masks: {train_masks} - Existe: {train_masks.exists()}")
    print(f"✓ Test masks: {test_masks} - Existe: {test_masks.exists()}")
    print()
    
    # Contar casos en cada split
    if train_images.exists():
        train_cases = [d for d in train_images.iterdir() if d.is_dir()]
        print(f"📁 Train cases encontrados: {len(train_cases)}")
        if len(train_cases) > 0:
            print(f"   Ejemplo: {train_cases[0].name}")
            # Verificar archivos dentro del primer caso
            example_case = train_cases[0]
            files = list(example_case.glob("*.nii.gz"))
            print(f"   Archivos en {example_case.name}: {len(files)}")
            for f in files[:3]:  # Mostrar primeros 3
                print(f"     - {f.name}")
    
    if test_images.exists():
        test_cases = [d for d in test_images.iterdir() if d.is_dir()]
        print(f"📁 Test cases encontrados: {len(test_cases)}")
        if len(test_cases) > 0:
            print(f"   Ejemplo: {test_cases[0].name}")
    
    if train_masks.exists():
        train_mask_cases = [d for d in train_masks.iterdir() if d.is_dir()]
        print(f"🎭 Train masks encontradas: {len(train_mask_cases)}")
        if len(train_mask_cases) > 0:
            example_mask_case = train_mask_cases[0]
            mask_files = list(example_mask_case.glob("*.nii.gz"))
            print(f"   Máscaras en {example_mask_case.name}: {len(mask_files)}")
            for f in mask_files:
                print(f"     - {f.name}")
    
    if test_masks.exists():
        test_mask_cases = [d for d in test_masks.iterdir() if d.is_dir()]
        print(f"🎭 Test masks encontradas: {len(test_mask_cases)}")

if __name__ == "__main__":
    verificar_estructura()