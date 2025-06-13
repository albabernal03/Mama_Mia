#!/usr/bin/env python3
"""
Test universal para verificar mapeo de archivos
"""

from pathlib import Path
import pandas as pd

def find_image_file(base_path, case_id):
    """Buscar archivo de imagen usando diferentes patrones de nombres"""
    base_path = Path(base_path)
    
    # Patrones posibles
    patterns = [
        f"{case_id}.nii.gz",                    # DUKE_019.nii.gz
        f"{case_id.lower()}.nii.gz",            # duke_019.nii.gz
        f"{case_id.lower()}_0000.nii.gz",       # duke_019_0000.nii.gz
    ]
    
    # Si es DUKE_XXX, probar duke_XXX_0000.nii.gz
    if case_id.startswith('DUKE_'):
        duke_num = case_id.split('_')[1]
        patterns.extend([
            f"duke_{duke_num.zfill(3)}_0000.nii.gz",  # duke_019_0000.nii.gz
            f"duke_{duke_num}_0000.nii.gz",           # duke_19_0000.nii.gz
        ])
    
    # Si es ISPY1_XXXX, probar ispy1_xxxx_0000.nii.gz
    if case_id.startswith('ISPY1_'):
        ispy_num = case_id.split('_')[1]
        patterns.extend([
            f"ispy1_{ispy_num}_0000.nii.gz",
            f"ispy1_{ispy_num.zfill(4)}_0000.nii.gz",
        ])
        
    # Si es ISPY2_XXXXXX, probar ispy2_xxxxxx_0000.nii.gz
    if case_id.startswith('ISPY2_'):
        ispy_num = case_id.split('_')[1]
        patterns.extend([
            f"ispy2_{ispy_num}_0000.nii.gz",
            f"ispy2_{ispy_num.zfill(6)}_0000.nii.gz",
        ])
        
    # Si es NACT_XX, probar nact_xx_0000.nii.gz
    if case_id.startswith('NACT_'):
        nact_num = case_id.split('_')[1]
        patterns.extend([
            f"nact_{nact_num}_0000.nii.gz",
            f"nact_{nact_num.zfill(2)}_0000.nii.gz",
        ])
    
    # Buscar archivos que coincidan
    for pattern in patterns:
        candidate = base_path / pattern
        if candidate.exists():
            return candidate
            
    return None

def test_file_mapping():
    print("TEST DE MAPEO DE ARCHIVOS")
    print("=" * 40)
    
    # Cargar casos de test
    csv_path = Path(r"C:\Users\usuario\Documents\Mama_Mia\datos\train_test_splits.csv")
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        test_cases = df['test_split'].dropna().tolist()[:10]  # Primeros 10
    else:
        test_cases = ["DUKE_019", "ISPY1_1005", "ISPY2_108939", "NACT_02"]
    
    images_path = Path(r"C:\Users\usuario\Documents\Mama_Mia\datos\images")
    
    print(f"Testeando {len(test_cases)} casos...")
    
    found = 0
    for case_id in test_cases:
        image_path = find_image_file(images_path, case_id)
        
        if image_path:
            print(f"OK {case_id} -> {image_path.name}")
            found += 1
        else:
            print(f"FAIL {case_id} -> No encontrado")
    
    print(f"\nResultado: {found}/{len(test_cases)} casos encontrados")
    
    if found == 0:
        # Mostrar archivos disponibles
        available = list(images_path.glob("*_0000.nii.gz"))[:10]
        print(f"\nArchivos disponibles (primeros 10):")
        for f in available:
            print(f"  {f.name}")
    
    return found > 0

if __name__ == "__main__":
    success = test_file_mapping()
    if success:
        print("\nTEST EXITOSO - Mapeo funcionando!")
    else:
        print("\nTEST FALLIDO - Revisar patrones de nombres")