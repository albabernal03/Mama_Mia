# quick_test.py - Script para probar tu estructura de datos

import pandas as pd
from pathlib import Path

def quick_debug():
    """Test rápido de tu estructura de datos"""
    
    # TUS RUTAS (ajusta si es necesario)
    SPLITS_CSV = r"C:\Users\usuario\Documents\Mama_Mia\datos\train_test_splits.csv"
    PCR_LABELS_CSV = r"C:\Users\usuario\Documents\Mama_Mia\ganadores_pcr\pcr_labels.csv"
    CROPPED_DATA_DIR = r"C:\Users\usuario\Documents\Mama_Mia\datos\mask_cropped_data"
    
    print("🔍 VERIFICANDO ESTRUCTURA DE DATOS")
    print("=" * 50)
    
    # 1. Check train_test_splits.csv
    print(f"\n📄 Checking train_test_splits.csv:")
    try:
        splits_df = pd.read_csv(SPLITS_CSV)
        print(f"   ✅ {len(splits_df)} filas encontradas")
        print(f"   Columnas: {list(splits_df.columns)}")
        print(f"   Primeras 3 filas:")
        for i, row in splits_df.head(3).iterrows():
            print(f"     Row {i}: train={row.get('train_split', 'N/A')}, test={row.get('test_split', 'N/A')}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return
    
    # 2. Check pcr_labels.csv  
    print(f"\n📄 Checking pcr_labels.csv:")
    try:
        pcr_df = pd.read_csv(PCR_LABELS_CSV)
        print(f"   ✅ {len(pcr_df)} pacientes con labels")
        print(f"   Columnas: {list(pcr_df.columns)}")
        
        # Adivinar nombre de columnas
        patient_col = None
        pcr_col = None
        
        for col in pcr_df.columns:
            if 'patient' in col.lower() or 'id' in col.lower():
                patient_col = col
            if 'pcr' in col.lower():
                pcr_col = col
        
        if patient_col and pcr_col:
            print(f"   Columnas detectadas: PatientID='{patient_col}', pCR='{pcr_col}'")
            
            # Distribución de clases
            pcr_dist = pcr_df[pcr_col].value_counts()
            print(f"   Distribución pCR: {dict(pcr_dist)}")
        else:
            print(f"   ⚠️  No se pudieron detectar columnas automáticamente")
            print(f"   Primeras 3 filas:")
            for i, row in pcr_df.head(3).iterrows():
                print(f"     Row {i}: {dict(row)}")
                
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return
    
    # 3. Check cropped data structure
    print(f"\n📁 Checking cropped data structure:")
    cropped_path = Path(CROPPED_DATA_DIR)
    
    if not cropped_path.exists():
        print(f"   ❌ Directory not found: {cropped_path}")
        return
    
    print(f"   ✅ Directory exists: {cropped_path}")
    
    # Check images and segmentations
    for data_type in ['images', 'segmentations']:
        type_path = cropped_path / data_type
        if type_path.exists():
            print(f"   📂 {data_type}/")
            
            for split in ['train_split', 'test_split']:
                split_path = type_path / split
                if split_path.exists():
                    patients = [p for p in split_path.iterdir() if p.is_dir()]
                    print(f"     📂 {split}/: {len(patients)} pacientes")
                    
                    # Sample patient analysis
                    if patients:
                        sample_patient = patients[0]
                        files = list(sample_patient.glob("*.nii.gz"))
                        print(f"       Ejemplo {sample_patient.name}: {len(files)} archivos .nii.gz")
                        
                        # Show file types
                        if data_type == 'images':
                            dce_files = [f for f in files if 'seg' not in f.name]
                            print(f"         - {len(dce_files)} archivos DCE")
                            if dce_files:
                                print(f"         - Ejemplo: {dce_files[0].name}")
                        else:
                            seg_files = [f for f in files if 'seg' in f.name]
                            print(f"         - {len(seg_files)} archivos segmentación")
                            if seg_files:
                                print(f"         - Ejemplo: {seg_files[0].name}")
                else:
                    print(f"     ❌ {split}/ not found")
        else:
            print(f"   ❌ {data_type}/ not found")
    
    # 4. Cross-check patients
    print(f"\n🔗 Cross-checking pacientes entre fuentes:")
    
    # Get patients from splits
    train_patients = set()
    test_patients = set()
    
    for _, row in splits_df.iterrows():
        if pd.notna(row.get('train_split')):
            train_patients.add(str(row['train_split']).strip().upper())
        if pd.notna(row.get('test_split')):
            test_patients.add(str(row['test_split']).strip().upper())
    
    # Get patients from pcr labels
    pcr_patients = set()
    if patient_col:
        for patient in pcr_df[patient_col]:
            if pd.notna(patient):
                pcr_patients.add(str(patient).strip().upper())
    
    # Get patients from cropped data
    cropped_train = set()
    cropped_test = set()
    
    train_img_path = cropped_path / "images" / "train_split"
    test_img_path = cropped_path / "images" / "test_split"
    
    if train_img_path.exists():
        cropped_train = {p.name.upper() for p in train_img_path.iterdir() if p.is_dir()}
    if test_img_path.exists():
        cropped_test = {p.name.upper() for p in test_img_path.iterdir() if p.is_dir()}
    
    print(f"   Splits CSV: {len(train_patients)} train, {len(test_patients)} test")
    print(f"   PCR labels: {len(pcr_patients)} pacientes")
    print(f"   Cropped data: {len(cropped_train)} train, {len(cropped_test)} test")
    
    # Find overlaps
    train_with_pcr = train_patients & pcr_patients
    test_with_pcr = test_patients & pcr_patients
    train_with_data = train_patients & cropped_train
    test_with_data = test_patients & cropped_test
    
    print(f"   ✅ Train válidos (splits+pcr+data): {len(train_with_pcr & cropped_train)}")
    print(f"   ✅ Test válidos (splits+pcr+data): {len(test_with_pcr & cropped_test)}")
    
    if len(train_with_pcr & cropped_train) < 5:
        print(f"   ⚠️  Muy pocos casos de training válidos!")
        print(f"      Train patients: {train_patients}")
        print(f"      PCR patients: {pcr_patients}")
        print(f"      Cropped train: {cropped_train}")
    
    print(f"\n🎯 LISTO PARA CONTINUAR!")
    return True

if __name__ == "__main__":
    quick_debug()