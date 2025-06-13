# extract_pcr_labels.py
"""
Extraer etiquetas pCR del archivo Excel clinical_and_imaging_info.xlsx
y generar CSV para el pipeline de entrenamiento
"""

import pandas as pd
import numpy as np
from pathlib import Path

def extract_pcr_labels(excel_path: str, output_csv: str = "pcr_labels.csv"):
    """Extraer etiquetas pCR del Excel y guardar como CSV limpio"""
    
    print("🎯 EXTRAYENDO ETIQUETAS pCR DEL EXCEL")
    print("=" * 50)
    
    # 1. CARGAR EXCEL
    try:
        # Leer la hoja 'dataset_info'
        df = pd.read_excel(excel_path, sheet_name='dataset_info')
        print(f"✅ Excel cargado: {len(df)} filas")
        
        # Verificar columnas necesarias
        required_cols = ['patient_id', 'pcr']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            print(f"❌ Columnas faltantes: {missing_cols}")
            print(f"📋 Columnas disponibles: {list(df.columns)}")
            return None
        
        print(f"✅ Columnas encontradas: patient_id, pcr")
        
    except Exception as e:
        print(f"❌ Error cargando Excel: {e}")
        return None
    
    # 2. LIMPIAR DATOS
    print(f"\n📊 LIMPIANDO DATOS:")
    
    # Filtrar filas con patient_id válido
    df_clean = df.dropna(subset=['patient_id']).copy()
    print(f"   Pacientes con ID válido: {len(df_clean)}")
    
    # Verificar valores de pCR
    print(f"   Valores únicos de pCR: {sorted(df_clean['pcr'].unique())}")
    
    # Filtrar solo valores binarios válidos (0, 1)
    valid_pcr_values = [0, 1, 0.0, 1.0, '0', '1']
    df_valid = df_clean[df_clean['pcr'].isin(valid_pcr_values)].copy()
    
    print(f"   Pacientes con pCR válido: {len(df_valid)}")
    print(f"   Pacientes excluidos: {len(df_clean) - len(df_valid)}")
    
    # 3. CONVERTIR A FORMATO ESTÁNDAR
    pcr_labels = df_valid[['patient_id', 'pcr']].copy()
    
    # Limpiar patient_id
    pcr_labels['patient_id'] = pcr_labels['patient_id'].astype(str).str.strip()
    
    # Convertir pCR a entero binario
    pcr_labels['pcr_response'] = pcr_labels['pcr'].astype(int)
    
    # Mantener solo columnas necesarias
    pcr_labels = pcr_labels[['patient_id', 'pcr_response']]
    
    # 4. ANÁLISIS DE DISTRIBUCIÓN
    print(f"\n📈 DISTRIBUCIÓN FINAL:")
    pcr_counts = pcr_labels['pcr_response'].value_counts().sort_index()
    total = len(pcr_labels)
    
    for pcr_value, count in pcr_counts.items():
        percentage = (count / total * 100)
        label = "pCR" if pcr_value == 1 else "No pCR"
        print(f"   {label} ({pcr_value}): {count:,} pacientes ({percentage:.1f}%)")
    
    # Ratio de desbalance
    pcr_positive = pcr_counts.get(1, 0)
    pcr_negative = pcr_counts.get(0, 0)
    if pcr_positive > 0:
        ratio = pcr_negative / pcr_positive
        print(f"   Ratio desbalance: {ratio:.2f}:1")
    
    # 5. VERIFICAR CALIDAD
    print(f"\n🔍 VERIFICACIÓN DE CALIDAD:")
    
    # Verificar duplicados
    duplicates = pcr_labels['patient_id'].duplicated().sum()
    print(f"   Pacientes duplicados: {duplicates}")
    
    # Verificar valores faltantes
    missing = pcr_labels.isnull().sum().sum()
    print(f"   Valores faltantes: {missing}")
    
    # Verificar formato patient_id
    sample_ids = pcr_labels['patient_id'].head().tolist()
    print(f"   Ejemplos de IDs: {sample_ids}")
    
    # 6. GUARDAR CSV
    try:
        pcr_labels.to_csv(output_csv, index=False)
        print(f"\n💾 CSV GUARDADO: {output_csv}")
        print(f"   Total registros: {len(pcr_labels):,}")
        
        # Mostrar muestra del CSV
        print(f"\n📄 MUESTRA DEL CSV:")
        print(pcr_labels.head(10).to_string(index=False))
        
        return pcr_labels
        
    except Exception as e:
        print(f"❌ Error guardando CSV: {e}")
        return None

def validate_pcr_labels(csv_path: str, splits_csv: str):
    """Validar que las etiquetas pCR coincidan con los splits de entrenamiento"""
    
    print(f"\n🔍 VALIDANDO ETIQUETAS CON SPLITS DE ENTRENAMIENTO")
    print("=" * 60)
    
    try:
        # Cargar etiquetas pCR
        pcr_df = pd.read_csv(csv_path)
        print(f"✅ Etiquetas pCR cargadas: {len(pcr_df)} pacientes")
        
        # Cargar splits
        splits_df = pd.read_csv(splits_csv)
        print(f"✅ Splits cargados: {len(splits_df)} filas")
        
        # Extraer todos los patient_ids de los splits
        train_patients = splits_df['train_split'].dropna().astype(str).str.strip().tolist()
        test_patients = splits_df['test_split'].dropna().astype(str).str.strip().tolist()
        all_split_patients = set(train_patients + test_patients)
        
        print(f"   Pacientes en train_split: {len(train_patients)}")
        print(f"   Pacientes en test_split: {len(test_patients)}")
        print(f"   Pacientes únicos en splits: {len(all_split_patients)}")
        
        # Verificar coincidencias
        pcr_patients = set(pcr_df['patient_id'].astype(str).str.strip())
        
        # Pacientes con pCR que están en splits
        in_both = pcr_patients & all_split_patients
        only_in_pcr = pcr_patients - all_split_patients
        only_in_splits = all_split_patients - pcr_patients
        
        print(f"\n📊 ANÁLISIS DE COINCIDENCIAS:")
        print(f"   Pacientes con pCR Y en splits: {len(in_both)}")
        print(f"   Pacientes con pCR pero NO en splits: {len(only_in_pcr)}")
        print(f"   Pacientes en splits pero SIN pCR: {len(only_in_splits)}")
        
        # Cobertura
        coverage = (len(in_both) / len(all_split_patients) * 100) if all_split_patients else 0
        print(f"   Cobertura de etiquetas: {coverage:.1f}%")
        
        if len(only_in_splits) > 0:
            print(f"\n⚠️  PACIENTES SIN ETIQUETAS pCR (primeros 10):")
            for patient in list(only_in_splits)[:10]:
                print(f"     {patient}")
        
        if coverage >= 90:
            print(f"\n✅ VALIDACIÓN EXITOSA - Cobertura suficiente para entrenamiento")
        else:
            print(f"\n⚠️  COBERTURA BAJA - Verificar etiquetas faltantes")
        
        return {
            'coverage': coverage,
            'patients_with_labels': len(in_both),
            'missing_labels': len(only_in_splits),
            'total_splits': len(all_split_patients)
        }
        
    except Exception as e:
        print(f"❌ Error en validación: {e}")
        return None

def main():
    """Ejecutar extracción completa de etiquetas pCR"""
    
    # Rutas de archivos
    excel_path = r"C:\Users\usuario\Documents\Mama_Mia\datos\clinical_and_imaging_info.xlsx"
    output_csv = "pcr_labels.csv"
    splits_csv = r"C:\Users\usuario\Documents\Mama_Mia\datos\train_test_splits.csv"
    
    print("🎯 EXTRACCIÓN COMPLETA DE ETIQUETAS pCR")
    print("=" * 60)
    
    # 1. Extraer etiquetas del Excel
    pcr_labels = extract_pcr_labels(excel_path, output_csv)
    
    if pcr_labels is None:
        print("❌ Error en extracción - Abortando")
        return
    
    # 2. Validar con splits de entrenamiento
    if Path(splits_csv).exists():
        validation_results = validate_pcr_labels(output_csv, splits_csv)
        
        if validation_results and validation_results['coverage'] >= 90:
            print(f"\n🎯 PROCESO COMPLETADO EXITOSAMENTE")
            print(f"✅ Archivo CSV generado: {output_csv}")
            print(f"✅ Cobertura: {validation_results['coverage']:.1f}%")
            print(f"✅ Pacientes listos para entrenamiento: {validation_results['patients_with_labels']}")
            print(f"\n🚀 SIGUIENTE PASO: Ejecutar pipeline de entrenamiento")
        else:
            print(f"\n⚠️  VERIFICAR ETIQUETAS FALTANTES ANTES DE CONTINUAR")
    else:
        print(f"\n⚠️  Archivo de splits no encontrado: {splits_csv}")
        print(f"✅ Archivo CSV generado: {output_csv}")

if __name__ == "__main__":
    main()