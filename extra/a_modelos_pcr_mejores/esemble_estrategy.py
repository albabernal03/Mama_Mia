# investigate_val_test_gap.py
"""
INVESTIGAR: ¿Por qué gap tan grande entre Val (0.64) y Test (0.57)?

Gap normal: 0.01-0.02
Gap actual: 0.06-0.07 (ANORMAL)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from sklearn.model_selection import train_test_split

def analyze_val_test_gap():
    """Investigar el gap anormalmente grande entre val y test"""
    
    print("🔍 INVESTIGANDO GAP VAL-TEST ANORMAL")
    print("🎯 Gap normal: 0.01-0.02, Gap actual: 0.06-0.07")
    print("=" * 60)
    
    # Cargar splits oficiales
    OFFICIAL_SPLITS_CSV = Path(r"C:\Users\usuario\Documents\Mama_Mia\datos\train_test_splits.csv")
    
    if not OFFICIAL_SPLITS_CSV.exists():
        print(f"❌ CSV no encontrado")
        return
    
    df = pd.read_csv(OFFICIAL_SPLITS_CSV)
    train_ids_csv = df['train_split'].dropna().unique().tolist()
    test_ids_csv = df['test_split'].dropna().unique().tolist()
    
    # Cargar labels
    with open(Path(r"D:\clinical_data_complete.json"), 'r') as f:
        pcr_list = json.load(f)
    
    pcr_data = {item['patient_id']: item for item in pcr_list}
    
    def analyze_split(patient_ids, split_name):
        """Analizar características de un split"""
        valid_ids, labels, centers, ages = [], [], [], []
        
        for pid in patient_ids:
            pid = str(pid)
            if pid in pcr_data and 'pcr' in pcr_data[pid]:
                if pcr_data[pid]['pcr'] in ["0", "1"]:
                    valid_ids.append(pid)
                    labels.append(int(pcr_data[pid]['pcr']))
                    
                    # Extraer centro (primeras letras del ID)
                    center = pid.split('_')[0] if '_' in pid else pid[:4]
                    centers.append(center)
                    
                    # Extraer edad si disponible
                    age = pcr_data[pid].get('age', None)
                    ages.append(age)
        
        # Estadísticas
        pcr_rate = np.mean(labels)
        center_counts = pd.Series(centers).value_counts()
        
        print(f"\n📊 {split_name.upper()}:")
        print(f"  👥 Pacientes: {len(valid_ids)}")
        print(f"  📈 pCR rate: {pcr_rate:.1%}")
        print(f"  🏥 Centros únicos: {len(center_counts)}")
        
        # Top centros
        print(f"  🏆 Top centros:")
        for center, count in center_counts.head(5).items():
            center_pcr = np.mean([labels[i] for i, c in enumerate(centers) if c == center])
            print(f"    {center}: {count} pacientes ({center_pcr:.1%} pCR)")
        
        # Edades
        valid_ages = [a for a in ages if a is not None]
        if valid_ages:
            print(f"  🎂 Edad promedio: {np.mean(valid_ages):.1f} ± {np.std(valid_ages):.1f}")
        
        return {
            'ids': valid_ids,
            'labels': labels,
            'centers': centers,
            'pcr_rate': pcr_rate,
            'center_distribution': center_counts.to_dict()
        }
    
    # Analizar train y test oficiales
    train_analysis = analyze_split(train_ids_csv, 'TRAIN oficial')
    test_analysis = analyze_split(test_ids_csv, 'TEST oficial')
    
    # Crear val split como en entrenamiento
    train_train_ids, val_ids, train_train_labels, val_labels = train_test_split(
        train_analysis['ids'], train_analysis['labels'], 
        test_size=0.2, random_state=42, stratify=train_analysis['labels']
    )
    
    val_analysis = analyze_split(val_ids, 'VAL creado')
    
    # Comparar distribuciones
    print(f"\n🔍 COMPARACIÓN DE DISTRIBUCIONES:")
    print("=" * 50)
    
    print(f"📈 pCR rates:")
    print(f"  Train: {train_analysis['pcr_rate']:.1%}")
    print(f"  Val:   {np.mean(val_labels):.1%}")
    print(f"  Test:  {test_analysis['pcr_rate']:.1%}")
    
    # Overlap de centros
    train_centers = set(train_analysis['centers'])
    test_centers = set(test_analysis['centers'])
    val_centers = set([test_analysis['centers'][i] for i, pid in enumerate(test_analysis['ids']) if pid in val_ids])
    
    overlap_val_test = len(train_centers & test_centers)
    total_test_centers = len(test_centers)
    
    print(f"\n🏥 Análisis de centros:")
    print(f"  Centros en Train: {len(train_centers)}")
    print(f"  Centros en Test: {len(test_centers)}")
    print(f"  Overlap Train-Test: {overlap_val_test}/{total_test_centers} ({overlap_val_test/total_test_centers:.1%})")
    
    # Buscar diferencias significativas
    print(f"\n⚠️ POSIBLES CAUSAS DEL GAP:")
    
    # 1. Diferencia de pCR rate
    pcr_diff = abs(np.mean(val_labels) - test_analysis['pcr_rate'])
    if pcr_diff > 0.02:
        print(f"  📈 pCR rate muy diferente: Val {np.mean(val_labels):.1%} vs Test {test_analysis['pcr_rate']:.1%}")
    
    # 2. Distribución de centros muy diferente
    if overlap_val_test / total_test_centers < 0.8:
        print(f"  🏥 Distribución de centros muy diferente")
    
    # 3. Test set demasiado pequeño
    if len(test_analysis['ids']) < 200:
        print(f"  👥 Test set pequeño: {len(test_analysis['ids'])} pacientes")
    
    # 4. Overfitting severo
    if True:  # Siempre mostrar
        print(f"  🧠 Overfitting severo: Gap {0.64 - 0.57:.2f} > normal (0.02)")
    
    return train_analysis, val_analysis, test_analysis

def main():
    train_analysis, val_analysis, test_analysis = analyze_val_test_gap()
    
    print(f"\n💡 RECOMENDACIONES:")
    print("=" * 30)
    print(f"1. 🔄 Cross-validation para validar overfitting")
    print(f"2. 📊 Usar test set de baseline original (no splits CSV)")
    print(f"3. 🎯 Entrenar con más regularización")
    print(f"4. 🏥 Análisis por centro médico")

if __name__ == "__main__":
    main()