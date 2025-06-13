# diagnostic_debug.py
"""
DIAGNÓSTICO URGENTE: ¿Por qué bajó de 0.6440 a 0.5833?

Investigar todas las posibles causas del problema
"""

import torch
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from pathlib import Path
import json
import hashlib

# Importar código base
from mejorar_resnet3d import *

def check_model_files():
    """Verificar qué modelos están disponibles"""
    print("🔍 VERIFICANDO ARCHIVOS DE MODELOS")
    print("=" * 50)
    
    model_files = list(Path('.').glob('*.pth'))
    
    if not model_files:
        print("❌ No se encontraron archivos .pth")
        return None
    
    print(f"📁 Modelos encontrados: {len(model_files)}")
    for f in model_files:
        size_mb = f.stat().st_size / (1024*1024)
        print(f"  📄 {f.name} ({size_mb:.1f} MB)")
    
    return model_files

def check_splits_consistency():
    """Verificar si los splits son consistentes"""
    print("\n🔍 VERIFICANDO CONSISTENCIA DE SPLITS")
    print("=" * 50)
    
    # Cargar CSV oficial
    OFFICIAL_SPLITS_CSV = Path(r"C:\Users\usuario\Documents\Mama_Mia\datos\train_test_splits.csv")
    
    if not OFFICIAL_SPLITS_CSV.exists():
        print(f"❌ CSV oficial no encontrado: {OFFICIAL_SPLITS_CSV}")
        return None
    
    df = pd.read_csv(OFFICIAL_SPLITS_CSV)
    train_ids_csv = df['train_split'].dropna().unique()
    test_ids_csv = df['test_split'].dropna().unique()
    
    print(f"📊 IDs en CSV:")
    print(f"  Train: {len(train_ids_csv)}")
    print(f"  Test: {len(test_ids_csv)}")
    
    # Cargar labels y verificar pacientes válidos
    with open(FastConfig.LABELS_FILE, 'r') as f:
        pcr_list = json.load(f)
    
    pcr_data = {item['patient_id']: item for item in pcr_list}
    
    def count_valid(patient_ids, split_name):
        valid_count = 0
        pcr_count = 0
        for pid in patient_ids:
            pid = str(pid)
            if pid in pcr_data and 'pcr' in pcr_data[pid]:
                if pcr_data[pid]['pcr'] in ["0", "1"]:
                    tensor_path = FastConfig.DATA_DIR / pid / f"{pid}_tensor_3ch.nii.gz"
                    if tensor_path.exists():
                        valid_count += 1
                        if pcr_data[pid]['pcr'] == "1":
                            pcr_count += 1
        
        pcr_rate = pcr_count / valid_count if valid_count > 0 else 0
        print(f"📊 {split_name}:")
        print(f"  Válidos: {valid_count}")
        print(f"  pCR rate: {pcr_rate:.1%}")
        
        return valid_count, pcr_rate
    
    train_valid, train_pcr = count_valid(train_ids_csv, "TRAIN")
    test_valid, test_pcr = count_valid(test_ids_csv, "TEST")
    
    # Verificar si coinciden con resultados anteriores
    expected_test_count = 306  # Del resultado anterior
    if test_valid != expected_test_count:
        print(f"⚠️ INCONSISTENCIA: Test esperado {expected_test_count}, actual {test_valid}")
    else:
        print(f"✅ Test count consistente: {test_valid}")
    
    return {
        'train_valid': train_valid,
        'test_valid': test_valid,
        'train_pcr': train_pcr,
        'test_pcr': test_pcr
    }

def test_model_loading():
    """Probar cargar diferentes modelos y ver sus predicciones"""
    print("\n🔍 PROBANDO CARGA DE MODELOS")
    print("=" * 50)
    
    model_files = [
        'best_model_resnet14t_80cubed_3ch_WINNER.pth',
        'best_model_OFFICIAL_SPLITS_CORRECTED.pth',
        'best_fast_model.pth'
    ]
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Configurar para modelo esperado
    FastConfig.TARGET_SHAPE = (80, 80, 80)
    FastConfig.INPUT_CHANNELS = 3
    FastConfig.MODEL_TYPE = 'resnet14t'
    FastConfig.BATCH_SIZE = 12
    
    results = {}
    
    for model_file in model_files:
        model_path = Path(model_file)
        if not model_path.exists():
            print(f"❌ {model_file}: No existe")
            continue
        
        print(f"\n📊 Probando: {model_file}")
        
        try:
            # Crear modelo
            model = FastTimmModel(
                model_name=FastConfig.MODEL_TYPE,
                in_channels=FastConfig.INPUT_CHANNELS,
                num_classes=1
            ).to(device)
            
            # Cargar weights
            model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
            model.eval()
            
            # Crear un tensor de prueba
            test_tensor = torch.randn(1, 3, 80, 80, 80).to(device)
            
            with torch.no_grad():
                output = model(test_tensor)
                prob = torch.sigmoid(output)
            
            print(f"✅ {model_file}: Carga OK, output shape: {output.shape}")
            print(f"   Sample prediction: {prob.item():.4f}")
            
            results[model_file] = 'OK'
            
        except Exception as e:
            print(f"❌ {model_file}: Error - {e}")
            results[model_file] = f'Error: {e}'
    
    return results

def compare_evaluation_methods():
    """Comparar evaluación actual vs original"""
    print("\n🔍 COMPARANDO MÉTODOS DE EVALUACIÓN")
    print("=" * 50)
    
    # Método 1: Recrear exactamente la evaluación original
    print("📊 Método 1: Evaluación original (batch_size=12)")
    auc1 = evaluate_with_original_method()
    
    # Método 2: Evaluación de TTA (batch_size=1)
    print("\n📊 Método 2: Evaluación TTA (batch_size=1)")
    auc2 = evaluate_with_tta_method()
    
    print(f"\n📊 COMPARACIÓN:")
    print(f"  Método original: {auc1:.4f}")
    print(f"  Método TTA:      {auc2:.4f}")
    print(f"  Diferencia:      {auc2 - auc1:+.4f}")
    
    if abs(auc1 - auc2) > 0.01:
        print("⚠️ GRAN DIFERENCIA - Hay un problema en el método")
    else:
        print("✅ Métodos consistentes")
    
    return auc1, auc2

def evaluate_with_original_method():
    """Recrear exactamente la evaluación que dio 0.6440"""
    
    # Configuración exacta del modelo ganador
    FastConfig.TARGET_SHAPE = (80, 80, 80)
    FastConfig.INPUT_CHANNELS = 3
    FastConfig.MODEL_TYPE = 'resnet14t'
    FastConfig.BATCH_SIZE = 12
    
    # Cargar splits oficiales
    from corrected_split_loader import create_splits_from_csv
    splits = create_splits_from_csv()
    test_ids, test_labels = splits['test']
    
    # Dataset exacto
    test_dataset = FastDataset(
        FastConfig.DATA_DIR, test_ids, test_labels,
        augment=False,
        target_shape=FastConfig.TARGET_SHAPE,
        input_channels=FastConfig.INPUT_CHANNELS
    )
    
    # DataLoader exacto
    test_loader = DataLoader(
        test_dataset, batch_size=FastConfig.BATCH_SIZE,
        shuffle=False, num_workers=2, pin_memory=True
    )
    
    # Modelo exacto
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = FastTimmModel(
        model_name=FastConfig.MODEL_TYPE,
        in_channels=FastConfig.INPUT_CHANNELS,
        num_classes=1
    ).to(device)
    
    # Probar diferentes archivos de modelo
    model_files = [
        'best_model_resnet14t_80cubed_3ch_WINNER.pth',
        'best_model_OFFICIAL_SPLITS_CORRECTED.pth'
    ]
    
    for model_file in model_files:
        if Path(model_file).exists():
            print(f"📄 Usando: {model_file}")
            model.load_state_dict(torch.load(model_file, map_location=device, weights_only=True))
            break
    else:
        print("❌ No se encontró ningún modelo válido")
        return 0.0
    
    # Evaluación exacta
    model.eval()
    test_preds, test_targets = [], []
    
    trainer = FastTrainer()
    with torch.no_grad():
        for batch in test_loader:
            tensors = batch['tensor'].to(device, non_blocking=True)
            targets = batch['target'].to(device, non_blocking=True).float()
            
            if trainer.use_amp:
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    outputs = model(tensors).squeeze()
            else:
                outputs = model(tensors).squeeze()
            
            probs = torch.sigmoid(outputs)
            test_preds.extend(safe_to_numpy(probs))
            test_targets.extend(safe_to_numpy(targets))
    
    auc = roc_auc_score(test_targets, test_preds)
    print(f"   Pacientes evaluados: {len(test_targets)}")
    print(f"   AUC obtenido: {auc:.4f}")
    
    return auc

def evaluate_with_tta_method():
    """Método usado en TTA (batch_size=1)"""
    
    # Misma configuración pero batch_size=1
    FastConfig.TARGET_SHAPE = (80, 80, 80)
    FastConfig.INPUT_CHANNELS = 3
    FastConfig.MODEL_TYPE = 'resnet14t'
    
    from corrected_split_loader import create_splits_from_csv
    splits = create_splits_from_csv()
    test_ids, test_labels = splits['test']
    
    test_dataset = FastDataset(
        FastConfig.DATA_DIR, test_ids, test_labels,
        augment=False,
        target_shape=FastConfig.TARGET_SHAPE,
        input_channels=FastConfig.INPUT_CHANNELS
    )
    
    # Batch size 1 (como en TTA)
    test_loader = DataLoader(
        test_dataset, batch_size=1,
        shuffle=False, num_workers=2, pin_memory=True
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = FastTimmModel(
        model_name=FastConfig.MODEL_TYPE,
        in_channels=FastConfig.INPUT_CHANNELS,
        num_classes=1
    ).to(device)
    
    # Cargar modelo
    model_file = 'best_model_resnet14t_80cubed_3ch_WINNER.pth'
    if Path(model_file).exists():
        model.load_state_dict(torch.load(model_file, map_location=device, weights_only=True))
    else:
        print(f"❌ Modelo no encontrado: {model_file}")
        return 0.0
    
    model.eval()
    test_preds = []
    
    with torch.no_grad():
        for batch in test_loader:
            tensor = batch['tensor'].to(device, non_blocking=True)
            output = model(tensor).squeeze()
            prob = torch.sigmoid(output)
            test_preds.append(safe_to_numpy(prob).item())
    
    auc = roc_auc_score(test_labels, test_preds)
    print(f"   Pacientes evaluados: {len(test_preds)}")
    print(f"   AUC obtenido: {auc:.4f}")
    
    return auc

def main():
    print("🚨 DIAGNÓSTICO URGENTE")
    print("🎯 Objetivo: Encontrar por qué bajó de 0.6440 a 0.5833")
    print("=" * 60)
    
    # 1. Verificar archivos
    model_files = check_model_files()
    
    # 2. Verificar splits
    splits_info = check_splits_consistency()
    
    # 3. Probar carga de modelos
    model_results = test_model_loading()
    
    # 4. Comparar métodos de evaluación
    auc1, auc2 = compare_evaluation_methods()
    
    # 5. Diagnóstico final
    print(f"\n🎯 DIAGNÓSTICO FINAL:")
    print("=" * 40)
    
    if abs(auc1 - 0.6440) < 0.01:
        print(f"✅ Método original reproduce 0.6440: {auc1:.4f}")
        print(f"❌ Problema está en el método TTA: {auc2:.4f}")
    elif abs(auc2 - 0.6440) < 0.01:
        print(f"❌ Problema en método original: {auc1:.4f}")
        print(f"✅ Método TTA está bien: {auc2:.4f}")
    else:
        print(f"❌ AMBOS MÉTODOS FALLAN:")
        print(f"   Original: {auc1:.4f}")
        print(f"   TTA: {auc2:.4f}")
        print(f"   Esperado: 0.6440")
        print(f"💡 Problema fundamental en splits o modelo")

if __name__ == "__main__":
    main()