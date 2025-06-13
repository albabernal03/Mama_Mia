# corrected_split_loader.py
"""
CARGAR SPLITS OFICIALES - FORMATO CORRECTO

Formato CSV: train_split,test_split
DUKE_001,DUKE_019
...
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import json
from pathlib import Path
import time

# Importar código base
from mejorar_resnet3d import *

# Ruta del CSV oficial
OFFICIAL_SPLITS_CSV = Path(r"C:\Users\usuario\Documents\Mama_Mia\datos\train_test_splits.csv")

def load_and_inspect_csv():
    """Cargar e inspeccionar el CSV real"""
    print("🔍 INSPECCIONANDO CSV OFICIAL")
    print("=" * 50)
    
    if not OFFICIAL_SPLITS_CSV.exists():
        print(f"❌ Archivo no encontrado: {OFFICIAL_SPLITS_CSV}")
        return None
    
    # Cargar CSV
    df = pd.read_csv(OFFICIAL_SPLITS_CSV)
    print(f"📄 Archivo: {OFFICIAL_SPLITS_CSV}")
    print(f"📊 Filas: {len(df)}")
    print(f"📋 Columnas: {list(df.columns)}")
    print()
    
    # Mostrar primeras filas
    print("👀 Primeras 5 filas:")
    print(df.head())
    print()
    
    return df

def create_splits_from_csv():
    """Crear splits desde el CSV con formato train_split,test_split"""
    print("🎯 CREANDO SPLITS DESDE CSV OFICIAL")
    print("=" * 50)
    
    # Cargar CSV
    df = pd.read_csv(OFFICIAL_SPLITS_CSV)
    
    # Extraer todos los IDs únicos de cada columna
    train_ids_from_csv = df['train_split'].dropna().unique().tolist()
    test_ids_from_csv = df['test_split'].dropna().unique().tolist()
    
    print(f"📊 IDs en train_split: {len(train_ids_from_csv)}")
    print(f"📊 IDs en test_split: {len(test_ids_from_csv)}")
    
    # Verificar overlap
    overlap = set(train_ids_from_csv) & set(test_ids_from_csv)
    if overlap:
        print(f"⚠️ Overlap entre train y test: {len(overlap)} pacientes")
        print(f"Ejemplos: {list(overlap)[:5]}")
    else:
        print("✅ No hay overlap entre train y test")
    
    # Cargar labels
    with open(FastConfig.LABELS_FILE, 'r') as f:
        pcr_list = json.load(f)
    
    pcr_data = {item['patient_id']: item for item in pcr_list}
    
    def filter_valid_patients(patient_ids, split_name):
        """Filtrar pacientes válidos y obtener labels"""
        valid_ids, valid_labels = [], []
        missing_data, missing_files = 0, 0
        
        for pid in patient_ids:
            pid = str(pid)  # Asegurar que es string
            
            # Verificar si tiene label
            if pid not in pcr_data or 'pcr' not in pcr_data[pid]:
                missing_data += 1
                continue
                
            if pcr_data[pid]['pcr'] not in ["0", "1"]:
                missing_data += 1
                continue
            
            # Verificar si existe el archivo
            tensor_path = FastConfig.DATA_DIR / pid / f"{pid}_tensor_3ch.nii.gz"
            if not tensor_path.exists():
                missing_files += 1
                continue
            
            # Paciente válido
            valid_ids.append(pid)
            valid_labels.append(int(pcr_data[pid]['pcr']))
        
        print(f"📊 {split_name.upper()}:")
        print(f"  En CSV: {len(patient_ids)}")
        print(f"  Válidos: {len(valid_ids)}")
        print(f"  Sin datos: {missing_data}")
        print(f"  Sin archivos: {missing_files}")
        print(f"  pCR rate: {np.mean(valid_labels):.1%}")
        print()
        
        return valid_ids, valid_labels
    
    # Procesar train y test
    train_ids, train_labels = filter_valid_patients(train_ids_from_csv, 'train')
    test_ids, test_labels = filter_valid_patients(test_ids_from_csv, 'test')
    
    # Crear split de validación desde train (20% del train)
    # Ya que no hay val explícito en el CSV
    train_train_ids, val_ids, train_train_labels, val_labels = train_test_split(
        train_ids, train_labels, test_size=0.2, random_state=42, stratify=train_labels
    )
    
    print(f"📊 SPLIT DE VALIDACIÓN CREADO:")
    print(f"  Train final: {len(train_train_ids)} ({np.mean(train_train_labels):.1%} pCR)")
    print(f"  Val creado: {len(val_ids)} ({np.mean(val_labels):.1%} pCR)")
    print()
    
    splits = {
        'train': (train_train_ids, train_train_labels),
        'val': (val_ids, val_labels),
        'test': (test_ids, test_labels)
    }
    
    return splits

def train_with_corrected_splits():
    """Entrenar usando los splits correctos"""
    print("🚀 ENTRENAMIENTO CON SPLITS OFICIALES CORREGIDOS")
    print("🏆 Configuración: resnet14t + 80³ + 3 canales")
    print("=" * 60)
    
    # Configurar la mejor configuración
    FastConfig.TARGET_SHAPE = (80, 80, 80)
    FastConfig.INPUT_CHANNELS = 3
    FastConfig.MODEL_TYPE = 'resnet14t'
    FastConfig.BATCH_SIZE = 12
    FastConfig.EPOCHS = 50
    FastConfig.LEARNING_RATE = 8e-4
    FastConfig.WEIGHT_DECAY = 1e-4
    FastConfig.PATIENCE = 10
    
    print(f"✅ Configuración aplicada:")
    print(f"  📏 Resolución: {FastConfig.TARGET_SHAPE}")
    print(f"  🔢 Canales: {FastConfig.INPUT_CHANNELS}")
    print(f"  🏗️ Modelo: {FastConfig.MODEL_TYPE}")
    print()
    
    # Cargar splits
    splits = create_splits_from_csv()
    train_ids, train_labels = splits['train']
    val_ids, val_labels = splits['val']
    
    if len(train_ids) == 0 or len(val_ids) == 0:
        print("❌ Error: Splits vacíos")
        return None, None
    
    # Crear datasets
    train_dataset = FastDataset(
        FastConfig.DATA_DIR, train_ids, train_labels,
        augment=True, target_shape=FastConfig.TARGET_SHAPE,
        input_channels=FastConfig.INPUT_CHANNELS
    )
    val_dataset = FastDataset(
        FastConfig.DATA_DIR, val_ids, val_labels,
        augment=False, target_shape=FastConfig.TARGET_SHAPE,
        input_channels=FastConfig.INPUT_CHANNELS
    )
    
    # DataLoaders
    train_loader = DataLoader(
        train_dataset, batch_size=FastConfig.BATCH_SIZE,
        shuffle=True, num_workers=2, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=FastConfig.BATCH_SIZE,
        shuffle=False, num_workers=2, pin_memory=True
    )
    
    # Crear modelo y trainer
    trainer = FastTrainer()
    model = trainer.create_model()
    
    # Loss y optimizer
    device = trainer.device
    pos_weight = torch.tensor([2.39]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=FastConfig.LEARNING_RATE,
        weight_decay=FastConfig.WEIGHT_DECAY
    )
    
    scheduler = trainer.create_scheduler(optimizer, len(train_loader))
    
    # Training loop
    best_auc = 0
    patience_counter = 0
    start_time = time.time()
    
    print(f"🎯 INICIANDO ENTRENAMIENTO")
    print(f"📊 Train: {len(train_ids)} | Val: {len(val_ids)}")
    print()
    
    for epoch in range(FastConfig.EPOCHS):
        # Training
        model.train()
        train_losses, train_preds, train_targets = [], [], []
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{FastConfig.EPOCHS}")
        for batch in pbar:
            tensors = batch['tensor'].to(device, non_blocking=True)
            targets = batch['target'].to(device, non_blocking=True).float()
            
            optimizer.zero_grad()
            
            if trainer.use_amp:
                with torch.autocast(device_type='cuda', dtype=trainer.dtype):
                    outputs = model(tensors).squeeze()
                    loss = criterion(outputs, targets)
                
                trainer.scaler.scale(loss).backward()
                
                if FastConfig.USE_GRADIENT_CLIPPING:
                    trainer.scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), FastConfig.CLIP_VALUE)
                
                trainer.scaler.step(optimizer)
                trainer.scaler.update()
            else:
                outputs = model(tensors).squeeze()
                loss = criterion(outputs, targets)
                loss.backward()
                
                if FastConfig.USE_GRADIENT_CLIPPING:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), FastConfig.CLIP_VALUE)
                
                optimizer.step()
            
            if scheduler:
                scheduler.step()
            
            # Métricas
            train_losses.append(loss.item())
            probs = torch.sigmoid(outputs)
            train_preds.extend(safe_to_numpy(probs.detach()))
            train_targets.extend(safe_to_numpy(targets))
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{optimizer.param_groups[0]["lr"]:.2e}'
            })
        
        # Validation
        model.eval()
        val_preds, val_targets = [], []
        
        with torch.no_grad():
            for batch in val_loader:
                tensors = batch['tensor'].to(device, non_blocking=True)
                targets = batch['target'].to(device, non_blocking=True).float()
                
                if trainer.use_amp:
                    with torch.autocast(device_type='cuda', dtype=trainer.dtype):
                        outputs = model(tensors).squeeze()
                else:
                    outputs = model(tensors).squeeze()
                
                probs = torch.sigmoid(outputs)
                val_preds.extend(safe_to_numpy(probs))
                val_targets.extend(safe_to_numpy(targets))
        
        # Métricas
        train_auc = roc_auc_score(train_targets, train_preds)
        val_auc = roc_auc_score(val_targets, val_preds)
        val_acc = accuracy_score(val_targets, [1 if p > 0.5 else 0 for p in val_preds])
        
        print(f"Epoch {epoch+1}:")
        print(f"  Loss: {np.mean(train_losses):.4f}")
        print(f"  Train AUC: {train_auc:.4f}, Val AUC: {val_auc:.4f}")
        print(f"  Val Acc: {val_acc:.4f}")
        
        # Early stopping
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            print(f"  🎯 New best AUC: {val_auc:.4f}")
            torch.save(model.state_dict(), 'best_model_OFFICIAL_SPLITS_CORRECTED.pth')
        else:
            patience_counter += 1
            if patience_counter >= FastConfig.PATIENCE:
                print(f"  ⏹️ Early stopping at epoch {epoch+1}")
                break
    
    training_time = time.time() - start_time
    
    print(f"\n🎯 ENTRENAMIENTO COMPLETADO:")
    print(f"🏆 Best Val AUC: {best_auc:.4f}")
    print(f"⏱️ Tiempo: {training_time/60:.1f} minutos")
    print(f"💾 Modelo guardado: best_model_OFFICIAL_SPLITS_CORRECTED.pth")
    
    return best_auc, splits

def evaluate_on_official_test():
    """Evaluar en test oficial"""
    print(f"\n🧪 EVALUACIÓN EN TEST OFICIAL")
    print("=" * 50)
    
    # Cargar splits
    splits = create_splits_from_csv()
    test_ids, test_labels = splits['test']
    
    print(f"📊 Test oficial: {len(test_ids)} pacientes")
    print(f"📈 Test pCR rate: {np.mean(test_labels):.1%}")
    
    # Dataset de test
    test_dataset = FastDataset(
        FastConfig.DATA_DIR, test_ids, test_labels,
        augment=False,
        target_shape=FastConfig.TARGET_SHAPE,
        input_channels=FastConfig.INPUT_CHANNELS
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=FastConfig.BATCH_SIZE,
        shuffle=False, num_workers=2, pin_memory=True
    )
    
    # Cargar modelo
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = FastTimmModel(
        model_name=FastConfig.MODEL_TYPE,
        in_channels=FastConfig.INPUT_CHANNELS,
        num_classes=1
    ).to(device)
    
    model.load_state_dict(torch.load('best_model_OFFICIAL_SPLITS_CORRECTED.pth', map_location=device, weights_only=True))
    print(f"✅ Modelo cargado correctamente")
    
    # Evaluación
    model.eval()
    test_preds, test_targets = [], []
    
    trainer = FastTrainer()
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Test evaluation"):
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
    
    # Métricas finales
    test_auc = roc_auc_score(test_targets, test_preds)
    test_acc = accuracy_score(test_targets, [1 if p > 0.5 else 0 for p in test_preds])
    
    # Comparación con baseline
    baseline_test = 0.5922
    improvement = ((test_auc - baseline_test) / baseline_test) * 100
    
    print(f"\n🎯 RESULTADOS FINALES CON SPLITS OFICIALES:")
    print("=" * 60)
    print(f"🎲 Test AUC (OFICIAL):  {test_auc:.4f}")
    print(f"🎯 Test Accuracy:       {test_acc:.4f}")
    print(f"👥 Pacientes evaluados: {len(test_targets)}")
    print(f"")
    print(f"📊 COMPARACIÓN:")
    print(f"📈 Tu baseline (test):  {baseline_test:.4f}")
    print(f"🚀 Mejora vs baseline:  {improvement:+.1f}%")
    
    if test_auc > baseline_test:
        print(f"\n🎉 ¡FELICITACIONES! SUPERASTE TU BASELINE CON SPLITS OFICIALES!")
        print(f"🏆 Este modelo está listo para el challenge!")
    else:
        gap = baseline_test - test_auc
        print(f"\n📈 Faltan {gap:.4f} puntos para superar baseline")
    
    # Guardar resultados
    results = {
        'test_auc_official': test_auc,
        'test_accuracy': test_acc,
        'baseline_test': baseline_test,
        'improvement_pct': improvement,
        'splits_used': 'OFFICIAL_CORRECTED',
        'test_patients': len(test_targets)
    }
    
    with open('FINAL_OFFICIAL_RESULTS.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Resultados guardados: FINAL_OFFICIAL_RESULTS.json")
    
    return test_auc

def main():
    print("🎯 ENTRENAMIENTO CON SPLITS OFICIALES (FORMATO CORRECTO)")
    print("📄 CSV: train_split,test_split")
    print("🏆 Configuración: resnet14t + 80³ + 3 canales")
    print("=" * 70)
    
    # 1. Inspeccionar CSV
    df = load_and_inspect_csv()
    if df is None:
        return
    
    # 2. Entrenar
    print("\nPASO 1: Entrenar con splits oficiales")
    val_auc, splits = train_with_corrected_splits()
    
    if val_auc is None:
        print("❌ Error en entrenamiento")
        return
    
    # 3. Evaluar en test
    print("\nPASO 2: Evaluar en test oficial")
    test_auc = evaluate_on_official_test()
    
    # 4. Resultado final
    print(f"\n" + "="*70)
    print(f"🏆 RESULTADOS FINALES:")
    print(f"   📊 Val AUC:  {val_auc:.4f}")
    print(f"   🎲 Test AUC: {test_auc:.4f}")
    print(f"   📈 vs Baseline: {((test_auc - 0.5922) / 0.5922 * 100):+.1f}%")
    
    if test_auc > 0.5922:
        print(f"\n🎉 ¡ÉXITO! Modelo listo para el challenge")
    else:
        print(f"\n📈 Modelo promisorio, evaluar optimizaciones")

if __name__ == "__main__":
    main()