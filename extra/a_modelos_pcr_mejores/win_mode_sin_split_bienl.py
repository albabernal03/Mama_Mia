# evaluate_best_model.py
"""
Re-entrenar y evaluar la MEJOR configuración encontrada:
resnet14t + 80³ + 3 canales (Val AUC 0.6356, +4.2% vs baseline)
"""

import torch
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from torch.utils.data import DataLoader
import json
from pathlib import Path
import time

# Importar desde tu archivo
from mejorar_resnet3d import *

def set_winner_config():
    """Configurar la MEJOR configuración encontrada"""
    print("🏆 CONFIGURANDO MEJOR SETUP ENCONTRADO")
    print("=" * 50)
    
    # LA CONFIGURACIÓN GANADORA: resnet14t + 80³ + 3 canales
    FastConfig.TARGET_SHAPE = (80, 80, 80)    # Resolución óptima
    FastConfig.INPUT_CHANNELS = 3              # CRÍTICO: 3 canales
    FastConfig.MODEL_TYPE = 'resnet14t'        # Modelo más grande
    FastConfig.BATCH_SIZE = 12                 # Ajustado para 80³
    FastConfig.EPOCHS = 50                     # Entrenamiento completo
    FastConfig.LEARNING_RATE = 8e-4            # LR optimizado
    FastConfig.WEIGHT_DECAY = 1e-4
    FastConfig.PATIENCE = 10                   # Más paciencia
    
    print(f"✅ Configuración GANADORA aplicada:")
    print(f"  📏 Resolución: {FastConfig.TARGET_SHAPE}")
    print(f"  🔢 Canales: {FastConfig.INPUT_CHANNELS}")
    print(f"  🏗️ Modelo: {FastConfig.MODEL_TYPE}")
    print(f"  📦 Batch size: {FastConfig.BATCH_SIZE}")
    print(f"  🎯 Objetivo: Superar Val AUC 0.6356")

def train_winner_model():
    """Re-entrenar la configuración ganadora"""
    print("\n🚀 RE-ENTRENANDO CONFIGURACIÓN GANADORA")
    print("=" * 50)
    
    set_winner_config()
    
    # Entrenar modelo
    start_time = time.time()
    trainer = FastTrainer()
    val_auc = trainer.train_fast_model()
    training_time = time.time() - start_time
    
    print(f"\n🎯 ENTRENAMIENTO COMPLETADO:")
    print(f"Val AUC obtenido: {val_auc:.4f}")
    print(f"Tiempo: {training_time/60:.1f} minutos")
    
    # Guardar con nombre específico
    winner_model_path = 'best_model_resnet14t_80cubed_3ch_WINNER.pth'
    # El modelo ya se guardó como 'best_fast_model.pth', copiarlo
    import shutil
    shutil.copy('best_fast_model.pth', winner_model_path)
    print(f"💾 Modelo ganador guardado: {winner_model_path}")
    
    return val_auc, winner_model_path

def evaluate_winner_on_test(model_path):
    """Evaluar modelo ganador en test"""
    print(f"\n🧪 EVALUANDO EN TEST: {model_path}")
    print("=" * 50)
    
    # Asegurar configuración correcta
    set_winner_config()
    
    # Crear datos de test
    with open(FastConfig.LABELS_FILE, 'r') as f:
        pcr_list = json.load(f)
    
    pcr_data = {item['patient_id']: item for item in pcr_list}
    patient_ids, labels = [], []
    
    for pid, info in pcr_data.items():
        if 'pcr' in info and info['pcr'] in ["0", "1"]:
            tensor_path = FastConfig.DATA_DIR / pid / f"{pid}_tensor_3ch.nii.gz"
            if tensor_path.exists():
                patient_ids.append(pid)
                labels.append(int(info['pcr']))
    
    # Split train/val/test
    train_val_ids, test_ids, train_val_labels, test_labels = train_test_split(
        patient_ids, labels, test_size=0.2, random_state=FastConfig.SEED, stratify=labels
    )
    
    print(f"📊 Test split: {len(test_ids)} pacientes ({np.mean(test_labels):.1%} pCR)")
    
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
    
    # Cargar weights
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
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
    
    # Métricas
    test_auc = roc_auc_score(test_targets, test_preds)
    test_acc = accuracy_score(test_targets, [1 if p > 0.5 else 0 for p in test_preds])
    
    # Comparación con baseline
    baseline_test = 0.5922
    baseline_val = 0.6102
    improvement = ((test_auc - baseline_test) / baseline_test) * 100
    
    print(f"\n🎯 RESULTADOS FINALES EN TEST:")
    print("=" * 60)
    print(f"🏆 CONFIGURACIÓN GANADORA:")
    print(f"   📏 Resolución: {FastConfig.TARGET_SHAPE}")
    print(f"   🔢 Canales: {FastConfig.INPUT_CHANNELS}")
    print(f"   🏗️ Modelo: {FastConfig.MODEL_TYPE}")
    print(f"")
    print(f"📊 RESULTADOS:")
    print(f"   🎲 Test AUC:      {test_auc:.4f}")
    print(f"   🎯 Test Accuracy: {test_acc:.4f}")
    print(f"   👥 Pacientes:     {len(test_targets)}")
    print(f"")
    print(f"📈 COMPARACIÓN:")
    print(f"   📊 Tu baseline (test): {baseline_test:.4f}")
    print(f"   🚀 Mejora vs baseline: {improvement:+.1f}%")
    
    if test_auc > baseline_test:
        print(f"\n🎉 ¡FELICITACIONES! SUPERASTE TU BASELINE EN TEST!")
        print(f"🏆 Este modelo está listo para el challenge!")
    else:
        gap = baseline_test - test_auc
        print(f"\n📈 Faltan {gap:.4f} puntos para superar baseline")
    
    # Guardar resultados
    results = {
        'test_auc': test_auc,
        'test_accuracy': test_acc,
        'baseline_test': baseline_test,
        'improvement_pct': improvement,
        'config': {
            'target_shape': list(FastConfig.TARGET_SHAPE),
            'input_channels': FastConfig.INPUT_CHANNELS,
            'model_type': FastConfig.MODEL_TYPE,
            'batch_size': FastConfig.BATCH_SIZE
        },
        'winner_model': True
    }
    
    with open('WINNER_test_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Resultados guardados: WINNER_test_results.json")
    
    return test_auc

def main():
    print("🏆 EVALUACIÓN DEL MODELO GANADOR")
    print("🎯 Configuración: resnet14t + 80³ + 3 canales")
    print("🎯 Val AUC objetivo: 0.6356 (+4.2% vs baseline)")
    print("🎯 Test objetivo: Superar 0.5922")
    print("=" * 60)
    
    # 1. Re-entrenar configuración ganadora
    print("PASO 1: Re-entrenar configuración ganadora")
    val_auc, model_path = train_winner_model()
    
    if val_auc < 0.62:
        print(f"⚠️ Val AUC ({val_auc:.4f}) menor a esperado. ¿Continuar con evaluación? (y/n)")
        response = input().lower()
        if response != 'y':
            print("❌ Evaluación cancelada")
            return
    
    # 2. Evaluar en test
    print(f"\nPASO 2: Evaluar en test")
    test_auc = evaluate_winner_on_test(model_path)
    
    # 3. Resultado final
    print(f"\n" + "="*60)
    print(f"🎯 RESULTADO FINAL:")
    print(f"   Val AUC:  {val_auc:.4f}")
    print(f"   Test AUC: {test_auc:.4f}")
    
    if test_auc > 0.5922:
        print(f"🎉 ¡ÉXITO! Modelo listo para submission")
    else:
        print(f"📈 Modelo promisorio, evaluar más optimizaciones")

if __name__ == "__main__":
    main()