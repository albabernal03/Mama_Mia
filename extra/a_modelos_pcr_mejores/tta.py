# corrected_tta_official_splits.py
"""
TTA CORREGIDO CON SPLITS OFICIALES

Objetivo: 0.5787 → 0.59-0.60+ usando Test Time Augmentation
"""

import torch
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
from pathlib import Path
import pandas as pd

from mejorar_resnet3d import *

def apply_tta_transforms(tensor):
    """Aplicar transformaciones médicas apropiadas para TTA"""
    
    transforms = []
    
    # Original
    transforms.append(tensor.clone())
    
    # Flips médicamente válidos
    transforms.append(torch.flip(tensor, [2]))  # Flip horizontal (left-right)
    transforms.append(torch.flip(tensor, [3]))  # Flip vertical (anterior-posterior)
    
    # Flip horizontal + vertical
    transforms.append(torch.flip(tensor, [2, 3]))
    
    # Rotaciones pequeñas (solo 90° que mantienen estructura)
    transforms.append(torch.rot90(tensor, 1, [2, 3]))
    transforms.append(torch.rot90(tensor, 2, [2, 3]))  # 180°
    transforms.append(torch.rot90(tensor, 3, [2, 3]))  # 270°
    
    # Flip sagital (depth) - cuidadoso para MRI temporal
    transforms.append(torch.flip(tensor, [1]))
    
    return transforms

def evaluate_with_tta_corrected(model_path, tta_transforms=8):
    """Evaluar modelo usando TTA con splits oficiales correctos"""
    
    print(f"🔄 TTA CON SPLITS OFICIALES CORRECTOS")
    print(f"🎯 Transformaciones: {tta_transforms}")
    print(f"📄 Modelo: {model_path}")
    print("=" * 60)
    
    # Configurar modelo (usar configuración del seed 42)
    FastConfig.TARGET_SHAPE = (80, 80, 80)
    FastConfig.INPUT_CHANNELS = 3
    FastConfig.MODEL_TYPE = 'resnet14t'
    FastConfig.BATCH_SIZE = 12
    
    # USAR SPLITS OFICIALES (no aleatorios)
    OFFICIAL_SPLITS_CSV = Path(r"C:\Users\usuario\Documents\Mama_Mia\datos\train_test_splits.csv")
    
    if not OFFICIAL_SPLITS_CSV.exists():
        print(f"❌ CSV oficial no encontrado: {OFFICIAL_SPLITS_CSV}")
        return None
    
    # Cargar splits oficiales
    df = pd.read_csv(OFFICIAL_SPLITS_CSV)
    test_ids_csv = df['test_split'].dropna().unique().tolist()
    
    # Cargar labels
    with open(FastConfig.LABELS_FILE, 'r') as f:
        pcr_list = json.load(f)
    
    pcr_data = {item['patient_id']: item for item in pcr_list}
    
    # Filtrar test IDs válidos
    test_ids, test_labels = [], []
    for pid in test_ids_csv:
        pid = str(pid)
        if pid in pcr_data and 'pcr' in pcr_data[pid]:
            if pcr_data[pid]['pcr'] in ["0", "1"]:
                tensor_path = FastConfig.DATA_DIR / pid / f"{pid}_tensor_3ch.nii.gz"
                if tensor_path.exists():
                    test_ids.append(pid)
                    test_labels.append(int(pcr_data[pid]['pcr']))
    
    print(f"📊 Test oficial: {len(test_ids)} pacientes ({np.mean(test_labels):.1%} pCR)")
    
    # Dataset de test con batch_size=1 para TTA individual
    test_dataset = FastDataset(
        FastConfig.DATA_DIR, test_ids, test_labels,
        augment=False,  # Sin augmentación en dataset
        target_shape=FastConfig.TARGET_SHAPE,
        input_channels=FastConfig.INPUT_CHANNELS
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=1,  # Importante: batch_size=1 para TTA
        shuffle=False, num_workers=2, pin_memory=True
    )
    
    # Cargar modelo
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = FastTimmModel(
        model_name=FastConfig.MODEL_TYPE,
        in_channels=FastConfig.INPUT_CHANNELS,
        num_classes=1
    ).to(device)
    
    # Verificar si existe el modelo
    if not Path(model_path).exists():
        print(f"❌ Modelo no encontrado: {model_path}")
        return None
    
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    
    print(f"✅ Modelo cargado: {model_path}")
    
    # Evaluación sin TTA (baseline del modelo)
    print("📊 Evaluando sin TTA...")
    simple_preds = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Sin TTA"):
            tensor = batch['tensor'].to(device, non_blocking=True)
            output = model(tensor).squeeze()
            prob = torch.sigmoid(output)
            simple_preds.append(safe_to_numpy(prob).item())
    
    simple_auc = roc_auc_score(test_labels, simple_preds)
    
    # Evaluación con TTA
    print(f"🔄 Evaluando con TTA ({tta_transforms} transformaciones)...")
    tta_preds = []
    
    trainer = FastTrainer()
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Con TTA"):
            tensor = batch['tensor'].to(device, non_blocking=True)
            
            # Aplicar múltiples transformaciones
            transforms = apply_tta_transforms(tensor.squeeze(0))  # Remove batch dim
            
            batch_predictions = []
            for i, transformed in enumerate(transforms[:tta_transforms]):
                # Add batch dimension back
                transformed_batch = transformed.unsqueeze(0)
                
                if trainer.use_amp:
                    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                        output = model(transformed_batch).squeeze()
                else:
                    output = model(transformed_batch).squeeze()
                
                prob = torch.sigmoid(output)
                batch_predictions.append(safe_to_numpy(prob).item())
            
            # Promedio de todas las transformaciones
            tta_pred = np.mean(batch_predictions)
            tta_preds.append(tta_pred)
    
    tta_auc = roc_auc_score(test_labels, tta_preds)
    
    # Comparación con baseline y resultado anterior
    baseline = 0.5922
    previous_result = 0.5787  # Resultado sin TTA
    
    improvement_tta = ((tta_auc - simple_auc) / simple_auc) * 100
    improvement_vs_baseline = ((tta_auc - baseline) / baseline) * 100
    improvement_vs_previous = ((tta_auc - previous_result) / previous_result) * 100
    
    print(f"\n🎯 RESULTADOS TTA CON SPLITS OFICIALES:")
    print("=" * 50)
    print(f"📊 Sin TTA:         {simple_auc:.4f}")
    print(f"🔄 Con TTA:         {tta_auc:.4f}")
    print(f"📈 Baseline:        {baseline:.4f}")
    print(f"📈 Resultado prev:  {previous_result:.4f}")
    print(f"")
    print(f"🚀 MEJORAS:")
    print(f"  TTA vs Sin TTA:    {improvement_tta:+.1f}%")
    print(f"  TTA vs Baseline:   {improvement_vs_baseline:+.1f}%")
    print(f"  TTA vs Anterior:   {improvement_vs_previous:+.1f}%")
    
    # Evaluación del éxito
    if tta_auc > baseline:
        print(f"\n🎉 ¡SUPERASTE EL BASELINE!")
        print(f"✅ Modelo listo para el challenge")
    elif tta_auc > baseline - 0.005:
        print(f"\n🎯 ¡MUY CERCA DEL BASELINE!")
        print(f"📈 Solo faltan {baseline - tta_auc:.4f} puntos")
    else:
        print(f"\n📈 Mejora pero aún por debajo de baseline")
        print(f"💡 Necesita estrategias adicionales")
    
    if tta_auc > simple_auc:
        print(f"✅ TTA mejora el resultado base")
    
    # Calcular métricas adicionales
    tta_acc = accuracy_score(test_labels, [1 if p > 0.5 else 0 for p in tta_preds])
    
    # Guardar resultados detallados
    results = {
        'simple_auc': simple_auc,
        'tta_auc': tta_auc,
        'tta_accuracy': tta_acc,
        'tta_transforms': tta_transforms,
        'baseline': baseline,
        'previous_result': previous_result,
        'improvement_tta_pct': improvement_tta,
        'improvement_vs_baseline_pct': improvement_vs_baseline,
        'improvement_vs_previous_pct': improvement_vs_previous,
        'beats_baseline': tta_auc > baseline,
        'test_patients': len(test_labels),
        'test_pcr_rate': np.mean(test_labels),
        'splits_used': 'OFFICIAL_CSV'
    }
    
    with open('TTA_OFFICIAL_RESULTS.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Guardar predicciones TTA para ensemble posterior
    np.save('tta_official_test_predictions.npy', tta_preds)
    
    print(f"\n💾 Resultados guardados:")
    print(f"  📄 TTA_OFFICIAL_RESULTS.json")
    print(f"  📊 tta_official_test_predictions.npy")
    
    return tta_auc

def ensemble_multiple_tta_configs():
    """Probar múltiples configuraciones de TTA y hacer ensemble"""
    
    print("\n🎯 ENSEMBLE DE MÚLTIPLES CONFIGURACIONES TTA")
    print("=" * 50)
    
    model_path = 'seed_42_auc0.6424.pth'
    
    if not Path(model_path).exists():
        print(f"❌ Modelo no encontrado: {model_path}")
        return None
    
    tta_configs = [
        {'transforms': 4, 'desc': 'Conservador (4 transforms)'},
        {'transforms': 6, 'desc': 'Moderado (6 transforms)'},
        {'transforms': 8, 'desc': 'Completo (8 transforms)'},
    ]
    
    all_predictions = []
    all_aucs = []
    
    for config in tta_configs:
        print(f"\n🔄 Probando: {config['desc']}")
        auc = evaluate_with_tta_corrected(model_path, config['transforms'])
        
        if auc:
            all_aucs.append(auc)
            # Cargar predicciones guardadas
            preds = np.load('tta_official_test_predictions.npy')
            all_predictions.append(preds)
    
    if len(all_predictions) > 1:
        # Ensemble de diferentes configuraciones TTA
        ensemble_preds = np.mean(all_predictions, axis=0)
        
        # Cargar labels de test
        OFFICIAL_SPLITS_CSV = Path(r"C:\Users\usuario\Documents\Mama_Mia\datos\train_test_splits.csv")
        df = pd.read_csv(OFFICIAL_SPLITS_CSV)
        test_ids_csv = df['test_split'].dropna().unique().tolist()
        
        with open(FastConfig.LABELS_FILE, 'r') as f:
            pcr_list = json.load(f)
        
        pcr_data = {item['patient_id']: item for item in pcr_list}
        
        test_labels = []
        for pid in test_ids_csv:
            pid = str(pid)
            if pid in pcr_data and 'pcr' in pcr_data[pid]:
                if pcr_data[pid]['pcr'] in ["0", "1"]:
                    tensor_path = FastConfig.DATA_DIR / pid / f"{pid}_tensor_3ch.nii.gz"
                    if tensor_path.exists():
                        test_labels.append(int(pcr_data[pid]['pcr']))
        
        ensemble_auc = roc_auc_score(test_labels, ensemble_preds)
        best_individual = max(all_aucs)
        
        print(f"\n🏆 ENSEMBLE TTA MULTI-CONFIG:")
        print("=" * 40)
        print(f"🎲 Ensemble AUC:    {ensemble_auc:.4f}")
        print(f"🥇 Mejor individual: {best_individual:.4f}")
        print(f"📈 Baseline:        0.5922")
        
        improvement = ((ensemble_auc - 0.5922) / 0.5922) * 100
        print(f"🚀 vs Baseline:     {improvement:+.1f}%")
        
        if ensemble_auc > best_individual:
            print(f"🎉 ¡ENSEMBLE MEJORA RESULTADO INDIVIDUAL!")
        
        # Guardar ensemble final
        np.save('tta_ensemble_final_predictions.npy', ensemble_preds)
        
        return ensemble_auc
    
    return max(all_aucs) if all_aucs else None

def main():
    print("🔄 TTA CORREGIDO CON SPLITS OFICIALES")
    print("🎯 Objetivo: 0.5787 → 0.59-0.60+")
    print("=" * 60)
    
    model_path = 'seed_42_auc0.6424.pth'
    
    # Verificar que existe el modelo
    if not Path(model_path).exists():
        print(f"❌ Modelo no encontrado: {model_path}")
        print(f"💡 Modelos disponibles:")
        for f in Path('.').glob('*.pth'):
            print(f"  📄 {f.name}")
        return
    
    # 1. TTA estándar
    tta_auc = evaluate_with_tta_corrected(model_path, tta_transforms=8)
    
    if tta_auc:
        print(f"\n📊 TTA ESTÁNDAR COMPLETADO: {tta_auc:.4f}")
        
        # 2. Si el resultado es prometedor, probar ensemble multi-config
        if tta_auc > 0.5800:
            print(f"\n🎯 Resultado prometedor, probando ensemble multi-config...")
            ensemble_auc = ensemble_multiple_tta_configs()
            
            if ensemble_auc:
                print(f"\n🏆 RESULTADO FINAL: {ensemble_auc:.4f}")
        
        # 3. Recomendación final
        final_auc = tta_auc
        baseline = 0.5922
        
        print(f"\n🎯 RECOMENDACIÓN FINAL:")
        print("=" * 30)
        
        if final_auc > baseline:
            print(f"🎉 ¡ÉXITO! AUC {final_auc:.4f} supera baseline")
            print(f"✅ Modelo listo para submission al challenge")
        elif final_auc > baseline - 0.01:
            print(f"🎯 MUY CERCA: {final_auc:.4f} vs {baseline:.4f}")
            print(f"💡 Probar ensemble de seeds o más optimizaciones")
        else:
            print(f"📈 Mejora pero necesita más trabajo")
            print(f"💡 Considerar ensemble de múltiples modelos")

if __name__ == "__main__":
    main()