# tta_strategy.py
"""
TEST TIME AUGMENTATION (TTA) 

Aplicar múltiples augmentaciones durante inferencia y promediar predicciones
Mejora esperada: +1-3% AUC con muy poco esfuerzo
"""

import torch
import numpy as np
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm
from win_model import *

def apply_tta_transforms(tensor):
    """Aplicar diferentes transformaciones para TTA"""
    
    transforms = []
    
    # Original
    transforms.append(tensor.clone())
    
    # Flip horizontal
    transforms.append(torch.flip(tensor, [2]))  # H
    
    # Flip vertical  
    transforms.append(torch.flip(tensor, [3]))  # W
    
    # Flip sagital
    transforms.append(torch.flip(tensor, [1]))  # D
    
    # Flip horizontal + vertical
    transforms.append(torch.flip(tensor, [2, 3]))
    
    # Rotaciones 90°
    transforms.append(torch.rot90(tensor, 1, [2, 3]))
    transforms.append(torch.rot90(tensor, 2, [2, 3]))
    transforms.append(torch.rot90(tensor, 3, [2, 3]))
    
    return transforms

def evaluate_with_tta(model_path, tta_transforms=8):
    """Evaluar modelo usando TTA"""
    
    print(f"🔄 EVALUACIÓN CON TEST TIME AUGMENTATION")
    print(f"🎯 Transformaciones: {tta_transforms}")
    print("=" * 50)
    
    # Configurar modelo (usar la configuración ganadora)
    FastConfig.TARGET_SHAPE = (80, 80, 80)
    FastConfig.INPUT_CHANNELS = 3
    FastConfig.MODEL_TYPE = 'resnet14t'
    FastConfig.BATCH_SIZE = 12
    
    # Cargar test data
    from corrected_split_loader import create_splits_from_csv
    splits = create_splits_from_csv()
    test_ids, test_labels = splits['test']
    
    # Dataset de test
    test_dataset = FastDataset(
        FastConfig.DATA_DIR, test_ids, test_labels,
        augment=False,  # Sin augmentación en dataset
        target_shape=FastConfig.TARGET_SHAPE,
        input_channels=FastConfig.INPUT_CHANNELS
    )
    
    # Usar batch_size=1 para TTA individual
    test_loader = DataLoader(
        test_dataset, batch_size=1,
        shuffle=False, num_workers=2, pin_memory=True
    )
    
    # Cargar modelo
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = FastTimmModel(
        model_name=FastConfig.MODEL_TYPE,
        in_channels=FastConfig.INPUT_CHANNELS,
        num_classes=1
    ).to(device)
    
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    
    # Evaluación sin TTA (baseline)
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
    
    # Comparación
    improvement = ((tta_auc - simple_auc) / simple_auc) * 100
    
    print(f"\n🎯 RESULTADOS TTA:")
    print("=" * 40)
    print(f"📊 Sin TTA:     {simple_auc:.4f}")
    print(f"🔄 Con TTA:     {tta_auc:.4f}")
    print(f"📈 Mejora TTA:  {improvement:+.1f}%")
    
    # Comparar con baseline original
    baseline = 0.5922
    improvement_vs_baseline = ((tta_auc - baseline) / baseline) * 100
    print(f"🚀 vs Baseline: {improvement_vs_baseline:+.1f}%")
    
    if tta_auc > simple_auc:
        print(f"\n🎉 ¡TTA MEJORA EL RESULTADO!")
    
    # Guardar resultados
    results = {
        'simple_auc': simple_auc,
        'tta_auc': tta_auc,
        'tta_transforms': tta_transforms,
        'improvement_pct': improvement,
        'improvement_vs_baseline': improvement_vs_baseline
    }
    
    with open('TTA_RESULTS.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Guardar predicciones TTA
    np.save('tta_test_predictions.npy', tta_preds)
    
    print(f"\n💾 Resultados guardados:")
    print(f"  📄 TTA_RESULTS.json") 
    print(f"  📊 tta_test_predictions.npy")
    
    return tta_auc

def main():
    print("🔄 TEST TIME AUGMENTATION")
    print("🎯 Objetivo: Mejorar 0.6440 con transformaciones")
    print("=" * 50)
    
    # Usar el modelo ganador actual
    model_path = 'best_model_resnet14t_80cubed_3ch_WINNER.pth'
    
    if not Path(model_path).exists():
        print(f"❌ Modelo no encontrado: {model_path}")
        return
    
    tta_auc = evaluate_with_tta(model_path, tta_transforms=8)
    
    print(f"\n🏆 RESULTADO FINAL:")
    print(f"Modelo simple: 0.6440")
    print(f"Con TTA:       {tta_auc:.4f}")
    print(f"Mejora:        {((tta_auc - 0.6440) / 0.6440 * 100):+.1f}%")

if __name__ == "__main__":
    main()