#!/usr/bin/env python3
"""
Re-entrenar modelo multi-canal con configuracion corregida
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from multichannel.multichannel_model import MultiChannelDCEProcessor

def retrain_multichannel():
    print("RE-ENTRENANDO MULTI-CANAL")
    print("=" * 40)
    
    processor = MultiChannelDCEProcessor()
    
    print("Eliminando modelo anterior...")
    model_path = processor.paths['models'] / "multichannel_dce_model.pth"
    if model_path.exists():
        model_path.unlink()
        print("Modelo anterior eliminado")
    
    print("Iniciando entrenamiento...")
    model = processor.train_multichannel_model(epochs=20, batch_size=2)  # Entrenamiento rapido
    
    if model:
        print("Re-entrenamiento completado")
        
        # Test rapido
        print("Testeando prediccion...")
        test_case = "DUKE_019"
        result = processor.predict_with_multichannel(model, test_case)
        
        if result:
            pred, _, _ = result
            print(f"Prediccion exitosa: shape {pred.shape}")
            print(f"   Rango: [{pred.min():.3f}, {pred.max():.3f}]")
            print(f"   Voxeles > 0.5: {(pred > 0.5).sum()}")
        else:
            print("Prediccion fallo")
    else:
        print("Re-entrenamiento fallo")

if __name__ == "__main__":
    retrain_multichannel()