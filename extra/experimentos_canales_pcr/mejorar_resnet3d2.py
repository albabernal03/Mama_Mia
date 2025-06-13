# quick_improvements.py
"""
MEJORAS RÁPIDAS PARA SUPERAR BASELINE (0.6102)

Resultado actual: 0.5750 (-5.8% vs baseline)
Objetivo: Superar 0.6102 en la próxima ejecución
"""

import time
import json
from pathlib import Path
from mejorar_resnet3d import *

# ==========================================
# 🎯 CONFIGURACIONES MEJORADAS
# ==========================================

class ImprovedConfigs:
    """Configuraciones optimizadas basadas en resultado 0.5750"""
    
    # Configuración 1: Más conservadora (recuperar información perdida)
    CONSERVATIVE = {
        'TARGET_SHAPE': (80, 80, 80),    # Más resolución
        'INPUT_CHANNELS': 3,              # Todos los canales
        'MODEL_TYPE': 'resnet14t',        # Modelo más grande
        'BATCH_SIZE': 12,                 # Ajustado para más resolución
        'EPOCHS': 40,                     # Más entrenamiento
        'LEARNING_RATE': 8e-4,            # LR ligeramente menor
        'description': 'Configuración conservadora - recuperar información'
    }
    
    # Configuración 2: Balance optimizado
    BALANCED = {
        'TARGET_SHAPE': (72, 72, 72),    # Resolución intermedia
        'INPUT_CHANNELS': 3,              # Todos los canales
        'MODEL_TYPE': 'resnet10t',        # Mantener modelo ligero
        'BATCH_SIZE': 14,                 
        'EPOCHS': 50,                     # Más tiempo de entrenamiento
        'LEARNING_RATE': 1e-3,            # LR original
        'description': 'Balance velocidad/calidad'
    }
    
    # Configuración 3: Solo incrementar resolución
    HIGHER_RES = {
        'TARGET_SHAPE': (96, 96, 96),    # Resolución alta
        'INPUT_CHANNELS': 2,              # Mantener 2 canales
        'MODEL_TYPE': 'resnet10t',        # Mantener modelo
        'BATCH_SIZE': 8,                  # Reducir batch por memoria
        'EPOCHS': 35,                     
        'LEARNING_RATE': 1e-3,
        'description': 'Alta resolución con 2 canales'
    }
    
    # Configuración 4: Probar 3 canales con resolución actual
    THREE_CHANNELS = {
        'TARGET_SHAPE': (64, 64, 64),    # Mantener resolución actual
        'INPUT_CHANNELS': 3,              # CAMBIO CLAVE: 3 canales
        'MODEL_TYPE': 'resnet10t',        
        'BATCH_SIZE': 16,
        'EPOCHS': 45,
        'LEARNING_RATE': 1e-3,
        'description': '3 canales con resolución 64³'
    }

def apply_config(config_dict):
    """Aplicar configuración a FastConfig"""
    for key, value in config_dict.items():
        if key != 'description':
            setattr(FastConfig, key, value)
    print(f"✅ Configuración aplicada: {config_dict['description']}")

def quick_test_configs():
    """Probar configuraciones mejoradas rápidamente"""
    configs = [
        ('THREE_CHANNELS', ImprovedConfigs.THREE_CHANNELS),
        ('BALANCED', ImprovedConfigs.BALANCED), 
        ('CONSERVATIVE', ImprovedConfigs.CONSERVATIVE),
        ('HIGHER_RES', ImprovedConfigs.HIGHER_RES)
    ]
    
    results = []
    
    print("🚀 PRUEBAS RÁPIDAS PARA SUPERAR BASELINE")
    print("🎯 Objetivo: AUC > 0.6102")
    print("=" * 50)
    
    for config_name, config in configs:
        print(f"\n🧪 PROBANDO: {config['description']}")
        print(f"📊 Resolución: {config['TARGET_SHAPE']}")
        print(f"🔢 Canales: {config['INPUT_CHANNELS']}")
        print(f"🏗️ Modelo: {config['MODEL_TYPE']}")
        
        start_time = time.time()
        
        try:
            # Aplicar configuración
            apply_config(config)
            
            # Reducir épocas para prueba rápida
            FastConfig.EPOCHS = min(config['EPOCHS'], 25)
            
            # Entrenar
            trainer = FastTrainer()
            auc = trainer.train_fast_model()
            
            elapsed = time.time() - start_time
            improvement = ((auc - 0.5750) / 0.5750) * 100
            vs_baseline = ((auc - 0.6102) / 0.6102) * 100
            
            result = {
                'config': config_name,
                'description': config['description'],
                'auc': auc,
                'time_minutes': elapsed / 60,
                'improvement_vs_current': improvement,
                'vs_baseline': vs_baseline,
                'success': auc > 0.5750
            }
            
            results.append(result)
            
            print(f"✅ AUC: {auc:.4f}")
            print(f"📈 vs actual: {improvement:+.1f}%")
            print(f"🎯 vs baseline: {vs_baseline:+.1f}%")
            
            if auc > 0.6102:
                print("🎉 ¡SUPERÓ EL BASELINE!")
            
        except Exception as e:
            print(f"❌ Error: {e}")
            results.append({
                'config': config_name,
                'description': config['description'],
                'auc': 0.0,
                'success': False,
                'error': str(e)
            })
    
    # Análisis final
    successful_results = [r for r in results if r.get('success', False)]
    
    if successful_results:
        best = max(successful_results, key=lambda x: x['auc'])
        
        print(f"\n🏆 MEJOR CONFIGURACIÓN:")
        print(f"🎯 {best['description']}")
        print(f"📊 AUC: {best['auc']:.4f}")
        print(f"⏱️ Tiempo: {best['time_minutes']:.1f} min")
        
        if best['auc'] > 0.6102:
            print(f"🎉 ¡BASELINE SUPERADO! (+{best['vs_baseline']:.1f}%)")
        else:
            print(f"📈 Mejora vs actual: +{best['improvement_vs_current']:.1f}%")
            print(f"🎯 Faltan {0.6102 - best['auc']:.4f} para superar baseline")
    
    else:
        print("\n⚠️ Ninguna configuración mejoró el resultado")
        print("💡 Recomendación: Verificar datos y preprocessing")
    
    return results

# ==========================================
# 🔧 DIAGNÓSTICOS RÁPIDOS
# ==========================================

def diagnose_low_performance():
    """Diagnosticar posibles causas del AUC bajo"""
    print("🔍 DIAGNÓSTICO DE PERFORMANCE BAJA")
    print("=" * 40)
    
    # Verificar datos
    print("📂 Verificando datos...")
    try:
        with open(FastConfig.LABELS_FILE, 'r') as f:
            pcr_list = json.load(f)
        
        pcr_data = {item['patient_id']: item for item in pcr_list}
        valid_patients = 0
        pcr_positive = 0
        
        for pid, info in pcr_data.items():
            if 'pcr' in info and info['pcr'] in ["0", "1"]:
                tensor_path = FastConfig.DATA_DIR / pid / f"{pid}_tensor_3ch.nii.gz"
                if tensor_path.exists():
                    valid_patients += 1
                    if info['pcr'] == "1":
                        pcr_positive += 1
        
        pcr_rate = pcr_positive / valid_patients if valid_patients > 0 else 0
        
        print(f"✅ Pacientes válidos: {valid_patients}")
        print(f"📊 pCR rate: {pcr_rate:.1%}")
        
        if pcr_rate < 0.25 or pcr_rate > 0.35:
            print("⚠️ pCR rate inusual - verificar labels")
        
        if valid_patients < 1000:
            print("⚠️ Pocos pacientes válidos - verificar rutas")
            
    except Exception as e:
        print(f"❌ Error verificando datos: {e}")
    
    # Verificar configuración actual
    print(f"\n⚙️ Configuración actual:")
    print(f"📏 Resolución: {FastConfig.TARGET_SHAPE}")
    print(f"🔢 Canales: {FastConfig.INPUT_CHANNELS}")
    print(f"🏗️ Modelo: {FastConfig.MODEL_TYPE}")
    print(f"📦 Batch size: {FastConfig.BATCH_SIZE}")
    print(f"🎯 Épocas: {FastConfig.EPOCHS}")
    print(f"📈 Learning rate: {FastConfig.LEARNING_RATE}")
    
    # Recomendaciones
    print(f"\n💡 RECOMENDACIONES INMEDIATAS:")
    print(f"1. 🔢 Probar 3 canales en lugar de {FastConfig.INPUT_CHANNELS}")
    print(f"2. 📏 Aumentar resolución de {FastConfig.TARGET_SHAPE} a (80,80,80)")
    print(f"3. 🎯 Más épocas: {FastConfig.EPOCHS} → 50")
    print(f"4. 🏗️ Modelo más grande: {FastConfig.MODEL_TYPE} → resnet14t")

# ==========================================
# 🎯 FUNCIÓN PRINCIPAL PARA MEJORA RÁPIDA
# ==========================================

def improve_quickly():
    """Función principal para mejorar rápidamente el AUC"""
    print("🎯 MEJORA RÁPIDA DE PERFORMANCE")
    print("=" * 40)
    
    # 1. Diagnóstico
    diagnose_low_performance()
    
    print("\n" + "="*50)
    
    # 2. Probar configuraciones mejoradas
    results = quick_test_configs()
    
    # 3. Recomendación final
    print(f"\n🚀 PRÓXIMOS PASOS:")
    print(f"1. Ejecutar la mejor configuración con más épocas")
    print(f"2. Si no supera baseline, ejecutar experimentos completos:")
    print(f"   python run_experiments.py --experiment channels")
    print(f"3. Considerar aumentar resolución a 96³ o 128³")
    
    return results

if __name__ == "__main__":
    improve_quickly()