import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import multiprocessing as mp
import torch
import json
import time
from datetime import datetime
from pathlib import Path

def run_experiment_on_gpu(gpu_id, experiment_type):
    """Ejecutar experimento individual en GPU específica"""
    # Configurar GPU específica
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    torch.cuda.set_device(0)  # Usar GPU 0 relativo después del filtro
    
    print(f"🚀 Iniciando {experiment_type} en GPU {gpu_id}")
    start_time = time.time()
    
    try:
        if experiment_type == 'postprocessing':
            from postprocessing.postprocessing_model import run_postprocessing_experiment
            result = run_postprocessing_experiment()
            
        elif experiment_type == 'multichannel':
            from multichannel.multichannel_model import run_multichannel_experiment
            result = run_multichannel_experiment()
            
        elif experiment_type == 'gan_augmentation':
            from gan_augmentation.gan_augmentation_model import run_gan_augmentation_experiment
            result = run_gan_augmentation_experiment()
            
        elif experiment_type == 'gan_refinement':
            from gan_refinement.gan_refinement_model import run_gan_refinement_experiment
            result = run_gan_refinement_experiment()
            
        else:
            print(f"❌ Experimento desconocido: {experiment_type}")
            return None
            
        end_time = time.time()
        duration = end_time - start_time
        
        if result:
            print(f"✅ {experiment_type} completado en GPU {gpu_id} ({duration:.1f}s)")
            print(f"📊 Mejora promedio: {result['statistics']['mean_improvement']:+.3f}")
            
            # Guardar resultado individual
            result_file = Path(f"./logs/{experiment_type}_gpu{gpu_id}_result.json")
            result_file.parent.mkdir(exist_ok=True)
            
            with open(result_file, 'w') as f:
                json.dump({
                    'experiment': experiment_type,
                    'gpu_id': gpu_id,
                    'duration': duration,
                    'result': result
                }, f, indent=2)
                
            return result
        else:
            print(f"❌ {experiment_type} falló en GPU {gpu_id}")
            return None
            
    except Exception as e:
        print(f"❌ Error en {experiment_type} (GPU {gpu_id}): {e}")
        return None

def launch_parallel_experiments():
    """Lanzar todos los experimentos en paralelo"""
    print("🔬 MAMA-MIA IMPROVEMENT FRAMEWORK")
    print("🎯 Objetivo: Baseline 0.762 → Improved 0.785+ (+3%)")
    print("🖥️  Hardware: 4x RTX A6000 en paralelo")
    print("=" * 60)
    
    experiments = [
        (0, 'postprocessing'),
        (1, 'multichannel'),
        (2, 'gan_augmentation'),
        (3, 'gan_refinement')
    ]
    
    print("🚀 Lanzando experimentos paralelos...")
    start_time = time.time()
    
    # Crear procesos para cada experimento
    processes = []
    for gpu_id, exp_type in experiments:
        print(f"   • {exp_type} → GPU {gpu_id}")
        p = mp.Process(target=run_experiment_on_gpu, args=(gpu_id, exp_type))
        p.start()
        processes.append((p, exp_type, gpu_id))
    
    print("\n⏳ Experimentos ejecutándose...")
    
    # Esperar a que terminen todos
    results = {}
    for p, exp_type, gpu_id in processes:
        p.join()
        
        # Cargar resultado del archivo
        result_file = Path(f"./logs/{exp_type}_gpu{gpu_id}_result.json")
        if result_file.exists():
            with open(result_file, 'r') as f:
                exp_data = json.load(f)
                results[exp_type] = exp_data['result']
        else:
            results[exp_type] = None
    
    total_time = time.time() - start_time
    
    # Análisis consolidado
    print("\n" + "=" * 60)
    print("📊 RESULTADOS CONSOLIDADOS")
    print("=" * 60)
    
    successful_experiments = []
    best_improvement = 0
    best_method = None
    
    for exp_type, result in results.items():
        if result:
            stats = result['statistics']
            improvement = stats['mean_improvement']
            successful_experiments.append(exp_type)
            
            print(f"\n🔬 {exp_type.upper()}:")
            print(f"   Casos: {stats['num_cases']}")
            print(f"   Baseline: {stats['baseline_mean']:.3f} ± {stats['baseline_std']:.3f}")
            print(f"   Mejorado: {stats['improved_mean']:.3f} ± {stats['improved_std']:.3f}")
            print(f"   Mejora: {improvement:+.3f} ± {stats['std_improvement']:.3f}")
            print(f"   Éxito: {stats['positive_improvements']}/{stats['num_cases']} casos ({100*stats['positive_improvements']/stats['num_cases']:.1f}%)")
            
            if improvement > best_improvement:
                best_improvement = improvement
                best_method = exp_type
                
            # Determinar estado del experimento
            if improvement > 0.01:
                print(f"   Estado: ✅ EXITOSO (+{improvement:.3f})")
            elif improvement > 0:
                print(f"   Estado: ⚠️  LEVE MEJORA (+{improvement:.3f})")
            else:
                print(f"   Estado: ❌ SIN MEJORA ({improvement:.3f})")
        else:
            print(f"\n🔬 {exp_type.upper()}: ❌ FALLÓ")
    
    # Resumen general
    print(f"\n{'='*60}")
    print("🎯 RESUMEN GENERAL:")
    print(f"Tiempo total: {total_time:.1f} segundos")
    print(f"Experimentos exitosos: {len(successful_experiments)}/4")
    
    if best_method:
        print(f"Mejor método: {best_method.upper()} (+{best_improvement:.3f})")
        
        if best_improvement > 0.02:
            print("🏆 ¡OBJETIVO ALCANZADO! Mejora significativa lograda")
        elif best_improvement > 0.01:
            print("✅ Mejora moderada alcanzada")
        else:
            print("⚠️  Mejora leve, considerar optimización")
    else:
        print("❌ Ningún experimento mejoró el baseline")
    
    # Recomendaciones para ensemble
    print(f"\n{'='*60}")
    print("🔧 RECOMENDACIONES PARA ENSEMBLE:")
    
    promising_methods = []
    for exp_type, result in results.items():
        if result and result['statistics']['mean_improvement'] > 0:
            promising_methods.append(exp_type)
    
    if len(promising_methods) >= 2:
        print(f"✅ Combinar métodos: {', '.join(promising_methods)}")
        print("📈 Ensemble esperado: +0.005 a +0.015 mejora adicional")
        
        # Crear ensemble automáticamente
        create_ensemble_recommendations(results, promising_methods)
    else:
        print("⚠️  Pocos métodos exitosos para ensemble")
        print("💡 Recomendación: Optimizar métodos individuales primero")
    
    # Guardar resumen completo
    save_consolidated_results(results, total_time, best_method, best_improvement)
    
    print(f"\n{'='*60}")
    print("🎉 ¡EXPERIMENTOS COMPLETADOS!")
    print("📄 Resultados guardados en ./logs/consolidated_results.json")
    print("📊 Revisar logs individuales para detalles específicos")
    print("🚀 Siguiente paso: Implementar ensemble de mejores métodos")
    print("="*60)

def create_ensemble_recommendations(results, promising_methods):
    """Crear recomendaciones para ensemble basado en resultados"""
    print("\n🎯 ESTRATEGIA DE ENSEMBLE:")
    
    # Analizar fortalezas de cada método
    method_strengths = {}
    
    for method in promising_methods:
        if results[method]:
            stats = results[method]['statistics']
            
            # Calcular métricas de calidad
            consistency = 1 - (stats['std_improvement'] / (abs(stats['mean_improvement']) + 0.001))
            success_rate = stats['positive_improvements'] / stats['num_cases']
            magnitude = stats['mean_improvement']
            
            method_strengths[method] = {
                'consistency': consistency,
                'success_rate': success_rate,
                'magnitude': magnitude,
                'score': consistency * success_rate * (1 + magnitude * 10)
            }
    
    # Ordenar por score
    sorted_methods = sorted(method_strengths.items(), key=lambda x: x[1]['score'], reverse=True)
    
    print("\n📊 RANKING DE MÉTODOS:")
    for i, (method, strengths) in enumerate(sorted_methods):
        print(f"{i+1}. {method.upper()}")
        print(f"   • Consistencia: {strengths['consistency']:.3f}")
        print(f"   • Tasa éxito: {strengths['success_rate']:.3f}")
        print(f"   • Magnitud: {strengths['magnitude']:+.3f}")
        print(f"   • Score: {strengths['score']:.3f}")
    
    # Estrategia de combinación
    print(f"\n🔧 ESTRATEGIA RECOMENDADA:")
    
    if len(sorted_methods) >= 3:
        print("🥇 Ensemble de 3 métodos (votación ponderada)")
        weights = [0.5, 0.3, 0.2]
        for i, (method, _) in enumerate(sorted_methods[:3]):
            print(f"   • {method}: peso {weights[i]}")
    elif len(sorted_methods) == 2:
        print("🥈 Ensemble de 2 métodos (promedio ponderado)")
        print(f"   • {sorted_methods[0][0]}: peso 0.7")
        print(f"   • {sorted_methods[1][0]}: peso 0.3")
    
    # Generar código de ensemble
    generate_ensemble_code(sorted_methods)

def generate_ensemble_code(sorted_methods):
    """Generar código para implementar ensemble"""
    
    ensemble_code = """
# CÓDIGO DE ENSEMBLE GENERADO AUTOMÁTICAMENTE
import numpy as np
import nibabel as nib
from pathlib import Path

def create_ensemble_prediction(case_id, method_weights):
    '''Crear predicción ensemble combinando múltiples métodos'''
    
    predictions = {}
    weights = {}
    
    # Cargar predicciones de cada método
"""
    
    for method, strengths in sorted_methods:
        ensemble_code += f"""
    # Cargar {method}
    {method}_file = Path(f"./results_{method}/{{case_id}}.nii.gz")
    if {method}_file.exists():
        predictions['{method}'] = nib.load({method}_file).get_fdata()
        weights['{method}'] = {strengths['score']:.3f}
"""
    
    ensemble_code += """
    
    if len(predictions) == 0:
        return None
        
    # Normalizar pesos
    total_weight = sum(weights.values())
    normalized_weights = {k: v/total_weight for k, v in weights.items()}
    
    # Crear ensemble
    ensemble_pred = np.zeros_like(list(predictions.values())[0])
    
    for method, pred in predictions.items():
        weight = normalized_weights[method]
        ensemble_pred += weight * pred
        
    return ensemble_pred

# Ejemplo de uso:
# ensemble_result = create_ensemble_prediction("case_001", method_weights)
"""
    
    # Guardar código de ensemble
    ensemble_file = Path("./ensemble_predictor.py")
    with open(ensemble_file, 'w') as f:
        f.write(ensemble_code)
        
    print(f"💾 Código de ensemble guardado: {ensemble_file}")

def save_consolidated_results(results, total_time, best_method, best_improvement):
    """Guardar resultados consolidados"""
    
    consolidado = {
        'experiment_info': {
            'timestamp': datetime.now().isoformat(),
            'total_duration': total_time,
            'hardware': '4x RTX A6000',
            'objective': 'Baseline 0.762 → Improved 0.785+ (+3%)'
        },
        'best_result': {
            'method': best_method,
            'improvement': best_improvement,
            'achieved_objective': best_improvement > 0.02
        },
        'individual_results': results,
        'summary': {
            'successful_experiments': len([r for r in results.values() if r is not None]),
            'total_experiments': len(results),
            'avg_improvement': np.mean([r['statistics']['mean_improvement'] 
                                     for r in results.values() if r is not None]) if any(results.values()) else 0
        }
    }
    
    # Guardar resultado consolidado
    results_file = Path("./logs/consolidated_results.json")
    results_file.parent.mkdir(exist_ok=True)
    
    with open(results_file, 'w') as f:
        json.dump(consolidado, f, indent=2)
    
    # Crear reporte markdown
    create_markdown_report(consolidado)

def create_markdown_report(consolidado):
    """Crear reporte en markdown"""
    
    report = f"""# MAMA-MIA Improvement Results

## Resumen Ejecutivo
- **Objetivo**: Baseline 0.762 → Improved 0.785+ (+3%)
- **Hardware**: 4x RTX A6000 en paralelo  
- **Duración**: {consolidado['experiment_info']['total_duration']:.1f} segundos
- **Mejor método**: {consolidado['best_result']['method'] or 'Ninguno'}
- **Mejor mejora**: {consolidado['best_result']['improvement']:+.3f}
- **Objetivo alcanzado**: {'✅ SÍ' if consolidado['best_result']['achieved_objective'] else '❌ NO'}

## Resultados por Experimento

"""
    
    for exp_name, result in consolidado['individual_results'].items():
        if result:
            stats = result['statistics']
            report += f"""### {exp_name.upper()}
- **Casos procesados**: {stats['num_cases']}
- **Baseline**: {stats['baseline_mean']:.3f} ± {stats['baseline_std']:.3f}
- **Mejorado**: {stats['improved_mean']:.3f} ± {stats['improved_std']:.3f}
- **Mejora promedio**: {stats['mean_improvement']:+.3f} ± {stats['std_improvement']:.3f}
- **Casos mejorados**: {stats['positive_improvements']}/{stats['num_cases']} ({100*stats['positive_improvements']/stats['num_cases']:.1f}%)

"""
        else:
            report += f"""### {exp_name.upper()}
❌ **EXPERIMENTO FALLÓ**

"""
    
    report += f"""## Análisis General
- **Experimentos exitosos**: {consolidado['summary']['successful_experiments']}/4
- **Mejora promedio global**: {consolidado['summary']['avg_improvement']:+.3f}

## Próximos Pasos
1. Implementar ensemble de mejores métodos
2. Optimizar hiperparámetros de métodos prometedores  
3. Evaluar en conjunto de validación independiente
4. Preparar publicación de resultados

---
*Reporte generado automáticamente el {consolidado['experiment_info']['timestamp']}*
"""
    
    # Guardar reporte
    report_file = Path("./logs/experiment_report.md")
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
        
    print(f"📄 Reporte guardado: {report_file}")

if __name__ == "__main__":
    # Configurar multiprocessing para Windows
    mp.set_start_method('spawn', force=True)
    
    print("🚀 Iniciando MAMA-MIA Improvement Framework...")
    print("⚠️  Asegúrate de que el baseline esté ejecutado en todos los casos de test")
    print("📂 Verifica que ./results_baseline/ contenga las segmentaciones base")
    
    input("📋 Presiona ENTER para continuar o Ctrl+C para cancelar...")
    
    launch_parallel_experiments()