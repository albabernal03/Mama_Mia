import SimpleITK as sitk
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import pandas as pd

def dice_coefficient(y_true, y_pred):
    """Calcular coeficiente Dice"""
    intersection = np.sum(y_true * y_pred)
    total = np.sum(y_true) + np.sum(y_pred)
    if total == 0:
        return 1.0 if np.sum(y_pred) == 0 else 0.0
    return (2.0 * intersection) / total

def find_correct_gt_folder(model_name):
    """Usar SIEMPRE Dataset111 para comparación justa (mismo test set)"""
    
    # TODOS usan el mismo test set para comparación justa
    base_path = Path(r'C:\Users\usuario\Documents\Mama_Mia\nnUNet_raw')
    gt_folder = base_path / 'Dataset111' / 'labelsTs'
    
    if gt_folder.exists():
        print(f"   📁 GT folder para {model_name}: Dataset111/labelsTs (test set común)")
        return gt_folder
    else:
        print(f"   ❌ No encontrado GT: {gt_folder}")
        return None

def evaluate_single_model(model_name, pred_folder_name):
    """Evaluar un modelo específico con su GT correspondiente"""
    
    pred_folder = Path(pred_folder_name)
    
    if not pred_folder.exists():
        print(f"❌ No existe: {pred_folder}")
        return None
    
    # Encontrar GT folder correcto para este modelo
    gt_folder = find_correct_gt_folder(model_name)
    if not gt_folder:
        return None
    
    print(f'🔍 EVALUANDO {model_name}')
    print('=' * 50)
    
    dice_scores = []
    pred_files = sorted([f for f in pred_folder.glob('*.nii.gz')])
    
    if not pred_files:
        print(f"❌ No hay archivos .nii.gz en {pred_folder}")
        return None
    
    for i, pred_file in enumerate(pred_files):
        gt_file = gt_folder / pred_file.name
        
        if gt_file.exists():
            try:
                pred_img = sitk.ReadImage(str(pred_file))
                pred_array = sitk.GetArrayFromImage(pred_img)
                
                gt_img = sitk.ReadImage(str(gt_file))
                gt_array = sitk.GetArrayFromImage(gt_img)
                
                dice = dice_coefficient(gt_array, pred_array)
                dice_scores.append(dice)
                
                # Mostrar progreso
                if i < 3 or (i + 1) % 100 == 0 or i == len(pred_files) - 1:
                    print(f'[{i+1:3d}/{len(pred_files)}] {pred_file.name}: {dice:.4f}')
                    
            except Exception as e:
                print(f'❌ Error en {pred_file.name}: {e}')
    
    if dice_scores:
        mean_dice = np.mean(dice_scores)
        std_dice = np.std(dice_scores)
        median_dice = np.median(dice_scores)
        min_dice = np.min(dice_scores)
        max_dice = np.max(dice_scores)
        
        print(f'📊 RESULTADOS {model_name}:')
        print(f'   Dice promedio: {mean_dice:.4f} ({mean_dice*100:.2f}%)')
        print(f'   Dice std: ±{std_dice:.4f}')
        print(f'   Dice mediana: {median_dice:.4f}')
        print(f'   Dice min/max: {min_dice:.4f} / {max_dice:.4f}')
        print(f'   Casos procesados: {len(dice_scores)}/{len(pred_files)}')
        print()
        
        return {
            'model_name': model_name,
            'dataset_trained_on': model_name,  # A1, A2, A3 (qué dataset usó para entrenar)
            'test_set_used': 'Dataset111',  # Todos evaluados con mismo test set
            'dice_mean': float(mean_dice),
            'dice_std': float(std_dice),
            'dice_median': float(median_dice),
            'dice_min': float(min_dice),
            'dice_max': float(max_dice),
            'dice_percentage': float(mean_dice * 100),
            'cases_processed': f"{len(dice_scores)}/{len(pred_files)}",
            'total_cases': len(pred_files),
            'successful_cases': len(dice_scores),
            'dice_scores': dice_scores  # Lista completa para análisis
        }
    else:
        print(f'❌ No se pudieron procesar casos para {model_name}')
        return None

def find_all_experiment_folders():
    """Buscar todas las carpetas de experimentos"""
    
    # RUTA CORREGIDA: donde están realmente los experimentos
    base_path = Path(r"C:\Users\usuario\Documents\Mama_Mia\experimentos")
    experiment_folders = []
    
    if not base_path.exists():
        print(f"❌ No existe la carpeta: {base_path}")
        return []
    
    print(f"🔍 BUSCANDO EN: {base_path}")
    
    # Buscar carpetas que contengan archivos .nii.gz
    for folder in base_path.iterdir():
        if folder.is_dir() and any(folder.glob("*.nii.gz")):
            experiment_folders.append(folder)
            print(f"   📁 Encontrada: {folder.name}")
    
    # También buscar por nombres específicos
    specific_names = ["A1_test_predictions", "A2_test_predictions", "A3_test_predictions", "A4_test_predictions"]
    
    for name in specific_names:
        folder = base_path / name
        if folder.exists() and folder.is_dir():
            if folder not in experiment_folders:
                experiment_folders.append(folder)
                print(f"   📁 Encontrada específica: {folder.name}")
    
    return experiment_folders

def evaluate_all_experiments():
    """Evaluar TODOS los experimentos encontrados"""
    
    print("🚀 EVALUANDO TODOS LOS EXPERIMENTOS")
    print("=" * 60)
    
    # Buscar experimentos
    experiment_folders = find_all_experiment_folders()
    
    if not experiment_folders:
        print("❌ No se encontraron carpetas de experimentos")
        return
    
    print(f"\n✅ Se encontraron {len(experiment_folders)} experimentos")
    
    all_results = {}
    
    # Evaluar cada experimento
    for folder in experiment_folders:
        model_name = folder.name.replace("_test_predictions", "").replace("_predictions", "")
        
        result = evaluate_single_model(model_name, str(folder))
        
        if result:
            all_results[model_name] = result
        
        print("-" * 60)
    
    return all_results

def save_comprehensive_results(all_results):
    """Guardar resultados completos de todos los experimentos"""
    
    if not all_results:
        print("❌ No hay resultados para guardar")
        return
    
    # Crear carpeta de resultados
    results_folder = Path("C:/experiment_results_complete")
    results_folder.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print(f"\n💾 GUARDANDO RESULTADOS COMPLETOS")
    print("=" * 50)
    
    # 1. GUARDAR RESULTADOS DETALLADOS (JSON)
    detailed_results = {
        "experiment_date": timestamp,
        "mama_mia_baseline": 0.7620,
        "total_models_evaluated": len(all_results),
        "models": all_results
    }
    
    json_file = results_folder / f"all_experiments_{timestamp}.json"
    
    # Guardar sin las listas completas de scores (muy grandes)
    json_data = {}
    for key, value in detailed_results.items():
        if key == "models":
            json_data[key] = {}
            for model_name, model_data in value.items():
                json_data[key][model_name] = {k: v for k, v in model_data.items() if k != 'dice_scores'}
        else:
            json_data[key] = value
    
    with open(json_file, 'w') as f:
        json.dump(json_data, f, indent=2)
    
    print(f"✅ JSON guardado: {json_file}")
    
    # 2. CREAR TABLA COMPARATIVA (CSV)
    comparison_data = []
    
    for model_name, results in all_results.items():
        comparison_data.append({
            'Model': model_name,
            'Trained_On': results.get('dataset_trained_on', model_name),
            'Evaluated_On': 'Dataset111 (común)',
            'Dice_Mean': results['dice_mean'],
            'Dice_Std': results['dice_std'], 
            'Dice_Percentage': results['dice_percentage'],
            'Cases_Processed': results['successful_cases'],
            'Total_Cases': results['total_cases'],
            'Gap_vs_Baseline': results['dice_mean'] - 0.7620,
            'Gap_Percentage': (results['dice_mean'] - 0.7620) * 100
        })
    
    df = pd.DataFrame(comparison_data)
    df = df.sort_values('Dice_Mean', ascending=False)
    
    csv_file = results_folder / f"model_comparison_{timestamp}.csv"
    df.to_csv(csv_file, index=False)
    
    print(f"✅ CSV guardado: {csv_file}")
    
    # 3. CREAR RESUMEN EJECUTIVO
    summary_file = results_folder / f"RESUMEN_TODOS_EXPERIMENTOS_{timestamp}.txt"
    
    best_model = max(all_results.items(), key=lambda x: x[1]['dice_mean'])
    worst_model = min(all_results.items(), key=lambda x: x[1]['dice_mean'])
    
    summary_content = f"""
🎯 RESUMEN COMPLETO - EXPERIMENTOS MAMA-MIA
Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*60}

📊 OBJETIVO: Superar MAMA-MIA baseline de 0.7620 (76.20%)

🏆 RESULTADOS GENERALES:
   Total modelos evaluados: {len(all_results)}
   
🥇 MEJOR MODELO: {best_model[0]}
   Dice: {best_model[1]['dice_mean']:.4f} ({best_model[1]['dice_percentage']:.2f}%)
   Gap vs baseline: {best_model[1]['dice_mean'] - 0.7620:+.4f} ({(best_model[1]['dice_mean'] - 0.7620)*100:+.2f}%)
   
📊 TODOS LOS MODELOS:
"""
    
    for model_name, results in sorted(all_results.items(), key=lambda x: x[1]['dice_mean'], reverse=True):
        gap = results['dice_mean'] - 0.7620
        gap_symbol = "✅" if gap >= 0 else "❌"
        dataset_used = results.get('dataset_used', 'Unknown')
        summary_content += f"""
   {gap_symbol} {model_name} (Dataset: {dataset_used}):
      Dice: {results['dice_mean']:.4f} ± {results['dice_std']:.4f} ({results['dice_percentage']:.2f}%)
      Gap: {gap:+.4f} ({gap*100:+.2f}%)
      Casos: {results['cases_processed']}
"""
    
    summary_content += f"""

🎯 ANÁLISIS:
   - Modelos que superan baseline: {sum(1 for r in all_results.values() if r['dice_mean'] > 0.7620)}
   - Modelos cercanos (±1%): {sum(1 for r in all_results.values() if abs(r['dice_mean'] - 0.7620) <= 0.01)}
   - Mejor mejora: {max((r['dice_mean'] - 0.7620)*100 for r in all_results.values()):.2f}%

💾 ARCHIVOS GENERADOS:
   📄 {json_file.name} - Resultados detallados
   📄 {csv_file.name} - Tabla comparativa  
   📄 {summary_file.name} - Este resumen
   
📍 Ubicación: {results_folder}
"""
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write(summary_content)
    
    print(f"✅ Resumen guardado: {summary_file}")
    
    # 4. MOSTRAR RESUMEN EN PANTALLA
    print(f"\n🏆 RANKING DE MODELOS:")
    print("-" * 50)
    
    for i, (model_name, results) in enumerate(sorted(all_results.items(), key=lambda x: x[1]['dice_mean'], reverse=True), 1):
        gap = results['dice_mean'] - 0.7620
        gap_symbol = "🥇" if i == 1 else "✅" if gap >= 0 else "❌"
        print(f"{i}. {gap_symbol} {model_name}: {results['dice_percentage']:.2f}% (gap: {gap*100:+.2f}%)")
    
    print(f"\n📁 TODOS LOS RESULTADOS EN: {results_folder}")
    
    return results_folder

if __name__ == "__main__":
    # Ejecutar evaluación completa
    all_results = evaluate_all_experiments()
    
    if all_results:
        results_folder = save_comprehensive_results(all_results)
        print(f"\n🎉 ¡EVALUACIÓN COMPLETA GUARDADA!")
        print(f"📂 Ubicación: {results_folder}")
    else:
        print("❌ No se pudieron evaluar experimentos")