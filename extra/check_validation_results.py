import os
import json
from pathlib import Path

def find_validation_results():
    """Buscar validation results de todos los experimentos"""
    
    print("🔍 BUSCANDO VALIDATION RESULTS")
    print("=" * 50)
    
    # Base path de nnUNet results
    base_path = Path(r"C:\Users\usuario\Documents\Mama_Mia\nnUNet_results")
    
    if not base_path.exists():
        print(f"❌ No existe: {base_path}")
        return
    
    # Buscar todos los datasets
    datasets = ['Dataset111', 'Dataset112', 'Dataset113']
    all_validation_results = {}
    
    for dataset in datasets:
        print(f"\n📁 Buscando en {dataset}:")
        
        # Posibles rutas
        possible_paths = [
            base_path / "nnUNet" / "3d_fullres" / dataset / "nnUNetTrainer__nnUNetPlans__3d_fullres" / "fold_0" / "validation",
            base_path / "nnUNet" / "3d_fullres" / dataset / "fold_0" / "validation",
            base_path / dataset / "nnUNetTrainer__nnUNetPlans__3d_fullres" / "fold_0" / "validation",
            base_path / dataset / "fold_0" / "validation"
        ]
        
        validation_found = False
        
        for val_path in possible_paths:
            if val_path.exists():
                print(f"   ✅ Encontrado: {val_path}")
                
                # Buscar archivos JSON con métricas
                json_files = list(val_path.glob("*.json"))
                
                if json_files:
                    print(f"   📄 JSON files encontrados: {len(json_files)}")
                    
                    # Intentar leer summary.json o similar
                    for json_file in json_files:
                        if 'summary' in json_file.name.lower():
                            try:
                                with open(json_file, 'r') as f:
                                    data = json.load(f)
                                
                                print(f"   📊 {json_file.name}:")
                                
                                # Buscar métricas Dice
                                if 'mean' in data:
                                    for key, value in data['mean'].items():
                                        if 'dice' in key.lower():
                                            print(f"      Dice: {value}")
                                            all_validation_results[dataset] = value
                                
                                elif 'results' in data:
                                    print(f"      Contiene 'results' - revisar manualmente")
                                
                                elif isinstance(data, dict):
                                    # Buscar cualquier clave que contenga 'dice'
                                    for key, value in data.items():
                                        if 'dice' in key.lower() and isinstance(value, (int, float)):
                                            print(f"      {key}: {value}")
                                            all_validation_results[dataset] = value
                                
                            except Exception as e:
                                print(f"   ❌ Error leyendo {json_file.name}: {e}")
                
                # Listar otros archivos disponibles
                other_files = [f for f in val_path.iterdir() if f.is_file() and not f.name.endswith('.json')]
                if other_files:
                    print(f"   📋 Otros archivos: {[f.name for f in other_files[:5]]}")
                
                validation_found = True
                break
        
        if not validation_found:
            print(f"   ❌ No se encontró validation para {dataset}")
    
    return all_validation_results

def find_training_logs():
    """Buscar logs de entrenamiento"""
    
    print(f"\n📋 BUSCANDO TRAINING LOGS")
    print("=" * 40)
    
    base_path = Path(r"C:\Users\usuario\Documents\Mama_Mia\nnUNet_results")
    
    # Buscar archivos de log
    log_files = []
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if 'training_log' in file or 'log' in file:
                log_files.append(Path(root) / file)
    
    print(f"📄 Log files encontrados: {len(log_files)}")
    
    for log_file in log_files[:5]:  # Mostrar primeros 5
        print(f"   📄 {log_file.name}")
        
        # Intentar leer últimas líneas si es texto
        try:
            if log_file.suffix in ['.txt', '.log']:
                with open(log_file, 'r') as f:
                    lines = f.readlines()
                    if lines:
                        print(f"      Últimas líneas: {lines[-2:] if len(lines) > 1 else lines}")
        except:
            pass

def compare_with_mama_mia(validation_results):
    """Comparar con baseline MAMA-MIA"""
    
    if not validation_results:
        print("\n❌ No se encontraron validation results")
        return
    
    print(f"\n🎯 COMPARACIÓN CON MAMA-MIA")
    print("=" * 40)
    
    mama_mia_baseline = 0.7620
    
    print(f"🏆 MAMA-MIA baseline: {mama_mia_baseline:.4f} (76.20%)")
    print(f"\n📊 TUS VALIDATION RESULTS:")
    
    for dataset, dice_score in validation_results.items():
        if isinstance(dice_score, (int, float)):
            gap = dice_score - mama_mia_baseline
            gap_percent = gap * 100
            
            if gap >= 0:
                status = "🎉 SUPERA"
            elif gap >= -0.01:
                status = "⚠️  MUY CERCA"
            else:
                status = "📈 NECESITA MEJORA"
            
            model_name = dataset.replace('Dataset11', 'A')  # Dataset111 -> A1
            
            print(f"   {status} {model_name}: {dice_score:.4f} ({dice_score*100:.2f}%)")
            print(f"      Gap: {gap:+.4f} ({gap_percent:+.2f}%)")

if __name__ == "__main__":
    # Buscar validation results
    validation_results = find_validation_results()
    
    # Buscar training logs
    find_training_logs()
    
    # Comparar con MAMA-MIA
    compare_with_mama_mia(validation_results)
    
    print(f"\n💡 NOTA:")
    print(f"Si no se encuentran métricas automáticamente,")
    print(f"puedes revisar manualmente los archivos JSON")
    print(f"en las carpetas validation/ mostradas arriba.")