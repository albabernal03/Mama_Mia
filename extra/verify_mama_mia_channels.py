import os
import json
import nibabel as nib
import numpy as np

def verify_mama_mia_setup():
    """Verifica la configuración real de MAMA-MIA"""
    
    weights_dir = "./mama_mia_weights/extracted"
    
    print("🔍 VERIFICANDO CONFIGURACIÓN DE MAMA-MIA\n")
    
    # 1. Buscar y leer dataset.json
    dataset_json_path = None
    for root, dirs, files in os.walk(weights_dir):
        for file in files:
            if file == "dataset.json":
                dataset_json_path = os.path.join(root, file)
                break
        if dataset_json_path:
            break
    
    if dataset_json_path:
        print(f"📄 Dataset.json encontrado: {dataset_json_path}")
        with open(dataset_json_path, 'r') as f:
            dataset_config = json.load(f)
        
        print("📋 Configuración del dataset:")
        print(f"  🔀 Canales: {dataset_config.get('channel_names', 'No encontrado')}")
        print(f"  🏷️  Labels: {dataset_config.get('labels', 'No encontrado')}")
        print(f"  📊 Casos entrenamiento: {dataset_config.get('numTraining', 'No encontrado')}")
        print(f"  📁 Extensión archivos: {dataset_config.get('file_ending', 'No encontrado')}")
        
        # Analizar canales
        channel_names = dataset_config.get('channel_names', {})
        num_channels = len(channel_names)
        print(f"\n🧠 ANÁLISIS DE CANALES:")
        print(f"  📈 Número de canales: {num_channels}")
        
        if num_channels == 1:
            print("  ⚠️  DISCREPANCIA: Solo 1 canal, pero el paper menciona múltiples fases")
            print("  💡 Posibles explicaciones:")
            print("     1. Modelo simplificado (solo primer post-contraste)")
            print("     2. Data augmentation no reflejada en configuración")
            print("     3. Versión diferente del modelo")
        else:
            print(f"  ✅ Múltiples canales como esperado del paper")
            
    else:
        print("❌ No se encontró dataset.json")
    
    # 2. Buscar archivos de modelo
    print(f"\n🧠 BUSCANDO ARCHIVOS DE MODELO:")
    model_files = []
    for root, dirs, files in os.walk(weights_dir):
        for file in files:
            if any(ext in file.lower() for ext in ['.pth', '.pkl', '.ckpt', '.pt']):
                model_files.append(os.path.join(root, file))
    
    if model_files:
        print("📦 Archivos de modelo encontrados:")
        for model_file in model_files:
            rel_path = os.path.relpath(model_file, weights_dir)
            size = os.path.getsize(model_file) / (1024*1024)
            print(f"  🧠 {rel_path} ({size:.1f} MB)")
            
            # Intentar cargar y analizar el modelo
            try:
                if model_file.endswith('.pkl'):
                    import pickle
                    with open(model_file, 'rb') as f:
                        model_data = pickle.load(f)
                    
                    print(f"    📋 Tipo: {type(model_data)}")
                    if hasattr(model_data, 'keys'):
                        print(f"    🔑 Keys: {list(model_data.keys())[:5]}...")
                        
            except Exception as e:
                print(f"    ❌ Error cargando: {e}")
    else:
        print("❌ No se encontraron archivos de modelo")
    
    # 3. Buscar archivos de configuración adicionales
    print(f"\n⚙️  BUSCANDO CONFIGURACIONES ADICIONALES:")
    config_files = []
    for root, dirs, files in os.walk(weights_dir):
        for file in files:
            if any(ext in file.lower() for ext in ['.json', '.yaml', '.yml', '.txt']):
                config_files.append(os.path.join(root, file))
    
    if config_files:
        print("📄 Archivos de configuración encontrados:")
        for config_file in config_files[:10]:  # Mostrar solo los primeros 10
            rel_path = os.path.relpath(config_file, weights_dir)
            print(f"  📋 {rel_path}")
    
    # 4. Verificar estructura de directorios
    print(f"\n📂 ESTRUCTURA DE DIRECTORIOS:")
    def print_tree(startpath, max_depth=3, current_depth=0):
        if current_depth >= max_depth:
            return
        
        items = []
        try:
            for item in sorted(os.listdir(startpath)):
                if not item.startswith('.'):
                    items.append(item)
        except PermissionError:
            return
            
        for i, item in enumerate(items[:10]):  # Limitar a 10 items por directorio
            path = os.path.join(startpath, item)
            is_last = i == len(items) - 1
            prefix = "└── " if is_last else "├── "
            print("  " * current_depth + prefix + item)
            
            if os.path.isdir(path) and current_depth < max_depth - 1:
                print_tree(path, max_depth, current_depth + 1)
    
    if os.path.exists(weights_dir):
        print_tree(weights_dir)
    else:
        print("❌ Directorio de pesos no encontrado")
    
    print(f"\n🎯 RECOMENDACIONES:")
    print("1. 📖 Revisar documentación adicional en los archivos descargados")
    print("2. 🧪 Probar inferencia con 1 canal vs múltiples canales") 
    print("3. 📧 Contactar a los autores para clarificación")
    print("4. 🔍 Buscar otros modelos en Synapse con configuración multi-canal")

if __name__ == "__main__":
    verify_mama_mia_setup()