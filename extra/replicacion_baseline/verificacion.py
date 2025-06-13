import os
import shutil

# Configurar estructura nnUNetv2
base_path = r"C:\Users\usuario\Documents\Mama_Mia"
weights_source = r"C:\Users\usuario\Documents\Mama_Mia\mama_mia_weights\extracted\Dataset501_full_image_dce_mri_tumor_segmentation"

# Crear estructura nnUNetv2
nnunet_results = os.path.join(base_path, "nnUNet_results")
dataset_folder = os.path.join(nnunet_results, "Dataset501_MAMA_MIA", "nnUNetTrainer__nnUNetPlans__3d_fullres")

print("CONFIGURANDO ESTRUCTURA nnUNetv2")
print("=" * 40)

# Crear carpetas
os.makedirs(dataset_folder, exist_ok=True)

print(f"Creando estructura en: {dataset_folder}")

# Copiar archivos de configuración
config_files = ['plans.json', 'dataset.json', 'dataset_fingerprint.json']

for config_file in config_files:
    source_file = os.path.join(weights_source, config_file)
    dest_file = os.path.join(dataset_folder, config_file)
    
    if os.path.exists(source_file):
        shutil.copy2(source_file, dest_file)
        print(f"✅ Copiado: {config_file}")
    else:
        print(f"❌ No encontrado: {config_file}")

# Copiar folds
for fold_num in range(5):
    fold_name = f"fold_{fold_num}"
    source_fold = os.path.join(weights_source, fold_name)
    dest_fold = os.path.join(dataset_folder, fold_name)
    
    if os.path.exists(source_fold):
        # Crear carpeta destino
        os.makedirs(dest_fold, exist_ok=True)
        
        # Copiar archivos del fold
        for file in os.listdir(source_fold):
            source_file = os.path.join(source_fold, file)
            dest_file = os.path.join(dest_fold, file)
            
            if os.path.isfile(source_file):
                shutil.copy2(source_file, dest_file)
        
        print(f"✅ Copiado: {fold_name}")
    else:
        print(f"❌ No encontrado: {fold_name}")

print("\n✅ Estructura nnUNetv2 configurada")

# Configurar variables de entorno
os.environ['nnUNet_raw'] = os.path.join(base_path, "nnUNet_raw")
os.environ['nnUNet_preprocessed'] = os.path.join(base_path, "nnUNet_preprocessed")
os.environ['nnUNet_results'] = nnunet_results

print("\n🚀 AHORA EJECUTA:")
print("nnUNetv2_predict -i .\\temp_input -o .\\results_output -d 501 -c 3d_fullres -f 0 1 2 3 4")

# Verificar estructura final
print("\n📁 Estructura creada:")
if os.path.exists(dataset_folder):
    items = os.listdir(dataset_folder)
    for item in items:
        item_path = os.path.join(dataset_folder, item)
        if os.path.isdir(item_path):
            print(f"   📁 {item}/")
        else:
            print(f"   📄 {item}")

print(f"\n📍 Ruta completa: {dataset_folder}")