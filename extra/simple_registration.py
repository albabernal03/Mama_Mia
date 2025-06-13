import SimpleITK as sitk
from pathlib import Path
from tqdm import tqdm
import shutil
import pandas as pd

def register_image_pair(fixed_path, moving_path, output_path):
    """Registra moving a fixed usando SimpleITK"""
    try:
        # Cargar imágenes y convertir a float32
        fixed = sitk.Cast(sitk.ReadImage(str(fixed_path)), sitk.sitkFloat32)
        moving = sitk.Cast(sitk.ReadImage(str(moving_path)), sitk.sitkFloat32)
        
        # Configurar registro
        registration = sitk.ImageRegistrationMethod()
        registration.SetMetricAsMeanSquares()
        registration.SetOptimizerAsRegularStepGradientDescent(
            learningRate=1.0,
            minStep=0.001,
            numberOfIterations=100
        )
        registration.SetInterpolator(sitk.sitkLinear)
        
        # Transform inicial
        initial_transform = sitk.CenteredTransformInitializer(
            fixed, moving, 
            sitk.Euler3DTransform(),
            sitk.CenteredTransformInitializerFilter.GEOMETRY
        )
        registration.SetInitialTransform(initial_transform)
        
        # Ejecutar registro
        final_transform = registration.Execute(fixed, moving)
        
        # Aplicar transformación
        registered = sitk.Resample(
            moving, fixed, final_transform, 
            sitk.sitkLinear, 0.0, moving.GetPixelID()
        )
        
        # Guardar
        sitk.WriteImage(registered, str(output_path))
        return True
        
    except Exception as e:
        print(f"Error en registro: {e}")
        return False

# Configuración directa
images_dir = Path(r"C:\Users\usuario\Documents\Mama_Mia\datos\images")
output_dir = Path(r"C:\Users\usuario\Documents\Mama_Mia\datos\images_registered")
splits_file = Path(r"C:\Users\usuario\Documents\Mama_Mia\datos\train_test_splits.csv")

print(f"Input: {images_dir}")
print(f"Output: {output_dir}")
print(f"Splits: {splits_file}")

# Leer casos del split oficial
splits_df = pd.read_csv(splits_file)
all_cases = list(splits_df['train_split']) + list(splits_df['test_split'])
all_cases = [case for case in all_cases if pd.notna(case)]

print(f"Procesando {len(all_cases)} casos del split oficial...")

success_count = 0
error_count = 0

for i, case_id in enumerate(tqdm(all_cases)):
    # Archivos de entrada
    pre_file = images_dir / f"{case_id}_0000.nii.gz"
    post_file = images_dir / f"{case_id}_0001.nii.gz"
    
    # Archivos de salida
    out_pre = output_dir / f"{case_id}_0000.nii.gz"
    out_post = output_dir / f"{case_id}_0001.nii.gz"
    
    # Verificar que existen
    if not pre_file.exists() or not post_file.exists():
        error_count += 1
        continue
    
    # Copiar PRE (referencia, sin cambios)
    shutil.copy2(pre_file, out_pre)
    
    # Registrar POST a PRE
    if register_image_pair(pre_file, post_file, out_post):
        success_count += 1
    else:
        error_count += 1
        # Si falla registro, copiar original
        shutil.copy2(post_file, out_post)
    
    # Progreso cada 100 casos
    if (i + 1) % 100 == 0:
        print(f"Procesados: {i+1}/{len(all_cases)} - Éxito: {success_count}, Errores: {error_count}")

print(f"\n{'='*50}")
print(f"REGISTRO COMPLETADO")
print(f"{'='*50}")
print(f"Total procesados: {len(all_cases)}")
print(f"Registros exitosos: {success_count}")
print(f"Errores: {error_count}")
print(f"Archivos guardados en: {output_dir}")