"""
SCRIPT: registrar_dataset_completo_FIXED.py
Registra todas las imágenes POST a sus correspondientes PRE
RUTAS ABSOLUTAS - Sin errores de directorio
"""
import SimpleITK as sitk
import pandas as pd
from pathlib import Path
import numpy as np
from tqdm import tqdm
import shutil

def register_dcemri_pair(pre_path, post_path, output_path):
    """Registra imagen POST a PRE usando SimpleITK"""
    try:
        # Cargar imágenes
        fixed_img = sitk.ReadImage(str(pre_path))
        moving_img = sitk.ReadImage(str(post_path))
        
        # Configurar registro
        registration_method = sitk.ImageRegistrationMethod()
        
        # Métrica de similitud
        registration_method.SetMetricAsMeanSquares()
        
        # Optimizador
        registration_method.SetOptimizerAsRegularStepGradientDescent(
            learningRate=1.0,
            minStep=0.001,
            numberOfIterations=200  # Reducido para velocidad
        )
        
        # Interpolador
        registration_method.SetInterpolator(sitk.sitkLinear)
        
        # Transform inicial (solo rigid - 6 DOF)
        initial_transform = sitk.CenteredTransformInitializer(
            fixed_img, moving_img, 
            sitk.Euler3DTransform(),
            sitk.CenteredTransformInitializerFilter.GEOMETRY
        )
        registration_method.SetInitialTransform(initial_transform)
        
        # Ejecutar registro
        final_transform = registration_method.Execute(fixed_img, moving_img)
        
        # Aplicar transformación
        registered_img = sitk.Resample(
            moving_img, fixed_img, final_transform, 
            sitk.sitkLinear, 0.0, moving_img.GetPixelID()
        )
        
        # Guardar resultado
        sitk.WriteImage(registered_img, str(output_path))
        
        return True, "Success"
        
    except Exception as e:
        return False, str(e)

def process_all_cases():
    """Procesa todos los casos del dataset"""
    
    # Configuración con rutas absolutas
    base_path = Path(r"C:\Users\usuario\Documents\Mama_Mia\datos")
    images_dir = base_path / "images"
    splits_file = Path(r"C:\Users\usuario\Documents\Mama_Mia\datos\train_test_splits.csv")
    
    # Crear directorio de salida
    registered_dir = base_path / "images_registered"
    
    print(f"Base path: {base_path}")
    print(f"Images dir: {images_dir}")
    print(f"Output dir: {registered_dir}")
    
    # Verificar que existen los directorios
    if not base_path.exists():
        print(f"ERROR: No existe {base_path}")
        return
    if not images_dir.exists():
        print(f"ERROR: No existe {images_dir}")
        return
    if not splits_file.exists():
        print(f"ERROR: No existe {splits_file}")
        return
    
    # Crear directorio de salida
    registered_dir.mkdir(exist_ok=True)
    print(f"Directorio de salida creado: {registered_dir}")
    
    # Leer split oficial
    splits_df = pd.read_csv(splits_file)
    all_cases = list(splits_df['train_split']) + list(splits_df['test_split'])
    all_cases = [case for case in all_cases if pd.notna(case)]
    
    print(f"Procesando {len(all_cases)} casos...")
    
    results = []
    
    for i, case_id in enumerate(tqdm(all_cases)):
        try:
            # Archivos de entrada
            pre_file = images_dir / f"{case_id}_0000.nii.gz"
            post_file = images_dir / f"{case_id}_0001.nii.gz"
            
            # Archivos de salida
            registered_pre = registered_dir / f"{case_id}_0000.nii.gz"
            registered_post = registered_dir / f"{case_id}_0001.nii.gz"
            
            # Verificar que existen
            if not pre_file.exists() or not post_file.exists():
                results.append({
                    'case_id': case_id,
                    'status': 'Missing files',
                    'pre_exists': pre_file.exists(),
                    'post_exists': post_file.exists()
                })
                continue
            
            # Copiar PRE como referencia (sin cambios)
            shutil.copy2(pre_file, registered_pre)
            
            # Registrar POST a PRE
            success, message = register_dcemri_pair(
                pre_file, post_file, registered_post
            )
            
            results.append({
                'case_id': case_id,
                'status': 'Success' if success else 'Failed',
                'message': message,
                'pre_exists': True,
                'post_exists': True
            })
            
            if (i + 1) % 50 == 0:
                print(f"Completados: {i+1}/{len(all_cases)}")
                
        except Exception as e:
            results.append({
                'case_id': case_id,
                'status': 'Error',
                'message': str(e),
                'pre_exists': pre_file.exists() if 'pre_file' in locals() else False,
                'post_exists': post_file.exists() if 'post_file' in locals() else False
            })
    
    # Guardar reporte
    results_df = pd.DataFrame(results)
    results_df.to_csv(registered_dir / "registration_report.csv", index=False)
    
    # Mostrar estadísticas
    print(f"\n{'='*50}")
    print(f"REPORTE DE REGISTRO")
    print(f"{'='*50}")
    print(f"Total casos procesados: {len(results)}")
    print(f"Exitosos: {len(results_df[results_df['status'] == 'Success'])}")
    print(f"Fallidos: {len(results_df[results_df['status'] == 'Failed'])}")
    print(f"Archivos faltantes: {len(results_df[results_df['status'] == 'Missing files'])}")
    print(f"Directorio de salida: {registered_dir}")

if __name__ == "__main__":
    process_all_cases()