import os
import subprocess
import sys
import time

# Configurar variables de entorno
base_path = r"C:\Users\usuario\Documents\Mama_Mia"
env_vars = {
    'nnUNet_raw': os.path.join(base_path, "nnUNet_raw"),
    'nnUNet_preprocessed': os.path.join(base_path, "nnUNet_preprocessed"), 
    'nnUNet_results': os.path.join(base_path, "nnUNet_results"),
}

for var, path in env_vars.items():
    os.environ[var] = path
    os.makedirs(path, exist_ok=True)

# Rutas
input_folder = r".\temp_input"
output_folder = r".\results_output"  
weights_path = r"C:\Users\usuario\Documents\Mama_Mia\mama_mia_weights\extracted\Dataset501_full_image_dce_mri_tumor_segmentation"

print("MAMA-MIA nnUNetv2 CORREGIDO")
print("=" * 40)

# Verificar archivos
if not os.path.exists(input_folder):
    print("ERROR: No existe temp_input")
    exit(1)

input_files = [f for f in os.listdir(input_folder) if f.endswith('.nii.gz')]
print(f"Archivos input: {len(input_files)}")

os.makedirs(output_folder, exist_ok=True)

# Código corregido para nnUNetv2
inference_code = f"""
import os
import sys

# Configurar environment
os.environ['nnUNet_raw'] = r'{env_vars['nnUNet_raw']}'
os.environ['nnUNet_preprocessed'] = r'{env_vars['nnUNet_preprocessed']}'
os.environ['nnUNet_results'] = r'{env_vars['nnUNet_results']}'

try:
    print("Buscando funciones disponibles en nnUNetv2...")
    
    # Método 1: Intentar import directo
    try:
        from nnunetv2.inference.predict_from_raw_data import predict_from_raw_data_how_to
        print("Encontrada: predict_from_raw_data_how_to")
    except ImportError:
        pass
    
    # Método 2: Comando de línea nnUNetv2
    try:
        from nnunetv2.inference.predict_from_raw_data import predict_entry_point
        print("Encontrada: predict_entry_point")
    except ImportError:
        pass
    
    # Método 3: API principal
    try:
        from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
        print("Encontrada: nnUNetPredictor")
        
        # Usar nnUNetPredictor
        predictor = nnUNetPredictor(
            tile_step_size=0.5,
            use_gaussian=True,
            use_mirroring=True,
            perform_everything_on_gpu=True,
            device=None,  # Auto-detect GPU
            verbose=True,
            verbose_preprocessing=True,
            allow_tqdm=True
        )
        
        print("Inicializando modelo...")
        predictor.initialize_from_trained_model_folder(
            model_training_output_dir=r'{weights_path}',
            use_folds=(0, 1, 2, 3, 4),
            checkpoint_name='checkpoint_final.pth'
        )
        
        print("Modelo inicializado, comenzando predicciones...")
        
        # Predecir cada archivo
        import glob
        input_files = glob.glob(r'{input_folder}/*.nii.gz')
        
        for input_file in input_files:
            output_file = r'{output_folder}/' + os.path.basename(input_file)
            
            print(f"Procesando: {{os.path.basename(input_file)}}")
            
            predictor.predict_from_files(
                list_of_list_of_images=[[input_file]],
                list_of_segmentations_from_previous_stage=[None],
                list_of_output_filenames=[output_file],
                save_probabilities=False,
                overwrite=True
            )
        
        print("SUCCESS_NNUNETV2_PREDICTOR")
        
    except ImportError as e:
        print(f"nnUNetPredictor no disponible: {{e}}")
        
        # Método 4: Comando sistema
        print("Intentando comando de sistema...")
        import subprocess
        
        cmd = [
            'nnUNetv2_predict',
            '-i', r'{input_folder}',
            '-o', r'{output_folder}',
            '-d', '501',  # Dataset ID
            '-c', '3d_fullres',
            '-f', '0', '1', '2', '3', '4'
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print("SUCCESS_COMMAND_LINE")
        else:
            print(f"ERROR_COMMAND: {{result.stderr}}")
            raise Exception("Comando falló")

except Exception as e:
    print(f"ERROR_GENERAL: {{str(e)}}")
    import traceback
    traceback.print_exc()
"""

try:
    start_time = time.time()
    
    print("Ejecutando inferencia nnUNetv2 corregida...")
    
    cmd = [sys.executable, "-c", inference_code]
    
    env = os.environ.copy()
    env.update(env_vars)
    
    result = subprocess.run(cmd, 
                          capture_output=True, 
                          text=True, 
                          timeout=600,
                          env=env,
                          encoding='utf-8',
                          errors='replace')
    
    end_time = time.time()
    print(f"Tiempo: {end_time - start_time:.1f} segundos")
    
    if "SUCCESS_NNUNETV2_PREDICTOR" in result.stdout or "SUCCESS_COMMAND_LINE" in result.stdout:
        print("INFERENCIA EXITOSA!")
        
        # Verificar outputs
        if os.path.exists(output_folder):
            outputs = [f for f in os.listdir(output_folder) if f.endswith('.nii.gz')]
            print(f"Archivos generados: {len(outputs)}")
            
            if outputs:
                print("Resultados:")
                for f in outputs[:3]:
                    print(f"  {f}")
                
                # Evaluación rápida
                try:
                    import nibabel as nib
                    import numpy as np
                    
                    case_id = outputs[0].replace('.nii.gz', '')
                    pred_file = os.path.join(output_folder, outputs[0])
                    gt_file = os.path.join(r'C:\Users\usuario\Documents\Mama_Mia\datos\segmentations\expert', case_id.lower() + '.nii.gz')
                    
                    pred = nib.load(pred_file).get_fdata()
                    
                    if os.path.exists(gt_file):
                        gt = nib.load(gt_file).get_fdata()
                        
                        pred_bin = pred > 0.5
                        gt_bin = gt > 0.5
                        intersection = np.sum(pred_bin * gt_bin)
                        total = np.sum(pred_bin) + np.sum(gt_bin)
                        
                        if total > 0:
                            dice = (2.0 * intersection) / total
                            print(f"\nDICE {case_id}: {dice:.3f}")
                            
                            if dice > 0.7:
                                print("BASELINE VALIDADO!")
                        
                except Exception as e:
                    print(f"Error evaluacion: {e}")
                
                print("\nBASELINE FUNCIONANDO!")
                
            else:
                print("No se generaron archivos")
    
    else:
        print("ERROR en inferencia:")
        print("STDOUT:")
        print(result.stdout)
        print("\nSTDERR:")
        print(result.stderr)

except Exception as e:
    print(f"Error: {e}")

print("\nSi sigue fallando, usemos comando directo:")
print("nnUNetv2_predict -i ./temp_input -o ./results_output -d 501 -c 3d_fullres -f 0 1 2 3 4")