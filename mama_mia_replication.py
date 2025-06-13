# Replicación exacta del preprocessing de MAMA-MIA CON SKIP AUTOMÁTICO
import SimpleITK as sitk
import numpy as np
import pandas as pd
from pathlib import Path
import os

def zscore_normalization_sitk(image_sitk, mean, std):
    """Z-score normalization exactamente como MAMA-MIA"""
    array = sitk.GetArrayFromImage(image_sitk)
    normalized_array = (array - mean) / std
    zscored_sitk = sitk.GetImageFromArray(normalized_array)
    zscored_sitk.CopyInformation(image_sitk)
    return zscored_sitk

def resample_sitk(image_sitk, new_spacing=[1.0, 1.0, 1.0], interpolator=sitk.sitkBSpline):
    """Resampling a 1x1x1mm exactamente como MAMA-MIA"""
    original_size = image_sitk.GetSize()
    original_spacing = image_sitk.GetSpacing()
    
    # Calcular nuevo tamaño
    new_size = [
        int(round(original_size[0] * original_spacing[0] / new_spacing[0])),
        int(round(original_size[1] * original_spacing[1] / new_spacing[1])),
        int(round(original_size[2] * original_spacing[2] / new_spacing[2]))
    ]
    
    # Configurar resampler
    resample = sitk.ResampleImageFilter()
    resample.SetInterpolator(interpolator)
    resample.SetOutputDirection(image_sitk.GetDirection())
    resample.SetOutputOrigin(image_sitk.GetOrigin())
    resample.SetOutputSpacing(new_spacing)
    resample.SetSize(new_size)
    resample.SetTransform(sitk.Transform())
    
    return resample.Execute(image_sitk)

def read_mri_phase(images_folder, patient_id, phase=0):
    """Leer fase específica de MRI - Adaptado para estructura DUKE"""
    # Convertir DUKE_001 -> duke_001_0000.nii.gz
    patient_id_lower = patient_id.lower()
    phase_path = f'{images_folder}/{patient_id_lower}_{phase:04d}.nii.gz'
    if os.path.exists(phase_path):
        return sitk.ReadImage(phase_path, sitk.sitkFloat32)
    else:
        raise FileNotFoundError(f"No se encontró: {phase_path}")

def read_segmentation(segmentations_folder, patient_id):
    """Leer segmentación ground truth - Adaptado para estructura DUKE"""
    # Probar primero en expert/, luego en automatic/
    patient_id_lower = patient_id.lower()
    
    # Intentar expert primero
    seg_path_expert = f'{segmentations_folder}/expert/{patient_id_lower}.nii.gz'
    if os.path.exists(seg_path_expert):
        return sitk.ReadImage(seg_path_expert, sitk.sitkUInt8)
    
    # Si no, intentar automatic
    seg_path_auto = f'{segmentations_folder}/automatic/{patient_id_lower}.nii.gz'
    if os.path.exists(seg_path_auto):
        return sitk.ReadImage(seg_path_auto, sitk.sitkUInt8)
    
    raise FileNotFoundError(f"No se encontró segmentación para {patient_id}")

def calculate_mean_std_all_phases(phases_list):
    """Calcular mean y std de TODAS las fases (clave de MAMA-MIA)"""
    all_voxels = []
    
    for phase_sitk in phases_list:
        array = sitk.GetArrayFromImage(phase_sitk)
        # Solo incluir voxels no-zero
        non_zero_voxels = array[array > 0]
        all_voxels.extend(non_zero_voxels.flatten())
    
    all_voxels = np.array(all_voxels)
    mean_all = np.mean(all_voxels)
    std_all = np.std(all_voxels)
    
    return mean_all, std_all

def check_patient_already_processed(output_folder, patient_id):
    """Verificar si paciente ya fue procesado completamente"""
    
    output_path = Path(output_folder) / patient_id
    
    if not output_path.exists():
        return False
    
    # Archivos mínimos requeridos
    required_files = [
        f"{patient_id}_pre_contrast.nii.gz",
        f"{patient_id}_post_contrast_1.nii.gz",
        f"{patient_id}_segmentation.nii.gz"
    ]
    
    # Verificar que todos los archivos existen y no están vacíos
    for required_file in required_files:
        file_path = output_path / required_file
        if not file_path.exists():
            return False
        
        # Verificar que el archivo no está vacío (>1KB)
        if file_path.stat().st_size < 1024:
            return False
    
    return True

def preprocess_patient_mama_mia_style(images_folder, segmentations_folder, patient_id, 
                                     output_folder, num_phases=4):
    """
    Preprocessing exacto de MAMA-MIA CON SKIP AUTOMÁTICO:
    1. Verificar si ya está procesado
    2. Leer todas las fases disponibles
    3. Calcular mean/std de TODAS las fases
    4. Z-score normalization
    5. Resampling a 1x1x1mm
    """
    
    # PASO 0: VERIFICAR SI YA ESTÁ PROCESADO
    if check_patient_already_processed(output_folder, patient_id):
        print(f"SKIP {patient_id} - YA PROCESADO")
        return True
    
    print(f"Procesando paciente: {patient_id}")
    
    # PASO 1: Leer todas las fases disponibles
    phases = []
    phase_names = []
    
    # Intentar leer pre-contraste y post-contraste
    for phase in range(num_phases):
        try:
            phase_sitk = read_mri_phase(images_folder, patient_id, phase)
            phases.append(phase_sitk)
            if phase == 0:
                phase_names.append('pre_contrast')
            else:
                phase_names.append(f'post_contrast_{phase}')
            print(f"  - Fase {phase}: OK")
        except FileNotFoundError:
            print(f"  - Fase {phase}: No encontrada")
            break
    
    if len(phases) < 2:
        print(f"  ERROR: Insuficientes fases para {patient_id}")
        return False
    
    # PASO 2: Calcular mean y std de TODAS las fases (CLAVE DE MAMA-MIA)
    print(f"  - Calculando estadísticas de {len(phases)} fases...")
    mean_all, std_all = calculate_mean_std_all_phases(phases)
    print(f"  - Mean: {mean_all:.4f}, Std: {std_all:.4f}")
    
    # PASO 3: Z-score normalization de cada fase
    normalized_phases = []
    for i, phase_sitk in enumerate(phases):
        normalized = zscore_normalization_sitk(phase_sitk, mean_all, std_all)
        normalized_phases.append(normalized)
        print(f"  - Normalizada fase {i}: OK")
    
    # PASO 4: Resampling a 1x1x1mm
    resampled_phases = []
    for i, normalized_sitk in enumerate(normalized_phases):
        resampled = resample_sitk(normalized_sitk, new_spacing=[1.0, 1.0, 1.0])
        resampled_phases.append(resampled)
        print(f"  - Resampled fase {i}: {resampled.GetSize()}")
    
    # PASO 5: Crear imagen de sustracción (post1 - pre)
    if len(resampled_phases) >= 2:
        subtraction = sitk.Subtract(resampled_phases[1], resampled_phases[0])  # post1 - pre
        resampled_phases.append(subtraction)
        phase_names.append('subtraction')
        print(f"  - Sustracción creada: OK")
    
    # PASO 6: Procesar segmentación ground truth
    try:
        segmentation_sitk = read_segmentation(segmentations_folder, patient_id)
        # Resampling con Nearest Neighbor para segmentaciones
        resampled_segmentation = resample_sitk(segmentation_sitk, new_spacing=[1.0, 1.0, 1.0], 
                                             interpolator=sitk.sitkNearestNeighbor)
        print(f"  - Segmentación procesada: OK")
    except FileNotFoundError:
        print(f"  - Segmentación no encontrada")
        resampled_segmentation = None
    
    # PASO 7: Guardar resultados
    output_path = Path(output_folder) / patient_id
    output_path.mkdir(parents=True, exist_ok=True)
    
    try:
        # Guardar cada fase procesada
        for i, (resampled_phase, name) in enumerate(zip(resampled_phases, phase_names)):
            output_file = output_path / f"{patient_id}_{name}.nii.gz"
            sitk.WriteImage(resampled_phase, str(output_file))
        
        # Guardar segmentación si existe
        if resampled_segmentation is not None:
            seg_output = output_path / f"{patient_id}_segmentation.nii.gz"
            sitk.WriteImage(resampled_segmentation, str(seg_output))
        
        print(f"  COMPLETADO: {patient_id}")
        return True
        
    except Exception as e:
        print(f"  ERROR guardando {patient_id}: {e}")
        # Limpiar archivos parciales
        if output_path.exists():
            import shutil
            try:
                shutil.rmtree(output_path)
            except:
                pass
        return False

def process_mama_mia_dataset(images_folder, segmentations_folder, patient_list, output_folder):
    """Procesar dataset completo estilo MAMA-MIA CON SKIP"""
    
    print("INICIANDO PREPROCESSING ESTILO MAMA-MIA")
    print("=" * 50)
    
    successful = 0
    failed = 0
    skipped = 0
    
    for i, patient_id in enumerate(patient_list, 1):
        print(f"\n[{i}/{len(patient_list)}]", end=" ")
        
        # Verificar si ya está procesado
        if check_patient_already_processed(output_folder, patient_id):
            print(f"SKIP {patient_id} - YA PROCESADO")
            skipped += 1
            continue
        
        try:
            success = preprocess_patient_mama_mia_style(
                images_folder, segmentations_folder, patient_id, output_folder
            )
            if success:
                successful += 1
            else:
                failed += 1
        except Exception as e:
            print(f"  ERROR procesando {patient_id}: {e}")
            failed += 1
        
        # Mostrar progreso cada 10 pacientes
        if i % 10 == 0:
            total_processed = successful + failed + skipped
            print(f"\nPROGRESO: {total_processed}/{len(patient_list)} | OK:{successful} SKIP:{skipped} ERROR:{failed}")
        
        print("-" * 30)
    
    print("\nRESUMEN FINAL:")
    print(f"EXITOSOS: {successful}")
    print(f"SALTADOS: {skipped}")
    print(f"FALLIDOS: {failed}")
    print(f"TASA DE EXITO: {successful/(successful+failed+skipped)*100:.1f}%")

def load_patient_list_from_csv(csv_path, split_type='train_split'):
    """Cargar lista de pacientes desde train_test_splits.csv"""
    df = pd.read_csv(csv_path)
    patient_list = df[split_type].dropna().tolist()
    return patient_list

# CONFIGURACIÓN PARA TU CASO
if __name__ == "__main__":
    
    # RUTAS AJUSTADAS A TU ESTRUCTURA
    IMAGES_FOLDER = r"C:\Users\usuario\Documents\Mama_Mia\datos\images"
    SEGMENTATIONS_FOLDER = r"C:\Users\usuario\Documents\Mama_Mia\datos\segmentations"
    SPLITS_CSV = r"C:\Users\usuario\Documents\Mama_Mia\datos\train_test_splits.csv"
    
    # RUTA CORREGIDA QUE SÍ FUNCIONA:
    OUTPUT_FOLDER = r"D:\preprocessed_mama_mia_style" #lo vomí porque no tenia espacio en el disco C
    
    # Cargar lista de pacientes desde CSV
    train_patients = load_patient_list_from_csv(SPLITS_CSV, 'train_split')
    test_patients = load_patient_list_from_csv(SPLITS_CSV, 'test_split')
    
    print(f"Pacientes train: {len(train_patients)}")
    print(f"Pacientes test: {len(test_patients)}")
    print(f"Primeros 5 train: {train_patients[:5]}")
    
    # Procesar TODOS los pacientes
    sample_patients = train_patients + test_patients  # TODOS los pacientes
    
    print(f"\nProcesando {len(sample_patients)} pacientes...")
    print(f"Output: {OUTPUT_FOLDER}")
    
    # Ejecutar preprocessing CON SKIP
    process_mama_mia_dataset(
        IMAGES_FOLDER, 
        SEGMENTATIONS_FOLDER, 
        sample_patients, 
        OUTPUT_FOLDER
    )