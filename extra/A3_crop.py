"""
PROCESAR A3_TEST_PREDICTIONS CON MODELO CROP TUMOR
Toma las predicciones de A3_test_predictions, las recorta con el mismo criterio
que se usó para el modelo crop tumor, y luego aplica el modelo crop tumor
"""
import numpy as np
import SimpleITK as sitk
import pandas as pd
from pathlib import Path
import json
import subprocess
import shutil
from datetime import datetime

print("=== PROCESAMIENTO A3_TEST_PREDICTIONS CON CROP TUMOR ===")

# ============ CONFIGURACIÓN ============
# Rutas principales
a3_predictions_dir = Path(r"C:\Users\usuario\Documents\Mama_Mia\experimentos\A3_test_predictions")
output_dir = Path(r"C:\Users\usuario\Documents\Mama_Mia\experimentos\A3_crop_tumor_results")

# Información del modelo crop tumor
crop_tumor_dataset_id = 114  # Dataset ID del modelo crop tumor
crop_tumor_model_name = "3d_fullres"
crop_tumor_fold = "0"  # o el fold que quieras usar

# Directorio temporal para procesamientos
temp_dir = Path(r"C:\Users\usuario\Documents\Mama_Mia\experimentos\temp_processing")

# ============ FUNCIONES PARA DETECTAR CRITERIO DE CROP ============
def load_crop_criteria():
    """
    Carga el criterio de crop que se usó para el modelo crop tumor
    Busca en varios lugares posibles donde se pudo haber guardado esta info
    """
    print("\n=== BUSCANDO CRITERIO DE CROP ORIGINAL ===")
    
    # Posibles ubicaciones de la información de crop
    possible_locations = [
        Path(r"C:\Users\usuario\Documents\Mama_Mia\crop_info.json"),
        Path(r"C:\Users\usuario\Documents\Mama_Mia\experimentos\crop_info.json"),
        Path(r"C:\Users\usuario\Documents\Mama_Mia\cropped_data\crop_info.json"),
        Path(r"C:\Users\usuario\Documents\Mama_Mia\datos\crop_info.json"),
        Path("./crop_info.json"),
        Path("./crop_criteria.json"),
        Path("./Dataset114_CropTumor/crop_info.json"),
        Path("../Dataset114_CropTumor/crop_info.json"),
    ]
    
    crop_info = None
    
    for location in possible_locations:
        if location.exists():
            try:
                with open(location, 'r') as f:
                    crop_info = json.load(f)
                print(f"✅ Encontrada info de crop en: {location}")
                break
            except Exception as e:
                print(f"Error leyendo {location}: {e}")
                continue
    
    if crop_info is None:
        print("⚠️ No se encontró información de crop guardada")
        print("   Se usará detección automática basada en contenido")
        return None
    
    return crop_info

def detect_tumor_region(image_array, margin=10):
    """
    Detecta la región del tumor en una predicción
    Optimizado para ser más rápido y robusto
    """
    # Encontrar todas las regiones no-zero (predicciones positivas)
    # Usar threshold para ser más eficiente con arrays grandes
    threshold = 0.1  # Umbral mínimo para considerar como predicción positiva
    binary_mask = image_array > threshold
    
    coords = np.where(binary_mask)
    
    if len(coords[0]) == 0:
        print("    ⚠️ No se encontraron predicciones positivas")
        return None
    
    # Calcular bounding box
    z_min, z_max = coords[0].min(), coords[0].max()
    y_min, y_max = coords[1].min(), coords[1].max()
    x_min, x_max = coords[2].min(), coords[2].max()
    
    # Añadir margen pero asegurar que no se salga de bounds
    z_min = max(0, z_min - margin)
    z_max = min(image_array.shape[0], z_max + margin + 1)  # +1 para incluir el último slice
    y_min = max(0, y_min - margin)
    y_max = min(image_array.shape[1], y_max + margin + 1)
    x_min = max(0, x_min - margin)
    x_max = min(image_array.shape[2], x_max + margin + 1)
    
    bbox = {
        'z': [int(z_min), int(z_max)],  # Convertir a int estándar de Python
        'y': [int(y_min), int(y_max)], 
        'x': [int(x_min), int(x_max)]
    }
    
    return bbox

def apply_crop_to_prediction(pred_file, crop_info=None):
    """
    Aplica crop a una predicción individual
    """
    print(f"  Procesando: {pred_file.name}")
    
    # Cargar predicción
    pred_sitk = sitk.ReadImage(str(pred_file))
    pred_array = sitk.GetArrayFromImage(pred_sitk)
    
    original_shape = pred_array.shape
    print(f"    Shape original: {original_shape}")
    
    # Determinar bounding box
    case_name = pred_file.stem.replace('.nii', '')
    
    if crop_info and case_name in crop_info:
        # Usar información guardada
        bbox = crop_info[case_name]
        print(f"    Usando bbox guardado: {bbox}")
    else:
        # Detectar automáticamente
        bbox = detect_tumor_region(pred_array)
        if bbox is None:
            return None, None, None
        print(f"    Bbox detectado: {bbox}")
    
    # Aplicar crop
    z_start, z_end = bbox['z']
    y_start, y_end = bbox['y']
    x_start, x_end = bbox['x']
    
    cropped_array = pred_array[z_start:z_end, y_start:y_end, x_start:x_end]
    print(f"    Shape cropped: {cropped_array.shape}")
    
    # Crear imagen SITK cropped
    cropped_sitk = sitk.GetImageFromArray(cropped_array.astype(pred_array.dtype))
    
    # Copiar metadatos compatibles (NO usar CopyInformation porque el tamaño cambió)
    try:
        # Copiar spacing (resolución)
        if hasattr(pred_sitk, 'GetSpacing'):
            cropped_sitk.SetSpacing(pred_sitk.GetSpacing())
        
        # Copiar dirección (orientación)
        if hasattr(pred_sitk, 'GetDirection'):
            cropped_sitk.SetDirection(pred_sitk.GetDirection())
        
        # Calcular nuevo origin basado en el crop
        if hasattr(pred_sitk, 'GetOrigin') and hasattr(pred_sitk, 'GetSpacing'):
            original_origin = pred_sitk.GetOrigin()
            spacing = pred_sitk.GetSpacing()
            
            # Calcular nuevo origin considerando el offset del crop
            new_origin = (
                original_origin[0] + x_start * spacing[0],
                original_origin[1] + y_start * spacing[1], 
                original_origin[2] + z_start * spacing[2]
            )
            cropped_sitk.SetOrigin(new_origin)
    except Exception as e:
        print(f"    ⚠️ No se pudieron copiar todos los metadatos: {e}")
        # Continuar sin metadatos si hay problemas
    
    return cropped_sitk, cropped_array, bbox

def process_all_a3_predictions():
    """
    Procesa todas las predicciones de A3_test_predictions aplicando crops
    """
    print(f"\n=== PROCESANDO PREDICCIONES A3 ===")
    
    if not a3_predictions_dir.exists():
        print(f"❌ No se encontró directorio: {a3_predictions_dir}")
        return None, 0
    
    # Buscar archivos de predicciones
    pred_files = list(a3_predictions_dir.glob("*.nii.gz"))
    if not pred_files:
        print(f"❌ No se encontraron archivos .nii.gz en {a3_predictions_dir}")
        return None, 0
    
    pred_files.sort()
    print(f"Encontradas {len(pred_files)} predicciones")
    
    # Cargar criterio de crop
    crop_info = load_crop_criteria()
    
    # Crear directorio de salida para predicciones cropped
    cropped_dir = temp_dir / "cropped_predictions"
    cropped_dir.mkdir(parents=True, exist_ok=True)
    
    processed_count = 0
    bbox_info = {}
    errors = []
    
    for i, pred_file in enumerate(pred_files):
        try:
            # Aplicar crop
            cropped_sitk, cropped_array, bbox = apply_crop_to_prediction(pred_file, crop_info)
            
            if cropped_sitk is not None:
                # Guardar predicción cropped
                output_name = f"cropped_{pred_file.name}"
                output_path = cropped_dir / output_name
                
                sitk.WriteImage(cropped_sitk, str(output_path))
                
                # Guardar info de bbox para referencia
                case_name = pred_file.stem.replace('.nii', '')
                bbox_info[case_name] = bbox
                
                processed_count += 1
                
                if i < 10 or (i + 1) % 5 == 0:  # Mostrar progreso más frecuentemente
                    print(f"  ✅ {i+1}/{len(pred_files)}: {output_name}")
                    print(f"     Progreso: {(i+1)/len(pred_files)*100:.1f}%")
            else:
                errors.append(f"{pred_file.name}: No se pudo detectar región")
                
        except Exception as e:
            errors.append(f"{pred_file.name}: {str(e)}")
            print(f"  ❌ Error: {pred_file.name} - {e}")
            continue
    
    # Guardar información de bboxes
    bbox_file = temp_dir / "applied_crops.json"
    with open(bbox_file, 'w') as f:
        json.dump(bbox_info, f, indent=2)
    
    print(f"\n✅ Procesamiento completado:")
    print(f"   Cropped exitosamente: {processed_count}/{len(pred_files)}")
    print(f"   Errores: {len(errors)}")
    print(f"   Guardadas en: {cropped_dir}")
    
    if errors:
        print("\nPrimeros errores:")
        for error in errors[:3]:
            print(f"   {error}")
    
    return cropped_dir, processed_count

def prepare_for_crop_tumor_model(cropped_dir):
    """
    Prepara las predicciones cropped para el modelo crop tumor
    Las renombra según el formato que espera nnU-Net
    """
    print(f"\n=== PREPARANDO PARA MODELO CROP TUMOR ===")
    
    if not cropped_dir or not cropped_dir.exists():
        print("❌ Directorio de predicciones cropped no válido")
        return None
    
    # Crear directorio para input del modelo crop tumor
    model_input_dir = temp_dir / "crop_tumor_input"
    model_input_dir.mkdir(parents=True, exist_ok=True)
    
    # Buscar predicciones cropped
    cropped_files = list(cropped_dir.glob("*.nii.gz"))
    cropped_files.sort()
    
    print(f"Preparando {len(cropped_files)} archivos para modelo crop tumor")
    
    for i, cropped_file in enumerate(cropped_files):
        try:
            # Generar nombre compatible con nnU-Net
            # Formato: case_XXX_0000.nii.gz
            new_name = f"case_{i:03d}_0000.nii.gz"
            new_path = model_input_dir / new_name
            
            # Copiar archivo
            shutil.copy2(cropped_file, new_path)
            
            if i < 5 or (i + 1) % 20 == 0:
                print(f"  {i+1}/{len(cropped_files)}: {new_name}")
                
        except Exception as e:
            print(f"  Error preparando {cropped_file.name}: {e}")
            continue
    
    print(f"✅ Archivos preparados en: {model_input_dir}")
    return model_input_dir

def run_crop_tumor_inference(input_dir):
    """
    Ejecuta el modelo crop tumor sobre las predicciones cropped
    """
    print(f"\n=== EJECUTANDO MODELO CROP TUMOR ===")
    
    if not input_dir or not input_dir.exists():
        print("❌ Directorio de input no válido")
        return False, None
    
    # Directorio de salida del modelo
    model_output_dir = temp_dir / "crop_tumor_output"
    model_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Comando nnU-Net
    cmd = [
        "nnUNetv2_predict",
        "-i", str(input_dir),
        "-o", str(model_output_dir),
        "-d", str(crop_tumor_dataset_id),
        "-c", crop_tumor_model_name,
        "-f", crop_tumor_fold,
        "--save_probabilities"
    ]
    
    print(f"Comando: {' '.join(cmd)}")
    print("Ejecutando modelo crop tumor...")
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=1800  # 30 minutos timeout
        )
        
        if result.returncode == 0:
            print("✅ Modelo crop tumor ejecutado exitosamente")
            
            # Verificar outputs
            output_files = list(model_output_dir.glob("*.nii.gz"))
            print(f"Generados {len(output_files)} archivos de predicción")
            
            return True, model_output_dir
        else:
            print("❌ Error ejecutando modelo crop tumor:")
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
            return False, None
            
    except subprocess.TimeoutExpired:
        print("❌ Timeout ejecutando modelo (>30 min)")
        return False, None
    except Exception as e:
        print(f"❌ Error: {e}")
        return False, None

def organize_final_results(model_output_dir):
    """
    Organiza los resultados finales en el directorio de salida
    """
    print(f"\n=== ORGANIZANDO RESULTADOS FINALES ===")
    
    if not model_output_dir or not model_output_dir.exists():
        print("❌ No hay resultados del modelo para organizar")
        return False
    
    # Crear directorio final
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Copiar resultados del modelo crop tumor
    final_predictions_dir = output_dir / "final_predictions"
    final_predictions_dir.mkdir(exist_ok=True)
    
    model_files = list(model_output_dir.glob("*.nii.gz"))
    
    for model_file in model_files:
        final_path = final_predictions_dir / model_file.name
        shutil.copy2(model_file, final_path)
    
    # Copiar archivos de información
    info_files = [
        temp_dir / "applied_crops.json"
    ]
    
    for info_file in info_files:
        if info_file.exists():
            shutil.copy2(info_file, output_dir / info_file.name)
    
    # Crear resumen
    summary = {
        'timestamp': datetime.now().isoformat(),
        'source': str(a3_predictions_dir),
        'crop_tumor_model': {
            'dataset_id': 114,
            'model_name': crop_tumor_model_name,
            'fold': crop_tumor_fold
        },
        'results': {
            'total_predictions': len(model_files),
            'output_directory': str(final_predictions_dir)
        }
    }
    
    summary_file = output_dir / "processing_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"✅ Resultados organizados en: {output_dir}")
    print(f"   Predicciones finales: {final_predictions_dir}")
    print(f"   Resumen: {summary_file}")
    
    return True

# ============ EJECUCIÓN PRINCIPAL ============
def main():
    start_time = datetime.now()
    print(f"Inicio: {start_time}")
    
    try:
        # 1. Procesar predicciones A3 aplicando crops
        cropped_dir, processed_count = process_all_a3_predictions()
        if processed_count == 0:
            print("❌ No se procesaron predicciones")
            return
        
        # 2. Preparar para modelo crop tumor
        model_input_dir = prepare_for_crop_tumor_model(cropped_dir)
        if model_input_dir is None:
            print("❌ No se pudo preparar input para modelo")
            return
        
        # 3. Ejecutar modelo crop tumor
        success, model_output_dir = run_crop_tumor_inference(model_input_dir)
        if not success:
            print("❌ Falló ejecución del modelo crop tumor")
            return
        
        # 4. Organizar resultados finales
        success = organize_final_results(model_output_dir)
        if not success:
            print("❌ No se pudieron organizar resultados")
            return
        
        # 5. Limpiar archivos temporales (opcional)
        # shutil.rmtree(temp_dir)  # Descomenta si quieres limpiar
        
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        
        print(f"\n{'='*60}")
        print(f"PROCESAMIENTO COMPLETADO EXITOSAMENTE")
        print(f"{'='*60}")
        print(f"Tiempo total: {execution_time:.1f} segundos")
        print(f"Predicciones procesadas: {processed_count}")
        print(f"Resultados finales en: {output_dir}")
        print(f"\n🎯 ¡LISTO! Ya tienes las predicciones del modelo crop tumor")
        print(f"   aplicado a las predicciones A3 recortadas")
        
    except Exception as e:
        print(f"❌ Error general: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()