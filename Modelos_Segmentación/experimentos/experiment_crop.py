"""
PIPELINE: MODELO A3 → CROP TUMOR
Extrae regiones de las predicciones A3 y las prepara para el segundo modelo
"""
import numpy as np
import SimpleITK as sitk
import pandas as pd
from pathlib import Path
import cv2
from scipy import ndimage
import json

print("=== PIPELINE A3 → CROP TUMOR ===")

# ============ CONFIGURACIÓN ============
# Rutas - AJUSTAR SEGÚN TUS DATOS
a3_predictions_dir = Path(r"C:\Users\usuario\Documents\Mama_Mia\experimentos\A3_test_predictions")
original_images_dir = Path(r"C:\Users\usuario\Documents\Mama_Mia\datos")
splits_csv = Path(r"C:\Users\usuario\Documents\Mama_Mia\datos\train_test_splits.csv")
output_crops_dir = Path(r"C:\Users\usuario\Documents\Mama_Mia\crops_for_tumor_model")

# Crear directorios de salida
output_crops_dir.mkdir(exist_ok=True)
(output_crops_dir / "images").mkdir(exist_ok=True)
(output_crops_dir / "metadata").mkdir(exist_ok=True)

# ============ FUNCIONES PRINCIPALES ============

def extract_bounding_box_from_mask(mask_array, margin=10):
    """
    Extrae bounding box de una máscara 3D
    
    Args:
        mask_array: Array 3D de la máscara (valores > 0 indican área de interés)
        margin: Píxeles extra alrededor del bounding box
    
    Returns:
        bbox: (z_min, z_max, y_min, y_max, x_min, x_max)
    """
    # Encontrar coordenadas donde hay valores > 0
    coords = np.where(mask_array > 0)
    
    if len(coords[0]) == 0:
        print("⚠️ Máscara vacía")
        return None
    
    # Calcular bounding box
    z_min, z_max = coords[0].min(), coords[0].max()
    y_min, y_max = coords[1].min(), coords[1].max()
    x_min, x_max = coords[2].min(), coords[2].max()
    
    # Añadir margen (sin salirse de la imagen)
    z_min = max(0, z_min - margin)
    z_max = min(mask_array.shape[0], z_max + margin)
    y_min = max(0, y_min - margin)
    y_max = min(mask_array.shape[1], y_max + margin)
    x_min = max(0, x_min - margin)
    x_max = min(mask_array.shape[2], x_max + margin)
    
    return (z_min, z_max, y_min, y_max, x_min, x_max)

def load_test_split_mapping():
    """
    Carga el mapeo desde train_test_splits.csv
    
    Returns:
        dict: {case_id: duke_case_name}
    """
    print(f"📋 Cargando mapeo desde {splits_csv}")
    
    # Leer CSV
    df = pd.read_csv(splits_csv)
    
    # Extraer test_split
    test_cases = df['test_split'].tolist()
    
    print(f"   Test cases en CSV: {len(test_cases)}")
    print(f"   Primeros 5: {test_cases[:5]}")
    
    # Crear mapeo case_XXX → duke_case
    mapping = {}
    
    for i, duke_case in enumerate(test_cases):
        case_id = f"case_{i:03d}"
        mapping[case_id] = duke_case
        
        if i < 5:  # Mostrar primeros 5
            print(f"   {case_id} → {duke_case}")
    
    if len(test_cases) > 5:
        print(f"   ... y {len(test_cases) - 5} más")
    
    return mapping, test_cases

def find_duke_phases_by_name(duke_case_name, original_dir):
    """
    Busca AMBAS fases del caso duke específico
    
    Args:
        duke_case_name: Nombre como "DUKE_001"
        original_dir: Directorio con imágenes
    
    Returns:
        tuple: (pre_phase_path, post_phase_path) o (None, None)
    """
    # Convertir a lowercase para búsqueda
    duke_lower = duke_case_name.lower()
    
    # Patrones para buscar ambas fases
    # Típicamente: duke_001_0000.nii.gz (pre) y duke_001_0001.nii.gz (post)
    pre_patterns = [
        f"{duke_lower}_0000.nii.gz",
        f"{duke_case_name}_0000.nii.gz",
        f"{duke_lower}_000.nii.gz",
        f"{duke_case_name}_000.nii.gz"
    ]
    
    post_patterns = [
        f"{duke_lower}_0001.nii.gz", 
        f"{duke_case_name}_0001.nii.gz",
        f"{duke_lower}_001.nii.gz",
        f"{duke_case_name}_001.nii.gz"
    ]
    
    def search_in_dir(directory, patterns):
        """Busca patrones en un directorio"""
        for pattern in patterns:
            matches = list(directory.glob(pattern))
            if matches:
                return matches[0]
        return None
    
    # Buscar en directorio principal
    pre_file = search_in_dir(original_dir, pre_patterns)
    post_file = search_in_dir(original_dir, post_patterns)
    
    # Si no se encuentra en directorio principal, buscar en subdirectorios
    if pre_file is None or post_file is None:
        # Buscar en subdirectorio images
        images_subdir = original_dir / "images"
        if images_subdir.exists():
            if pre_file is None:
                pre_file = search_in_dir(images_subdir, pre_patterns)
            if post_file is None:
                post_file = search_in_dir(images_subdir, post_patterns)
        
        # Búsqueda más amplia en todos los subdirectorios
        if pre_file is None or post_file is None:
            for subdir in original_dir.iterdir():
                if subdir.is_dir():
                    if pre_file is None:
                        pre_file = search_in_dir(subdir, pre_patterns)
                    if post_file is None:
                        post_file = search_in_dir(subdir, post_patterns)
    
    return pre_file, post_file

def crop_image_with_bbox(image_sitk, bbox):
    """
    Recorta imagen usando bounding box
    
    Args:
        image_sitk: Imagen SimpleITK
        bbox: (z_min, z_max, y_min, y_max, x_min, x_max)
    
    Returns:
        cropped_sitk: Imagen recortada
    """
    z_min, z_max, y_min, y_max, x_min, x_max = bbox
    
    # Convertir a array para recortar
    image_array = sitk.GetArrayFromImage(image_sitk)
    
    # Recortar
    cropped_array = image_array[z_min:z_max, y_min:y_max, x_min:x_max]
    
    # Convertir de vuelta a SimpleITK
    cropped_sitk = sitk.GetImageFromArray(cropped_array)
    
    # Actualizar origen y dirección (importante para coordenadas)
    original_origin = image_sitk.GetOrigin()
    original_spacing = image_sitk.GetSpacing()
    
    # Calcular nuevo origen
    new_origin = (
        original_origin[0] + x_min * original_spacing[0],
        original_origin[1] + y_min * original_spacing[1],
        original_origin[2] + z_min * original_spacing[2]
    )
    
    cropped_sitk.SetOrigin(new_origin)
    cropped_sitk.SetSpacing(original_spacing)
    cropped_sitk.SetDirection(image_sitk.GetDirection())
    
    return cropped_sitk

def process_a3_prediction_to_two_phase_crops(prediction_file, pre_phase_file, post_phase_file, output_dir, case_id):
    """
    Procesa una predicción A3 para extraer crops de DOS FASES para el modelo tumor
    
    Args:
        prediction_file: Archivo .nii.gz con predicción A3
        pre_phase_file: Imagen pre-contraste correspondiente
        post_phase_file: Imagen post-contraste correspondiente  
        output_dir: Directorio de salida
        case_id: ID del caso
    
    Returns:
        dict: Metadata del procesamiento
    """
    try:
        # Cargar predicción A3 (máscara)
        mask_sitk = sitk.ReadImage(str(prediction_file))
        mask_array = sitk.GetArrayFromImage(mask_sitk)
        
        print(f"📋 {case_id}: Máscara shape {mask_array.shape}, valores únicos: {np.unique(mask_array)}")
        
        # Extraer bounding box
        bbox = extract_bounding_box_from_mask(mask_array, margin=15)
        if bbox is None:
            return {"status": "error", "message": "Máscara vacía"}
        
        z_min, z_max, y_min, y_max, x_min, x_max = bbox
        crop_size = ((z_max-z_min), (y_max-y_min), (x_max-x_min))
        
        print(f"   📦 Bounding box: Z[{z_min}:{z_max}] Y[{y_min}:{y_max}] X[{x_min}:{x_max}]")
        print(f"   📏 Tamaño crop: {crop_size}")
        
        # Cargar AMBAS fases
        pre_sitk = sitk.ReadImage(str(pre_phase_file))
        post_sitk = sitk.ReadImage(str(post_phase_file))
        
        pre_array = sitk.GetArrayFromImage(pre_sitk)
        post_array = sitk.GetArrayFromImage(post_sitk)
        
        print(f"   🖼️ Fase PRE: {pre_array.shape}")
        print(f"   🖼️ Fase POST: {post_array.shape}")
        
        # Verificar que el bbox cabe en AMBAS imágenes
        for phase_name, phase_array in [("PRE", pre_array), ("POST", post_array)]:
            if (z_max > phase_array.shape[0] or 
                y_max > phase_array.shape[1] or 
                x_max > phase_array.shape[2]):
                return {"status": "error", "message": f"Bounding box fuera de imagen {phase_name}"}
        
        # Recortar AMBAS fases con el MISMO bounding box
        pre_cropped_sitk = crop_image_with_bbox(pre_sitk, bbox)
        post_cropped_sitk = crop_image_with_bbox(post_sitk, bbox)
        
        # Guardar ambos crops
        pre_crop_file = output_dir / "images" / f"{case_id}_pre_crop.nii.gz"
        post_crop_file = output_dir / "images" / f"{case_id}_post_crop.nii.gz"
        
        sitk.WriteImage(pre_cropped_sitk, str(pre_crop_file))
        sitk.WriteImage(post_cropped_sitk, str(post_crop_file))
        
        # Metadata
        metadata = {
            "case_id": case_id,
            "status": "success",
            "pre_original_shape": pre_array.shape,
            "post_original_shape": post_array.shape,
            "crop_shape": crop_size,
            "bbox": bbox,
            "pre_crop_file": str(pre_crop_file),
            "post_crop_file": str(post_crop_file),
            "pre_original_file": str(pre_phase_file),
            "post_original_file": str(post_phase_file),
            "prediction_file": str(prediction_file)
        }
        
        return metadata
        
    except Exception as e:
        return {"status": "error", "message": str(e), "case_id": case_id}

# ============ PROCESAMIENTO PRINCIPAL ============

def process_with_csv_mapping():
    """
    Procesa usando el mapeo del CSV train_test_splits.csv
    GENERA DOS FASES: pre_crop y post_crop para modelo crop tumor
    """
    # Cargar mapeo desde CSV
    try:
        mapping, test_cases = load_test_split_mapping()
    except Exception as e:
        print(f"❌ Error leyendo CSV: {e}")
        return []
    
    print(f"\n🔄 Procesando {len(mapping)} casos del test set (DOS FASES)...")
    
    results = []
    successful = 0
    errors = 0
    not_found_predictions = []
    not_found_originals = []
    
    for case_id, duke_case_name in mapping.items():
        print(f"\n🔄 Procesando {case_id} → {duke_case_name}...")
        
        # 1. Verificar que existe la predicción A3
        pred_file = a3_predictions_dir / f"{case_id}.nii.gz"
        
        if not pred_file.exists():
            print(f"❌ No existe predicción: {pred_file.name}")
            not_found_predictions.append(case_id)
            results.append({"case_id": case_id, "status": "error", "message": "Predicción no encontrada"})
            errors += 1
            continue
        
        # 2. Buscar AMBAS fases del archivo duke original
        pre_file, post_file = find_duke_phases_by_name(duke_case_name, original_images_dir)
        
        if pre_file is None or post_file is None:
            missing_phases = []
            if pre_file is None:
                missing_phases.append("PRE")
            if post_file is None:
                missing_phases.append("POST")
            
            print(f"❌ Fases no encontradas para {duke_case_name}: {missing_phases}")
            not_found_originals.append(f"{duke_case_name} (fases: {missing_phases})")
            results.append({"case_id": case_id, "status": "error", "message": f"Fases {missing_phases} no encontradas: {duke_case_name}"})
            errors += 1
            continue
        
        print(f"   ✅ Archivos encontrados:")
        print(f"      Predicción: {pred_file.name}")
        print(f"      Fase PRE: {pre_file.name}")
        print(f"      Fase POST: {post_file.name}")
        
        # 3. Procesar AMBAS fases
        result = process_a3_prediction_to_two_phase_crops(
            pred_file, pre_file, post_file, output_crops_dir, case_id
        )
        
        # Añadir info del mapeo
        result["duke_case_name"] = duke_case_name
        result["pre_duke_file"] = str(pre_file)
        result["post_duke_file"] = str(post_file)
        
        results.append(result)
        
        if result["status"] == "success":
            successful += 1
            print(f"   ✅ Crops guardados:")
            print(f"      PRE: {result['crop_shape']}")
            print(f"      POST: {result['crop_shape']}")
        else:
            errors += 1
            print(f"   ❌ Error: {result['message']}")
        
        # Progreso cada 20 casos
        if (successful + errors) % 20 == 0:
            print(f"\n📊 Progreso: {successful + errors}/{len(mapping)} - Exitosos: {successful}, Errores: {errors}")
    
    # Guardar metadata
    metadata_file = output_crops_dir / "metadata" / "processing_results.json"
    with open(metadata_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Guardar casos no encontrados
    if not_found_predictions:
        missing_preds_file = output_crops_dir / "metadata" / "missing_predictions.txt"
        with open(missing_preds_file, 'w') as f:
            f.write("Predicciones A3 no encontradas:\n")
            for case in not_found_predictions:
                f.write(f"{case}\n")
    
    if not_found_originals:
        missing_originals_file = output_crops_dir / "metadata" / "missing_originals.txt"
        with open(missing_originals_file, 'w') as f:
            f.write("Archivos duke originales no encontrados:\n")
            for case in not_found_originals:
                f.write(f"{case}\n")
    
    # Resumen final
    print(f"\n{'='*60}")
    print(f"PROCESAMIENTO COMPLETADO")
    print(f"{'='*60}")
    print(f"✅ Exitosos: {successful}")
    print(f"❌ Errores: {errors}")
    print(f"📁 Crops guardados en: {output_crops_dir / 'images'}")
    print(f"📋 Metadata en: {metadata_file}")
    
    if not_found_predictions:
        print(f"⚠️ Predicciones no encontradas: {len(not_found_predictions)}")
        print(f"   Lista en: {missing_preds_file}")
    
    if not_found_originals:
        print(f"⚠️ Originales no encontrados: {len(not_found_originals)}")
        print(f"   Lista en: {missing_originals_file}")
    
    return results

# ============ PREPARACIÓN PARA MODELO CROP TUMOR ============

def prepare_two_phase_crops_for_tumor_model(results, target_size=(128, 128, 128)):
    """
    Prepara los crops de DOS FASES para el modelo de tumor (redimensiona, normaliza)
    Genera: [pre_processed, post_processed] para cada caso
    """
    processed_dir = output_crops_dir / "processed_for_tumor_model"
    processed_dir.mkdir(exist_ok=True)
    
    successful_crops = [r for r in results if r["status"] == "success"]
    
    print(f"\n🔄 Preparando {len(successful_crops)} casos (DOS FASES) para modelo tumor...")
    
    for result in successful_crops:
        case_id = result["case_id"]
        pre_crop_file = Path(result["pre_crop_file"])
        post_crop_file = Path(result["post_crop_file"])
        
        try:
            # Cargar ambos crops
            pre_sitk = sitk.ReadImage(str(pre_crop_file))
            post_sitk = sitk.ReadImage(str(post_crop_file))
            
            pre_array = sitk.GetArrayFromImage(pre_sitk)
            post_array = sitk.GetArrayFromImage(post_sitk)
            
            print(f"   📋 {case_id}: PRE {pre_array.shape}, POST {post_array.shape}")
            
            # Procesar cada fase
            processed_phases = []
            
            for phase_name, phase_array in [("PRE", pre_array), ("POST", post_array)]:
                # Redimensionar a tamaño fijo (si tu modelo lo requiere)
                if target_size:
                    # Usar interpolación para redimensionar
                    zoom_factors = [
                        target_size[i] / phase_array.shape[i] 
                        for i in range(3)
                    ]
                    
                    resized_array = ndimage.zoom(phase_array, zoom_factors, order=1)
                else:
                    resized_array = phase_array
                
                # Normalización Z-score independiente para cada fase
                mean_val = np.mean(resized_array)
                std_val = np.std(resized_array)
                
                if std_val > 0:
                    normalized_array = (resized_array - mean_val) / std_val
                else:
                    normalized_array = resized_array - mean_val
                
                processed_phases.append(normalized_array.astype(np.float32))
            
            # Guardar ambas fases procesadas
            pre_processed_sitk = sitk.GetImageFromArray(processed_phases[0])
            post_processed_sitk = sitk.GetImageFromArray(processed_phases[1])
            
            pre_processed_file = processed_dir / f"{case_id}_pre_processed.nii.gz"
            post_processed_file = processed_dir / f"{case_id}_post_processed.nii.gz"
            
            sitk.WriteImage(pre_processed_sitk, str(pre_processed_file))
            sitk.WriteImage(post_processed_sitk, str(post_processed_file))
            
            print(f"   ✅ {case_id}: {pre_array.shape} → {processed_phases[0].shape} (ambas fases)")
            
        except Exception as e:
            print(f"   ❌ {case_id}: Error - {e}")
    
    print(f"📁 Crops procesados guardados en: {processed_dir}")
    print(f"📋 Cada caso tiene: case_XXX_pre_processed.nii.gz y case_XXX_post_processed.nii.gz")
    return processed_dir

# ============ EJECUCIÓN ============
if __name__ == "__main__":
    print("=== VERIFICANDO CONFIGURACIÓN ===")
    
    # Verificar rutas
    if not a3_predictions_dir.exists():
        print(f"❌ No existe: {a3_predictions_dir}")
        print("👉 Ajusta la variable 'a3_predictions_dir' en el código")
        exit()
    
    if not original_images_dir.exists():
        print(f"❌ No existe: {original_images_dir}")
        print("👉 Ajusta la variable 'original_images_dir' en el código")
        exit()
        
    if not splits_csv.exists():
        print(f"❌ No existe CSV: {splits_csv}")
        print("👉 Ajusta la variable 'splits_csv' en el código")
        exit()
    
    # Verificar predicciones A3
    prediction_files = list(a3_predictions_dir.glob("case_*.nii.gz"))
    print(f"🔍 Encontradas {len(prediction_files)} predicciones A3")
    
    if len(prediction_files) == 0:
        print(f"❌ No se encontraron archivos case_*.nii.gz en {a3_predictions_dir}")
        exit()
    
    # Mostrar algunos ejemplos
    print(f"   Ejemplos: {[f.name for f in prediction_files[:5]]}")
    if len(prediction_files) > 5:
        print(f"   ... y {len(prediction_files) - 5} más")
    
    print(f"✅ Configuración OK!")
    
    # Procesar usando mapeo CSV
    print(f"\n🚀 INICIANDO PROCESAMIENTO CON MAPEO CSV")
    
    results = process_with_csv_mapping()
    
    # Solo preparar para modelo tumor si hay casos exitosos
    successful_cases = [r for r in results if r["status"] == "success"]
    
    if successful_cases:
        print(f"\n✅ {len(successful_cases)} casos procesados exitosamente")
        
        # Preparar DOS FASES para modelo tumor
        processed_dir = prepare_two_phase_crops_for_tumor_model(results)
        
        print(f"\n🎯 LISTO PARA MODELO CROP TUMOR (DOS FASES)!")
        print(f"📁 Usar crops de: {processed_dir}")
        print(f"\n📋 Para cada caso tienes:")
        print(f"   - case_XXX_pre_processed.nii.gz")
        print(f"   - case_XXX_post_processed.nii.gz")
        print(f"\nPróximo paso - cargar modelo:")
        print(f"   pre_crop = sitk.ReadImage('case_000_pre_processed.nii.gz')")
        print(f"   post_crop = sitk.ReadImage('case_000_post_processed.nii.gz')")
        print(f"   prediction = crop_tumor_model([pre_crop, post_crop])")
    else:
        print(f"\n❌ No se pudo procesar ningún caso exitosamente")
        print(f"👉 Revisar errores y estructura de archivos")