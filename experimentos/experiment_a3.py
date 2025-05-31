#!/usr/bin/env python3
"""
EXPERIMENTO A3: MULTI-CANAL [Pre, Post1, Subtraction]
Tres canales de entrada
"""

import numpy as np
import SimpleITK as sitk
from pathlib import Path
import json
import shutil

def create_experiment_A3(base_dir, output_dir="Dataset103_A3_MultiChannel", num_cases=None):
    """
    EXPERIMENTO A3: Multi-canal [Pre, Post1, Resta]
    """
    print("=== EXPERIMENTO A3: MULTI-CANAL [Pre, Post1, Resta] ===")
    
    base_path = Path(base_dir)
    images_dir = base_path / "images"
    seg_dir = base_path / "segmentations" / "expert"
    
    # Crear estructura nnU-Net
    output_path = Path(output_dir)
    (output_path / "imagesTr").mkdir(parents=True, exist_ok=True)
    (output_path / "labelsTr").mkdir(parents=True, exist_ok=True)
    
    # Obtener pacientes
    files_0000 = sorted(list(images_dir.glob("*_0000.nii.gz")))
    if num_cases:
        files_0000 = files_0000[:num_cases]
    
    processed = 0
    for file in files_0000:
        patient_id = file.name.replace("_0000.nii.gz", "")
        post_file = images_dir / f"{patient_id}_0001.nii.gz"
        seg_file = seg_dir / f"{patient_id}.nii.gz"
        
        if not post_file.exists() or not seg_file.exists():
            continue
            
        try:
            print(f"Procesando A3: {patient_id} ({processed+1})")
            
            # Cargar imágenes
            pre_sitk = sitk.Cast(sitk.ReadImage(str(file)), sitk.sitkFloat32)
            post_sitk = sitk.Cast(sitk.ReadImage(str(post_file)), sitk.sitkFloat32)
            
            pre_array = sitk.GetArrayFromImage(pre_sitk)
            post_array = sitk.GetArrayFromImage(post_sitk) 
            sub_array = post_array - pre_array
            
            # Normalizar cada canal por separado
            def normalize_channel(array):
                mask = array > 0
                if np.sum(mask) > 0:
                    mean_val = np.mean(array[mask])
                    std_val = np.std(array[mask])
                    if std_val > 0:
                        return (array - mean_val) / std_val
                return array
            
            pre_norm = normalize_channel(pre_array)
            post_norm = normalize_channel(post_array)
            sub_norm = normalize_channel(sub_array)
            
            # Guardar 3 canales
            for i, (array, suffix) in enumerate([(pre_norm, "0000"), (post_norm, "0001"), (sub_norm, "0002")]):
                result_sitk = sitk.GetImageFromArray(array.astype(np.float32))
                result_sitk.CopyInformation(post_sitk)
                
                output_file = output_path / "imagesTr" / f"case_{processed:03d}_{suffix}.nii.gz"
                sitk.WriteImage(result_sitk, str(output_file))
            
            # Copiar segmentación
            shutil.copy2(seg_file, output_path / "labelsTr" / f"case_{processed:03d}.nii.gz")
            
            processed += 1
            
        except Exception as e:
            print(f"Error A3 {patient_id}: {e}")
            continue
    
    # Dataset JSON
    dataset_json = {
        "channel_names": {
            "0": "Pre_Contrast",
            "1": "Post_Contrast", 
            "2": "Subtraction"
        },
        "labels": {"background": 0, "tumor": 1},
        "numTraining": processed,
        "file_ending": ".nii.gz",
        "dataset_name": "A3_Multi_Channel",
        "description": "Multi-channel: Pre + Post1 + Subtraction"
    }
    
    with open(output_path / "dataset.json", 'w') as f:
        json.dump(dataset_json, f, indent=2)
    
    print(f"A3 COMPLETADO: {processed} casos en {output_path}")
    return processed

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str, required=True)
    parser.add_argument("--num_cases", type=int, default=None)
    args = parser.parse_args()
    
    create_experiment_A3(args.base_dir, num_cases=args.num_cases)
    print("LISTO PARA: nnUNetv2_plan_and_preprocess -d 103")
    print("ENTRENAR: nnUNetv2_train 103 3d_fullres 0")