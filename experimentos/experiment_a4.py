#!/usr/bin/env python3
"""
EXPERIMENTO A4: TEMPORAL [Pre, Post1, Post2, Post3]
Secuencia temporal completa
"""

import numpy as np
import SimpleITK as sitk
from pathlib import Path
import json
import shutil

def create_experiment_A4(base_dir, output_dir="Dataset105_A4_Temporal", num_cases=None):
    """
    EXPERIMENTO A4: Temporal [Pre, Post1, Post2, Post3]
    """
    print("=== EXPERIMENTO A4: TEMPORAL [Pre, Post1, Post2, Post3] ===")
    
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
        seg_file = seg_dir / f"{patient_id}.nii.gz"
        
        if not seg_file.exists():
            continue
            
        try:
            print(f"Procesando A4: {patient_id} ({processed+1})")
            
            # Buscar todas las fases disponibles
            phases = []
            for phase in ["0000", "0001", "0002", "0003", "0004"]:
                phase_file = images_dir / f"{patient_id}_{phase}.nii.gz"
                if phase_file.exists():
                    phases.append((phase_file, phase))
            
            if len(phases) < 2:
                print(f"  Skipping {patient_id}: solo {len(phases)} fases")
                continue
            
            print(f"  Encontradas {len(phases)} fases: {[p[1] for p in phases]}")
            
            # Procesar y guardar cada fase
            reference_sitk = None
            for i, (phase_file, phase_name) in enumerate(phases):
                # Cargar y normalizar
                phase_sitk = sitk.Cast(sitk.ReadImage(str(phase_file)), sitk.sitkFloat32)
                phase_array = sitk.GetArrayFromImage(phase_sitk)
                
                # Z-score normalization
                mask = phase_array > 0
                if np.sum(mask) > 0:
                    mean_val = np.mean(phase_array[mask])
                    std_val = np.std(phase_array[mask])
                    if std_val > 0:
                        phase_array = (phase_array - mean_val) / std_val
                
                # Guardar
                result_sitk = sitk.GetImageFromArray(phase_array.astype(np.float32))
                result_sitk.CopyInformation(phase_sitk)
                
                # Usar el nombre de fase original para nnU-Net
                output_file = output_path / "imagesTr" / f"case_{processed:03d}_{phase_name}.nii.gz"
                sitk.WriteImage(result_sitk, str(output_file))
                
                if reference_sitk is None:
                    reference_sitk = phase_sitk
            
            # Copiar segmentación
            shutil.copy2(seg_file, output_path / "labelsTr" / f"case_{processed:03d}.nii.gz")
            
            processed += 1
            print(f"  SUCCESS: {patient_id} con {len(phases)} fases")
            
        except Exception as e:
            print(f"Error A4 {patient_id}: {e}")
            continue
    
    # Dataset JSON - Hasta 5 canales posibles
    channel_names = {
        "0": "Pre_Contrast",
        "1": "Post1_Contrast",
        "2": "Post2_Contrast", 
        "3": "Post3_Contrast",
        "4": "Post4_Contrast"
    }
    
    dataset_json = {
        "channel_names": channel_names,
        "labels": {"background": 0, "tumor": 1},
        "numTraining": processed,
        "file_ending": ".nii.gz",
        "dataset_name": "A4_Temporal_Sequence",
        "description": "Temporal sequence: Pre + Post1 + Post2 + Post3 + Post4"
    }
    
    with open(output_path / "dataset.json", 'w') as f:
        json.dump(dataset_json, f, indent=2)
    
    print(f"A4 COMPLETADO: {processed} casos en {output_path}")
    return processed

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str, required=True)
    parser.add_argument("--num_cases", type=int, default=None)
    args = parser.parse_args()
    
    create_experiment_A4(args.base_dir, num_cases=args.num_cases)
    print("LISTO PARA: nnUNetv2_plan_and_preprocess -d 105")
    print("ENTRENAR: nnUNetv2_train 105 3d_fullres 0")