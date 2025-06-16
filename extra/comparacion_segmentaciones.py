import os
import csv
import time
import glob
import numpy as np
import slicer

# Rutas
BASE_DIR = "C:/Users/FX507/Documents/GitHub/Mama_Mia/datos"
IMAGES_DIR = os.path.join(BASE_DIR, "images")
AUTO_SEG_DIR = os.path.join(BASE_DIR, "segmentations/automatic")
EXPERT_SEG_DIR = os.path.join(BASE_DIR, "segmentations/expert")
RESULTS_FILE = os.path.join(BASE_DIR, "resultados_ROI_automatizado.csv")

def get_bounding_box_mask(mask1, mask2):
    combined = np.logical_or(mask1, mask2)
    coords = np.argwhere(combined)
    if coords.size == 0:
        return None
    zmin, ymin, xmin = coords.min(axis=0)
    zmax, ymax, xmax = coords.max(axis=0) + 1
    return slice(zmin, zmax), slice(ymin, ymax), slice(xmin, xmax)

def procesarPaciente(patient_id):
    print(f"Procesando paciente: {patient_id}")
    slicer.mrmlScene.Clear(0)

    ruta_imagen = os.path.join(IMAGES_DIR, f"{patient_id}_0001.nii.gz")
    ruta_seg_auto = os.path.join(AUTO_SEG_DIR, f"{patient_id}.nii.gz")
    ruta_seg_expert = os.path.join(EXPERT_SEG_DIR, f"{patient_id}.nii.gz")

    if not os.path.exists(ruta_imagen):
        print(f"  ADVERTENCIA: Imagen no encontrada: {ruta_imagen}")
        return None
    if not os.path.exists(ruta_seg_auto):
        print(f"  ADVERTENCIA: Segmentación automática no encontrada: {ruta_seg_auto}")
        return None
    if not os.path.exists(ruta_seg_expert):
        print(f"  ADVERTENCIA: Segmentación experta no encontrada: {ruta_seg_expert}")
        return None

    try:
        imagen = slicer.util.loadVolume(ruta_imagen)
        auto_node = slicer.util.loadLabelVolume(ruta_seg_auto)
        expert_node = slicer.util.loadLabelVolume(ruta_seg_expert)

        array_auto = slicer.util.arrayFromVolume(auto_node)
        array_expert = slicer.util.arrayFromVolume(expert_node)

        mask_auto = array_auto > 0
        mask_expert = array_expert > 0

        roi_slices = get_bounding_box_mask(mask_auto, mask_expert)
        if roi_slices is None:
            print("  Segmentaciones vacías, se omite.")
            return None

        z, y, x = roi_slices
        roi_auto = mask_auto[z, y, x]
        roi_expert = mask_expert[z, y, x]

        intersection = np.logical_and(roi_auto, roi_expert).sum()
        size_auto = roi_auto.sum()
        size_expert = roi_expert.sum()
        total_voxels = roi_auto.size

        dice = 2.0 * intersection / (size_auto + size_expert) if (size_auto + size_expert) > 0 else 0.0

        true_positives = intersection
        false_positives = size_auto - intersection
        false_negatives = size_expert - intersection
        true_negatives = total_voxels - true_positives - false_positives - false_negatives

        tp_percent = (true_positives / total_voxels) * 100
        tn_percent = (true_negatives / total_voxels) * 100
        fp_percent = (false_positives / total_voxels) * 100
        fn_percent = (false_negatives / total_voxels) * 100

        spacing = expert_node.GetSpacing()
        voxel_volume_cm3 = (spacing[0] * spacing[1] * spacing[2]) / 1000.0

        volume_auto = size_auto * voxel_volume_cm3
        volume_expert = size_expert * voxel_volume_cm3

        slicer.mrmlScene.RemoveNode(imagen)
        slicer.mrmlScene.RemoveNode(auto_node)
        slicer.mrmlScene.RemoveNode(expert_node)

        return {
            'paciente_id': patient_id,
            'dice': round(dice, 4),
            'true_positives': round(tp_percent, 3),
            'true_negatives': round(tn_percent, 3),
            'false_positives': round(fp_percent, 3),
            'false_negatives': round(fn_percent, 3),
            'reference_volume': round(volume_expert, 3),
            'compare_volume': round(volume_auto, 3)
        }

    except Exception as e:
        print(f"  ERROR al procesar {patient_id}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def obtenerListaPacientes():
    expert_segmentations = glob.glob(os.path.join(EXPERT_SEG_DIR, "*.nii.gz"))
    return [os.path.basename(f).replace(".nii.gz", "") for f in expert_segmentations]

def main():
    start = time.time()
    print("Iniciando procesamiento automatizado (solo ROI)")

    patient_ids = obtenerListaPacientes()
    print(f"Pacientes detectados: {len(patient_ids)}")


    with open(RESULTS_FILE, 'w', newline='') as csv_file:
        campos = [
            'paciente_id', 'dice',
            'true_positives', 'true_negatives',
            'false_positives', 'false_negatives',
            'reference_volume', 'compare_volume'
        ]
        writer = csv.DictWriter(csv_file, fieldnames=campos)
        writer.writeheader()

        for patient_id in patient_ids:
            resultado = procesarPaciente(patient_id)
            if resultado:
                writer.writerow(resultado)

    duracion = time.time() - start
    print(f"Finalizado en {duracion:.2f} segundos.")
    print(f"Resultados guardados en: {RESULTS_FILE}")

if __name__ == "__main__":
    main()
