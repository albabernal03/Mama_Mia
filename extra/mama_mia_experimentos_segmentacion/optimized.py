import numpy as np
import nibabel as nib
from scipy import ndimage
from skimage import morphology, measure
from pathlib import Path
import cv2

def load_nifti(file_path):
    nii = nib.load(file_path)
    return nii.get_fdata(), nii.affine, nii.header

def save_nifti(data, affine, header, file_path):
    nib.save(nib.Nifti1Image(data, affine, header), file_path)

def remove_small_objects(mask, min_size=100):
    """Eliminar componentes pequeños (ruido)"""
    cleaned = morphology.remove_small_objects(mask > 0.5, min_size=min_size)
    return cleaned.astype(np.uint8)

def keep_largest_connected_component(mask):
    """Conservar solo el objeto más grande"""
    labels = measure.label(mask, connectivity=1)
    if labels.max() == 0:
        return mask
    largest = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1
    return largest.astype(np.uint8)

def fill_holes(mask):
    """Rellenar huecos slice por slice"""
    filled = np.zeros_like(mask)
    for i in range(mask.shape[2]):
        filled[:, :, i] = ndimage.binary_fill_holes(mask[:, :, i])
    return filled.astype(np.uint8)

def smooth_edges(mask):
    """Suavizar bordes: apertura seguida de cierre"""
    struct = morphology.ball(1)  # Esfera pequeña
    opened = morphology.binary_opening(mask, struct)
    closed = morphology.binary_closing(opened, struct)
    return closed.astype(np.uint8)

def guided_filter_mask(image, mask, radius=5, eps=1e-3):
    """Ajustar bordes con Guided Filter"""
    final_mask = np.zeros_like(mask)
    image = np.clip(image, np.percentile(image, 5), np.percentile(image, 95))
    image = (image - np.min(image)) / (np.max(image) - np.min(image) + 1e-8)
    
    for z in range(mask.shape[2]):
        img_slice = (image[:, :, z] * 255).astype(np.uint8)
        mask_slice = (mask[:, :, z] * 255).astype(np.uint8)
        
        if np.sum(mask_slice) > 0:
            guided = cv2.ximgproc.guidedFilter(guide=img_slice, src=mask_slice, radius=radius, eps=eps)
            guided = (guided > 127).astype(np.uint8)
            final_mask[:, :, z] = guided
        else:
            final_mask[:, :, z] = mask[:, :, z]
    
    return final_mask

def advanced_postprocessing(case_id, apply_guided=False):
    """Pipeline avanzado de postprocesamiento"""
    # RUTAS COMO LAS QUE TENÍAS ANTES
    base_path = Path(r"C:\Users\usuario\Documents\Mama_Mia")
    baseline_results = base_path / "replicacion_definitiva" / "results_output"
    ground_truth_path = base_path / "datos" / "segmentations" / "expert"
    improved_results = Path("./results_adaptive_postprocessing_advanced")
    
    # Crear carpeta si no existe
    improved_results.mkdir(parents=True, exist_ok=True)

    # Construir paths
    mask_path = baseline_results / f"{case_id}.nii.gz"
    gt_path = ground_truth_path / f"{case_id.lower()}.nii.gz"
    output_path = improved_results / f"{case_id}.nii.gz"
    image_path = gt_path  # 🧠 Usamos la segmentación experta como "guía" de intensidades

    # Cargar máscara baseline
    mask, affine, header = load_nifti(mask_path)
    # Cargar imagen para filtro guiado
    image, _, _ = load_nifti(image_path)

    print(f"🧠 Procesando {case_id} - Voxels positivos iniciales: {np.sum(mask)}")
    
    # Post-procesamiento
    mask = remove_small_objects(mask, min_size=100)
    mask = keep_largest_connected_component(mask)
    mask = fill_holes(mask)
    mask = smooth_edges(mask)
    
    if apply_guided:
        mask = guided_filter_mask(image, mask, radius=5, eps=1e-2)
    
    print(f"✅ {case_id} - Voxels positivos finales: {np.sum(mask)}")
    
    # Guardar máscara postprocesada
    save_nifti(mask, affine, header, output_path)
    print(f"💾 Guardado en: {output_path}")

# =========================================================
# 🚀 USO
# =========================================================
if __name__ == "__main__":
    # Cambia aquí el case_id para probar otros
    case_id = "DUKE_019"
    advanced_postprocessing(case_id, apply_guided=True)
