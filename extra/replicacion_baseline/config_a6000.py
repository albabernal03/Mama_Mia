import os
import torch

# Configurar variables de entorno nnU-Net
base_path = r"C:\Users\usuario\Documents\Mama_Mia"
os.environ['nnUNet_raw_data_base'] = os.path.join(base_path, "nnUNet_raw_data_base")
os.environ['nnUNet_preprocessed'] = os.path.join(base_path, "nnUNet_preprocessed") 
os.environ['nnUNet_RESULTS_FOLDER'] = os.path.join(base_path, "nnUNet_results")

# Crear carpetas nnU-Net si no existen
for path in [os.environ['nnUNet_raw_data_base'], 
             os.environ['nnUNet_preprocessed'], 
             os.environ['nnUNet_RESULTS_FOLDER']]:
    os.makedirs(path, exist_ok=True)

# RUTAS PRINCIPALES
MAMA_MIA_ROOT = r"C:\Users\usuario\Documents\Mama_Mia\datos"
WEIGHTS_PATH = r"C:\Users\usuario\Documents\Mama_Mia\mama_mia_weights\extracted\Dataset501_full_image_dce_mri_tumor_segmentation"

# Rutas de datos
IMAGES_PATH = os.path.join(MAMA_MIA_ROOT, "images")
EXPERT_SEGS_PATH = os.path.join(MAMA_MIA_ROOT, "segmentations", "expert")
AUTO_SEGS_PATH = os.path.join(MAMA_MIA_ROOT, "segmentations", "automatic")
SPLITS_CSV = os.path.join(MAMA_MIA_ROOT, "train_test_splits.csv")

# Configuraciones optimizadas
INFERENCE_CONFIG = {
    'batch_size_per_gpu': 8,
    'mixed_precision': True,
    'num_threads_preprocessing': 32,
    'num_threads_nifti_save': 16,
}

# Carpetas temporales
TEMP_INPUT = "./temp_input"
RESULTS_OUTPUT = "./results_output"

def setup_torch_a6000():
    """Configuración optimizada para RTX A6000"""
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    torch.cuda.empty_cache()
    
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
    
    print("🚀 PyTorch configurado para 4x RTX A6000")
    for i in range(4):
        props = torch.cuda.get_device_properties(i)
        print(f"GPU {i}: {props.name} - {props.total_memory/1e9:.1f}GB")

def convert_case_id(case_id):
    """Convierte DUKE_019 a duke_019 para buscar archivos"""
    return case_id.lower()

def find_case_files(case_id):
    """Encuentra todos los archivos de un caso (múltiples fases)"""
    case_lower = convert_case_id(case_id)
    
    if not os.path.exists(IMAGES_PATH):
        return []
    
    # Buscar todos los archivos que empiecen con el case_id
    all_files = os.listdir(IMAGES_PATH)
    case_files = [f for f in all_files if f.startswith(case_lower) and f.endswith('.nii.gz')]
    case_files.sort()  # Ordenar por fase
    
    return case_files

def get_post_contrast_file(case_id):
    """Obtiene el archivo de primera fase post-contraste"""
    case_files = find_case_files(case_id)
    
    if not case_files:
        return None
    
    # Buscar primera fase post-contraste (generalmente _0001.nii.gz)
    for file in case_files:
        if '_0001.nii.gz' in file:  # Primera post-contraste
            return os.path.join(IMAGES_PATH, file)
    
    # Si no hay _0001, usar el segundo archivo disponible
    if len(case_files) >= 2:
        return os.path.join(IMAGES_PATH, case_files[1])
    
    # Fallback: usar el primer archivo
    if case_files:
        return os.path.join(IMAGES_PATH, case_files[0])
    
    return None

def find_expert_segmentation(case_id):
    """Encuentra la segmentación experta para un caso"""
    case_lower = convert_case_id(case_id)
    
    # Buscar archivo de segmentación experta
    expert_file = f"{case_lower}.nii.gz"
    expert_path = os.path.join(EXPERT_SEGS_PATH, expert_file)
    
    if os.path.exists(expert_path):
        return expert_path
    
    # Buscar variaciones del nombre
    if os.path.exists(EXPERT_SEGS_PATH):
        all_segs = os.listdir(EXPERT_SEGS_PATH)
        for seg_file in all_segs:
            if case_lower in seg_file.lower():
                return os.path.join(EXPERT_SEGS_PATH, seg_file)
    
    return None