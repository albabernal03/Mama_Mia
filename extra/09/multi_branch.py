import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from sklearn.metrics import roc_auc_score, accuracy_score, balanced_accuracy_score
import cv2
from scipy import ndimage
import warnings
import pandas as pd
import nibabel as nib
from pathlib import Path
warnings.filterwarnings('ignore')

# ============================================================================
# 1. MULTI-REGION DATA EXTRACTION
# ============================================================================

class MultiRegionExtractor:
    """Extrae múltiples regiones de interés usando las máscaras cropped"""
    
    def __init__(self, margin_sizes=[0, 5, 10]):
        self.margin_sizes = margin_sizes  # mm margins for periphery regions
        
    def extract_tumor_core(self, dcemri_volume, mask):
        """Extrae la región tumoral central"""
        # Asegurar que mask es binaria
        binary_mask = (mask > 0.5).astype(np.uint8)
        
        # Aplicar máscara a todas las fases DCE
        tumor_roi = dcemri_volume * binary_mask[np.newaxis, ...]
        
        return tumor_roi
    
    def extract_tumor_periphery(self, dcemri_volume, mask, margin_mm=10, voxel_size=1.0):
        """Extrae la periferia tumoral (anillo alrededor del tumor)"""
        binary_mask = (mask > 0.5).astype(np.uint8)
        
        # Calcular dilatación en voxels
        margin_voxels = int(margin_mm / voxel_size)
        
        # Crear estructura para dilatación 3D
        struct = ndimage.generate_binary_structure(3, 1)
        
        # Dilatar máscara
        dilated_mask = ndimage.binary_dilation(
            binary_mask, 
            structure=struct, 
            iterations=margin_voxels
        ).astype(np.uint8)
        
        # Periferia = dilatada - original
        periphery_mask = dilated_mask - binary_mask
        
        # Aplicar a volumen DCE
        periphery_roi = dcemri_volume * periphery_mask[np.newaxis, ...]
        
        return periphery_roi
    
    def extract_enhancement_region(self, dcemri_volume, mask, enhancement_threshold=0.3):
        """Extrae región con mayor enhancement (wash-in patterns)"""
        if dcemri_volume.shape[0] < 2:
            return self.extract_tumor_core(dcemri_volume, mask)
        
        # Calcular enhancement = (post-contrast - pre-contrast) / pre-contrast
        pre_contrast = dcemri_volume[0]  # Primera fase
        post_contrast = dcemri_volume[1]  # Segunda fase
        
        # Evitar división por cero
        enhancement = np.divide(
            post_contrast - pre_contrast,
            pre_contrast + 1e-8,
            out=np.zeros_like(pre_contrast),
            where=(pre_contrast > 0)
        )
        
        # Crear máscara de enhancement
        enhancement_mask = (enhancement > enhancement_threshold).astype(np.uint8)
        
        # Combinar con máscara tumoral
        binary_tumor_mask = (mask > 0.5).astype(np.uint8)
        combined_mask = enhancement_mask * binary_tumor_mask
        
        # Aplicar a volumen
        enhancement_roi = dcemri_volume * combined_mask[np.newaxis, ...]
        
        return enhancement_roi

# ============================================================================
# 2. PREPARACIÓN DE DATOS DESDE CSVs
# ============================================================================

def load_patient_data(splits_csv, pcr_labels_csv, cropped_data_dir):
    """Carga información de pacientes desde CSVs"""
    
    # Leer CSVs
    splits_df = pd.read_csv(splits_csv)
    pcr_df = pd.read_csv(pcr_labels_csv)
    
    # Crear diccionario de etiquetas pCR
    pcr_dict = {}
    for _, row in pcr_df.iterrows():
        patient_id = str(row['patient_id']).strip().upper()  # Normalizar IDs
        pcr_label = int(row['pcr_response'])  # Asumir columna 'pCR' con 0/1
        pcr_dict[patient_id] = pcr_label
    
    train_data = []
    test_data = []
    
    # Procesar splits
    for _, row in splits_df.iterrows():
        # Train split
        if pd.notna(row['train_split']):
            patient_id = str(row['train_split']).strip().upper()
            if patient_id in pcr_dict:
                patient_path = Path(cropped_data_dir) / "images" / "train_split" / patient_id
                if patient_path.exists():
                    train_data.append({
                        'patient_id': patient_id,
                        'path': patient_path,
                        'pcr_label': pcr_dict[patient_id],
                        'split': 'train'
                    })
        
        # Test split  
        if pd.notna(row['test_split']):
            patient_id = str(row['test_split']).strip().upper()
            if patient_id in pcr_dict:
                patient_path = Path(cropped_data_dir) / "images" / "test_split" / patient_id
                if patient_path.exists():
                    test_data.append({
                        'patient_id': patient_id,
                        'path': patient_path,
                        'pcr_label': pcr_dict[patient_id],
                        'split': 'test'
                    })
    
    print(f"📊 Datos cargados:")
    print(f"   Train: {len(train_data)} pacientes")
    print(f"   Test: {len(test_data)} pacientes")
    
    # Mostrar distribución de clases
    train_pcr = sum(d['pcr_label'] for d in train_data)
    test_pcr = sum(d['pcr_label'] for d in test_data)
    
    print(f"📈 Distribución pCR:")
    print(f"   Train: {train_pcr}/{len(train_data)} ({train_pcr/len(train_data)*100:.1f}% pCR)")
    print(f"   Test: {test_pcr}/{len(test_data)} ({test_pcr/len(test_data)*100:.1f}% pCR)")
    
    return train_data, test_data

# ============================================================================
# 3. DATASET MULTI-REGIÓN ADAPTADO
# ============================================================================

class MultiRegionDCEDataset(Dataset):
    """Dataset que carga múltiples regiones para cada caso desde estructura cropped"""
    
    def __init__(self, patient_data, cropped_data_dir, transform=None, target_size=(64, 64, 32)):
        self.patient_data = patient_data
        self.cropped_data_dir = Path(cropped_data_dir)
        self.transform = transform
        self.target_size = target_size
        self.region_extractor = MultiRegionExtractor()
        
        # Filtrar pacientes válidos
        self.valid_patients = self._validate_patients()
        print(f"✅ {len(self.valid_patients)}/{len(patient_data)} pacientes válidos encontrados")
        
    def _validate_patients(self):
        """Valida que los pacientes tengan datos DCE completos"""
        valid_patients = []
        
        for patient_info in self.patient_data:
            patient_id = patient_info['patient_id']
            split = patient_info['split']
            
            # Buscar archivos DCE del paciente
            patient_img_dir = self.cropped_data_dir / "images" / f"{split}_split" / patient_id
            patient_seg_dir = self.cropped_data_dir / "segmentations" / f"{split}_split" / patient_id
            
            if not patient_img_dir.exists() or not patient_seg_dir.exists():
                continue
                
            # Buscar archivos DCE (formato: duke_xxx_000X_cropped.nii.gz)
            dce_files = list(patient_img_dir.glob(f"{patient_id.lower()}_*_cropped.nii.gz"))
            
            # Buscar archivo de segmentación
            seg_files = list(patient_seg_dir.glob(f"{patient_id.lower()}_seg_cropped.nii.gz"))
            
            if len(dce_files) >= 2 and len(seg_files) == 1:  # Mínimo 2 fases DCE
                # Ordenar archivos DCE por número de fase
                dce_files_sorted = sorted(dce_files, 
                    key=lambda x: int(x.stem.split('_')[-2]))
                
                patient_info['dce_files'] = dce_files_sorted
                patient_info['seg_file'] = seg_files[0]
                valid_patients.append(patient_info)
            else:
                print(f"⚠️  {patient_id}: DCE files: {len(dce_files)}, Seg files: {len(seg_files)}")
        
        return valid_patients
        
    def __len__(self):
        return len(self.valid_patients)
    
    def load_dcemri_data(self, patient_info):
        """Carga datos DCE-MRI desde archivos .nii.gz croppeados"""
        
        dce_files = patient_info['dce_files']
        seg_file = patient_info['seg_file']
        
        # Cargar fases DCE
        dcemri_phases = []
        
        for dce_file in dce_files:
            nii_img = nib.load(dce_file)
            img_data = nii_img.get_fdata().astype(np.float32)
            dcemri_phases.append(img_data)
        
        # Stack fases: (n_phases, H, W, D)
        dcemri_volume = np.stack(dcemri_phases, axis=0)
        
        # Cargar segmentación
        seg_nii = nib.load(seg_file)
        mask = seg_nii.get_fdata().astype(np.float32)
        
        # Verificar dimensiones
        if dcemri_volume.shape[1:] != mask.shape:
            raise ValueError(f"Shape mismatch: DCE {dcemri_volume.shape[1:]} vs Mask {mask.shape}")
        
        return dcemri_volume, mask
    
    def resize_volume(self, volume, target_size):
        """Redimensiona volumen a tamaño objetivo"""
        if volume.ndim == 4:  # (phases, H, W, D)
            resized_phases = []
            for phase in volume:
                resized_phase = self.resize_3d(phase, target_size)
                resized_phases.append(resized_phase)
            return np.stack(resized_phases)
        else:  # (H, W, D)
            return self.resize_3d(volume, target_size)
    
    def resize_3d(self, volume_3d, target_size):
        """Redimensiona volumen 3D usando interpolación"""
        from scipy.ndimage import zoom
        
        current_size = volume_3d.shape
        zoom_factors = [t/c for t, c in zip(target_size, current_size)]
        
        return zoom(volume_3d, zoom_factors, order=1)
    
    def normalize_phases(self, dcemri_volume, target_phases=6):
        """Normaliza cada fase por separado y asegura número consistente de fases"""
        
        current_phases = dcemri_volume.shape[0]
        
        # Ajustar número de fases si es necesario
        if current_phases < target_phases:
            # Repetir última fase si faltan
            last_phase = dcemri_volume[-1:]
            repeats_needed = target_phases - current_phases
            repeated_phases = np.repeat(last_phase, repeats_needed, axis=0)
            dcemri_volume = np.concatenate([dcemri_volume, repeated_phases], axis=0)
        elif current_phases > target_phases:
            # Tomar solo las primeras fases
            dcemri_volume = dcemri_volume[:target_phases]
        
        # Normalizar cada fase por separado (Z-score)
        normalized_phases = []
        
        for i, phase in enumerate(dcemri_volume):
            # Z-score normalization por fase
            mean_val = np.mean(phase)
            std_val = np.std(phase)
            
            if std_val > 1e-8:  # Evitar división por cero
                normalized_phase = (phase - mean_val) / std_val
            else:
                normalized_phase = phase - mean_val
            
            # Clip valores extremos
            normalized_phase = np.clip(normalized_phase, -5, 5)
            
            normalized_phases.append(normalized_phase)
        
        return np.stack(normalized_phases)
    
    def __getitem__(self, idx):
        patient_info = self.valid_patients[idx]
        label = patient_info['pcr_label']
        
        try:
            # Cargar datos DCE-MRI del paciente
            dcemri_volume, mask = self.load_dcemri_data(patient_info)
            
            # Redimensionar
            dcemri_volume = self.resize_volume(dcemri_volume, self.target_size)
            mask = self.resize_volume(mask, self.target_size)
            
            # Normalizar por fases (asegurar 6 fases)
            dcemri_volume = self.normalize_phases(dcemri_volume, target_phases=6)
            
            # Extraer múltiples regiones
            tumor_core = self.region_extractor.extract_tumor_core(dcemri_volume, mask)
            tumor_periphery = self.region_extractor.extract_tumor_periphery(dcemri_volume, mask)
            enhancement_region = self.region_extractor.extract_enhancement_region(dcemri_volume, mask)
            
            # Convertir a tensores
            tumor_core = torch.FloatTensor(tumor_core)
            tumor_periphery = torch.FloatTensor(tumor_periphery)
            enhancement_region = torch.FloatTensor(enhancement_region)
            
            # Aplicar transformaciones si existen
            if self.transform:
                # Nota: Las transformaciones deben ser compatibles con datos 3D+tiempo
                pass
            
            return {
                'tumor_core': tumor_core,
                'tumor_periphery': tumor_periphery, 
                'enhancement_region': enhancement_region,
                'label': torch.FloatTensor([label]),
                'patient_id': patient_info['patient_id']
            }
            
        except Exception as e:
            print(f"❌ Error loading patient {patient_info['patient_id']}: {str(e)}")
            # Return dummy data to avoid crash
            dummy_shape = (6,) + self.target_size  # (phases, H, W, D)
            return {
                'tumor_core': torch.zeros(dummy_shape),
                'tumor_periphery': torch.zeros(dummy_shape), 
                'enhancement_region': torch.zeros(dummy_shape),
                'label': torch.FloatTensor([0]),
                'patient_id': 'ERROR'
            }

# ============================================================================
# 3. MODELO MULTI-BRANCH
# ============================================================================

class ResNet3DBackbone(nn.Module):
    """Backbone 3D ResNet simplificado"""
    
    def __init__(self, in_channels=6):  # 6 fases DCE
        super().__init__()
        
        self.conv1 = nn.Conv3d(in_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        
        # Bloques ResNet simplificados
        self.layer1 = self._make_layer(64, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        
    def _make_layer(self, inplanes, planes, blocks, stride=1):
        layers = []
        layers.append(self._make_basic_block(inplanes, planes, stride))
        
        for _ in range(1, blocks):
            layers.append(self._make_basic_block(planes, planes))
            
        return nn.Sequential(*layers)
    
    def _make_basic_block(self, inplanes, planes, stride=1):
        return nn.Sequential(
            nn.Conv3d(inplanes, planes, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm3d(planes),
            nn.ReLU(inplace=True),
            nn.Conv3d(planes, planes, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(planes)
        )
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        return x

class MultiBranchPCRPredictor(nn.Module):
    """Modelo Multi-Branch para predicción de pCR"""
    
    def __init__(self, num_phases=6, feature_dim=512):
        super().__init__()
        
        # Backbone compartido
        self.backbone = ResNet3DBackbone(in_channels=num_phases)
        
        # Branches específicas para cada región
        self.tumor_core_branch = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        self.tumor_periphery_branch = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        self.enhancement_branch = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Attention-based fusion
        self.attention_weights = nn.Parameter(torch.ones(3) / 3)  # Inicializar con pesos iguales
        
        # Cross-attention entre regiones
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=128, num_heads=8, batch_first=True
        )
        
        # Clasificador final
        self.classifier = nn.Sequential(
            nn.Linear(128 * 3, 256),  # 3 regiones * 128 features
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)  # Predicción binaria pCR
        )
        
    def forward(self, tumor_core, tumor_periphery, enhancement_region):
        # Extraer features con backbone compartido
        core_features = self.backbone(tumor_core)
        periphery_features = self.backbone(tumor_periphery)
        enhancement_features = self.backbone(enhancement_region)
        
        # Procesar cada región con su branch específico
        core_processed = self.tumor_core_branch(core_features)
        periphery_processed = self.tumor_periphery_branch(periphery_features)
        enhancement_processed = self.enhancement_branch(enhancement_features)
        
        # Combinar features de todas las regiones
        all_features = torch.stack([
            core_processed, 
            periphery_processed, 
            enhancement_processed
        ], dim=1)  # (batch_size, 3, 128)
        
        # Cross-attention entre regiones
        attended_features, attention_weights = self.cross_attention(
            all_features, all_features, all_features
        )
        
        # Concatenar features atendidas
        fused_features = attended_features.flatten(1)  # (batch_size, 3*128)
        
        # Predicción final
        logits = self.classifier(fused_features)
        
        return logits, attention_weights

# ============================================================================
# 4. TRAINING LOOP OPTIMIZADO
# ============================================================================

class FocalLoss(nn.Module):
    """Focal Loss para manejar desbalance de clases"""
    
    def __init__(self, alpha=0.75, gamma=2.0, pos_weight=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight
        
    def forward(self, logits, targets):
        # Binary Cross Entropy with logits
        ce_loss = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight, reduction='none'
        )
        
        # Compute p_t
        p = torch.sigmoid(logits)
        p_t = p * targets + (1 - p) * (1 - targets)
        
        # Compute alpha_t
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        # Compute focal loss
        focal_loss = alpha_t * (1 - p_t) ** self.gamma * ce_loss
        
        return focal_loss.mean()

class PCRTrainer:
    """Trainer para el modelo Multi-Branch"""
    
    def __init__(self, model, train_loader, val_loader, device='cuda'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # Calcular pos_weight para balancear clases
        pos_weight = self.calculate_pos_weight()
        
        # Loss function
        self.criterion = FocalLoss(alpha=0.75, gamma=2.0, pos_weight=pos_weight)
        
        # Optimizer con weight decay
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=1e-3, 
            weight_decay=1e-4
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2
        )
        
        # Métricas
        self.train_losses = []
        self.val_losses = []
        self.val_aucs = []
        self.val_accs = []
        
    def calculate_pos_weight(self):
        """Calcula pos_weight para balancear clases"""
        total_pos = 0
        total_neg = 0
        
        for batch in self.train_loader:
            labels = batch['label']
            total_pos += labels.sum().item()
            total_neg += (1 - labels).sum().item()
        
        if total_pos > 0:
            pos_weight = torch.tensor([total_neg / total_pos]).to(self.device)
        else:
            pos_weight = torch.tensor([1.0]).to(self.device)
            
        print(f"Calculated pos_weight: {pos_weight.item():.3f}")
        return pos_weight
    
    def train_epoch(self):
        self.model.train()
        epoch_loss = 0
        num_batches = 0
        
        for batch in self.train_loader:
            # Mover datos a GPU
            tumor_core = batch['tumor_core'].to(self.device)
            tumor_periphery = batch['tumor_periphery'].to(self.device)
            enhancement_region = batch['enhancement_region'].to(self.device)
            labels = batch['label'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            logits, attention_weights = self.model(tumor_core, tumor_periphery, enhancement_region)
            
            # Calcular loss
            loss = self.criterion(logits.squeeze(), labels.squeeze())
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        return epoch_loss / num_batches
    
    def validate_epoch(self):
        self.model.eval()
        epoch_loss = 0
        all_preds = []
        all_labels = []
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                # Mover datos a GPU
                tumor_core = batch['tumor_core'].to(self.device)
                tumor_periphery = batch['tumor_periphery'].to(self.device)
                enhancement_region = batch['enhancement_region'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # Forward pass
                logits, _ = self.model(tumor_core, tumor_periphery, enhancement_region)
                
                # Calcular loss
                loss = self.criterion(logits.squeeze(), labels.squeeze())
                epoch_loss += loss.item()
                
                # Guardar predicciones
                probs = torch.sigmoid(logits).cpu().numpy()
                all_preds.extend(probs.flatten())
                all_labels.extend(labels.cpu().numpy().flatten())
                
                num_batches += 1
        
        # Calcular métricas
        avg_loss = epoch_loss / num_batches
        auc = roc_auc_score(all_labels, all_preds)
        
        # Accuracy con threshold óptimo
        from sklearn.metrics import precision_recall_curve
        precision, recall, thresholds = precision_recall_curve(all_labels, all_preds)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
        best_threshold = thresholds[np.argmax(f1_scores)]
        
        binary_preds = (np.array(all_preds) > best_threshold).astype(int)
        accuracy = accuracy_score(all_labels, binary_preds)
        balanced_acc = balanced_accuracy_score(all_labels, binary_preds)
        
        return avg_loss, auc, accuracy, balanced_acc, best_threshold
    
    def train(self, num_epochs=100, early_stopping_patience=15):
        """Entrenamiento completo con early stopping"""
        best_auc = 0
        patience_counter = 0
        
        print("Starting Multi-Branch pCR Prediction Training...")
        print("=" * 60)
        
        for epoch in range(num_epochs):
            # Entrenar época
            train_loss = self.train_epoch()
            
            # Validar época
            val_loss, val_auc, val_acc, val_balanced_acc, best_threshold = self.validate_epoch()
            
            # Actualizar scheduler
            self.scheduler.step()
            
            # Guardar métricas
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.val_aucs.append(val_auc)
            self.val_accs.append(val_balanced_acc)  # Usar balanced accuracy
            
            # Print progress
            if epoch % 5 == 0:
                print(f"Epoch {epoch:3d} | "
                      f"Train Loss: {train_loss:.4f} | "
                      f"Val Loss: {val_loss:.4f} | "
                      f"Val AUC: {val_auc:.4f} | "
                      f"Val Balanced Acc: {val_balanced_acc:.4f}")
            
            # Early stopping check
            if val_auc > best_auc:
                best_auc = val_auc
                patience_counter = 0
                # Guardar mejor modelo
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'best_auc': best_auc,
                    'best_threshold': best_threshold
                }, 'best_multi_branch_pcr_model.pth')
            else:
                patience_counter += 1
            
            if patience_counter >= early_stopping_patience:
                print(f"\nEarly stopping at epoch {epoch}")
                print(f"Best validation AUC: {best_auc:.4f}")
                break
        
        print("=" * 60)
        print(f"Training completed! Best AUC: {best_auc:.4f}")
        
        return best_auc

# ============================================================================
# 5. EJEMPLO DE USO
# ============================================================================

def main():
    """Ejemplo de cómo usar el pipeline completo con tus datos reales"""
    
    # ============================================================================
    # CONFIGURACIÓN - ADAPTA ESTAS RUTAS A TU SISTEMA
    # ============================================================================
    
    SPLITS_CSV = r"C:\Users\usuario\Documents\Mama_Mia\datos\train_test_splits.csv"
    PCR_LABELS_CSV = r"C:\Users\usuario\Documents\Mama_Mia\ganadores_pcr\pcr_labels.csv"
    CROPPED_DATA_DIR = r"C:\Users\usuario\Documents\Mama_Mia\datos\mask_cropped_data"
    
    # Configuración de entrenamiento
    BATCH_SIZE = 2  # Empezar pequeño para testing
    TARGET_SIZE = (32, 32, 16)  # Tamaño pequeño para testing inicial
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("🚀 Iniciando Multi-Branch pCR Prediction Pipeline")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"Target size: {TARGET_SIZE}")
    print(f"Batch size: {BATCH_SIZE}")
    
    # ============================================================================
    # PASO 1: Cargar datos desde CSVs
    # ============================================================================
    
    print("\n📂 Cargando datos desde CSVs...")
    try:
        train_data, test_data = load_patient_data(
            SPLITS_CSV, 
            PCR_LABELS_CSV, 
            CROPPED_DATA_DIR
        )
        
        if len(train_data) == 0:
            print("❌ No se encontraron datos de entrenamiento!")
            return
            
    except Exception as e:
        print(f"❌ Error cargando datos: {str(e)}")
        return
    
    # ============================================================================
    # PASO 2: Crear datasets
    # ============================================================================
    
    print("\n🔄 Creando datasets...")
    
    train_dataset = MultiRegionDCEDataset(
        train_data, 
        CROPPED_DATA_DIR,
        target_size=TARGET_SIZE
    )
    
    test_dataset = MultiRegionDCEDataset(
        test_data,
        CROPPED_DATA_DIR, 
        target_size=TARGET_SIZE
    )
    
    if len(train_dataset) == 0:
        print("❌ No hay pacientes válidos en el dataset de entrenamiento!")
        return
    
    # ============================================================================
    # PASO 3: Crear dataloaders
    # ============================================================================
    
    print(f"\n📊 Creando dataloaders...")
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,  # 0 para evitar problemas en Windows
        pin_memory=True if DEVICE == 'cuda' else False,
        drop_last=True  # ← ESTA LÍNEA ARREGLA EL ERROR
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True if DEVICE == 'cuda' else False,
        drop_last=False  # ← No drop para test (usamos model.eval())
    )
    
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Test batches: {len(test_loader)}")
    
    # ============================================================================
    # PASO 4: Test rápido de carga de datos
    # ============================================================================
    
    print(f"\n🧪 Testing data loading...")
    try:
        test_batch = next(iter(train_loader))
        
        print("✅ Batch cargado correctamente!")
        print(f"   Tumor core shape: {test_batch['tumor_core'].shape}")
        print(f"   Tumor periphery shape: {test_batch['tumor_periphery'].shape}")
        print(f"   Enhancement region shape: {test_batch['enhancement_region'].shape}")
        print(f"   Labels: {test_batch['label'].flatten().tolist()}")
        print(f"   Patient IDs: {test_batch['patient_id']}")
        
        # Verificar rangos de datos
        for region_name in ['tumor_core', 'tumor_periphery', 'enhancement_region']:
            tensor = test_batch[region_name]
            print(f"   {region_name}: min={tensor.min():.3f}, max={tensor.max():.3f}, mean={tensor.mean():.3f}")
        
    except Exception as e:
        print(f"❌ Error en test de carga: {str(e)}")
        return
    
    # ============================================================================
    # PASO 5: Crear y entrenar modelo
    # ============================================================================
    
    print(f"\n🧠 Creando modelo Multi-Branch...")
    
    # Detectar número de fases automáticamente
    num_phases = test_batch['tumor_core'].shape[1]  # (batch, phases, H, W, D)
    print(f"   Detected {num_phases} DCE phases")
    
    model = MultiBranchPCRPredictor(num_phases=num_phases)
    
    # Test forward pass
    try:
        model.eval()  # ← ESTA LÍNEA ARREGLA EL ERROR
        with torch.no_grad():
            logits, attention = model(
                test_batch['tumor_core'][:2],  # Usar 2 samples en lugar de 1
                test_batch['tumor_periphery'][:2],
                test_batch['enhancement_region'][:2]
            )
        print(f"✅ Forward pass successful! Output shape: {logits.shape}")
        print(f"   Attention weights shape: {attention.shape}")
        model.train()  # Volver a modo training
    except Exception as e:
        print(f"❌ Error en forward pass: {str(e)}")
        return
    
    # ============================================================================
    # PASO 6: Entrenar modelo
    # ============================================================================
    
    print(f"\n🏋️ Iniciando entrenamiento...")
    
    trainer = PCRTrainer(model, train_loader, test_loader, device=DEVICE)
    
    # Entrenar con pocos epochs primero para testing
    num_epochs = 20 if len(train_dataset) > 10 else 10
    best_auc = trainer.train(num_epochs=num_epochs, early_stopping_patience=10)
    
    print(f"\n🎉 Entrenamiento completado!")
    print(f"Mejor AUC obtenido: {best_auc:.4f}")
    print(f"Modelo guardado en: best_multi_branch_pcr_model.pth")
    
    # ============================================================================
    # PASO 7: Evaluación en test set
    # ============================================================================
    
    if len(test_dataset) > 0:
        print(f"\n📊 Evaluando en test set...")
        
        # Cargar mejor modelo
        checkpoint = torch.load('best_multi_branch_pcr_model.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Evaluar
        test_loss, test_auc, test_acc, test_balanced_acc, threshold = trainer.validate_epoch()
        
        print(f"📈 Resultados en Test Set:")
        print(f"   AUC: {test_auc:.4f}")
        print(f"   Accuracy: {test_acc:.4f}")
        print(f"   Balanced Accuracy: {test_balanced_acc:.4f}")
        print(f"   Optimal Threshold: {threshold:.4f}")

# ============================================================================
# FUNCIÓN DE DEBUGGING
# ============================================================================

def debug_data_structure(splits_csv, pcr_labels_csv, cropped_data_dir):
    """Función para debuggear la estructura de datos"""
    
    print("🔍 DEBUGGING DATA STRUCTURE")
    print("=" * 50)
    
    # Check CSVs
    print(f"\n📄 Checking {splits_csv}:")
    if Path(splits_csv).exists():
        splits_df = pd.read_csv(splits_csv)
        print(f"   ✅ Found {len(splits_df)} rows")
        print(f"   Columns: {list(splits_df.columns)}")
        print(f"   Sample: {splits_df.head(2).to_dict('records')}")
    else:
        print("   ❌ File not found!")
        return
    
    print(f"\n📄 Checking {pcr_labels_csv}:")
    if Path(pcr_labels_csv).exists():
        pcr_df = pd.read_csv(pcr_labels_csv)
        print(f"   ✅ Found {len(pcr_df)} rows")
        print(f"   Columns: {list(pcr_df.columns)}")
        print(f"   Sample: {pcr_df.head(2).to_dict('records')}")
        
        # Check label distribution
        if 'pCR' in pcr_df.columns:
            pcr_dist = pcr_df['pCR'].value_counts()
            print(f"   pCR distribution: {pcr_dist.to_dict()}")
    else:
        print("   ❌ File not found!")
        return
    
    # Check cropped data structure
    print(f"\n📁 Checking {cropped_data_dir}:")
    cropped_path = Path(cropped_data_dir)
    if cropped_path.exists():
        print("   ✅ Directory exists")
        
        # Check subdirectories
        for subdir in ['images', 'segmentations']:
            subdir_path = cropped_path / subdir
            if subdir_path.exists():
                print(f"   📂 {subdir}/")
                for split in ['train_split', 'test_split']:
                    split_path = subdir_path / split
                    if split_path.exists():
                        patients = list(split_path.iterdir())
                        print(f"     📂 {split}/: {len(patients)} patients")
                        
                        # Show sample patient
                        if patients:
                            sample_patient = patients[0]
                            files = list(sample_patient.iterdir())
                            print(f"       Sample patient {sample_patient.name}: {len(files)} files")
                            for f in files[:3]:  # Show first 3 files
                                print(f"         - {f.name}")
                    else:
                        print(f"     ❌ {split}/ not found")
            else:
                print(f"   ❌ {subdir}/ not found")
    else:
        print("   ❌ Directory not found!")

if __name__ == "__main__":
    
    # OPCIÓN 1: Debugging (ejecutar primero)
    # debug_data_structure(
    #     r"C:\Users\usuario\Documents\Mama_Mia\datos\train_test_splits.csv",
    #     r"C:\Users\usuario\Documents\Mama_Mia\ganadores_pcr\pcr_labels.csv", 
    #     r"C:\Users\usuario\Documents\Mama_Mia\datos\mask_cropped_data"
    # )
    
    # OPCIÓN 2: Entrenamiento completo
    main()