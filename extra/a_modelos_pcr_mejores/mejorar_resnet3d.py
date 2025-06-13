# fast_pcr_prediction_fixed.py
"""
MODELO RÁPIDO Y LIGERO - CONSEJOS APLICADOS (CORREGIDO):

✅ Consejos implementados:
1. bfloat16 mixed precision (CORREGIDO)
2. Resolución reducida (64³ en lugar de 128³)
3. Modelos timm pequeños (resnet10t, mobilenetv3)
4. Entrenamiento rápido para experimentar
5. Cosine LR con linear warmup
6. Gradient clipping + EMA
7. Opción 1-2 canales en lugar de 3
8. Simplificado y enfocado en velocidad

Meta: Entrenamiento 3x más rápido, experimentación ágil
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import nibabel as nib
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, accuracy_score
from tqdm import tqdm
from pathlib import Path
import random
import json
import math

# Intentar importar timm para modelos ligeros
try:
    import timm
    TIMM_AVAILABLE = True
    print("✅ TIMM disponible - Usando modelos optimizados")
except ImportError:
    TIMM_AVAILABLE = False
    print("⚠️ TIMM no disponible - pip install timm")

# ==========================================
# 🆕 CONFIGURACIÓN OPTIMIZADA
# ==========================================
class FastConfig:
    # Data paths (usando tu pipeline existente)
    DATA_DIR = Path(r"D:\mama_mia_final_corrected")
    LABELS_FILE = Path(r"D:\clinical_data_complete.json")
    
    # Tamaños optimizados (3x más rápido)
    TARGET_SHAPE = (64, 64, 64)  # Reducido de 128³ a 64³
    BATCH_SIZE = 16  # Más grande para aprovechar GPU
    
    # Modelos rápidos para experimentar
    MODEL_TYPE = 'resnet10t'  # resnet10t, resnet14t, mobilenetv3_small
    INPUT_CHANNELS = 2  # Probar 2 canales: [post, mask] (quitar pre)
    
    # Training optimizado
    EPOCHS = 50  # Menos épocas para experimentar rápido
    LEARNING_RATE = 1e-3  # LR más alto para convergencia rápida
    WEIGHT_DECAY = 1e-4
    PATIENCE = 8  # Early stopping más agresivo
    
    # Técnicas de optimización
    USE_BFLOAT16 = True  # Más estable que float16
    USE_COSINE_SCHEDULE = True
    USE_WARMUP = True
    WARMUP_EPOCHS = 5
    USE_GRADIENT_CLIPPING = True
    CLIP_VALUE = 1.0
    
    # EMA simplificado
    EMA_DECAY = 0.99  # Menos agresivo
    
    # Augmentaciones simplificadas
    AUG_PROB = 0.5
    
    SEED = 42

# ==========================================
# 🆕 MODELOS RÁPIDOS CON TIMM
# ==========================================
class FastTimmModel(nn.Module):
    """Modelos rápidos usando timm"""
    def __init__(self, model_name='resnet10t', in_channels=2, num_classes=1):
        super().__init__()
        
        if not TIMM_AVAILABLE:
            # Fallback a modelo simple
            self.use_timm = False
            self.backbone = self._create_simple_cnn(in_channels)
            self.classifier = nn.Linear(512, num_classes)
        else:
            self.use_timm = True
            # Crear modelo timm 2D y adaptarlo
            self.backbone = timm.create_model(
                model_name, 
                pretrained=False,  # Sin pretrain para experimentos rápidos
                num_classes=0,  # Sin clasificador
                in_chans=in_channels,
                features_only=False
            )
            
            # Adaptar a 3D con pooling global
            self.global_pool = nn.AdaptiveAvgPool3d(1)
            self.classifier = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(self.backbone.num_features, 256),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(256, num_classes)
            )
        
        print(f"✅ Using {model_name} with {in_channels} channels")
    
    def _create_simple_cnn(self, in_channels):
        """CNN simple como fallback"""
        return nn.Sequential(
            nn.Conv3d(in_channels, 32, 3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(2),
            
            nn.Conv3d(32, 64, 3, padding=1),
            nn.BatchNorm3d(64), 
            nn.ReLU(),
            nn.MaxPool3d(2),
            
            nn.Conv3d(64, 128, 3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.MaxPool3d(2),
            
            nn.Conv3d(128, 256, 3, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten()
        )
    
    def forward(self, x):
        if self.use_timm:
            # Para timm: promediar a lo largo de la dimensión depth
            b, c, d, h, w = x.shape
            
            # Método 1: Promedio de slices centrales
            mid_slices = d // 4
            start_idx = d // 2 - mid_slices // 2
            end_idx = start_idx + mid_slices
            x_2d = x[:, :, start_idx:end_idx, :, :].mean(dim=2)  # [B, C, H, W]
            
            features = self.backbone(x_2d)
            
        else:
            # CNN 3D simple
            features = self.backbone(x)
        
        logits = self.classifier(features)
        return logits

# ==========================================
# 🆕 DATASET OPTIMIZADO PARA VELOCIDAD
# ==========================================
class FastDataset(Dataset):
    """Dataset optimizado para velocidad"""
    def __init__(self, data_dir, patient_ids, labels, augment=False, 
                 target_shape=(64, 64, 64), input_channels=2):
        self.data_dir = Path(data_dir)
        self.patient_ids = patient_ids
        self.labels = labels
        self.target_shape = target_shape
        self.input_channels = input_channels
        self.augment = augment
        
        # Cache pequeño para datos más usados
        self.cache = {}
        self.max_cache_size = 50
        
        # Verificar archivos válidos
        self.valid_indices = self._check_valid_files()
        
        print(f"Dataset: {len(self.valid_indices)}/{len(patient_ids)} pacientes válidos")
        if labels:
            valid_labels = [labels[i] for i in self.valid_indices]
            print(f"  pCR rate: {np.mean(valid_labels):.1%}")
            print(f"  Input channels: {input_channels} ({'[post, mask]' if input_channels == 2 else '[pre, post, mask]'})")
    
    def _check_valid_files(self):
        """Verificar archivos válidos"""
        valid = []
        for i, pid in enumerate(self.patient_ids):
            tensor_file = self.data_dir / pid / f"{pid}_tensor_3ch.nii.gz"
            if tensor_file.exists():
                valid.append(i)
        return valid
    
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        actual_idx = self.valid_indices[idx]
        patient_id = self.patient_ids[actual_idx]
        label = self.labels[actual_idx] if self.labels else 0
        
        # Verificar cache
        if patient_id in self.cache:
            tensor = self.cache[patient_id].clone()
        else:
            # Cargar tensor
            tensor_file = self.data_dir / patient_id / f"{patient_id}_tensor_3ch.nii.gz"
            img = nib.load(tensor_file).get_fdata().astype(np.float32)
            tensor = torch.from_numpy(img)
            
            # Redimensionar a tamaño pequeño (más rápido)
            tensor = self._fast_resize(tensor)
            
            # Cache si hay espacio
            if len(self.cache) < self.max_cache_size:
                self.cache[patient_id] = tensor.clone()
        
        # Seleccionar canales según configuración
        if self.input_channels == 1:
            # Solo post-contraste
            tensor = tensor[1:2]  # Canal 1 (post)
        elif self.input_channels == 2:
            # Post + mask (más informativo que pre + post)
            tensor = torch.stack([tensor[1], tensor[2]])  # [post, mask]
        # Si input_channels == 3, usar todos
        
        # Augmentaciones ligeras
        if self.augment:
            tensor = self._fast_augment(tensor)
        
        return {
            'tensor': tensor.float(),
            'target': torch.tensor(label, dtype=torch.long),
            'patient_id': patient_id
        }
    
    def _fast_resize(self, tensor):
        """Redimensionamiento rápido usando interpolación"""
        current_shape = tensor.shape[1:]  # Sin canal
        target_shape = self.target_shape
        
        if current_shape != target_shape:
            # Usar interpolación trilinear (más rápida que padding/cropping)
            tensor = F.interpolate(
                tensor.unsqueeze(0),  # Add batch dim
                size=target_shape,
                mode='trilinear',
                align_corners=False
            ).squeeze(0)
        
        return tensor
    
    def _fast_augment(self, tensor):
        """Augmentaciones rápidas"""
        if torch.rand(1) < FastConfig.AUG_PROB:
            # Solo flips (más rápidos)
            if torch.rand(1) < 0.5:
                tensor = torch.flip(tensor, [1])  # Flip H
            if torch.rand(1) < 0.3:
                tensor = torch.flip(tensor, [2])  # Flip W
        
        if torch.rand(1) < FastConfig.AUG_PROB * 0.3:
            # Rotación 90° ocasional
            k = torch.randint(1, 4, (1,)).item()
            tensor = torch.rot90(tensor, k, [1, 2])
        
        return tensor

# ==========================================
# 🆕 FUNCIÓN PARA CONVERTIR BFLOAT16 A NUMPY
# ==========================================
def safe_to_numpy(tensor):
    """Convierte tensor a numpy de forma segura, manejando bfloat16"""
    if tensor.dtype == torch.bfloat16:
        return tensor.float().cpu().numpy()
    else:
        return tensor.cpu().numpy()

# ==========================================
# 🆕 TRAINER RÁPIDO (CORREGIDO)
# ==========================================
class FastTrainer:
    """Trainer optimizado para experimentación rápida"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Device: {self.device}")
        
        # Configurar mixed precision (CORREGIDO)
        if FastConfig.USE_BFLOAT16 and torch.cuda.is_available():
            self.use_amp = True
            # Usar la nueva API
            self.scaler = torch.amp.GradScaler('cuda')
            self.dtype = torch.bfloat16
            print("✅ Using bfloat16 mixed precision")
        else:
            self.use_amp = False
            self.dtype = torch.float32
    
    def create_model(self):
        """Crear modelo optimizado"""
        if TIMM_AVAILABLE:
            model = FastTimmModel(
                model_name=FastConfig.MODEL_TYPE,
                in_channels=FastConfig.INPUT_CHANNELS,
                num_classes=1
            )
        else:
            # Fallback simple
            model = FastTimmModel(
                model_name='simple_cnn',
                in_channels=FastConfig.INPUT_CHANNELS,
                num_classes=1
            )
        
        return model.to(self.device)
    
    def create_scheduler(self, optimizer, steps_per_epoch):
        """Crear scheduler con warmup"""
        if not FastConfig.USE_COSINE_SCHEDULE:
            return None
        
        total_steps = FastConfig.EPOCHS * steps_per_epoch
        warmup_steps = FastConfig.WARMUP_EPOCHS * steps_per_epoch
        
        def lr_lambda(step):
            if step < warmup_steps:
                # Linear warmup
                return step / warmup_steps
            else:
                # Cosine annealing
                progress = (step - warmup_steps) / (total_steps - warmup_steps)
                return 0.5 * (1 + math.cos(math.pi * progress))
        
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    def train_fast_model(self):
        """Entrenamiento rápido para experimentación"""
        print("🚀 ENTRENAMIENTO RÁPIDO Y LIGERO")
        print("=" * 50)
        
        # Datos
        with open(FastConfig.LABELS_FILE, 'r') as f:
            pcr_list = json.load(f)
        
        pcr_data = {item['patient_id']: item for item in pcr_list}
        patient_ids, labels = [], []
        
        for pid, info in pcr_data.items():
            if 'pcr' in info and info['pcr'] in ["0", "1"]:
                tensor_path = FastConfig.DATA_DIR / pid / f"{pid}_tensor_3ch.nii.gz"
                if tensor_path.exists():
                    patient_ids.append(pid)
                    labels.append(int(info['pcr']))
        
        print(f"Pacientes: {len(patient_ids)}")
        print(f"pCR rate: {np.mean(labels):.1%}")
        
        # Split
        train_ids, val_ids, train_labels, val_labels = train_test_split(
            patient_ids, labels, test_size=0.2, random_state=FastConfig.SEED, stratify=labels
        )
        
        # Datasets
        train_dataset = FastDataset(
            FastConfig.DATA_DIR, train_ids, train_labels,
            augment=True, target_shape=FastConfig.TARGET_SHAPE,
            input_channels=FastConfig.INPUT_CHANNELS
        )
        val_dataset = FastDataset(
            FastConfig.DATA_DIR, val_ids, val_labels,
            augment=False, target_shape=FastConfig.TARGET_SHAPE,
            input_channels=FastConfig.INPUT_CHANNELS
        )
        
        # DataLoaders
        train_loader = DataLoader(
            train_dataset, batch_size=FastConfig.BATCH_SIZE,
            shuffle=True, num_workers=2, pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=FastConfig.BATCH_SIZE,
            shuffle=False, num_workers=2, pin_memory=True
        )
        
        # Model
        model = self.create_model()
        
        # Loss (simple y efectivo)
        pos_weight = torch.tensor([2.39]).to(self.device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        
        # Optimizer
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=FastConfig.LEARNING_RATE,
            weight_decay=FastConfig.WEIGHT_DECAY
        )
        
        # Scheduler
        scheduler = self.create_scheduler(optimizer, len(train_loader))
        
        # Training loop
        best_auc = 0
        patience_counter = 0
        
        print(f"Configuración:")
        print(f"  Resolución: {FastConfig.TARGET_SHAPE}")
        print(f"  Canales: {FastConfig.INPUT_CHANNELS}")
        print(f"  Modelo: {FastConfig.MODEL_TYPE}")
        print(f"  Batch size: {FastConfig.BATCH_SIZE}")
        print()
        
        for epoch in range(FastConfig.EPOCHS):
            # Training
            model.train()
            train_losses, train_preds, train_targets = [], [], []
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
            for batch in pbar:
                tensors = batch['tensor'].to(self.device, non_blocking=True)
                targets = batch['target'].to(self.device, non_blocking=True).float()
                
                optimizer.zero_grad()
                
                if self.use_amp:
                    with torch.autocast(device_type='cuda', dtype=self.dtype):
                        outputs = model(tensors).squeeze()
                        loss = criterion(outputs, targets)
                    
                    self.scaler.scale(loss).backward()
                    
                    if FastConfig.USE_GRADIENT_CLIPPING:
                        self.scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), FastConfig.CLIP_VALUE)
                    
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    outputs = model(tensors).squeeze()
                    loss = criterion(outputs, targets)
                    loss.backward()
                    
                    if FastConfig.USE_GRADIENT_CLIPPING:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), FastConfig.CLIP_VALUE)
                    
                    optimizer.step()
                
                if scheduler:
                    scheduler.step()
                
                # Métricas (CORREGIDO para bfloat16)
                train_losses.append(loss.item())
                probs = torch.sigmoid(outputs)
                train_preds.extend(safe_to_numpy(probs.detach()))
                train_targets.extend(safe_to_numpy(targets))
                
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'lr': f'{optimizer.param_groups[0]["lr"]:.2e}'
                })
            
            # Validation
            model.eval()
            val_preds, val_targets = [], []
            
            with torch.no_grad():
                for batch in val_loader:
                    tensors = batch['tensor'].to(self.device, non_blocking=True)
                    targets = batch['target'].to(self.device, non_blocking=True).float()
                    
                    if self.use_amp:
                        with torch.autocast(device_type='cuda', dtype=self.dtype):
                            outputs = model(tensors).squeeze()
                    else:
                        outputs = model(tensors).squeeze()
                    
                    probs = torch.sigmoid(outputs)
                    # CORREGIDO para bfloat16
                    val_preds.extend(safe_to_numpy(probs))
                    val_targets.extend(safe_to_numpy(targets))
            
            # Métricas
            train_auc = roc_auc_score(train_targets, train_preds)
            val_auc = roc_auc_score(val_targets, val_preds)
            val_acc = accuracy_score(val_targets, [1 if p > 0.5 else 0 for p in val_preds])
            
            print(f"Epoch {epoch+1}:")
            print(f"  Loss: {np.mean(train_losses):.4f}")
            print(f"  Train AUC: {train_auc:.4f}, Val AUC: {val_auc:.4f}")
            print(f"  Val Acc: {val_acc:.4f}")
            
            # Early stopping
            if val_auc > best_auc:
                best_auc = val_auc
                patience_counter = 0
                print(f"  🎯 New best AUC: {val_auc:.4f}")
                torch.save(model.state_dict(), 'best_fast_model.pth')
            else:
                patience_counter += 1
                if patience_counter >= FastConfig.PATIENCE:
                    print(f"  ⏹️ Early stopping at epoch {epoch+1}")
                    break
        
        print(f"\n🎯 RESULTADOS FINALES:")
        print(f"Best AUC: {best_auc:.4f}")
        print(f"Configuración óptima:")
        print(f"  Modelo: {FastConfig.MODEL_TYPE}")
        print(f"  Canales: {FastConfig.INPUT_CHANNELS}")
        print(f"  Resolución: {FastConfig.TARGET_SHAPE}")
        
        baseline = 0.6102
        improvement = ((best_auc - baseline) / baseline) * 100
        print(f"  Mejora vs baseline: {improvement:+.1f}%")
        
        return best_auc

def main():
    """Ejecutar entrenamiento rápido"""
    # Set seed
    torch.manual_seed(FastConfig.SEED)
    np.random.seed(FastConfig.SEED)
    random.seed(FastConfig.SEED)
    
    # Crear trainer
    trainer = FastTrainer()
    
    # Entrenar
    best_auc = trainer.train_fast_model()
    
    print(f"\n✅ Entrenamiento completado!")
    print(f"🎯 Best AUC: {best_auc:.4f}")
    print(f"⚡ Modelo rápido para experimentación guardado")

if __name__ == "__main__":
    main()