# ==========================
# Importaciones necesarias
# ==========================
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.parallel import DataParallel
import os
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import gc

# Importa tu dataset (asegúrate que tienes este archivo!)
from dataset import MAMA_MIA_Dataset_MultiGPU

# ==========================
# Modelo Beast U-Net 3D
# ==========================
class BeastUNet3D(nn.Module):
    def __init__(self, in_channels=8, num_classes=2, base_features=64):
        super().__init__()
        self.enc1 = self.conv_block(in_channels, base_features)
        self.enc2 = self.conv_block(base_features, base_features * 2)
        self.enc3 = self.conv_block(base_features * 2, base_features * 4)
        self.enc4 = self.conv_block(base_features * 4, base_features * 8)
        self.enc5 = self.conv_block(base_features * 8, base_features * 16)
        self.enc6 = self.conv_block(base_features * 16, base_features * 32)
        self.bridge = self.conv_block(base_features * 32, base_features * 64)

        self.up6 = nn.ConvTranspose3d(base_features * 64, base_features * 32, 2, 2)
        self.dec6 = self.conv_block(base_features * 64, base_features * 32)
        self.up5 = nn.ConvTranspose3d(base_features * 32, base_features * 16, 2, 2)
        self.dec5 = self.conv_block(base_features * 32, base_features * 16)
        self.up4 = nn.ConvTranspose3d(base_features * 16, base_features * 8, 2, 2)
        self.dec4 = self.conv_block(base_features * 16, base_features * 8)
        self.up3 = nn.ConvTranspose3d(base_features * 8, base_features * 4, 2, 2)
        self.dec3 = self.conv_block(base_features * 8, base_features * 4)
        self.up2 = nn.ConvTranspose3d(base_features * 4, base_features * 2, 2, 2)
        self.dec2 = self.conv_block(base_features * 4, base_features * 2)
        self.up1 = nn.ConvTranspose3d(base_features * 2, base_features, 2, 2)
        self.dec1 = self.conv_block(base_features * 2, base_features)

        self.final_conv = nn.Conv3d(base_features, base_features // 2, 3, padding=1)
        self.final = nn.Conv3d(base_features // 2, num_classes, 1)

        self.deep_sup_4 = nn.Conv3d(base_features * 4, num_classes, 1)
        self.deep_sup_2 = nn.Conv3d(base_features * 2, num_classes, 1)

        self.attention_4 = self.attention_block(base_features * 8, base_features * 4)
        self.attention_2 = self.attention_block(base_features * 4, base_features * 2)
        self.attention_1 = self.attention_block(base_features * 2, base_features)

        self.dropout = nn.Dropout3d(0.2)
        self.bridge_dropout = nn.Dropout3d(0.3)

    def conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True)
        )

    def attention_block(self, F_g, F_l):
        return nn.Sequential(
            nn.Conv3d(F_g, F_l, 1),
            nn.BatchNorm3d(F_l),
            nn.Sigmoid()
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(nn.MaxPool3d(2)(e1))
        e3 = self.enc3(nn.MaxPool3d(2)(e2))
        e4 = self.enc4(nn.MaxPool3d(2)(e3))
        e5 = self.enc5(nn.MaxPool3d(2)(e4))
        e6 = self.enc6(nn.MaxPool3d(2)(e5))
        bridge = self.bridge(nn.MaxPool3d(2)(e6))
        bridge = self.bridge_dropout(bridge)

        d6 = self.up6(bridge)
        d6 = torch.cat([d6, e6], dim=1)
        d6 = self.dec6(d6)

        d5 = self.up5(d6)
        d5 = torch.cat([d5, e5], dim=1)
        d5 = self.dec5(d5)

        d4 = self.up4(d5)
        att4 = self.attention_4(d4) * e4
        d4 = torch.cat([d4, att4], dim=1)
        d4 = self.dec4(d4)
        d4 = self.dropout(d4)

        d3 = self.up3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        att2 = self.attention_2(d2) * e2
        d2 = torch.cat([d2, att2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        att1 = self.attention_1(d1) * e1
        d1 = torch.cat([d1, att1], dim=1)
        d1 = self.dec1(d1)

        final_features = self.final_conv(d1)
        output = self.final(final_features)

        if self.training:
            aux_4 = self.deep_sup_4(d4)
            aux_2 = self.deep_sup_2(d2)
            aux_4 = nn.functional.interpolate(aux_4, size=output.shape[2:], mode='trilinear', align_corners=False)
            aux_2 = nn.functional.interpolate(aux_2, size=output.shape[2:], mode='trilinear', align_corners=False)
            return output, aux_4, aux_2
        else:
            return output

# ==========================
# Combined Loss
# ==========================
class CombinedLoss(nn.Module):
    def __init__(self, ce_weight=0.5, dice_weight=0.3, focal_weight=0.2, deep_sup_weight=0.3, class_weights=None):
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.deep_sup_weight = deep_sup_weight
        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights) if class_weights is not None else nn.CrossEntropyLoss()

    def dice_loss(self, pred, target, smooth=1e-6):
        pred_soft = torch.softmax(pred, dim=1)[:, 1:2]
        target_one_hot = (target == 1).float().unsqueeze(1)
        intersection = (pred_soft * target_one_hot).sum()
        union = pred_soft.sum() + target_one_hot.sum()
        dice = (2 * intersection + smooth) / (union + smooth)
        return 1 - dice

    def focal_loss(self, pred, target, alpha=0.25, gamma=2.0):
        ce_loss = nn.functional.cross_entropy(pred, target, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = alpha * (1 - pt) ** gamma * ce_loss
        return focal_loss.mean()

    def forward(self, outputs, targets):
        if isinstance(outputs, tuple):
            main_output, aux_4, aux_2 = outputs
            main_loss = (self.ce_weight * self.ce_loss(main_output, targets) +
                         self.dice_weight * self.dice_loss(main_output, targets) +
                         self.focal_weight * self.focal_loss(main_output, targets))
            aux_4_loss = self.ce_loss(aux_4, targets) + self.dice_loss(aux_4, targets)
            aux_2_loss = self.ce_loss(aux_2, targets) + self.dice_loss(aux_2, targets)
            total_loss = main_loss + self.deep_sup_weight * (aux_4_loss + aux_2_loss)
            return total_loss
        else:
            ce = self.ce_loss(outputs, targets)
            dice = self.dice_loss(outputs, targets)
            focal = self.focal_loss(outputs, targets)
            return self.ce_weight * ce + self.dice_weight * dice + self.focal_weight * focal

# ==========================
# Métricas avanzadas
# ==========================
def calculate_advanced_metrics(pred, target, smooth=1e-6):
    if pred.shape[1] > 1:
        pred_prob = torch.softmax(pred, dim=1)[:, 1:2]
    else:
        pred_prob = torch.sigmoid(pred)
    pred_binary = (pred_prob > 0.5).float()
    target_binary = (target > 0).float()

    intersection = (pred_binary * target_binary).sum()
    union = pred_binary.sum() + target_binary.sum()
    dice = (2 * intersection + smooth) / (union + smooth)

    iou = intersection / (union - intersection + smooth)
    sensitivity = intersection / (target_binary.sum() + smooth)
    tn = ((1 - pred_binary) * (1 - target_binary)).sum()
    fp = (pred_binary * (1 - target_binary)).sum()
    specificity = tn / (tn + fp + smooth)
    precision = intersection / (pred_binary.sum() + smooth)

    return {
        'dice': dice.item(),
        'iou': iou.item(),
        'sensitivity': sensitivity.item(),
        'specificity': specificity.item(),
        'precision': precision.item(),
    }

# ==========================
# Entrenamiento y Validación
# ==========================
## ==========================
# Función para una época de entrenamiento
# ==========================
def train_epoch_multi_gpu(model, dataloader, optimizer, criterion, device, epoch, scaler=None):
    model.train()
    running_loss = 0.0
    running_metrics = {'dice': 0.0, 'iou': 0.0, 'sensitivity': 0.0, 'precision': 0.0}

    progress_bar = tqdm(dataloader, desc=f'Epoch {epoch}')

    for batch_idx, (images, masks, case_ids) in enumerate(progress_bar):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        optimizer.zero_grad()

        if scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, masks)
        else:
            outputs = model(images)
            loss = criterion(outputs, masks)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        main_output = outputs[0] if isinstance(outputs, tuple) else outputs
        metrics = calculate_advanced_metrics(main_output, masks)

        running_loss += loss.item()
        for key in running_metrics:
            running_metrics[key] += metrics[key]

        progress_bar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Dice': f'{metrics["dice"]:.4f}',
            'IoU': f'{metrics["iou"]:.4f}',
            'GPU_GB': f'{torch.cuda.memory_allocated() / 1024**3:.1f}'
        })

    epoch_loss = running_loss / len(dataloader)
    epoch_metrics = {key: val / len(dataloader) for key, val in running_metrics.items()}

    return epoch_loss, epoch_metrics

# ==========================
# Función para validación
# ==========================
def validate_epoch_multi_gpu(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_metrics = {'dice': 0.0, 'iou': 0.0, 'sensitivity': 0.0, 'precision': 0.0}

    with torch.no_grad():
        for batch_idx, (images, masks, case_ids) in enumerate(tqdm(dataloader, desc='Validation')):
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            with torch.cuda.amp.autocast():
                outputs = model(images)

            main_output = outputs[0] if isinstance(outputs, tuple) else outputs
            loss = criterion(main_output, masks)

            metrics = calculate_advanced_metrics(main_output, masks)

            running_loss += loss.item()
            for key in running_metrics:
                running_metrics[key] += metrics[key]

    epoch_loss = running_loss / len(dataloader)
    epoch_metrics = {key: val / len(dataloader) for key, val in running_metrics.items()}

    return epoch_loss, epoch_metrics

# ==========================
# Main Training Loop
# ==========================
def main():
    BASE_DATA_PATH = r"C:\Users\usuario\Documents\Mama_Mia\cropped_data"

    config = {
        'batch_size': 16,
        'learning_rate': 3e-4,
        'epochs': 120,
        'num_gpus': 4,
        'device': 'cuda',
        'save_every': 10,
        'mixed_precision': True,
        'high_resolution': True,
        'deep_supervision': True,
    }

    print("\n🚀 BEAST MODE TRAINING - 4x RTX A6000")
    print("=" * 70)

    if not torch.cuda.is_available():
        print("❌ CUDA no disponible!")
        return

    print(f"🔥 GPUs detectadas: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")

    output_dir = Path("outputs_beast")
    output_dir.mkdir(exist_ok=True)

    print("📦 Cargando datasets HIGH-RES...")
    train_dataset = MAMA_MIA_Dataset_MultiGPU(BASE_DATA_PATH, split='train_split', transform=True, high_res=True)
    test_dataset = MAMA_MIA_Dataset_MultiGPU(BASE_DATA_PATH, split='test_split', transform=False, high_res=True)

    if len(train_dataset) == 0:
        print("❌ ERROR: No se encontraron datos de entrenamiento!")
        return

    print(f"✅ Train cases: {len(train_dataset)}")
    print(f"✅ Test cases: {len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=8, pin_memory=True, persistent_workers=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True) if len(test_dataset) > 0 else None

    print("🧠 Inicializando BEAST MODEL...")

    model = BeastUNet3D(in_channels=8, num_classes=2, base_features=64)
    model = model.to('cuda:0')

    if torch.cuda.device_count() > 1:
        print(f"🔥 Usando DataParallel en {torch.cuda.device_count()} GPUs")
        model = DataParallel(model)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📊 Total parámetros: {total_params:,}")
    print(f"📊 Parámetros entrenables: {trainable_params:,}")
    print(f"💾 Modelo size aprox: {total_params * 4 / 1024**3:.2f} GB")

    class_weights = torch.tensor([0.1, 0.9]).to(config['device'])
    criterion = CombinedLoss(ce_weight=0.4, dice_weight=0.4, focal_weight=0.2, deep_sup_weight=0.3, class_weights=class_weights)

    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=1e-4, betas=(0.9, 0.999))

    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=config['learning_rate'] * 3, epochs=config['epochs'], steps_per_epoch=len(train_loader), pct_start=0.1, anneal_strategy='cos', div_factor=10, final_div_factor=100)

    scaler = torch.cuda.amp.GradScaler() if config['mixed_precision'] else None

    best_dice = 0.0
    train_losses, train_metrics = [], []
    test_losses, test_metrics = [], []

    print("\n🏋️ INICIANDO BEAST TRAINING...")
    start_time = time.time()

    for epoch in range(config['epochs']):
        print(f"\n📅 Epoch {epoch+1}/{config['epochs']}")
        print("-" * 60)

        for i in range(torch.cuda.device_count()):
            memory_gb = torch.cuda.memory_allocated(i) / 1024**3
            memory_max = torch.cuda.max_memory_allocated(i) / 1024**3
            print(f"🖥️  GPU {i} Memory: {memory_gb:.2f}/{memory_max:.2f} GB")

        train_loss, train_metrics_epoch = train_epoch_multi_gpu(model, train_loader, optimizer, criterion, config['device'], epoch+1, scaler)

        if test_loader is not None:
            test_loss, test_metrics_epoch = validate_epoch_multi_gpu(model, test_loader, criterion, config['device'])
        else:
            test_loss, test_metrics_epoch = 0.0, {k: 0.0 for k in train_metrics_epoch.keys()}

        scheduler.step()

        train_losses.append(train_loss)
        train_metrics.append(train_metrics_epoch)
        test_losses.append(test_loss)
        test_metrics.append(test_metrics_epoch)

        print(f"📊 TRAIN - Loss: {train_loss:.4f}")
        print(f"   Dice: {train_metrics_epoch['dice']:.4f}, IoU: {train_metrics_epoch['iou']:.4f}")
        print(f"   Sensitivity: {train_metrics_epoch['sensitivity']:.4f}, Precision: {train_metrics_epoch['precision']:.4f}")

        if test_loader is not None:
            print(f"📊 TEST  - Loss: {test_loss:.4f}")
            print(f"   Dice: {test_metrics_epoch['dice']:.4f}, IoU: {test_metrics_epoch['iou']:.4f}")

        print(f"🔧 Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")

        current_dice = test_metrics_epoch['dice'] if test_loader is not None else train_metrics_epoch['dice']
        if current_dice > best_dice:
            best_dice = current_dice
            model_state = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
            torch.save({
                'epoch': epoch,
                'model_state_dict': model_state,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_dice': best_dice,
                'config': config,
            }, output_dir / 'best_model_beast.pth')
            print(f"💾 ✅ NUEVO MEJOR MODELO! Dice: {best_dice:.4f}")

        if (epoch + 1) % config['save_every'] == 0:
            model_state = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
            torch.save({
                'epoch': epoch,
                'model_state_dict': model_state,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_losses': train_losses,
                'train_metrics': train_metrics,
                'test_losses': test_losses,
                'test_metrics': test_metrics,
                'config': config,
            }, output_dir / f'checkpoint_epoch_{epoch+1}_beast.pth')
            print(f"💾 Checkpoint guardado: epoch_{epoch+1}")

    end_time = time.time()
    training_time = end_time - start_time

    print("\n" + "=" * 70)
    print("🎉 ¡BEAST TRAINING COMPLETADO!")
    print(f"⏱️  Tiempo total: {training_time/3600:.2f} horas")
    print(f"🏆 Mejor Dice Score: {best_dice:.4f}")
    print(f"📁 Modelos guardados en: {output_dir}")
    print("🚀 MULTI-GPU HIGH-RESOLUTION TRAINING FINALIZADO")

# ==========================
# Lanzador principal
# ==========================
if __name__ == "__main__":
    main()
