import numpy as np
import nibabel as nib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import pandas as pd
import json
from datetime import datetime
import os
from scipy.ndimage import zoom
import torch.nn.functional as F

def find_image_file(base_path, case_id):
    """Buscar archivo de imagen usando diferentes patrones de nombres"""
    base_path = Path(base_path)
    
    # Patrones posibles
    patterns = [
        f"{case_id}.nii.gz",                    # DUKE_019.nii.gz
        f"{case_id.lower()}.nii.gz",            # duke_019.nii.gz
        f"{case_id.lower()}_0000.nii.gz",       # duke_019_0000.nii.gz
    ]
    
    # Si es DUKE_XXX, probar duke_XXX_0000.nii.gz
    if case_id.startswith('DUKE_'):
        duke_num = case_id.split('_')[1]
        patterns.extend([
            f"duke_{duke_num.zfill(3)}_0000.nii.gz",  # duke_019_0000.nii.gz
            f"duke_{duke_num}_0000.nii.gz",           # duke_19_0000.nii.gz
        ])
    
    # Si es ISPY1_XXXX, probar ispy1_xxxx_0000.nii.gz
    if case_id.startswith('ISPY1_'):
        ispy_num = case_id.split('_')[1]
        patterns.extend([
            f"ispy1_{ispy_num}_0000.nii.gz",
            f"ispy1_{ispy_num.zfill(4)}_0000.nii.gz",
        ])
        
    # Si es ISPY2_XXXXXX, probar ispy2_xxxxxx_0000.nii.gz
    if case_id.startswith('ISPY2_'):
        ispy_num = case_id.split('_')[1]
        patterns.extend([
            f"ispy2_{ispy_num}_0000.nii.gz",
            f"ispy2_{ispy_num.zfill(6)}_0000.nii.gz",
        ])
        
    # Si es NACT_XX, probar nact_xx_0000.nii.gz
    if case_id.startswith('NACT_'):
        nact_num = case_id.split('_')[1]
        patterns.extend([
            f"nact_{nact_num}_0000.nii.gz",
            f"nact_{nact_num.zfill(2)}_0000.nii.gz",
        ])
    
    # Buscar archivos que coincidan
    for pattern in patterns:
        candidate = base_path / pattern
        if candidate.exists():
            return candidate
            
    return None
class SegmentationRefinementGenerator(nn.Module):
    """
    Generador GAN para refinar segmentaciones de mama
    Entrada: Segmentación cruda + Imagen original
    Salida: Segmentación refinada
    """
    
    def __init__(self, input_channels=2):  # Segmentación + Imagen
        super().__init__()
        
        # Encoder
        self.enc1 = self.conv_block(input_channels, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)
        
        # Bottleneck
        self.bottleneck = self.conv_block(512, 1024)
        
        # Decoder con skip connections
        self.dec4 = self.upconv_block(1024 + 512, 512)
        self.dec3 = self.upconv_block(512 + 256, 256)
        self.dec2 = self.upconv_block(256 + 128, 128)
        self.dec1 = self.upconv_block(128 + 64, 64)
        
        # Output layer
        self.final_conv = nn.Sequential(
            nn.Conv3d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        self.maxpool = nn.MaxPool3d(2)
        
    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def upconv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        # Encoder con skip connections
        e1 = self.enc1(x)
        e2 = self.enc2(self.maxpool(e1))
        e3 = self.enc3(self.maxpool(e2))
        e4 = self.enc4(self.maxpool(e3))
        
        # Bottleneck
        b = self.bottleneck(self.maxpool(e4))
        
        # Decoder con skip connections
        d4 = self.dec4(torch.cat([F.interpolate(b, e4.shape[2:], mode='trilinear'), e4], dim=1))
        d3 = self.dec3(torch.cat([F.interpolate(d4, e3.shape[2:], mode='trilinear'), e3], dim=1))
        d2 = self.dec2(torch.cat([F.interpolate(d3, e2.shape[2:], mode='trilinear'), e2], dim=1))
        d1 = self.dec1(torch.cat([F.interpolate(d2, e1.shape[2:], mode='trilinear'), e1], dim=1))
        
        # Output refinado
        refined = self.final_conv(d1)
        
        return refined

class SegmentationDiscriminator(nn.Module):
    """
    Discriminador para distinguir entre segmentaciones reales y refinadas
    """
    
    def __init__(self, input_channels=3):  # Imagen + Segmentación + Ground Truth
        super().__init__()
        
        self.conv_layers = nn.Sequential(
            # Primera capa
            nn.Conv3d(input_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Segunda capa
            nn.Conv3d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Tercera capa
            nn.Conv3d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Cuarta capa
            nn.Conv3d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(512),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Quinta capa
            nn.Conv3d(512, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        # Clasificador final
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        features = self.conv_layers(x)
        output = self.classifier(features)
        return output

class SegmentationDataset(Dataset):
    """
    Dataset para entrenar GAN de refinamiento
    """
    
    def __init__(self, image_paths, baseline_paths, gt_paths, target_size=(64, 64, 64)):
        self.image_paths = image_paths
        self.baseline_paths = baseline_paths
        self.gt_paths = gt_paths
        self.target_size = target_size
        
    def __len__(self):
        return len(self.image_paths)
        
    def normalize_volume(self, volume):
        """Normalizar volumen entre 0 y 1"""
        vmin, vmax = volume.min(), volume.max()
        if vmax > vmin:
            return (volume - vmin) / (vmax - vmin)
        return volume
        
    def __getitem__(self, idx):
        try:
            # Cargar imagen original
            img = nib.load(self.image_paths[idx]).get_fdata()
            
            # Cargar segmentación baseline
            baseline_seg = nib.load(self.baseline_paths[idx]).get_fdata()
            
            # Cargar ground truth
            gt_seg = nib.load(self.gt_paths[idx]).get_fdata()
            
            # Redimensionar todos los volúmenes
            volumes = [img, baseline_seg, gt_seg]
            resized_volumes = []
            
            for vol in volumes:
                zoom_factors = [self.target_size[i] / vol.shape[i] for i in range(3)]
                resized_vol = zoom(vol, zoom_factors, order=1)
                resized_volumes.append(resized_vol)
                
            img_resized, baseline_resized, gt_resized = resized_volumes
            
            # Normalizar
            img_norm = self.normalize_volume(img_resized)
            baseline_norm = baseline_resized  # Ya está entre 0-1
            gt_norm = gt_resized  # Ya está entre 0-1
            
            # Convertir a tensores
            img_tensor = torch.FloatTensor(img_norm).unsqueeze(0)
            baseline_tensor = torch.FloatTensor(baseline_norm).unsqueeze(0)
            gt_tensor = torch.FloatTensor(gt_norm).unsqueeze(0)
            
            # Input para generador: imagen + segmentación baseline
            generator_input = torch.cat([img_tensor, baseline_tensor], dim=0)
            
            # Input para discriminador: imagen + segmentación + ground truth
            discriminator_real = torch.cat([img_tensor, gt_tensor], dim=0)
            
            return {
                'generator_input': generator_input,
                'discriminator_real': discriminator_real,
                'baseline_seg': baseline_tensor,
                'gt_seg': gt_tensor,
                'image': img_tensor
            }
            
        except Exception as e:
            # Retornar datos aleatorios en caso de error
            return {
                'generator_input': torch.randn(2, *self.target_size),
                'discriminator_real': torch.randn(2, *self.target_size),
                'baseline_seg': torch.randn(1, *self.target_size),
                'gt_seg': torch.randn(1, *self.target_size),
                'image': torch.randn(1, *self.target_size)
            }

class SegmentationRefinementGAN:
    """
    GAN completa para refinar segmentaciones de mama
    """
    
    def __init__(self, base_path=r"C:\Users\usuario\Documents\Mama_Mia"):
        self.base_path = Path(base_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.target_size = (64, 64, 64)
        
        self.setup_paths()
        self.load_cases()
        
        # Inicializar modelos
        self.generator = SegmentationRefinementGenerator(input_channels=2).to(self.device)
        self.discriminator = SegmentationDiscriminator(input_channels=2).to(self.device)
        
        # Optimizadores
        self.g_optimizer = optim.Adam(self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.d_optimizer = optim.Adam(self.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        
        # Funciones de pérdida
        self.adversarial_loss = nn.BCELoss()
        self.l1_loss = nn.L1Loss()
        self.dice_loss_fn = self.dice_loss
        
    def setup_paths(self):
        """Configurar rutas necesarias"""
        self.paths = {
            'images': Path(r"C:\Users\usuario\Documents\Mama_Mia\datos\images"),
            'ground_truth': Path(r"C:\Users\usuario\Documents\Mama_Mia\datos\segmentations\expert"),
            'baseline_results': Path(r"C:\Users\usuario\Documents\Mama_Mia\replicacion_definitiva\results_output"),
            'improved_results': Path("./results_gan_refinement"),
            'models': Path("./models"),
            'logs': Path("./logs")
        }
        
        for path in self.paths.values():
            path.mkdir(parents=True, exist_ok=True)
            
    def load_cases(self):
        """Cargar casos de entrenamiento y test"""
        csv_path = Path(r"C:\Users\usuario\Documents\Mama_Mia\datos\train_test_splits.csv")
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            self.train_cases = df['train_split'].dropna().tolist()
            self.test_cases = df['test_split'].dropna().tolist()
        else:
            self.train_cases = []
            self.test_cases = []
            
    def dice_loss(self, pred, target, smooth=1e-6):
        """Función de pérdida Dice"""
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        intersection = (pred_flat * target_flat).sum()
        dice = (2.0 * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)
        
        return 1 - dice
        
    def generator_loss(self, fake_output, refined_seg, gt_seg):
        """Pérdida combinada del generador"""
        # Pérdida adversarial
        adversarial = self.adversarial_loss(fake_output, torch.ones_like(fake_output))
        
        # Pérdida L1 (suavidad)
        l1 = self.l1_loss(refined_seg, gt_seg)
        
        # Pérdida Dice (similitud geométrica)
        dice = self.dice_loss_fn(refined_seg, gt_seg)
        
        # Combinar pérdidas
        total_loss = adversarial + 10.0 * l1 + 20.0 * dice
        
        return total_loss, adversarial, l1, dice
        
    def prepare_training_data(self):
        """Preparar datos de entrenamiento"""
        image_paths = []
        baseline_paths = []
        gt_paths = []
        
        # Usar subset de casos para entrenamiento rápido
        train_subset = self.train_cases[:30] if len(self.train_cases) > 30 else self.train_cases
        
        for case_id in train_subset:
            img_path = find_image_file(self.paths['images'], case_id)
            baseline_path = self.paths['baseline_results'] / f"{case_id}.nii.gz"
            gt_path = self.paths['ground_truth'] / f"{case_id.lower()}.nii.gz"
            
            if img_path.exists() and baseline_path.exists() and gt_path.exists():
                image_paths.append(img_path)
                baseline_paths.append(baseline_path)
                gt_paths.append(gt_path)
                
        return image_paths, baseline_paths, gt_paths
        
    def train_refinement_gan(self, epochs=100, batch_size=2):
        """Entrenar GAN de refinamiento"""
        print("🎯 Entrenando GAN para refinamiento de segmentaciones...")
        
        # Preparar datos
        image_paths, baseline_paths, gt_paths = self.prepare_training_data()
        
        if len(image_paths) == 0:
            print("❌ No se encontraron datos para entrenar")
            return False
            
        print(f"📊 Datos de entrenamiento: {len(image_paths)} casos")
        
        dataset = SegmentationDataset(image_paths, baseline_paths, gt_paths, self.target_size)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        
        # Listas para guardar pérdidas
        g_losses = []
        d_losses = []
        dice_scores = []
        
        for epoch in range(epochs):
            epoch_g_loss = 0
            epoch_d_loss = 0
            epoch_dice = 0
            num_batches = 0
            
            for batch in dataloader:
                try:
                    generator_input = batch['generator_input'].to(self.device)
                    discriminator_real = batch['discriminator_real'].to(self.device)
                    gt_seg = batch['gt_seg'].to(self.device)
                    image = batch['image'].to(self.device)
                    
                    batch_size_actual = generator_input.size(0)
                    
                    # Etiquetas
                    real_labels = torch.ones(batch_size_actual, 1).to(self.device)
                    fake_labels = torch.zeros(batch_size_actual, 1).to(self.device)
                    
                    # =================== Entrenar Discriminador ===================
                    self.d_optimizer.zero_grad()
                    
                    # Discriminador en segmentaciones reales
                    real_output = self.discriminator(discriminator_real)
                    d_loss_real = self.adversarial_loss(real_output, real_labels)
                    
                    # Generar segmentaciones refinadas
                    refined_seg = self.generator(generator_input)
                    
                    # Discriminador en segmentaciones refinadas
                    fake_input = torch.cat([image, refined_seg.detach()], dim=1)
                    fake_output = self.discriminator(fake_input)
                    d_loss_fake = self.adversarial_loss(fake_output, fake_labels)
                    
                    # Pérdida total discriminador
                    d_loss = (d_loss_real + d_loss_fake) / 2
                    d_loss.backward()
                    self.d_optimizer.step()
                    
                    # =================== Entrenar Generador ===================
                    self.g_optimizer.zero_grad()
                    
                    # Generador trata de engañar al discriminador
                    fake_input = torch.cat([image, refined_seg], dim=1)
                    fake_output = self.discriminator(fake_input)
                    
                    # Pérdida combinada del generador
                    g_loss, adv_loss, l1_loss, dice_loss = self.generator_loss(fake_output, refined_seg, gt_seg)
                    
                    g_loss.backward()
                    self.g_optimizer.step()
                    
                    # Calcular Dice score
                    with torch.no_grad():
                        dice_score = 1 - dice_loss.item()
                        
                    # Acumular pérdidas
                    epoch_g_loss += g_loss.item()
                    epoch_d_loss += d_loss.item()
                    epoch_dice += dice_score
                    num_batches += 1
                    
                except Exception as e:
                    print(f"Error en batch: {e}")
                    continue
                    
            # Promediar métricas
            if num_batches > 0:
                avg_g_loss = epoch_g_loss / num_batches
                avg_d_loss = epoch_d_loss / num_batches
                avg_dice = epoch_dice / num_batches
                
                g_losses.append(avg_g_loss)
                d_losses.append(avg_d_loss)
                dice_scores.append(avg_dice)
                
                if epoch % 20 == 0:
                    print(f"Época {epoch}/{epochs} - G_Loss: {avg_g_loss:.4f}, D_Loss: {avg_d_loss:.4f}, Dice: {avg_dice:.3f}")
                    
        # Guardar modelos entrenados
        torch.save({
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'g_losses': g_losses,
            'd_losses': d_losses,
            'dice_scores': dice_scores
        }, self.paths['models'] / 'segmentation_refinement_gan.pth')
        
        print("✅ GAN de refinamiento entrenada y guardada")
        return True
        
    def refine_segmentation(self, case_id):
        """Refinar segmentación individual usando GAN"""
        try:
            # Cargar datos
            img_path = find_image_file(self.paths['images'], case_id)
            baseline_path = self.paths['baseline_results'] / f"{case_id}.nii.gz"
            
            if not img_path.exists() or not baseline_path.exists():
                return None
                
            # Cargar volúmenes
            img_nii = nib.load(img_path)
            img = img_nii.get_fdata()
            
            baseline_nii = nib.load(baseline_path)
            baseline_seg = baseline_nii.get_fdata()
            
            # Preparar para refinamiento
            original_shape = img.shape
            
            # Redimensionar
            zoom_factors = [self.target_size[i] / original_shape[i] for i in range(3)]
            img_resized = zoom(img, zoom_factors, order=1)
            baseline_resized = zoom(baseline_seg, zoom_factors, order=1)
            
            # Normalizar
            img_norm = (img_resized - img_resized.min()) / (img_resized.max() - img_resized.min() + 1e-8)
            
            # Convertir a tensores
            img_tensor = torch.FloatTensor(img_norm).unsqueeze(0).unsqueeze(0).to(self.device)
            baseline_tensor = torch.FloatTensor(baseline_resized).unsqueeze(0).unsqueeze(0).to(self.device)
            
            # Input para generador
            generator_input = torch.cat([img_tensor, baseline_tensor], dim=1)
            
            # Refinamiento
            self.generator.eval()
            with torch.no_grad():
                refined_tensor = self.generator(generator_input)
                refined_resized = refined_tensor.squeeze().cpu().numpy()
                
            # Redimensionar de vuelta al tamaño original
            zoom_back_factors = [original_shape[i] / self.target_size[i] for i in range(3)]
            refined_full = zoom(refined_resized, zoom_back_factors, order=1)
            
            return refined_full, baseline_nii.affine, baseline_nii.header
            
        except Exception as e:
            print(f"Error refinando {case_id}: {e}")
            return None
            
    def calculate_dice(self, pred, gt):
        """Calcular coeficiente Dice"""
        pred_bin = pred > 0.5
        gt_bin = gt > 0.5
        
        intersection = np.sum(pred_bin * gt_bin)
        total = np.sum(pred_bin) + np.sum(gt_bin)
        
        if total > 0:
            return (2.0 * intersection) / total
        return 0.0
        
    def evaluate_gan_refinement(self):
        """Evaluar GAN de refinamiento"""
        print("🎯 EXPERIMENTO: GAN Refinement de Segmentaciones")
        print("=" * 60)
        
        # Cargar o entrenar GAN
        gan_model_path = self.paths['models'] / 'segmentation_refinement_gan.pth'
        
        if not gan_model_path.exists():
            print("Entrenando nueva GAN de refinamiento...")
            success = self.train_refinement_gan(epochs=50)
            if not success:
                return None
        else:
            print("Cargando GAN de refinamiento pre-entrenada...")
            checkpoint = torch.load(gan_model_path)
            self.generator.load_state_dict(checkpoint['generator_state_dict'])
            self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
            
        # Evaluar en casos de test
        results = []
        
        for i, case_id in enumerate(self.test_cases):
            print(f"Procesando {i+1}/{len(self.test_cases)}: {case_id}")
            
            try:
                # Cargar ground truth
                gt_file = self.paths['ground_truth'] / f"{case_id.lower()}.nii.gz"
                baseline_file = self.paths['baseline_results'] / f"{case_id}.nii.gz"
                
                if not gt_file.exists() or not baseline_file.exists():
                    continue
                    
                baseline_seg = nib.load(baseline_file).get_fdata()
                gt_seg = nib.load(gt_file).get_fdata()
                
                # Refinar segmentación
                refine_result = self.refine_segmentation(case_id)
                
                if refine_result is None:
                    continue
                    
                refined_seg, affine, header = refine_result
                
                # Guardar resultado refinado
                refined_file = self.paths['improved_results'] / f"{case_id}.nii.gz"
                refined_nii = nib.Nifti1Image(refined_seg, affine, header)
                nib.save(refined_nii, refined_file)
                
                # Calcular métricas
                baseline_dice = self.calculate_dice(baseline_seg, gt_seg)
                refined_dice = self.calculate_dice(refined_seg, gt_seg)
                improvement = refined_dice - baseline_dice
                
                result = {
                    'case_id': case_id,
                    'baseline_dice': float(baseline_dice),
                    'improved_dice': float(refined_dice),
                    'improvement': float(improvement),
                    'method': 'gan_refinement'
                }
                
                results.append(result)
                print(f"✅ {case_id}: {baseline_dice:.3f} → {refined_dice:.3f} (+{improvement:+.3f})")
                
            except Exception as e:
                print(f"❌ Error procesando {case_id}: {e}")
                continue
                
        # Análisis de resultados
        if results:
            baseline_dices = [r['baseline_dice'] for r in results]
            improved_dices = [r['improved_dice'] for r in results]
            improvements = [r['improvement'] for r in results]
            
            stats = {
                'num_cases': len(results),
                'baseline_mean': float(np.mean(baseline_dices)),
                'baseline_std': float(np.std(baseline_dices)),
                'improved_mean': float(np.mean(improved_dices)),
                'improved_std': float(np.std(improved_dices)),
                'mean_improvement': float(np.mean(improvements)),
                'std_improvement': float(np.std(improvements)),
                'positive_improvements': sum(1 for imp in improvements if imp > 0),
                'significant_improvements': sum(1 for imp in improvements if imp > 0.01)
            }
            
            # Guardar resultados
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = self.paths['logs'] / f"gan_refinement_results_{timestamp}.json"
            
            full_results = {
                'experiment': 'gan_refinement',
                'timestamp': timestamp,
                'statistics': stats,
                'detailed_results': results
            }
            
            with open(results_file, 'w') as f:
                json.dump(full_results, f, indent=2)
                
            # Mostrar resumen
            print("\n" + "=" * 60)
            print("📊 RESULTADOS GAN REFINEMENT:")
            print(f"Casos procesados: {stats['num_cases']}")
            print(f"Baseline: {stats['baseline_mean']:.3f} ± {stats['baseline_std']:.3f}")
            print(f"Mejorado: {stats['improved_mean']:.3f} ± {stats['improved_std']:.3f}")
            print(f"Mejora promedio: {stats['mean_improvement']:+.3f} ± {stats['std_improvement']:.3f}")
            print(f"Casos mejorados: {stats['positive_improvements']}/{stats['num_cases']} ({100*stats['positive_improvements']/stats['num_cases']:.1f}%)")
            print(f"Mejoras significativas (>0.01): {stats['significant_improvements']}")
            
            if stats['mean_improvement'] > 0:
                print("🎯 ¡GAN REFINEMENT EXITOSO!")
            else:
                print("⚠️  GAN refinement no mejoró el promedio")
                
            print("=" * 60)
            
            return full_results
            
        else:
            print("❌ No se pudieron procesar casos")
            return None

# Función principal para ser llamada desde el launcher
def run_gan_refinement_experiment():
    """Función principal para ejecutar experimento GAN refinement"""
    gan_processor = SegmentationRefinementGAN()
    return gan_processor.evaluate_gan_refinement()

if __name__ == "__main__":
    run_gan_refinement_experiment()