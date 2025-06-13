import numpy as np
import nibabel as nib
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import pandas as pd
import json
from datetime import datetime
from scipy.ndimage import zoom, sobel

def find_image_file(base_path, case_id):
    base_path = Path(base_path)
    patterns = [f"{case_id}.nii.gz", f"{case_id.lower()}.nii.gz", f"{case_id.lower()}_0000.nii.gz"]
    
    if case_id.startswith('DUKE_'):
        duke_num = case_id.split('_')[1]
        patterns.extend([
            f"duke_{duke_num.zfill(3)}_0000.nii.gz",
            f"duke_{duke_num}_0000.nii.gz"
        ])
    
    for pattern in patterns:
        candidate = base_path / pattern
        if candidate.exists():
            return candidate
    return None

class MultiChannelDCENet(nn.Module):
    def __init__(self, in_channels=4, out_channels=1):
        super().__init__()
        self.encoder1 = self.conv_block(in_channels, 64)
        self.encoder2 = self.conv_block(64, 128)
        self.encoder3 = self.conv_block(128, 256)
        self.encoder4 = self.conv_block(256, 512)
        self.bottleneck = self.conv_block(512, 1024)
        self.decoder4 = self.upconv_block(1024, 512)
        self.decoder3 = self.upconv_block(512, 256)
        self.decoder2 = self.upconv_block(256, 128)
        self.decoder1 = self.upconv_block(128, 64)
        self.final_conv = nn.Conv3d(64, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
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
        e1 = self.encoder1(x)
        e2 = self.encoder2(self.maxpool(e1))
        e3 = self.encoder3(self.maxpool(e2))
        e4 = self.encoder4(self.maxpool(e3))
        b = self.bottleneck(self.maxpool(e4))
        d4 = self.decoder4(b) + e4
        d3 = self.decoder3(d4) + e3
        d2 = self.decoder2(d3) + e2
        d1 = self.decoder1(d2) + e1
        output = self.final_conv(d1)
        return self.sigmoid(output)

class MultiChannelDCEProcessor:
    def __init__(self, base_path=r"C:\Users\usuario\Documents\Mama_Mia"):
        self.base_path = Path(base_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.setup_paths()
        self.load_test_cases()
    
    def setup_paths(self):
        self.paths = {
            'images': Path(r"C:\Users\usuario\Documents\Mama_Mia\datos\images"),
            'ground_truth': Path(r"C:\Users\usuario\Documents\Mama_Mia\datos\segmentations\expert"),
            'baseline_results': Path(r"C:\Users\usuario\Documents\Mama_Mia\replicacion_definitiva\results_output"),
            'improved_results': Path("./results_multichannel"),
            'models': Path("./models"),
            'logs': Path("./logs")
        }
        for path in self.paths.values():
            path.mkdir(parents=True, exist_ok=True)
    
    def load_test_cases(self):
        csv_path = Path(r"C:\Users\usuario\Documents\Mama_Mia\datos\train_test_splits.csv")
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            self.train_cases = df['train_split'].dropna().tolist()
            self.test_cases = df['test_split'].dropna().tolist()
        else:
            self.train_cases = []
            self.test_cases = []
    
    def normalize_image(self, image):
        p5, p95 = np.percentile(image, [5, 95])
        if p95 > p5:
            return np.clip((image - p5) / (p95 - p5), 0, 1)
        return np.zeros_like(image)
    
    def create_multichannel_image(self, image_path):
        try:
            nii = nib.load(image_path)
            img_data = nii.get_fdata()
            if len(img_data.shape) == 4:
                img_data = img_data[:, :, :, 0]
            img_norm = self.normalize_image(img_data)
            channel1 = img_norm
            channel2 = self.normalize_image(np.abs(sobel(img_norm, axis=0)))
            channel3 = self.normalize_image(np.abs(sobel(img_norm, axis=1)))
            channel4 = self.normalize_image(np.abs(sobel(img_norm, axis=2)))
            multichannel = np.stack([channel1, channel2, channel3, channel4], axis=0)
            return multichannel, nii.affine, nii.header
        except Exception as e:
            print(f"Error: {e}")
            return None, None, None
    
    def dice_loss(self, pred, target, smooth=1e-6):
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        intersection = (pred_flat * target_flat).sum()
        dice = (2.0 * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)
        return 1 - dice
    
    def combined_loss(self, pred, target):
        dice = self.dice_loss(pred, target)
        bce = F.binary_cross_entropy(pred, target)
        return 0.7 * dice + 0.3 * bce
    
    def train_multichannel_model(self, epochs=30, batch_size=2):
        print("Entrenando modelo multi-canal...")
        model = MultiChannelDCENet(in_channels=4, out_channels=1).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
        train_subset = self.train_cases
        train_losses = []
        print(f'Entrenando con {len(train_subset)} casos por {epochs} epocas')
        
        for epoch in range(epochs):
            model.train()
            epoch_loss = 0
            valid_batches = 0
            for case_id in train_subset:
                try:
                    image_path = find_image_file(self.paths['images'], case_id)
                    gt_path = self.paths['ground_truth'] / f"{case_id.lower()}.nii.gz"
                    if image_path is None or not image_path.exists() or not gt_path.exists():
                        continue
                    multichannel, _, _ = self.create_multichannel_image(image_path)
                    if multichannel is None:
                        continue
                    gt = nib.load(gt_path).get_fdata()
                    target_size = (64, 64, 32)
                    zoom_factors = [target_size[i] / multichannel.shape[i+1] for i in range(3)]
                    multichannel_resized = np.stack([zoom(multichannel[c], zoom_factors, order=1) for c in range(4)])
                    gt_zoom_factors = [target_size[i] / gt.shape[i] for i in range(3)]
                    gt_resized = zoom(gt, gt_zoom_factors, order=0)
                    x = torch.FloatTensor(multichannel_resized).unsqueeze(0).to(self.device)
                    y = torch.FloatTensor(gt_resized).unsqueeze(0).unsqueeze(0).to(self.device)
                    optimizer.zero_grad()
                    pred = model(x)
                    loss = self.combined_loss(pred, y)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                    valid_batches += 1
                except Exception as e:
                    continue
            if valid_batches > 0:
                avg_loss = epoch_loss / valid_batches
                train_losses.append(avg_loss)
                if epoch % 5 == 0:
                    print(f"Epoch {epoch}/{epochs}, Loss: {avg_loss:.4f}")
        
        model_path = self.paths['models'] / "multichannel_dce_model.pth"
        torch.save({'model_state_dict': model.state_dict(), 'train_losses': train_losses}, model_path)
        print(f"Modelo guardado: {model_path}")
        return model
    
    def predict_with_multichannel(self, model, case_id):
        try:
            image_path = find_image_file(self.paths['images'], case_id)
            if image_path is None or not image_path.exists():
                return None
            multichannel, affine, header = self.create_multichannel_image(image_path)
            if multichannel is None:
                return None
            original_shape = multichannel.shape[1:]
            target_size = (64, 64, 32)
            zoom_factors = [target_size[i] / original_shape[i] for i in range(3)]
            multichannel_resized = np.stack([zoom(multichannel[c], zoom_factors, order=1) for c in range(4)])
            x = torch.FloatTensor(multichannel_resized).unsqueeze(0).to(self.device)
            model.eval()
            with torch.no_grad():
                pred_resized = model(x).squeeze().cpu().numpy()
            zoom_back_factors = [original_shape[i] / target_size[i] for i in range(3)]
            pred_full = zoom(pred_resized, zoom_back_factors, order=1)
            return pred_full, affine, header
        except Exception as e:
            print(f"Error: {e}")
            return None
    
    def calculate_dice(self, pred, gt):
        pred_bin = pred > 0.5
        gt_bin = gt > 0.5
        intersection = np.sum(pred_bin * gt_bin)
        total = np.sum(pred_bin) + np.sum(gt_bin)
        if total > 0:
            return (2.0 * intersection) / total
        return 0.0
    
    def evaluate_multichannel(self):
        print("EXPERIMENTO: Multi-Canal DCE-MRI")
        print("=" * 60)
        model_path = self.paths['models'] / "multichannel_dce_model.pth"
        if model_path.exists():
            print("Cargando modelo pre-entrenado...")
            model = MultiChannelDCENet(in_channels=4, out_channels=1).to(self.device)
            checkpoint = torch.load(model_path)
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            print("Entrenando nuevo modelo...")
            model = self.train_multichannel_model()
        
        results = []
        for i, case_id in enumerate(self.test_cases):
            print(f"Procesando {i+1}/{len(self.test_cases)}: {case_id}")
            try:
                baseline_file = self.paths['baseline_results'] / f"{case_id}.nii.gz"
                gt_file = self.paths['ground_truth'] / f"{case_id.lower()}.nii.gz"
                if not baseline_file.exists() or not gt_file.exists():
                    continue
                baseline_seg = nib.load(baseline_file).get_fdata()
                gt_seg = nib.load(gt_file).get_fdata()
                pred_result = self.predict_with_multichannel(model, case_id)
                if pred_result is None:
                    continue
                improved_seg, affine, header = pred_result
                improved_file = self.paths['improved_results'] / f"{case_id}.nii.gz"
                improved_nii = nib.Nifti1Image(improved_seg, affine, header)
                nib.save(improved_nii, improved_file)
                baseline_dice = self.calculate_dice(baseline_seg, gt_seg)
                improved_dice = self.calculate_dice(improved_seg, gt_seg)
                improvement = improved_dice - baseline_dice
                result = {
                    'case_id': case_id,
                    'baseline_dice': float(baseline_dice),
                    'improved_dice': float(improved_dice),
                    'improvement': float(improvement),
                    'method': 'multichannel_dce'
                }
                results.append(result)
                print(f"OK {case_id}: {baseline_dice:.3f} -> {improved_dice:.3f} (+{improvement:+.3f})")
            except Exception as e:
                continue
        
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
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = self.paths['logs'] / f"multichannel_results_{timestamp}.json"
            full_results = {
                'experiment': 'multichannel_dce',
                'timestamp': timestamp,
                'statistics': stats,
                'detailed_results': results
            }
            with open(results_file, 'w') as f:
                json.dump(full_results, f, indent=2)
            print("RESULTADOS MULTI-CANAL DCE:")
            print(f"Casos procesados: {stats['num_cases']}")
            print(f"Baseline: {stats['baseline_mean']:.3f} ± {stats['baseline_std']:.3f}")
            print(f"Mejorado: {stats['improved_mean']:.3f} ± {stats['improved_std']:.3f}")
            print(f"Mejora promedio: {stats['mean_improvement']:+.3f} ± {stats['std_improvement']:.3f}")
            if stats['mean_improvement'] > 0:
                print("MULTI-CANAL DCE EXITOSO!")
            return full_results
        return None

def run_multichannel_experiment():
    processor = MultiChannelDCEProcessor()
    return processor.evaluate_multichannel()

if __name__ == "__main__":
    run_multichannel_experiment()
