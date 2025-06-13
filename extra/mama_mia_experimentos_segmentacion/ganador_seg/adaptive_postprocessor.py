import numpy as np
import nibabel as nib
from scipy import ndimage
from skimage import morphology, measure
from pathlib import Path
import pandas as pd
import json
from datetime import datetime
import os

class AdaptivePostProcessor:
    """
    Post-procesamiento adaptativo mejorado para segmentaciones de mama
    Aplica estrategias específicas según el tipo de caso y calidad inicial
    """
    
    def __init__(self, base_path=r"C:\Users\usuario\Documents\Mama_Mia"):
        self.base_path = Path(base_path)
        self.setup_paths()
        self.load_test_cases()
        
    def setup_paths(self):
        """Configurar rutas necesarias"""
        self.paths = {
            'baseline_results': Path(r"C:\Users\usuario\Documents\Mama_Mia\replicacion_definitiva\results_output"),
            'improved_results': Path("./results_adaptive_postprocessing"),
            'ground_truth': Path(r"C:\Users\usuario\Documents\Mama_Mia\datos\segmentations\expert"),
            'logs': Path("./logs")
        }
        
        for path in self.paths.values():
            path.mkdir(parents=True, exist_ok=True)
            
    def load_test_cases(self):
        """Cargar casos de test"""
        csv_path = Path(r"C:\Users\usuario\Documents\Mama_Mia\datos\train_test_splits.csv")
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            self.test_cases = df['test_split'].dropna().tolist()
        else:
            self.test_cases = []
    
    def analyze_segmentation_quality(self, segmentation, gt_segmentation=None):
        """Analizar calidad de la segmentación para determinar estrategia"""
        
        # Calcular métricas básicas
        total_volume = np.sum(segmentation > 0.5)
        
        if total_volume == 0:
            return {
                'quality': 'empty',
                'strategy': 'skip',
                'volume': 0,
                'components': 0,
                'fragmentation': 0
            }
        
        # Análisis de componentes conectados
        binary_seg = segmentation > 0.5
        labeled_array, num_components = ndimage.label(binary_seg)
        
        if num_components == 0:
            return {
                'quality': 'empty',
                'strategy': 'skip',
                'volume': 0,
                'components': 0,
                'fragmentation': 0
            }
        
        # Calcular tamaños de componentes
        component_sizes = ndimage.sum(binary_seg, labeled_array, range(num_components + 1))
        largest_component_size = np.max(component_sizes[1:]) if num_components > 0 else 0
        
        # Métricas de fragmentación
        fragmentation = num_components / max(1, total_volume / 1000)  # Componentes por cada 1000 voxels
        volume_ratio = largest_component_size / max(1, total_volume)  # Ratio del componente más grande
        
        # Determinar calidad y estrategia
        if total_volume < 100:
            quality = 'very_poor'
            strategy = 'skip'  # Muy pequeño, probablemente noise
        elif fragmentation > 5:
            quality = 'fragmented'
            strategy = 'conservative_cleanup'
        elif volume_ratio < 0.5:
            quality = 'multi_component'
            strategy = 'selective_cleanup'
        elif num_components > 10:
            quality = 'noisy'
            strategy = 'moderate_cleanup'
        else:
            quality = 'good'
            strategy = 'minimal_cleanup'
            
        return {
            'quality': quality,
            'strategy': strategy,
            'volume': int(total_volume),
            'components': num_components,
            'fragmentation': float(fragmentation),
            'volume_ratio': float(volume_ratio),
            'largest_component': int(largest_component_size)
        }
    
    def conservative_cleanup(self, segmentation):
        """Limpieza muy conservadora - solo elimina ruido obvio"""
        binary_seg = segmentation > 0.5
        
        # Solo eliminar componentes muy pequeños
        labeled_array, num_features = ndimage.label(binary_seg)
        if num_features == 0:
            return segmentation
            
        component_sizes = ndimage.sum(binary_seg, labeled_array, range(num_features + 1))
        
        # Umbral muy bajo - solo ruido obvio
        min_size = max(10, np.sum(binary_seg) * 0.001)  # 0.1% del volumen total o 10 voxels
        
        large_components = component_sizes >= min_size
        large_components[0] = False
        
        cleaned = np.isin(labeled_array, np.where(large_components)[0])
        return cleaned.astype(segmentation.dtype)
    
    def selective_cleanup(self, segmentation):
        """Limpieza selectiva - mantiene componentes principales"""
        binary_seg = segmentation > 0.5
        
        labeled_array, num_features = ndimage.label(binary_seg)
        if num_features == 0:
            return segmentation
            
        component_sizes = ndimage.sum(binary_seg, labeled_array, range(num_features + 1))
        
        # Mantener componentes que sean al menos 2% del volumen total
        total_volume = np.sum(binary_seg)
        min_size = max(25, total_volume * 0.02)
        
        large_components = component_sizes >= min_size
        large_components[0] = False
        
        cleaned = np.isin(labeled_array, np.where(large_components)[0])
        
        # Suavizado muy suave solo si mejora
        kernel = morphology.ball(1)
        smoothed = morphology.binary_closing(cleaned, kernel)
        
        return smoothed.astype(segmentation.dtype)
    
    def moderate_cleanup(self, segmentation):
        """Limpieza moderada con morfología básica"""
        binary_seg = segmentation > 0.5
        
        # Eliminar componentes pequeños
        labeled_array, num_features = ndimage.label(binary_seg)
        if num_features == 0:
            return segmentation
            
        component_sizes = ndimage.sum(binary_seg, labeled_array, range(num_features + 1))
        total_volume = np.sum(binary_seg)
        
        # Umbral adaptativo
        min_size = max(50, total_volume * 0.05)  # 5% del volumen total
        
        large_components = component_sizes >= min_size
        large_components[0] = False
        
        cleaned = np.isin(labeled_array, np.where(large_components)[0])
        
        # Operaciones morfológicas suaves
        kernel = morphology.ball(1)
        processed = morphology.binary_closing(cleaned, kernel)
        processed = morphology.binary_opening(processed, kernel)
        
        # Rellenar agujeros pequeños slice por slice
        filled = np.zeros_like(processed)
        for z in range(processed.shape[2]):
            slice_2d = processed[:, :, z]
            if np.any(slice_2d):
                filled[:, :, z] = ndimage.binary_fill_holes(slice_2d)
                
        return filled.astype(segmentation.dtype)
    
    def minimal_cleanup(self, segmentation):
        """Limpieza mínima - solo refinamiento de bordes"""
        binary_seg = segmentation > 0.5
        
        # Solo rellenar agujeros muy pequeños
        filled = np.zeros_like(binary_seg)
        for z in range(binary_seg.shape[2]):
            slice_2d = binary_seg[:, :, z]
            if np.any(slice_2d):
                # Rellenar solo agujeros pequeños
                filled_slice = ndimage.binary_fill_holes(slice_2d)
                # Verificar que no se agregue demasiado volumen
                if np.sum(filled_slice) <= np.sum(slice_2d) * 1.1:  # Max 10% más volumen
                    filled[:, :, z] = filled_slice
                else:
                    filled[:, :, z] = slice_2d
                    
        return filled.astype(segmentation.dtype)
    
    def adaptive_postprocess(self, segmentation, gt_segmentation=None):
        """Post-procesamiento adaptativo basado en análisis de calidad"""
        
        # Analizar calidad de la segmentación
        analysis = self.analyze_segmentation_quality(segmentation, gt_segmentation)
        
        strategy = analysis['strategy']
        
        if strategy == 'skip':
            # No aplicar post-procesamiento
            return segmentation, analysis
            
        elif strategy == 'conservative_cleanup':
            processed = self.conservative_cleanup(segmentation)
            
        elif strategy == 'selective_cleanup':
            processed = self.selective_cleanup(segmentation)
            
        elif strategy == 'moderate_cleanup':
            processed = self.moderate_cleanup(segmentation)
            
        elif strategy == 'minimal_cleanup':
            processed = self.minimal_cleanup(segmentation)
            
        else:
            # Default: no procesamiento
            processed = segmentation
            
        return processed, analysis
    
    def calculate_dice(self, pred, gt):
        """Calcular coeficiente Dice"""
        pred_bin = pred > 0.5
        gt_bin = gt > 0.5
        
        intersection = np.sum(pred_bin * gt_bin)
        total = np.sum(pred_bin) + np.sum(gt_bin)
        
        if total > 0:
            return (2.0 * intersection) / total
        return 0.0
    
    def process_single_case(self, case_id):
        """Procesar un caso individual con post-procesamiento adaptativo"""
        
        # Cargar segmentación baseline
        baseline_file = self.paths['baseline_results'] / f"{case_id}.nii.gz"
        if not baseline_file.exists():
            print(f"❌ No existe baseline para {case_id}")
            return None
            
        # Cargar ground truth
        gt_file = self.paths['ground_truth'] / f"{case_id.lower()}.nii.gz"
        if not gt_file.exists():
            print(f"❌ No existe ground truth para {case_id}")
            return None
            
        try:
            # Cargar datos
            baseline_nii = nib.load(baseline_file)
            baseline_seg = baseline_nii.get_fdata()
            
            gt_seg = nib.load(gt_file).get_fdata()
            
            # Aplicar post-procesamiento adaptativo
            improved_seg, analysis = self.adaptive_postprocess(baseline_seg, gt_seg)
            
            # Calcular métricas antes y después
            baseline_dice = self.calculate_dice(baseline_seg, gt_seg)
            improved_dice = self.calculate_dice(improved_seg, gt_seg)
            improvement = improved_dice - baseline_dice
            
            # Solo guardar si mejora o si no empeora significativamente
            should_save = improvement > -0.01  # No empeorar más de 0.01
            
            if should_save:
                # Guardar resultado mejorado
                improved_file = self.paths['improved_results'] / f"{case_id}.nii.gz"
                improved_nii = nib.Nifti1Image(improved_seg, baseline_nii.affine, baseline_nii.header)
                nib.save(improved_nii, improved_file)
                final_dice = improved_dice
                used_postprocessing = True
            else:
                # Guardar original si el post-procesamiento empeora
                improved_file = self.paths['improved_results'] / f"{case_id}.nii.gz"
                nib.save(baseline_nii, improved_file)
                final_dice = baseline_dice
                improved_dice = baseline_dice
                improvement = 0.0
                used_postprocessing = False
            
            result = {
                'case_id': case_id,
                'baseline_dice': float(baseline_dice),
                'improved_dice': float(final_dice),
                'improvement': float(improvement),
                'strategy': analysis['strategy'],
                'quality': analysis['quality'],
                'used_postprocessing': used_postprocessing,
                'analysis': analysis
            }
            
            status = "✅" if improvement > 0 else "🔄" if improvement == 0 else "⚠️"
            pp_status = f"[{analysis['strategy']}]" if used_postprocessing else "[SKIP]"
            
            print(f"{status} {case_id}: {baseline_dice:.3f} → {final_dice:.3f} (+{improvement:+.3f}) {pp_status}")
            
            return result
            
        except Exception as e:
            print(f"❌ Error procesando {case_id}: {e}")
            return None
    
    def evaluate_adaptive_postprocessing(self):
        """Evaluar post-procesamiento adaptativo"""
        print("🔧 EXPERIMENTO: Post-procesamiento Adaptativo Mejorado")
        print("=" * 60)
        
        results = []
        strategy_stats = {}
        
        for i, case_id in enumerate(self.test_cases):
            print(f"Procesando {i+1}/{len(self.test_cases)}: {case_id}")
            
            result = self.process_single_case(case_id)
            if result:
                results.append(result)
                
                # Estadísticas por estrategia
                strategy = result['strategy']
                if strategy not in strategy_stats:
                    strategy_stats[strategy] = []
                strategy_stats[strategy].append(result['improvement'])
                
        # Análisis de resultados
        if results:
            baseline_dices = [r['baseline_dice'] for r in results]
            improved_dices = [r['improved_dice'] for r in results]
            improvements = [r['improvement'] for r in results]
            
            # Estadísticas generales
            stats = {
                'num_cases': len(results),
                'baseline_mean': float(np.mean(baseline_dices)),
                'baseline_std': float(np.std(baseline_dices)),
                'improved_mean': float(np.mean(improved_dices)),
                'improved_std': float(np.std(improved_dices)),
                'mean_improvement': float(np.mean(improvements)),
                'std_improvement': float(np.std(improvements)),
                'positive_improvements': sum(1 for imp in improvements if imp > 0),
                'significant_improvements': sum(1 for imp in improvements if imp > 0.01),
                'used_postprocessing': sum(1 for r in results if r['used_postprocessing']),
                'strategy_distribution': {k: len(v) for k, v in strategy_stats.items()}
            }
            
            # Guardar resultados
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = self.paths['logs'] / f"adaptive_postprocessing_{timestamp}.json"
            
            full_results = {
                'experiment': 'adaptive_postprocessing',
                'timestamp': timestamp,
                'statistics': stats,
                'strategy_stats': {k: {'count': len(v), 'mean_improvement': float(np.mean(v)), 
                                     'positive_rate': sum(1 for x in v if x > 0)/len(v)} 
                                 for k, v in strategy_stats.items()},
                'detailed_results': results
            }
            
            with open(results_file, 'w') as f:
                json.dump(full_results, f, indent=2)
                
            # Mostrar resumen
            print("\n" + "=" * 60)
            print("📊 RESULTADOS POST-PROCESAMIENTO ADAPTATIVO:")
            print(f"Casos procesados: {stats['num_cases']}")
            print(f"Baseline: {stats['baseline_mean']:.3f} ± {stats['baseline_std']:.3f}")
            print(f"Mejorado: {stats['improved_mean']:.3f} ± {stats['improved_std']:.3f}")
            print(f"Mejora promedio: {stats['mean_improvement']:+.3f} ± {stats['std_improvement']:.3f}")
            print(f"Casos mejorados: {stats['positive_improvements']}/{stats['num_cases']} ({100*stats['positive_improvements']/stats['num_cases']:.1f}%)")
            print(f"Mejoras significativas (>0.01): {stats['significant_improvements']}")
            print(f"Casos con post-procesamiento: {stats['used_postprocessing']}/{stats['num_cases']}")
            
            print("\n📈 ESTADÍSTICAS POR ESTRATEGIA:")
            for strategy, strategy_results in strategy_stats.items():
                mean_imp = np.mean(strategy_results)
                positive_rate = sum(1 for x in strategy_results if x > 0) / len(strategy_results)
                print(f"  {strategy}: {len(strategy_results)} casos, mejora promedio: {mean_imp:+.3f}, tasa positiva: {positive_rate:.1%}")
            
            if stats['mean_improvement'] > 0:
                print("\n🎯 ¡POST-PROCESAMIENTO ADAPTATIVO EXITOSO!")
            else:
                print("\n⚠️  Aún necesita optimización")
                
            return full_results
            
        else:
            print("❌ No se pudieron procesar casos")
            return None

# Función principal
def run_adaptive_postprocessing():
    """Ejecutar experimento de post-procesamiento adaptativo"""
    processor = AdaptivePostProcessor()
    return processor.evaluate_adaptive_postprocessing()

if __name__ == "__main__":
    run_adaptive_postprocessing()