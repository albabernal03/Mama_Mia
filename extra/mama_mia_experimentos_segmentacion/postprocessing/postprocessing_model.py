import numpy as np
import nibabel as nib
from scipy import ndimage
from skimage import morphology, measure
from pathlib import Path
import pandas as pd
import json
from datetime import datetime
import os

class IntelligentPostProcessor:
    """
    Post-procesamiento anatómico inteligente para segmentaciones de mama
    Corrige errores comunes manteniendo la anatomía correcta
    """
    
    def __init__(self, base_path=r"C:\Users\usuario\Documents\Mama_Mia"):
        self.base_path = Path(base_path)
        self.setup_paths()
        self.load_test_cases()
        
    def setup_paths(self):
        """Configurar rutas necesarias"""
        self.paths = {
            'baseline_results': Path(r"C:\Users\usuario\Documents\Mama_Mia\replicacion_definitiva\results_output"),
            'improved_results': Path("./results_postprocessing"),
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
            
    def remove_small_components(self, segmentation, min_size=50):
        """Eliminar componentes conectados pequeños"""
        # Etiquetar componentes conectados
        labeled_array, num_features = ndimage.label(segmentation)
        
        if num_features == 0:
            return segmentation
            
        # Calcular tamaños de componentes
        component_sizes = ndimage.sum(segmentation, labeled_array, range(num_features + 1))
        
        # Crear máscara para componentes grandes
        large_components = component_sizes >= min_size
        large_components[0] = False  # Excluir background
        
        # Filtrar componentes pequeños
        filtered = np.isin(labeled_array, np.where(large_components)[0])
        
        return filtered.astype(segmentation.dtype)
        
    def fill_holes(self, segmentation):
        """Rellenar agujeros internos en la segmentación"""
        # Procesar slice por slice (axial)
        filled = np.zeros_like(segmentation)
        
        for z in range(segmentation.shape[2]):
            slice_2d = segmentation[:, :, z]
            if np.any(slice_2d):
                # Rellenar agujeros en 2D
                filled_slice = ndimage.binary_fill_holes(slice_2d)
                filled[:, :, z] = filled_slice
                
        return filled.astype(segmentation.dtype)
        
    def morphological_smoothing(self, segmentation, iterations=1):
        """Suavizado morfológico para eliminar irregularidades"""
        # Operaciones morfológicas suaves
        kernel = morphology.ball(1)  # Kernel esférico pequeño
        
        # Closing para rellenar pequeños agujeros
        smoothed = morphology.binary_closing(segmentation, kernel)
        
        # Opening para eliminar pequeñas protuberancias
        smoothed = morphology.binary_opening(smoothed, kernel)
        
        return smoothed.astype(segmentation.dtype)
        
    def anatomical_constraints(self, segmentation, image_shape):
        """Aplicar restricciones anatómicas específicas de mama"""
        
        # 1. Eliminar regiones en bordes de la imagen (artefactos)
        margin = 5
        constrained = segmentation.copy()
        
        # Bordes X
        constrained[:margin, :, :] = 0
        constrained[-margin:, :, :] = 0
        
        # Bordes Y  
        constrained[:, :margin, :] = 0
        constrained[:, -margin:, :] = 0
        
        # Bordes Z
        constrained[:, :, :margin] = 0
        constrained[:, :, -margin:] = 0
        
        # 2. Restricción de tamaño máximo (evitar segmentaciones irrealistas)
        total_volume = np.sum(constrained)
        max_volume = 0.3 * np.prod(image_shape)  # Max 30% del volumen total
        
        if total_volume > max_volume:
            # Mantener solo los componentes más grandes
            labeled_array, num_features = ndimage.label(constrained)
            if num_features > 0:
                component_sizes = ndimage.sum(constrained, labeled_array, range(num_features + 1))
                largest_component = np.argmax(component_sizes[1:]) + 1
                constrained = (labeled_array == largest_component).astype(constrained.dtype)
                
        return constrained
        
    def edge_refinement(self, segmentation, original_image=None):
        """Refinar bordes usando información de la imagen original"""
        if original_image is None:
            return segmentation
            
        # Calcular gradientes de la imagen
        grad_x = ndimage.sobel(original_image, axis=0)
        grad_y = ndimage.sobel(original_image, axis=1) 
        grad_z = ndimage.sobel(original_image, axis=2)
        
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2 + grad_z**2)
        
        # Normalizar gradientes
        if np.max(gradient_magnitude) > 0:
            gradient_magnitude = gradient_magnitude / np.max(gradient_magnitude)
            
        # Refinar bordes donde hay gradientes altos
        distance_map = ndimage.distance_transform_edt(segmentation)
        edge_mask = (distance_map <= 2) & (distance_map > 0)
        
        # Ajustar bordes basado en gradientes
        refined = segmentation.copy()
        strong_edges = gradient_magnitude > 0.3
        
        # Expandir donde hay bordes fuertes cerca
        expansion_mask = edge_mask & strong_edges
        refined[expansion_mask] = 1
        
        return refined.astype(segmentation.dtype)
        
    def intelligent_postprocess(self, segmentation, original_image=None):
        """Pipeline completo de post-procesamiento inteligente"""
        
        # 1. Eliminar componentes pequeños
        processed = self.remove_small_components(segmentation, min_size=100)
        
        # 2. Rellenar agujeros
        processed = self.fill_holes(processed)
        
        # 3. Suavizado morfológico
        processed = self.morphological_smoothing(processed, iterations=1)
        
        # 4. Aplicar restricciones anatómicas
        processed = self.anatomical_constraints(processed, segmentation.shape)
        
        # 5. Refinar bordes (si tenemos imagen original)
        if original_image is not None:
            processed = self.edge_refinement(processed, original_image)
            
        # 6. Post-procesamiento final
        processed = self.remove_small_components(processed, min_size=50)
        
        return processed
        
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
        """Procesar un caso individual"""
        
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
            
            # Aplicar post-procesamiento inteligente
            improved_seg = self.intelligent_postprocess(baseline_seg)
            
            # Guardar resultado mejorado
            improved_file = self.paths['improved_results'] / f"{case_id}.nii.gz"
            improved_nii = nib.Nifti1Image(improved_seg, baseline_nii.affine, baseline_nii.header)
            nib.save(improved_nii, improved_file)
            
            # Calcular métricas
            baseline_dice = self.calculate_dice(baseline_seg, gt_seg)
            improved_dice = self.calculate_dice(improved_seg, gt_seg)
            improvement = improved_dice - baseline_dice
            
            result = {
                'case_id': case_id,
                'baseline_dice': float(baseline_dice),
                'improved_dice': float(improved_dice),
                'improvement': float(improvement),
                'method': 'intelligent_postprocessing'
            }
            
            print(f"✅ {case_id}: {baseline_dice:.3f} → {improved_dice:.3f} (+{improvement:+.3f})")
            
            return result
            
        except Exception as e:
            print(f"❌ Error procesando {case_id}: {e}")
            return None
            
    def evaluate_postprocessing(self):
        """Evaluar post-procesamiento en todos los casos de test"""
        print("🔧 EXPERIMENTO: Post-procesamiento Anatómico Inteligente")
        print("=" * 60)
        
        results = []
        
        for i, case_id in enumerate(self.test_cases):
            print(f"Procesando {i+1}/{len(self.test_cases)}: {case_id}")
            
            result = self.process_single_case(case_id)
            if result:
                results.append(result)
                
        # Análisis de resultados
        if results:
            baseline_dices = [r['baseline_dice'] for r in results]
            improved_dices = [r['improved_dice'] for r in results]
            improvements = [r['improvement'] for r in results]
            
            # Estadísticas
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
            
            # Guardar resultados detallados
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = self.paths['logs'] / f"postprocessing_results_{timestamp}.json"
            
            full_results = {
                'experiment': 'intelligent_postprocessing',
                'timestamp': timestamp,
                'statistics': stats,
                'detailed_results': results
            }
            
            with open(results_file, 'w') as f:
                json.dump(full_results, f, indent=2)
                
            # Mostrar resumen
            print("\n" + "=" * 60)
            print("📊 RESULTADOS POST-PROCESAMIENTO:")
            print(f"Casos procesados: {stats['num_cases']}")
            print(f"Baseline: {stats['baseline_mean']:.3f} ± {stats['baseline_std']:.3f}")
            print(f"Mejorado: {stats['improved_mean']:.3f} ± {stats['improved_std']:.3f}")
            print(f"Mejora promedio: {stats['mean_improvement']:+.3f} ± {stats['std_improvement']:.3f}")
            print(f"Casos mejorados: {stats['positive_improvements']}/{stats['num_cases']} ({100*stats['positive_improvements']/stats['num_cases']:.1f}%)")
            print(f"Mejoras significativas (>0.01): {stats['significant_improvements']}")
            
            if stats['mean_improvement'] > 0:
                print("🎯 ¡POST-PROCESAMIENTO EXITOSO!")
            else:
                print("⚠️  Post-procesamiento no mejoró el promedio")
                
            print("=" * 60)
            
            return full_results
            
        else:
            print("❌ No se pudieron procesar casos")
            return None

# Función principal para ser llamada desde el launcher
def run_postprocessing_experiment():
    """Función principal para ejecutar experimento de post-procesamiento"""
    processor = IntelligentPostProcessor()
    return processor.evaluate_postprocessing()

if __name__ == "__main__":
    run_postprocessing_experiment()