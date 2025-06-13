import numpy as np
import nibabel as nib
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import ndimage
import json

class SegmentationAnalyzer:
    """
    Analizador para identificar patrones en casos problemáticos
    """
    
    def __init__(self, base_path=r"C:\Users\usuario\Documents\Mama_Mia"):
        self.base_path = Path(base_path)
        self.setup_paths()
        self.load_test_cases()
        
    def setup_paths(self):
        """Configurar rutas"""
        self.paths = {
            'baseline_results': Path(r"C:\Users\usuario\Documents\Mama_Mia\replicacion_definitiva\results_output"),
            'ground_truth': Path(r"C:\Users\usuario\Documents\Mama_Mia\datos\segmentations\expert"),
            'analysis': Path("./analysis_output")
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
    
    def analyze_single_case(self, case_id):
        """Analizar un caso individual en detalle"""
        
        baseline_file = self.paths['baseline_results'] / f"{case_id}.nii.gz"
        gt_file = self.paths['ground_truth'] / f"{case_id.lower()}.nii.gz"
        
        if not baseline_file.exists() or not gt_file.exists():
            return None
            
        try:
            # Cargar datos
            baseline_seg = nib.load(baseline_file).get_fdata()
            gt_seg = nib.load(gt_file).get_fdata()
            
            # Análisis detallado
            baseline_binary = baseline_seg > 0.5
            gt_binary = gt_seg > 0.5
            
            # Métricas básicas
            baseline_volume = np.sum(baseline_binary)
            gt_volume = np.sum(gt_binary)
            
            # Análisis de componentes
            labeled_baseline, num_components_baseline = ndimage.label(baseline_binary)
            labeled_gt, num_components_gt = ndimage.label(gt_binary)
            
            # Tamaños de componentes
            if num_components_baseline > 0:
                component_sizes_baseline = ndimage.sum(baseline_binary, labeled_baseline, 
                                                     range(num_components_baseline + 1))[1:]
                largest_component_baseline = np.max(component_sizes_baseline)
                fragmentation_baseline = num_components_baseline / max(1, baseline_volume / 1000)
            else:
                largest_component_baseline = 0
                fragmentation_baseline = 0
                
            if num_components_gt > 0:
                component_sizes_gt = ndimage.sum(gt_binary, labeled_gt, range(num_components_gt + 1))[1:]
                largest_component_gt = np.max(component_sizes_gt)
            else:
                largest_component_gt = 0
            
            # Dice score
            intersection = np.sum(baseline_binary * gt_binary)
            total = np.sum(baseline_binary) + np.sum(gt_binary)
            dice = (2.0 * intersection) / total if total > 0 else 0.0
            
            # Análisis espacial
            if baseline_volume > 0:
                # Centro de masa
                baseline_com = ndimage.center_of_mass(baseline_binary)
                gt_com = ndimage.center_of_mass(gt_binary) if gt_volume > 0 else (0, 0, 0)
                
                # Distancia entre centros de masa
                com_distance = np.sqrt(sum((a - b)**2 for a, b in zip(baseline_com, gt_com)))
                
                # Bounding box
                baseline_coords = np.where(baseline_binary)
                if len(baseline_coords[0]) > 0:
                    bbox_baseline = [
                        (np.min(baseline_coords[0]), np.max(baseline_coords[0])),
                        (np.min(baseline_coords[1]), np.max(baseline_coords[1])),
                        (np.min(baseline_coords[2]), np.max(baseline_coords[2]))
                    ]
                    bbox_volume_baseline = ((bbox_baseline[0][1] - bbox_baseline[0][0]) * 
                                          (bbox_baseline[1][1] - bbox_baseline[1][0]) * 
                                          (bbox_baseline[2][1] - bbox_baseline[2][0]))
                    compactness = baseline_volume / max(1, bbox_volume_baseline)
                else:
                    compactness = 0
                    com_distance = float('inf')
            else:
                compactness = 0
                com_distance = float('inf')
            
            # Determinar tipo de caso
            case_type = 'OTHER'
            if 'ISPY' in case_id.upper():
                case_type = 'ISPY'
            elif 'NACT' in case_id.upper():
                case_type = 'NACT'
            
            # Determinar categoría de problema
            problem_category = 'GOOD'
            if dice < 0.1:
                problem_category = 'VERY_POOR'
            elif dice < 0.3:
                problem_category = 'POOR'  
            elif dice < 0.6:
                problem_category = 'MODERATE'
            elif dice < 0.8:
                problem_category = 'GOOD'
            else:
                problem_category = 'EXCELLENT'
            
            analysis = {
                'case_id': case_id,
                'case_type': case_type,
                'problem_category': problem_category,
                'dice_score': float(dice),
                'baseline_volume': int(baseline_volume),
                'gt_volume': int(gt_volume),
                'volume_ratio': float(baseline_volume / max(1, gt_volume)),
                'num_components_baseline': num_components_baseline,
                'num_components_gt': num_components_gt,
                'largest_component_baseline': int(largest_component_baseline),
                'largest_component_gt': int(largest_component_gt),
                'fragmentation_baseline': float(fragmentation_baseline),
                'compactness': float(compactness),
                'com_distance': float(com_distance),
                'shape': list(baseline_seg.shape)
            }
            
            return analysis
            
        except Exception as e:
            print(f"Error analizando {case_id}: {e}")
            return None
    
    def analyze_all_cases(self):
        """Analizar todos los casos y generar reporte"""
        print("🔍 ANÁLISIS DETALLADO DE SEGMENTACIONES")
        print("=" * 50)
        
        analyses = []
        
        for i, case_id in enumerate(self.test_cases):
            if i % 50 == 0:
                print(f"Analizando {i+1}/{len(self.test_cases)}...")
                
            analysis = self.analyze_single_case(case_id)
            if analysis:
                analyses.append(analysis)
        
        if not analyses:
            print("❌ No se pudieron analizar casos")
            return None
            
        # Convertir a DataFrame para análisis
        df = pd.DataFrame(analyses)
        
        # Análisis por tipo de caso
        print("\n📊 ANÁLISIS POR TIPO DE CASO:")
        type_analysis = df.groupby('case_type').agg({
            'dice_score': ['count', 'mean', 'std', 'min', 'max'],
            'volume_ratio': ['mean', 'std'],
            'fragmentation_baseline': ['mean', 'std'],
            'num_components_baseline': ['mean', 'std']
        }).round(3)
        
        print(type_analysis)
        
        # Análisis por categoría de problema
        print("\n🎯 ANÁLISIS POR CATEGORÍA DE PROBLEMA:")
        problem_analysis = df.groupby('problem_category').agg({
            'case_id': 'count',
            'dice_score': ['mean', 'std'],
            'volume_ratio': ['mean', 'std'],
            'fragmentation_baseline': ['mean', 'std']
        }).round(3)
        
        print(problem_analysis)
        
        # Casos problemáticos específicos
        print("\n⚠️  CASOS MÁS PROBLEMÁTICOS:")
        worst_cases = df.nsmallest(10, 'dice_score')[['case_id', 'case_type', 'dice_score', 
                                                     'volume_ratio', 'fragmentation_baseline']]
        print(worst_cases.to_string(index=False))
        
        # Estadísticas de casos ISPY
        ispy_cases = df[df['case_type'] == 'ISPY']
        if len(ispy_cases) > 0:
            print(f"\n🔍 ANÁLISIS ESPECÍFICO CASOS ISPY ({len(ispy_cases)} casos):")
            print(f"Dice promedio: {ispy_cases['dice_score'].mean():.3f} ± {ispy_cases['dice_score'].std():.3f}")
            print(f"Casos con dice < 0.1: {sum(ispy_cases['dice_score'] < 0.1)}")
            print(f"Fragmentación promedio: {ispy_cases['fragmentation_baseline'].mean():.3f}")
            print(f"Ratio de volumen promedio: {ispy_cases['volume_ratio'].mean():.3f}")
        
        # Guardar análisis
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        
        # CSV con todos los análisis
        csv_file = self.paths['analysis'] / f"segmentation_analysis_{timestamp}.csv"
        df.to_csv(csv_file, index=False)
        
        # JSON con resumen
        summary = {
            'timestamp': timestamp,
            'total_cases': len(df),
            'type_distribution': df['case_type'].value_counts().to_dict(),
            'problem_distribution': df['problem_category'].value_counts().to_dict(),
            'overall_stats': {
                'mean_dice': float(df['dice_score'].mean()),
                'std_dice': float(df['dice_score'].std()),
                'min_dice': float(df['dice_score'].min()),
                'max_dice': float(df['dice_score'].max())
            },
            'recommendations': self.generate_recommendations(df)
        }
        
        json_file = self.paths['analysis'] / f"analysis_summary_{timestamp}.json"
        with open(json_file, 'w') as f:
            json.dump(summary, f, indent=2)
            
        print(f"\n📁 Análisis guardado en:")
        print(f"  CSV: {csv_file}")
        print(f"  JSON: {json_file}")
        
        return df, summary
    
    def generate_recommendations(self, df):
        """Generar recomendaciones basadas en el análisis"""
        recommendations = []
        
        # Análisis de casos ISPY
        ispy_cases = df[df['case_type'] == 'ISPY']
        if len(ispy_cases) > 0:
            poor_ispy = sum(ispy_cases['dice_score'] < 0.1)
            if poor_ispy > len(ispy_cases) * 0.5:
                recommendations.append({
                    'issue': 'ISPY_cases_poor_performance',
                    'description': f'{poor_ispy}/{len(ispy_cases)} casos ISPY con dice < 0.1',
                    'recommendation': 'Casos ISPY necesitan estrategia específica - posible skip o procesamiento muy conservativo'
                })
        
        # Análisis de fragmentación
        high_frag = df[df['fragmentation_baseline'] > 5]
        if len(high_frag) > 0:
            recommendations.append({
                'issue': 'high_fragmentation',
                'description': f'{len(high_frag)} casos con alta fragmentación',
                'recommendation': 'Usar cleanup muy conservativo para casos fragmentados'
            })
        
        # Análisis de ratio de volumen
        low_volume = df[df['volume_ratio'] < 0.1] 
        if len(low_volume) > 0:
            recommendations.append({
                'issue': 'very_small_predictions',
                'description': f'{len(low_volume)} casos con predicciones muy pequeñas',
                'recommendation': 'Skip post-processing para predicciones < 10% del ground truth'
            })
        
        # Casos con muchos componentes
        many_components = df[df['num_components_baseline'] > 20]
        if len(many_components) > 0:
            recommendations.append({
                'issue': 'too_many_components', 
                'description': f'{len(many_components)} casos con >20 componentes',
                'recommendation': 'Aplicar cleanup agresivo solo a componentes muy pequeños'
            })
            
        return recommendations

def run_segmentation_analysis():
    """Ejecutar análisis completo de segmentaciones"""
    analyzer = SegmentationAnalyzer()
    return analyzer.analyze_all_cases()

if __name__ == "__main__":
    run_segmentation_analysis()