import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class MamaMiaFairnessAnalyzer:
    """
    Analizador de fairness específico para el dataset Mama Mia
    Con mapeo correcto usando train_test_splits.csv
    """
    
    def __init__(self):
        self.results_df = None
        self.demographics_df = None
        self.splits_df = None
        self.combined_df = None
        self.test_ids = None
        
    def load_data(self, results_csv, demographics_excel, splits_csv):
        """Cargar todos los archivos necesarios"""
        print("🚀 CARGANDO DATOS DEL DATASET MAMA MIA")
        print("=" * 50)
        
        # 1. Resultados (case_000, case_001, etc.)
        self.results_df = pd.read_csv(results_csv)
        print(f"📊 Resultados cargados: {len(self.results_df)} casos")
        print(f"   Formato IDs: {self.results_df['patient_id'].iloc[0]} a {self.results_df['patient_id'].iloc[-1]}")
        print(f"   Dice promedio: {self.results_df['dice'].mean():.3f}")
        
        # 2. Train/Test splits (la clave del mapeo)
        self.splits_df = pd.read_csv(splits_csv)
        print(f"\n🔗 Splits cargados: {len(self.splits_df)} pares")
        print(f"   Columnas: {list(self.splits_df.columns)}")
        
        # Extraer solo los test IDs
        self.test_ids = self.splits_df['test_split'].tolist()
        print(f"   Test IDs: {len(self.test_ids)} casos")
        print(f"   Ejemplos: {self.test_ids[:5]}")
        
        # 3. Demographics (DUKE_001, ISPY1_001, etc.)
        self.demographics_df = pd.read_excel(demographics_excel, sheet_name='dataset_info')
        print(f"\n📊 Demographics cargados: {len(self.demographics_df)} casos")
        print(f"   Datasets: {self.demographics_df['dataset'].value_counts().to_dict()}")
        
        return self
    
    def create_mapping(self):
        """Crear mapeo correcto entre case_XXX y test_split IDs"""
        print(f"\n🔗 CREANDO MAPEO CORRECTO")
        print("-" * 30)
        
        # Verificar que tengamos suficientes test IDs
        n_results = len(self.results_df)
        n_test_ids = len(self.test_ids)
        
        print(f"Casos en resultados: {n_results}")
        print(f"IDs en test_split: {n_test_ids}")
        
        if n_test_ids < n_results:
            print(f"⚠️  WARNING: Menos test IDs ({n_test_ids}) que resultados ({n_results})")
            print(f"   Se usarán solo los primeros {n_test_ids} resultados")
            n_results = n_test_ids
        
        # Crear mapeo directo: case_000 -> primer test_id, case_001 -> segundo test_id, etc.
        mapping_data = []
        successful_mappings = 0
        
        for i in range(n_results):
            case_id = self.results_df.iloc[i]['patient_id']
            dice_score = self.results_df.iloc[i]['dice']
            test_id = self.test_ids[i] if i < len(self.test_ids) else None
            
            if test_id:
                # Buscar demographics para este test_id
                demo_row = self.demographics_df[self.demographics_df['patient_id'] == test_id]
                
                if not demo_row.empty:
                    demo_data = demo_row.iloc[0]
                    
                    mapping_data.append({
                        'case_id': case_id,
                        'original_id': test_id,
                        'dice': dice_score,
                        'age': demo_data['age'],
                        'ethnicity': demo_data['ethnicity'],
                        'menopause': demo_data['menopause'],
                        'bmi_group': demo_data['bmi_group'],
                        'tumor_subtype': demo_data['tumor_subtype'],
                        'dataset': demo_data['dataset'],
                        'nottingham_grade': demo_data['nottingham_grade'],
                        'hr': demo_data['hr'],
                        'er': demo_data['er'],
                        'pr': demo_data['pr'],
                        'her2': demo_data['her2']
                    })
                    successful_mappings += 1
                else:
                    print(f"   ⚠️  No se encontró demographic para {test_id}")
            else:
                print(f"   ⚠️  No hay test_id para {case_id}")
        
        self.combined_df = pd.DataFrame(mapping_data)
        
        print(f"\n✅ MAPEO COMPLETADO")
        print(f"   Mapeos exitosos: {successful_mappings}/{n_results}")
        print(f"   Tasa de éxito: {successful_mappings/n_results*100:.1f}%")
        
        if successful_mappings > 0:
            print(f"\n📊 DISTRIBUCIÓN FINAL:")
            print(f"   Datasets: {self.combined_df['dataset'].value_counts().to_dict()}")
            print(f"   Etnicidades: {self.combined_df['ethnicity'].value_counts().to_dict()}")
            
        return self.combined_df
    
    def analyze_fairness(self):
        """Análisis completo de fairness"""
        if self.combined_df is None or len(self.combined_df) == 0:
            print("❌ No hay datos combinados para analizar")
            return None
            
        print(f"\n📊 ANÁLISIS DE FAIRNESS")
        print("=" * 50)
        print(f"Total casos analizados: {len(self.combined_df)}")
        
        # Limpiar datos
        df = self.combined_df.dropna(subset=['dice']).copy()
        
        # Variables para análisis
        fairness_variables = {
            'ethnicity': 'Etnicidad',
            'menopause': 'Estado Menopáusico', 
            'bmi_group': 'Grupo BMI',
            'tumor_subtype': 'Subtipo de Tumor',
            'dataset': 'Dataset'
        }
        
        fairness_results = {}
        
        # Análisis por variable categórica
        for var, var_name in fairness_variables.items():
            if var in df.columns and df[var].notna().sum() > 1:
                print(f"\n🔍 {var_name.upper()}:")
                
                # Estadísticas descriptivas
                stats_by_group = df.groupby(var)['dice'].agg([
                    'count', 'mean', 'std', 'min', 'max'
                ]).round(3)
                
                print(stats_by_group)
                
                # Test estadístico (ANOVA)
                groups = [group['dice'].values for name, group in df.groupby(var) 
                         if len(group) > 1]
                
                if len(groups) > 1:
                    try:
                        f_stat, p_value = stats.f_oneway(*groups)
                        print(f"   📊 F-statistic: {f_stat:.3f}, p-value: {p_value:.4f}")
                        
                        # Interpretar resultado
                        if p_value < 0.05:
                            print("   ⚠️  DIFERENCIAS ESTADÍSTICAMENTE SIGNIFICATIVAS")
                            
                            # Identificar mejor y peor grupo
                            group_means = df.groupby(var)['dice'].mean().sort_values(ascending=False)
                            best_group = group_means.index[0]
                            worst_group = group_means.index[-1]
                            gap = group_means[best_group] - group_means[worst_group]
                            
                            print(f"   📈 Mejor rendimiento: {best_group} (dice={group_means[best_group]:.3f})")
                            print(f"   📉 Peor rendimiento: {worst_group} (dice={group_means[worst_group]:.3f})")
                            print(f"   📊 Brecha de equidad: {gap:.3f} ({gap/df['dice'].mean()*100:.1f}% relativa)")
                            
                            # Calcular métricas de fairness
                            fairness_metrics = self.calculate_fairness_metrics(df, var)
                            for metric_name, metric_value in fairness_metrics.items():
                                print(f"   📏 {metric_name}: {metric_value:.3f}")
                        else:
                            print("   ✅ No hay diferencias significativas - EQUITATIVO")
                        
                        fairness_results[var] = {
                            'f_statistic': f_stat,
                            'p_value': p_value,
                            'significant': p_value < 0.05,
                            'group_stats': stats_by_group.to_dict(),
                            'fairness_concern': p_value < 0.05
                        }
                        
                    except Exception as e:
                        print(f"   ❌ Error en análisis estadístico: {e}")
        
        # Análisis de edad (variable continua)
        print(f"\n🔍 EDAD (Variable Continua):")
        age_median = df['age'].median()
        age_young = df[df['age'] <= age_median]['dice']
        age_old = df[df['age'] > age_median]['dice']
        
        print(f"   Jóvenes (≤{age_median:.0f} años): n={len(age_young)}, dice={age_young.mean():.3f}±{age_young.std():.3f}")
        print(f"   Mayores (>{age_median:.0f} años): n={len(age_old)}, dice={age_old.mean():.3f}±{age_old.std():.3f}")
        
        try:
            t_stat, p_value = stats.ttest_ind(age_young, age_old)
            print(f"   📊 T-test: t={t_stat:.3f}, p={p_value:.4f}")
            
            if p_value < 0.05:
                print("   ⚠️  DIFERENCIAS SIGNIFICATIVAS por edad")
                age_gap = abs(age_young.mean() - age_old.mean())
                print(f"   📊 Brecha etaria: {age_gap:.3f}")
            else:
                print("   ✅ No hay diferencias significativas por edad")
                
            fairness_results['age'] = {
                't_statistic': t_stat,
                'p_value': p_value,
                'significant': p_value < 0.05
            }
        except Exception as e:
            print(f"   ❌ Error en t-test de edad: {e}")
        
        return fairness_results
    
    def calculate_fairness_metrics(self, df, protected_attribute):
        """Calcular métricas específicas de fairness"""
        metrics = {}
        
        # 1. Demographic Parity Difference
        group_rates = df.groupby(protected_attribute)['dice'].mean()
        overall_rate = df['dice'].mean()
        max_diff = group_rates.max() - group_rates.min()
        metrics['Demographic_Parity_Diff'] = max_diff
        
        # 2. Equalized Odds (aproximación usando percentiles)
        # Considerar alto rendimiento como dice > percentil 75
        high_performance_threshold = df['dice'].quantile(0.75)
        
        group_high_perf_rates = df.groupby(protected_attribute).apply(
            lambda x: (x['dice'] > high_performance_threshold).mean()
        )
        
        if len(group_high_perf_rates) > 1:
            metrics['Equalized_Odds_Diff'] = group_high_perf_rates.max() - group_high_perf_rates.min()
        
        # 3. Coefficient of Variation (dispersión relativa)
        metrics['CV_between_groups'] = group_rates.std() / group_rates.mean()
        
        return metrics
    
    def create_fairness_visualizations(self):
        """Crear visualizaciones comprehensivas de fairness"""
        if self.combined_df is None:
            print("❌ No hay datos para visualizar")
            return
            
        print(f"\n📈 CREANDO VISUALIZACIONES DE FAIRNESS")
        
        # Configurar estilo
        plt.style.use('default')
        sns.set_palette("Set2")
        
        # Figura principal: 2x3 subplots
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('Análisis de Fairness - Dataset Mama Mia\nDistribución de Dice Score por Variables Demográficas', 
                     fontsize=16, fontweight='bold')
        
        df = self.combined_df.dropna(subset=['dice'])
        
        # 1. Etnicidad
        if 'ethnicity' in df.columns and df['ethnicity'].notna().sum() > 0:
            sns.boxplot(data=df, x='ethnicity', y='dice', ax=axes[0,0])
            axes[0,0].set_title('Dice Score por Etnicidad', fontweight='bold')
            axes[0,0].tick_params(axis='x', rotation=45)
            
            # Agregar medias
            eth_means = df.groupby('ethnicity')['dice'].mean()
            for i, (eth, mean_val) in enumerate(eth_means.items()):
                axes[0,0].text(i, mean_val + 0.02, f'{mean_val:.3f}', 
                              ha='center', va='bottom', fontweight='bold', color='red')
        
        # 2. BMI
        if 'bmi_group' in df.columns and df['bmi_group'].notna().sum() > 0:
            sns.boxplot(data=df, x='bmi_group', y='dice', ax=axes[0,1])
            axes[0,1].set_title('Dice Score por Grupo BMI', fontweight='bold')
            axes[0,1].tick_params(axis='x', rotation=45)
        
        # 3. Subtipo tumor
        if 'tumor_subtype' in df.columns and df['tumor_subtype'].notna().sum() > 0:
            sns.boxplot(data=df, x='tumor_subtype', y='dice', ax=axes[0,2])
            axes[0,2].set_title('Dice Score por Subtipo de Tumor', fontweight='bold')
            axes[0,2].tick_params(axis='x', rotation=45)
        
        # 4. Edad (grupos)
        age_median = df['age'].median()
        df_viz = df.copy()
        df_viz['age_group'] = np.where(
            df_viz['age'] <= age_median, 
            f'Jóvenes (≤{age_median:.0f})', 
            f'Mayores (>{age_median:.0f})'
        )
        sns.boxplot(data=df_viz, x='age_group', y='dice', ax=axes[1,0])
        axes[1,0].set_title('Dice Score por Grupo de Edad', fontweight='bold')
        
        # 5. Dataset
        if 'dataset' in df.columns:
            sns.boxplot(data=df, x='dataset', y='dice', ax=axes[1,1])
            axes[1,1].set_title('Dice Score por Dataset', fontweight='bold')
            axes[1,1].tick_params(axis='x', rotation=45)
        
        # 6. Distribución general + estadísticas
        axes[1,2].hist(df['dice'], bins=25, alpha=0.7, edgecolor='black', color='skyblue')
        axes[1,2].set_title('Distribución General de Dice Score', fontweight='bold')
        axes[1,2].set_xlabel('Dice Score')
        axes[1,2].set_ylabel('Frecuencia')
        
        # Líneas de estadísticas
        mean_dice = df['dice'].mean()
        median_dice = df['dice'].median()
        axes[1,2].axvline(mean_dice, color='red', linestyle='--', linewidth=2, label=f'Media: {mean_dice:.3f}')
        axes[1,2].axvline(median_dice, color='orange', linestyle='--', linewidth=2, label=f'Mediana: {median_dice:.3f}')
        axes[1,2].legend()
        
        # Agregar texto con estadísticas
        stats_text = f'n = {len(df)}\nStd = {df["dice"].std():.3f}\nMin = {df["dice"].min():.3f}\nMax = {df["dice"].max():.3f}'
        axes[1,2].text(0.02, 0.98, stats_text, transform=axes[1,2].transAxes, 
                      verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.show()
        
        # Figura adicional: Correlación y scatter plots
        self.create_correlation_analysis(df)
        
        return fig
    
    def create_correlation_analysis(self, df):
        """Análisis de correlación detallado"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('Análisis de Correlación - Variables Demográficas vs Rendimiento', fontsize=14, fontweight='bold')
        
        # 1. Scatter plot: Edad vs Dice coloreado por etnicidad
        if 'ethnicity' in df.columns:
            sns.scatterplot(data=df, x='age', y='dice', hue='ethnicity', alpha=0.7, s=50, ax=axes[0])
            axes[0].set_title('Dice Score vs Edad (por Etnicidad)')
            axes[0].set_xlabel('Edad (años)')
            axes[0].set_ylabel('Dice Score')
            
            # Línea de tendencia
            z = np.polyfit(df['age'], df['dice'], 1)
            p = np.poly1d(z)
            axes[0].plot(df['age'], p(df['age']), "r--", alpha=0.8, linewidth=2)
            
            # Correlación
            corr_age_dice = df['age'].corr(df['dice'])
            axes[0].text(0.05, 0.95, f'Correlación edad-dice: {corr_age_dice:.3f}', 
                        transform=axes[0].transAxes, bbox=dict(boxstyle='round', facecolor='lightblue'))
        
        # 2. Heatmap de variables categóricas
        # Crear matriz de variables categóricas codificadas
        categorical_vars = ['ethnicity', 'menopause', 'bmi_group', 'tumor_subtype']
        available_vars = [var for var in categorical_vars if var in df.columns and df[var].notna().sum() > 0]
        
        if available_vars:
            df_encoded = df.copy()
            for var in available_vars:
                df_encoded[f'{var}_code'] = pd.Categorical(df_encoded[var]).codes
            
            corr_vars = ['dice', 'age'] + [f'{var}_code' for var in available_vars]
            correlation_matrix = df_encoded[corr_vars].corr()
            
            # Renombrar para mejor visualización
            new_labels = {f'{var}_code': var for var in available_vars}
            correlation_matrix = correlation_matrix.rename(index=new_labels, columns=new_labels)
            
            sns.heatmap(correlation_matrix, annot=True, cmap='RdBu_r', center=0, 
                       square=True, fmt='.3f', ax=axes[1])
            axes[1].set_title('Matriz de Correlación\nVariables vs Dice Score')
            axes[1].tick_params(axis='x', rotation=45)
            axes[1].tick_params(axis='y', rotation=0)
        
        plt.tight_layout()
        plt.show()
    
    def generate_fairness_report(self, fairness_results):
        """Generar reporte ejecutivo de fairness"""
        print(f"\n📋 REPORTE EJECUTIVO DE FAIRNESS")
        print("=" * 60)
        
        if not fairness_results:
            print("❌ No hay resultados de fairness para reportar")
            return
        
        # Clasificar variables por nivel de preocupación
        high_concern = []
        medium_concern = []
        low_concern = []
        
        for var, results in fairness_results.items():
            if results.get('significant', False):
                p_val = results.get('p_value', 1.0)
                if p_val < 0.01:
                    high_concern.append((var, p_val))
                elif p_val < 0.05:
                    medium_concern.append((var, p_val))
            else:
                low_concern.append(var)
        
        # Reporte por niveles
        if high_concern:
            print(f"🚨 ALTA PREOCUPACIÓN DE FAIRNESS:")
            for var, p_val in high_concern:
                print(f"   • {var}: p = {p_val:.4f} (muy significativo)")
        
        if medium_concern:
            print(f"\n⚠️  PREOCUPACIÓN MODERADA:")
            for var, p_val in medium_concern:
                print(f"   • {var}: p = {p_val:.4f} (significativo)")
        
        if low_concern:
            print(f"\n✅ SIN PREOCUPACIONES DE FAIRNESS:")
            for var in low_concern:
                print(f"   • {var}: No hay diferencias significativas")
        
        # Recomendaciones
        print(f"\n💡 RECOMENDACIONES:")
        
        if high_concern or medium_concern:
            print(f"   🔍 ACCIONES INMEDIATAS:")
            print(f"   • Investigar las causas de las diferencias en: {', '.join([var for var, _ in high_concern + medium_concern])}")
            print(f"   • Considerar re-balanceo del dataset de entrenamiento")
            print(f"   • Implementar técnicas de fairness-aware machine learning")
            print(f"   • Validar resultados con expertos clínicos")
            
            print(f"\n   📊 MÉTRICAS A MONITOREAR:")
            print(f"   • Brecha máxima de rendimiento entre grupos")
            print(f"   • Paridad demográfica")
            print(f"   • Igualdad de oportunidades")
        else:
            print(f"   🎯 El modelo muestra un comportamiento equitativo")
            print(f"   • Continuar monitoreando en nuevos datos")
            print(f"   • Mantener diversidad en el dataset")
        
        # Estadísticas finales
        print(f"\n📊 ESTADÍSTICAS GENERALES:")
        df = self.combined_df
        print(f"   • Total casos analizados: {len(df)}")
        print(f"   • Dice Score promedio: {df['dice'].mean():.3f} ± {df['dice'].std():.3f}")
        print(f"   • Rango de rendimiento: {df['dice'].min():.3f} - {df['dice'].max():.3f}")
        print(f"   • Coeficiente de variación: {df['dice'].std()/df['dice'].mean()*100:.1f}%")

# FUNCIÓN PRINCIPAL PARA EJECUTAR TODO
def run_complete_fairness_analysis():
    """Ejecutar análisis completo de fairness para Mama Mia"""
    
    print("🎯 ANÁLISIS COMPLETO DE FAIRNESS - MAMA MIA DATASET")
    print("=" * 70)
    
    # Rutas de archivos
    results_csv = "resultados_a3.csv"
    demographics_excel = r"C:\Users\usuario\Documents\Mama_Mia\datos\clinical_and_imaging_info.xlsx" 
    splits_csv = r"C:\Users\usuario\Documents\Mama_Mia\datos\train_test_splits.csv"  # ← ARCHIVO CLAVE
    
    try:
        # 1. Inicializar y cargar datos
        analyzer = MamaMiaFairnessAnalyzer()
        analyzer.load_data(results_csv, demographics_excel, splits_csv)
        
        # 2. Crear mapeo correcto
        combined_df = analyzer.create_mapping()
        
        if combined_df is None or len(combined_df) == 0:
            print("❌ FALLO EN EL MAPEO DE DATOS")
            return None
        
        # 3. Análisis de fairness
        fairness_results = analyzer.analyze_fairness()
        
        # 4. Visualizaciones
        analyzer.create_fairness_visualizations()
        
        # 5. Reporte ejecutivo
        analyzer.generate_fairness_report(fairness_results)
        
        print(f"\n🎉 ANÁLISIS COMPLETADO EXITOSAMENTE")
        print(f"✅ {len(combined_df)} casos analizados para fairness")
        
        return analyzer, fairness_results
        
    except Exception as e:
        print(f"❌ ERROR EN EL ANÁLISIS: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    # Ejecutar análisis completo
    results = run_complete_fairness_analysis()