import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
from tqdm import tqdm

def detectar_movimiento_rapido():
    """
    Identifica casos con mucho/poco movimiento usando SOLO imágenes originales
    Métrica: Diferencia cruda entre PRE y POST
    """
    
    images_dir = Path(r"C:\Users\usuario\Documents\Mama_Mia\datos\images")
    splits_file = Path(r"C:\Users\usuario\Documents\Mama_Mia\datos\train_test_splits.csv")
    
    # Leer casos del split oficial
    splits_df = pd.read_csv(splits_file)
    all_cases = list(splits_df['train_split']) + list(splits_df['test_split'])
    all_cases = [case for case in all_cases if pd.notna(case)]
    
    print(f"Analizando movimiento en {len(all_cases)} casos...")
    
    movimientos = []
    
    # Analizar muestra representativa (no todos para velocidad)
    casos_muestra = all_cases[::10]  # Cada 10 casos
    print(f"Usando muestra de {len(casos_muestra)} casos para análisis rápido")
    
    for case_id in tqdm(casos_muestra):
        try:
            pre_file = images_dir / f"{case_id}_0000.nii.gz"
            post_file = images_dir / f"{case_id}_0001.nii.gz"
            
            if not pre_file.exists() or not post_file.exists():
                continue
            
            # Cargar solo slice central para velocidad
            pre_img = sitk.ReadImage(str(pre_file))
            post_img = sitk.ReadImage(str(post_file))
            
            pre_array = sitk.GetArrayFromImage(pre_img)
            post_array = sitk.GetArrayFromImage(post_img)
            
            # Analizar solo slice central (mucho más rápido)
            slice_central = pre_array.shape[0] // 2
            pre_slice = pre_array[slice_central]
            post_slice = post_array[slice_central]
            
            # Métricas de movimiento
            diff_cruda = np.abs(post_slice - pre_slice)
            
            # Métricas rápidas
            movement_score = np.percentile(diff_cruda, 95)  # Percentil 95
            mean_diff = diff_cruda.mean()
            std_diff = diff_cruda.std()
            max_diff = diff_cruda.max()
            
            # Correlación (indica alineamiento)
            correlation = np.corrcoef(pre_slice.flatten(), post_slice.flatten())[0,1]
            
            # Centro de masa (detecta traslaciones)
            def centro_masa(img):
                img_thresh = img > np.percentile(img, 50)  # Threshold simple
                if np.sum(img_thresh) == 0:
                    return (0, 0)
                y, x = np.where(img_thresh)
                return (np.mean(y), np.mean(x))
            
            cm_pre = centro_masa(pre_slice)
            cm_post = centro_masa(post_slice)
            desplazamiento_cm = np.sqrt((cm_pre[0] - cm_post[0])**2 + (cm_pre[1] - cm_post[1])**2)
            
            movimientos.append({
                'case_id': case_id,
                'movement_score': movement_score,
                'mean_diff': mean_diff,
                'std_diff': std_diff,
                'max_diff': max_diff,
                'correlation': correlation,
                'desplazamiento_cm': desplazamiento_cm,
                'centro': case_id.split('_')[0]  # DUKE, ISPY1, etc.
            })
            
        except Exception as e:
            print(f"Error con {case_id}: {e}")
            continue
    
    # Convertir a DataFrame y analizar
    df_movement = pd.DataFrame(movimientos)
    
    if len(df_movement) == 0:
        print("❌ No se pudo analizar ningún caso")
        return None
    
    # Ordenar por score de movimiento
    df_movement = df_movement.sort_values('movement_score')
    
    print(f"\n{'='*60}")
    print(f"ANÁLISIS DE MOVIMIENTO - {len(df_movement)} casos analizados")
    print(f"{'='*60}")
    
    # Estadísticas generales
    print(f"Movement score - Min: {df_movement['movement_score'].min():.1f}, Max: {df_movement['movement_score'].max():.1f}")
    print(f"Correlación - Min: {df_movement['correlation'].min():.3f}, Max: {df_movement['correlation'].max():.3f}")
    print(f"Desplazamiento CM - Min: {df_movement['desplazamiento_cm'].min():.1f}, Max: {df_movement['desplazamiento_cm'].max():.1f}")
    
    # Por centro médico
    print(f"\nPor centro médico:")
    print(df_movement.groupby('centro')['movement_score'].agg(['count', 'mean', 'std']).round(2))
    
    # Casos extremos
    print(f"\n🟢 CASOS CON POCO MOVIMIENTO (fáciles para registro):")
    casos_faciles = df_movement.head(3)
    for _, caso in casos_faciles.iterrows():
        print(f"  {caso['case_id']}: Score={caso['movement_score']:.1f}, Corr={caso['correlation']:.3f}, Desp={caso['desplazamiento_cm']:.1f}")
    
    print(f"\n🔴 CASOS CON MUCHO MOVIMIENTO (difíciles para registro):")
    casos_dificiles = df_movement.tail(3)
    for _, caso in casos_dificiles.iterrows():
        print(f"  {caso['case_id']}: Score={caso['movement_score']:.1f}, Corr={caso['correlation']:.3f}, Desp={caso['desplazamiento_cm']:.1f}")
    
    return df_movement

def visualizar_casos_extremos(df_movement):
    """Visualiza ejemplos de casos fáciles vs difíciles"""
    
    if df_movement is None or len(df_movement) == 0:
        return
    
    images_dir = Path(r"C:\Users\usuario\Documents\Mama_Mia\datos\images")
    
    # Seleccionar casos extremos
    caso_facil = df_movement.iloc[0]['case_id']
    caso_dificil = df_movement.iloc[-1]['case_id']
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    
    for i, (caso, label) in enumerate([(caso_facil, "FÁCIL"), (caso_dificil, "DIFÍCIL")]):
        try:
            # Cargar imágenes
            pre_img = sitk.GetArrayFromImage(sitk.ReadImage(str(images_dir / f"{caso}_0000.nii.gz")))
            post_img = sitk.GetArrayFromImage(sitk.ReadImage(str(images_dir / f"{caso}_0001.nii.gz")))
            
            slice_central = pre_img.shape[0] // 2
            pre_slice = pre_img[slice_central]
            post_slice = post_img[slice_central]
            diff_slice = np.abs(post_slice - pre_slice)
            
            # Normalizar para visualización
            def normalize(img):
                p1, p99 = np.percentile(img, [1, 99])
                img_norm = np.clip(img, p1, p99)
                return (img_norm - img_norm.min()) / (img_norm.max() - img_norm.min())
            
            # Plots
            axes[i, 0].imshow(normalize(pre_slice), cmap='gray')
            axes[i, 0].set_title(f'{label} - {caso}\nPRE')
            axes[i, 0].axis('off')
            
            axes[i, 1].imshow(normalize(post_slice), cmap='gray')
            axes[i, 1].set_title(f'POST')
            axes[i, 1].axis('off')
            
            axes[i, 2].imshow(diff_slice, cmap='hot')
            axes[i, 2].set_title(f'Diferencia Absoluta\nMax: {diff_slice.max():.0f}')
            axes[i, 2].axis('off')
            
            # Histograma de diferencias
            axes[i, 3].hist(diff_slice.flatten(), bins=50, alpha=0.7)
            axes[i, 3].set_title(f'Distribución Diferencias')
            axes[i, 3].set_xlabel('Intensidad')
            
            # Métricas en el caso
            caso_info = df_movement[df_movement['case_id'] == caso].iloc[0]
            print(f"\n{label} - {caso}:")
            print(f"  Movement score: {caso_info['movement_score']:.1f}")
            print(f"  Correlación: {caso_info['correlation']:.3f}")
            print(f"  Desplazamiento CM: {caso_info['desplazamiento_cm']:.1f}")
            
        except Exception as e:
            print(f"Error visualizando {caso}: {e}")
    
    plt.tight_layout()
    plt.savefig('casos_extremos_movimiento.png', dpi=150, bbox_inches='tight')
    plt.show()

# Función principal
if __name__ == "__main__":
    print("🔍 DETECTOR RÁPIDO DE MOVIMIENTO")
    print("Analizando casos para identificar extremos...")
    
    # Detectar movimiento
    df_casos = detectar_movimiento_rapido()
    
    if df_casos is not None:
        # Visualizar casos extremos
        print("\n📊 Generando visualización de casos extremos...")
        visualizar_casos_extremos(df_casos)
        
        # Guardar resultados
        df_casos.to_csv('analisis_movimiento_casos.csv', index=False)
        print(f"\n💾 Resultados guardados en 'analisis_movimiento_casos.csv'")
        
        print(f"\n🎯 RECOMENDACIONES:")
        print(f"- Para probar registro: usar casos difíciles")
        print(f"- Para validar: comparar fáciles vs difíciles")
        print(f"- Si registro mejora casos difíciles → vale la pena")
        print(f"- Si solo mejora fáciles → no vale la pena")
    
    else:
        print("❌ No se pudieron analizar casos")