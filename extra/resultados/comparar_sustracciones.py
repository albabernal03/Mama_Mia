import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def comparar_sustracciones_visual(case_id="DUKE_001"):
    """
    Compara sustracción original vs registrada visualmente
    """
    # Directorios
    original_dir = Path(r"C:\Users\usuario\Documents\Mama_Mia\datos\images")
    registered_dir = Path(r"C:\Users\usuario\Documents\Mama_Mia\datos\images_registered")
    
    print(f"Comparando caso: {case_id}")
    
    # === IMÁGENES ORIGINALES ===
    pre_orig_path = original_dir / f"{case_id}_0000.nii.gz"
    post_orig_path = original_dir / f"{case_id}_0001.nii.gz"
    
    # === IMÁGENES REGISTRADAS ===
    pre_reg_path = registered_dir / f"{case_id}_0000.nii.gz"
    post_reg_path = registered_dir / f"{case_id}_0001.nii.gz"
    
    # Verificar que existen
    if not all([p.exists() for p in [pre_orig_path, post_orig_path]]):
        print(f"❌ Archivos originales no encontrados para {case_id}")
        return
        
    if not all([p.exists() for p in [pre_reg_path, post_reg_path]]):
        print(f"❌ Archivos registrados no encontrados para {case_id}")
        print("Usa un caso que ya esté procesado en el registro")
        return
    
    # Cargar imágenes originales
    pre_orig = sitk.GetArrayFromImage(sitk.ReadImage(str(pre_orig_path)))
    post_orig = sitk.GetArrayFromImage(sitk.ReadImage(str(post_orig_path)))
    
    # Cargar imágenes registradas
    pre_reg = sitk.GetArrayFromImage(sitk.ReadImage(str(pre_reg_path)))
    post_reg = sitk.GetArrayFromImage(sitk.ReadImage(str(post_reg_path)))
    
    # Crear sustracciones
    sub_original = post_orig - pre_orig
    sub_registered = post_reg - pre_reg
    
    # Normalizar para visualización
    def normalize_for_display(img):
        img_norm = img.copy()
        # Clip extremos
        p1, p99 = np.percentile(img_norm, [1, 99])
        img_norm = np.clip(img_norm, p1, p99)
        # Normalizar 0-1
        img_norm = (img_norm - img_norm.min()) / (img_norm.max() - img_norm.min())
        return img_norm
    
    # Slice central para visualización
    slice_central = pre_orig.shape[0] // 2
    
    # Visualización completa
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    
    # Fila 1: Imágenes originales
    axes[0,0].imshow(normalize_for_display(pre_orig[slice_central]), cmap='gray')
    axes[0,0].set_title('PRE Original')
    axes[0,0].axis('off')
    
    axes[0,1].imshow(normalize_for_display(post_orig[slice_central]), cmap='gray')
    axes[0,1].set_title('POST Original')
    axes[0,1].axis('off')
    
    axes[0,2].imshow(sub_original[slice_central], cmap='RdBu_r', vmin=-200, vmax=200)
    axes[0,2].set_title('SUSTRACCIÓN Original\n(POST - PRE)')
    axes[0,2].axis('off')
    
    axes[0,3].imshow(np.abs(sub_original[slice_central]), cmap='hot')
    axes[0,3].set_title('SUSTRACCIÓN Absoluta\n(Artifacts visibles)')
    axes[0,3].axis('off')
    
    # Fila 2: Imágenes registradas
    axes[1,0].imshow(normalize_for_display(pre_reg[slice_central]), cmap='gray')
    axes[1,0].set_title('PRE Registrado\n(Copia del original)')
    axes[1,0].axis('off')
    
    axes[1,1].imshow(normalize_for_display(post_reg[slice_central]), cmap='gray')
    axes[1,1].set_title('POST Registrado\n(Alineado a PRE)')
    axes[1,1].axis('off')
    
    axes[1,2].imshow(sub_registered[slice_central], cmap='RdBu_r', vmin=-200, vmax=200)
    axes[1,2].set_title('SUSTRACCIÓN Registrada\n(POST_reg - PRE)')
    axes[1,2].axis('off')
    
    axes[1,3].imshow(np.abs(sub_registered[slice_central]), cmap='hot')
    axes[1,3].set_title('SUSTRACCIÓN Absoluta\n(Menos artifacts)')
    axes[1,3].axis('off')
    
    # Fila 3: Comparaciones directas
    diferencia_pre = np.abs(pre_orig[slice_central] - pre_reg[slice_central])
    diferencia_post = np.abs(post_orig[slice_central] - post_reg[slice_central])
    diferencia_sub = np.abs(sub_original[slice_central] - sub_registered[slice_central])
    
    axes[2,0].imshow(diferencia_pre, cmap='hot')
    axes[2,0].set_title(f'Diferencia PRE\nMax: {diferencia_pre.max():.1f}')
    axes[2,0].axis('off')
    
    axes[2,1].imshow(diferencia_post, cmap='hot')
    axes[2,1].set_title(f'Diferencia POST\nMax: {diferencia_post.max():.1f}')
    axes[2,1].axis('off')
    
    axes[2,2].imshow(diferencia_sub, cmap='hot')
    axes[2,2].set_title(f'Diferencia SUSTRACCIÓN\nMax: {diferencia_sub.max():.1f}')
    axes[2,2].axis('off')
    
    # Histograma de diferencias en sustracción
    axes[2,3].hist(sub_original.flatten()[::1000], bins=50, alpha=0.7, label='Original', density=True)
    axes[2,3].hist(sub_registered.flatten()[::1000], bins=50, alpha=0.7, label='Registrada', density=True)
    axes[2,3].set_title('Distribución Sustracciones')
    axes[2,3].legend()
    axes[2,3].set_xlabel('Intensidad')
    axes[2,3].set_ylabel('Densidad')
    
    plt.tight_layout()
    plt.suptitle(f'COMPARACIÓN REGISTRO - {case_id}', fontsize=16, y=0.98)
    plt.savefig(f'comparacion_registro_{case_id}.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Estadísticas cuantitativas
    print(f"\n{'='*50}")
    print(f"ESTADÍSTICAS CUANTITATIVAS - {case_id}")
    print(f"{'='*50}")
    print(f"Diferencia máxima POST original vs registrado: {diferencia_post.max():.2f}")
    print(f"Diferencia promedio POST original vs registrado: {diferencia_post.mean():.2f}")
    print(f"Diferencia máxima SUSTRACCIÓN: {diferencia_sub.max():.2f}")
    print(f"Diferencia promedio SUSTRACCIÓN: {diferencia_sub.mean():.2f}")
    
    # Métricas de calidad de sustracción
    std_orig = np.std(sub_original)
    std_reg = np.std(sub_registered)
    print(f"Variabilidad sustracción original: {std_orig:.2f}")
    print(f"Variabilidad sustracción registrada: {std_reg:.2f}")
    print(f"Reducción de variabilidad: {((std_orig-std_reg)/std_orig*100):.1f}%")
    
    return diferencia_post.mean(), diferencia_sub.mean()

# Función para comparar múltiples casos
def comparar_multiples_casos():
    """Compara varios casos procesados"""
    registered_dir = Path(r"C:\Users\usuario\Documents\Mama_Mia\datos\images_registered")
    
    # Buscar casos ya procesados
    casos_procesados = []
    for file in registered_dir.glob("*_0000.nii.gz"):
        case_id = file.name.replace("_0000.nii.gz", "")
        casos_procesados.append(case_id)
    
    print(f"Casos disponibles para comparar: {len(casos_procesados)}")
    print(f"Primeros 10: {casos_procesados[:10]}")
    
    # Comparar primeros 3 casos
    for case_id in casos_procesados[:3]:
        try:
            print(f"\n{'='*60}")
            comparar_sustracciones_visual(case_id)
        except Exception as e:
            print(f"Error con {case_id}: {e}")

if __name__ == "__main__":
    # Usar un caso específico o buscar casos disponibles
    print("Buscando casos procesados...")
    comparar_multiples_casos()