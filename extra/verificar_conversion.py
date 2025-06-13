import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def verificar_conversion_tipo():
    """Verifica que la conversión int16 → float32 no afecte los datos"""
    
    # Usar una imagen de ejemplo
    images_dir = Path(r"C:\Users\usuario\Documents\Mama_Mia\datos\images")
    sample_file = list(images_dir.glob("*_0000.nii.gz"))[0]
    
    print(f"Verificando archivo: {sample_file.name}")
    
    # Cargar imagen original (int16)
    original_img = sitk.ReadImage(str(sample_file))
    original_array = sitk.GetArrayFromImage(original_img)
    
    # Convertir a float32
    converted_img = sitk.Cast(original_img, sitk.sitkFloat32)
    converted_array = sitk.GetArrayFromImage(converted_img)
    
    print(f"\n{'='*50}")
    print(f"VERIFICACIÓN DE CONVERSIÓN")
    print(f"{'='*50}")
    print(f"Tipo original: {original_img.GetPixelIDTypeAsString()}")
    print(f"Tipo convertido: {converted_img.GetPixelIDTypeAsString()}")
    print(f"Shape original: {original_array.shape}")
    print(f"Shape convertido: {converted_array.shape}")
    print(f"Min original: {original_array.min()}")
    print(f"Max original: {original_array.max()}")
    print(f"Min convertido: {converted_array.min()}")
    print(f"Max convertido: {converted_array.max()}")
    
    # Verificar que son idénticos
    diferencia_max = np.max(np.abs(original_array - converted_array))
    son_identicos = np.array_equal(original_array, converted_array)
    
    print(f"Diferencia máxima: {diferencia_max}")
    print(f"¿Son idénticos?: {son_identicos}")
    
    # Visualización comparativa
    slice_medio = original_array.shape[0] // 2
    
    plt.figure(figsize=(15, 5))
    
    # Imagen original
    plt.subplot(1, 3, 1)
    plt.imshow(original_array[slice_medio], cmap='gray')
    plt.title(f'Original (Int16)\nMin: {original_array.min()}, Max: {original_array.max()}')
    plt.colorbar()
    
    # Imagen convertida
    plt.subplot(1, 3, 2)
    plt.imshow(converted_array[slice_medio], cmap='gray')
    plt.title(f'Convertida (Float32)\nMin: {converted_array.min()}, Max: {converted_array.max()}')
    plt.colorbar()
    
    # Diferencia (debería ser todo negro = 0)
    plt.subplot(1, 3, 3)
    diferencia = np.abs(original_array[slice_medio] - converted_array[slice_medio])
    plt.imshow(diferencia, cmap='hot')
    plt.title(f'Diferencia Absoluta\nMax diff: {diferencia.max()}')
    plt.colorbar()
    
    plt.tight_layout()
    plt.savefig('verificacion_conversion.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Verificar estadísticas
    print(f"\n{'='*30}")
    print(f"ESTADÍSTICAS DETALLADAS")
    print(f"{'='*30}")
    print(f"Media original: {np.mean(original_array):.6f}")
    print(f"Media convertida: {np.mean(converted_array):.6f}")
    print(f"Std original: {np.std(original_array):.6f}")
    print(f"Std convertida: {np.std(converted_array):.6f}")
    
    # Verificar histograma
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.hist(original_array.flatten()[::1000], bins=50, alpha=0.7, label='Original Int16')
    plt.hist(converted_array.flatten()[::1000], bins=50, alpha=0.7, label='Convertida Float32')
    plt.xlabel('Valor de Intensidad')
    plt.ylabel('Frecuencia')
    plt.title('Distribución de Intensidades')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    # Plot de correlación punto a punto
    sample_indices = np.random.choice(original_array.size, 10000, replace=False)
    plt.scatter(original_array.flatten()[sample_indices], 
                converted_array.flatten()[sample_indices], 
                alpha=0.1, s=1)
    plt.plot([original_array.min(), original_array.max()], 
             [original_array.min(), original_array.max()], 'r--', label='y=x')
    plt.xlabel('Original Int16')
    plt.ylabel('Convertida Float32')
    plt.title('Correlación Punto a Punto')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('histograma_conversion.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Conclusión
    print(f"\n{'='*50}")
    print(f"CONCLUSIÓN")
    print(f"{'='*50}")
    if son_identicos:
        print("✅ CONVERSIÓN PERFECTA: Los datos son idénticos")
        print("✅ No se requiere normalización adicional")
        print("✅ Seguro para usar en registro y entrenamiento")
    else:
        print("⚠️  HAY DIFERENCIAS (revisar)")
        
    return son_identicos, diferencia_max

if __name__ == "__main__":
    resultado = verificar_conversion_tipo()