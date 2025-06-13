import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def test_registro_caso_especifico(case_id):
    """
    Prueba diferentes métodos de registro en un caso específico
    """
    print(f"🔬 PROBANDO REGISTRO EN: {case_id}")
    
    images_dir = Path(r"C:\Users\usuario\Documents\Mama_Mia\datos\images")
    
    # Cargar imágenes
    pre_path = images_dir / f"{case_id}_0000.nii.gz"
    post_path = images_dir / f"{case_id}_0001.nii.gz"
    
    if not pre_path.exists() or not post_path.exists():
        print(f"❌ Archivos no encontrados para {case_id}")
        return
    
    fixed = sitk.Cast(sitk.ReadImage(str(pre_path)), sitk.sitkFloat32)
    moving = sitk.Cast(sitk.ReadImage(str(post_path)), sitk.sitkFloat32)
    
    fixed_array = sitk.GetArrayFromImage(fixed)
    moving_array = sitk.GetArrayFromImage(moving)
    
    # Slice central para análisis
    slice_central = fixed_array.shape[0] // 2
    
    # Métodos a probar
    metodos = {}
    
    # MÉTODO 0: Sin registro (baseline)
    metodos['Sin registro'] = {
        'imagen': moving_array,
        'tiempo': 0,
        'exito': True
    }
    
    # MÉTODO 1: Registro básico (tu método actual)
    print("  Probando método básico...")
    try:
        import time
        start = time.time()
        
        registration = sitk.ImageRegistrationMethod()
        registration.SetMetricAsMeanSquares()
        registration.SetOptimizerAsRegularStepGradientDescent(1.0, 0.001, 100)
        registration.SetInterpolator(sitk.sitkLinear)
        
        initial_transform = sitk.CenteredTransformInitializer(
            fixed, moving, sitk.Euler3DTransform()
        )
        registration.SetInitialTransform(initial_transform)
        
        final_transform = registration.Execute(fixed, moving)
        registered = sitk.Resample(moving, fixed, final_transform)
        
        metodos['Básico (actual)'] = {
            'imagen': sitk.GetArrayFromImage(registered),
            'tiempo': time.time() - start,
            'exito': True,
            'metrica': registration.GetMetricValue()
        }
        print(f"    ✅ Básico: {time.time() - start:.1f}s")
        
    except Exception as e:
        metodos['Básico (actual)'] = {'imagen': moving_array, 'tiempo': 0, 'exito': False, 'error': str(e)}
        print(f"    ❌ Básico falló: {e}")
    
    # MÉTODO 2: Multi-resolución
    print("  Probando multi-resolución...")
    try:
        start = time.time()
        
        registration = sitk.ImageRegistrationMethod()
        registration.SetMetricAsMeanSquares()
        registration.SetOptimizerAsRegularStepGradientDescent(0.5, 0.0001, 50)
        registration.SetInterpolator(sitk.sitkLinear)
        
        # Multi-level
        registration.SetShrinkFactorsPerLevel([4, 2, 1])
        registration.SetSmoothingSigmasPerLevel([2.0, 1.0, 0.0])
        registration.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
        
        initial_transform = sitk.CenteredTransformInitializer(
            fixed, moving, sitk.Euler3DTransform()
        )
        registration.SetInitialTransform(initial_transform)
        
        final_transform = registration.Execute(fixed, moving)
        registered = sitk.Resample(moving, fixed, final_transform)
        
        metodos['Multi-resolución'] = {
            'imagen': sitk.GetArrayFromImage(registered),
            'tiempo': time.time() - start,
            'exito': True,
            'metrica': registration.GetMetricValue()
        }
        print(f"    ✅ Multi-res: {time.time() - start:.1f}s")
        
    except Exception as e:
        metodos['Multi-resolución'] = {'imagen': moving_array, 'tiempo': 0, 'exito': False, 'error': str(e)}
        print(f"    ❌ Multi-res falló: {e}")
    
    # MÉTODO 3: Correlación
    print("  Probando correlación...")
    try:
        start = time.time()
        
        registration = sitk.ImageRegistrationMethod()
        registration.SetMetricAsCorrelation()
        registration.SetOptimizerAsRegularStepGradientDescent(1.0, 0.001, 100)
        registration.SetInterpolator(sitk.sitkLinear)
        
        initial_transform = sitk.CenteredTransformInitializer(
            fixed, moving, sitk.Euler3DTransform()
        )
        registration.SetInitialTransform(initial_transform)
        
        final_transform = registration.Execute(fixed, moving)
        registered = sitk.Resample(moving, fixed, final_transform)
        
        metodos['Correlación'] = {
            'imagen': sitk.GetArrayFromImage(registered),
            'tiempo': time.time() - start,
            'exito': True,
            'metrica': registration.GetMetricValue()
        }
        print(f"    ✅ Correlación: {time.time() - start:.1f}s")
        
    except Exception as e:
        metodos['Correlación'] = {'imagen': moving_array, 'tiempo': 0, 'exito': False, 'error': str(e)}
        print(f"    ❌ Correlación falló: {e}")
    
    # Evaluar resultados
    resultados = []
    
    plt.figure(figsize=(20, 12))
    
    for i, (nombre, datos) in enumerate(metodos.items()):
        if not datos['exito']:
            continue
            
        registered_array = datos['imagen']
        
        # Calcular sustracción y métricas
        subtraction = registered_array - fixed_array
        
        # Métricas de calidad
        var_subtraction = np.var(subtraction)
        mean_abs_diff = np.mean(np.abs(subtraction))
        correlation = np.corrcoef(fixed_array.flatten(), registered_array.flatten())[0,1]
        
        resultados.append({
            'método': nombre,
            'varianza_sustracción': var_subtraction,
            'diferencia_abs_media': mean_abs_diff,
            'correlación': correlation,
            'tiempo': datos.get('tiempo', 0),
            'métrica_registro': datos.get('metrica', 0)
        })
        
        # Visualización
        plt.subplot(3, 4, i*4 + 1)
        plt.imshow(registered_array[slice_central], cmap='gray')
        plt.title(f'{nombre}\nPOST registrado')
        plt.axis('off')
        
        plt.subplot(3, 4, i*4 + 2)
        plt.imshow(subtraction[slice_central], cmap='RdBu_r', vmin=-200, vmax=200)
        plt.title(f'Sustracción\nVar: {var_subtraction:.0f}')
        plt.axis('off')
        
        plt.subplot(3, 4, i*4 + 3)
        plt.imshow(np.abs(subtraction[slice_central]), cmap='hot')
        plt.title(f'Abs Sustracción\nMedia: {mean_abs_diff:.1f}')
        plt.axis('off')
        
        plt.subplot(3, 4, i*4 + 4)
        plt.hist(subtraction.flatten()[::1000], bins=50, alpha=0.7)
        plt.title(f'Distribución\nCorr: {correlation:.3f}')
        plt.xlabel('Intensidad')
        
        print(f"  📊 {nombre}: Var={var_subtraction:.0f}, Corr={correlation:.3f}, Tiempo={datos.get('tiempo', 0):.1f}s")
    
    plt.tight_layout()
    plt.suptitle(f'COMPARACIÓN MÉTODOS - {case_id}', fontsize=16, y=0.98)
    plt.savefig(f'test_registro_{case_id}.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Mostrar ranking
    if resultados:
        import pandas as pd
        df_resultados = pd.DataFrame(resultados)
        df_resultados = df_resultados.sort_values('varianza_sustracción')
        
        print(f"\n{'='*60}")
        print(f"RANKING PARA {case_id} (menor varianza = mejor)")
        print(f"{'='*60}")
        print(df_resultados[['método', 'varianza_sustracción', 'correlación', 'tiempo']].to_string(index=False))
    
    return resultados

# Probar casos extremos
if __name__ == "__main__":
    print("🧪 PROBANDO MÉTODOS EN CASOS EXTREMOS")
    print("="*50)
    
    # Casos identificados previamente
    caso_facil = "ISPY1_1229"    # Score: 10.0
    caso_dificil = "ISPY2_467622"  # Score: 1613.0
    
    print(f"\n1. PROBANDO CASO FÁCIL: {caso_facil}")
    resultados_facil = test_registro_caso_especifico(caso_facil)
    
    print(f"\n2. PROBANDO CASO DIFÍCIL: {caso_dificil}")
    resultados_dificil = test_registro_caso_especifico(caso_dificil)
    
    print(f"\n🎯 CONCLUSIONES:")
    print(f"- Compara las imágenes generadas")
    print(f"- ¿Qué método es más robusto?")
    print(f"- ¿Vale la pena el registro en casos difíciles?")