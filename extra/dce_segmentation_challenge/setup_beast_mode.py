# setup_beast_mode.py - Configuración y verificación para 4x RTX A6000
import torch
import os
import psutil
import subprocess
import platform

def check_system_requirements():
    """Verificar requisitos del sistema para BEAST MODE"""
    
    print("🔍 VERIFICANDO CONFIGURACIÓN BEAST MODE")
    print("=" * 60)
    
    # Sistema operativo
    print(f"💻 OS: {platform.system()} {platform.release()}")
    
    # CPU
    cpu_count = psutil.cpu_count(logical=False)
    cpu_count_logical = psutil.cpu_count(logical=True)
    print(f"🧮 CPU: {cpu_count} cores físicos, {cpu_count_logical} lógicos")
    
    # RAM
    memory = psutil.virtual_memory()
    ram_gb = memory.total / (1024**3)
    print(f"💾 RAM: {ram_gb:.1f} GB total, {memory.available/(1024**3):.1f} GB disponible")
    
    # CUDA y PyTorch
    print(f"\n🔥 CONFIGURACIÓN CUDA:")
    print(f"   PyTorch version: {torch.__version__}")
    print(f"   CUDA disponible: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"   CUDA version: {torch.version.cuda}")
        print(f"   cuDNN version: {torch.backends.cudnn.version()}")
        print(f"   Número de GPUs: {torch.cuda.device_count()}")
        
        total_vram = 0
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            total_vram += gpu_memory
            print(f"   GPU {i}: {gpu_name} - {gpu_memory:.1f} GB VRAM")
        
        print(f"   🚀 VRAM total: {total_vram:.1f} GB")
        
        # Verificar que sean RTX A6000
        if "A6000" in torch.cuda.get_device_name(0):
            print("   ✅ RTX A6000 detectadas - BEAST MODE READY!")
        else:
            print("   ⚠️  No son RTX A6000, pero se puede adaptar")
    else:
        print("   ❌ CUDA no disponible!")
        return False
    
    return True

def optimize_system_settings():
    """Optimizar configuraciones del sistema para multi-GPU"""
    
    print("\n⚙️ OPTIMIZANDO CONFIGURACIONES")
    print("-" * 40)
    
    # Variables de entorno para multi-GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'  # Usar las 4 GPUs
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    
    # Optimizaciones de cuDNN
    torch.backends.cudnn.benchmark = True  # Optimizar para input size fijo
    torch.backends.cudnn.enabled = True
    
    # Configuraciones de memoria
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
    
    print("✅ Variables de entorno configuradas")
    print("✅ cuDNN optimizado")
    print("✅ Gestión de memoria configurada")

def test_multi_gpu_performance():
    """Test de rendimiento multi-GPU"""
    
    print("\n🧪 TESTING MULTI-GPU PERFORMANCE")
    print("-" * 40)
    
    if not torch.cuda.is_available():
        print("❌ CUDA no disponible para testing")
        return
    
    device_count = torch.cuda.device_count()
    print(f"🔥 Testing en {device_count} GPUs...")
    
    # Test de memoria
    for i in range(device_count):
        try:
            # Crear tensor grande para test
            test_tensor = torch.randn(1, 8, 320, 320, 256, device=f'cuda:{i}')
            memory_used = torch.cuda.memory_allocated(i) / (1024**3)
            print(f"   GPU {i}: Test tensor creado - {memory_used:.2f} GB usados")
            del test_tensor
            torch.cuda.empty_cache()
        except RuntimeError as e:
            print(f"   GPU {i}: Error - {e}")
    
    # Test de DataParallel
    try:
        from train import BeastUNet3D
        model = BeastUNet3D(in_channels=8, num_classes=2, base_features=32)  # Más pequeño para test
        
        if device_count > 1:
            model = torch.nn.DataParallel(model)
        
        model = model.cuda()
        
        # Test forward pass
        test_input = torch.randn(4, 8, 160, 160, 128).cuda()  # Batch pequeño para test
        
        import time
        start_time = time.time()
        
        with torch.no_grad():
            output = model(test_input)
        
        torch.cuda.synchronize()
        end_time = time.time()
        
        print(f"✅ DataParallel test OK - {end_time - start_time:.3f}s")
        print(f"   Input shape: {test_input.shape}")
        
        if isinstance(output, tuple):
            print(f"   Output shapes: {[o.shape for o in output]}")
        else:
            print(f"   Output shape: {output.shape}")
        
        del model, test_input, output
        torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"❌ DataParallel test failed: {e}")

def estimate_training_capacity():
    """Estimar capacidad de entrenamiento"""
    
    print("\n📊 ESTIMANDO CAPACIDAD DE ENTRENAMIENTO")
    print("-" * 45)
    
    if not torch.cuda.is_available():
        return
    
    # Parámetros del modelo BEAST
    model_params = 150_000_000  # ~150M parámetros estimados
    bytes_per_param = 4  # float32
    
    # Memoria por componente (por GPU)
    memory_per_gpu = 48  # GB RTX A6000
    
    model_memory = model_params * bytes_per_param / (1024**3)
    
    # Estimaciones por sample (alta resolución)
    input_memory_per_sample = 8 * 320 * 320 * 256 * 4 / (1024**3)  # 8 channels, float32
    gradients_memory = model_memory  # Aproximadamente igual
    optimizer_memory = model_memory * 2  # AdamW necesita 2x parámetros
    
    print(f"💾 Memoria por componente (por GPU):")
    print(f"   Modelo: {model_memory:.2f} GB")
    print(f"   Input por sample: {input_memory_per_sample:.3f} GB")
    print(f"   Gradientes: {gradients_memory:.2f} GB")
    print(f"   Optimizer states: {optimizer_memory:.2f} GB")
    
    # Cálculo de batch size óptimo
    fixed_memory = model_memory + gradients_memory + optimizer_memory
    available_per_gpu = memory_per_gpu * 0.85  # 85% usable
    memory_for_data = available_per_gpu - fixed_memory
    
    optimal_batch_per_gpu = int(memory_for_data / input_memory_per_sample)
    total_batch_size = optimal_batch_per_gpu * torch.cuda.device_count()
    
    print(f"\n🎯 BATCH SIZE ÓPTIMO:")
    print(f"   Por GPU: {optimal_batch_per_gpu}")
    print(f"   Total (4 GPUs): {total_batch_size}")
    print(f"   Memory usage por GPU: ~{available_per_gpu:.1f} GB")
    
    # Estimación de tiempo de entrenamiento
    samples_per_epoch = 200  # Casos de entrenamiento
    batches_per_epoch = samples_per_epoch / total_batch_size
    seconds_per_batch = 3.0  # Estimación para high-res
    
    time_per_epoch = batches_per_epoch * seconds_per_batch / 60  # minutos
    
    print(f"\n⏱️ ESTIMACIÓN DE TIEMPO:")
    print(f"   Batches por epoch: {batches_per_epoch:.1f}")
    print(f"   Tiempo por epoch: {time_per_epoch:.1f} minutos")
    print(f"   Tiempo 100 epochs: {time_per_epoch * 100 / 60:.1f} horas")

def setup_monitoring():
    """Configurar monitoreo del sistema"""
    
    print("\n📡 CONFIGURANDO MONITOREO")
    print("-" * 30)
    
    monitor_script = """
# monitor_beast.py - Script de monitoreo durante entrenamiento
import time
import torch
import psutil
import os

def monitor_system():
    while True:
        print("\\n" + "="*50)
        print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # CPU y RAM
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        print(f"CPU: {cpu_percent:.1f}% | RAM: {memory.percent:.1f}%")
        
        # GPUs
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                memory_used = torch.cuda.memory_allocated(i) / (1024**3)
                memory_total = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                temp = torch.cuda.temperature(i) if hasattr(torch.cuda, 'temperature') else 'N/A'
                print(f"GPU {i}: {memory_used:.1f}/{memory_total:.1f} GB | Temp: {temp}°C")
        
        time.sleep(10)  # Actualizar cada 10 segundos

if __name__ == "__main__":
    monitor_system()
"""
    
    with open("monitor_beast.py", "w") as f:
        f.write(monitor_script)
    
    print("✅ Script de monitoreo creado: monitor_beast.py")
    print("   Ejecutar en terminal separada: python monitor_beast.py")

def main():
    """Función principal de setup"""
    
    print("🚀 CONFIGURACIÓN BEAST MODE PARA 4x RTX A6000")
    print("=" * 60)
    
    # Verificar sistema
    if not check_system_requirements():
        print("❌ Sistema no cumple requisitos mínimos")
        return
    
    # Optimizar configuraciones
    optimize_system_settings()
    
    # Test de rendimiento
    test_multi_gpu_performance()
    
    # Estimar capacidad
    estimate_training_capacity()
    
    # Configurar monitoreo
    setup_monitoring()
    
    print("\n" + "=" * 60)
    print("✅ BEAST MODE SETUP COMPLETADO!")
    print("\n📋 SIGUIENTES PASOS:")
    print("1. Ejecutar: python dataset.py  (verificar dataset)")
    print("2. Ejecutar: python train.py   (iniciar entrenamiento)")
    print("3. En terminal separada: python monitor_beast.py")
    print("\n🎯 OBJETIVO: Superar baseline con alta resolución!")
    print("🚀 ¡BEAST MODE ACTIVATED!")

if __name__ == "__main__":
    main()