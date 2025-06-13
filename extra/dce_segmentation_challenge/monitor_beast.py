# -*- coding: utf-8 -*-
# monitor_beast.py - Script de monitoreo durante entrenamiento
import time
import torch
import psutil
import os

def monitor_system():
    while True:
        print("\n" + "="*50)
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
                print(f"GPU {i}: {memory_used:.1f}/{memory_total:.1f} GB | Temp: {temp} C")
        
        time.sleep(10)  # Actualizar cada 10 segundos

if __name__ == "__main__":
    monitor_system()
