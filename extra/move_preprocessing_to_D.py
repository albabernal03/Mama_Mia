import os
import shutil
from pathlib import Path

def move_preprocessing_to_d_drive():
    """Mover preprocessing existente de C: a D: y continuar ahí"""
    
    print("📁 MOVIENDO PREPROCESSING A DISCO D:")
    print("=" * 50)
    
    # Rutas
    old_folder = r"C:\Users\usuario\Documents\Mama_Mia\preprocessed_mama_mia_style"
    new_folder = r"D:\preprocessed_mama_mia_style"
    
    # 1. Verificar que la carpeta origen existe
    if not os.path.exists(old_folder):
        print(f"❌ No existe carpeta origen: {old_folder}")
        return False
    
    # 2. Contar archivos existentes
    print(f"🔍 Verificando carpeta origen...")
    existing_patients = [d for d in os.listdir(old_folder) 
                        if os.path.isdir(os.path.join(old_folder, d))]
    
    print(f"📊 Pacientes ya procesados: {len(existing_patients)}")
    if existing_patients:
        print(f"📋 Ejemplos: {existing_patients[:5]}")
    
    # 3. Verificar espacio en D:
    try:
        total, used, free = shutil.disk_usage("D:\\")
        free_gb = free // (1024**3)
        print(f"💾 Espacio libre en D:: {free_gb} GB")
        
        if free_gb < 10:
            print(f"⚠️ Poco espacio en D:")
            return False
        else:
            print(f"✅ Espacio suficiente en D:")
    except Exception as e:
        print(f"❌ Error verificando espacio en D:: {e}")
        return False
    
    # 4. Crear directorio destino
    try:
        os.makedirs(new_folder, exist_ok=True)
        print(f"📁 Directorio creado: {new_folder}")
    except Exception as e:
        print(f"❌ Error creando directorio: {e}")
        return False
    
    # 5. Copiar archivos (no mover, por seguridad)
    print(f"\n🔄 Copiando archivos...")
    
    copied_count = 0
    failed_count = 0
    
    for i, patient_id in enumerate(existing_patients, 1):
        try:
            source_path = Path(old_folder) / patient_id
            dest_path = Path(new_folder) / patient_id
            
            print(f"[{i}/{len(existing_patients)}] Copiando {patient_id}...", end="")
            
            # Copiar carpeta completa del paciente
            shutil.copytree(source_path, dest_path, dirs_exist_ok=True)
            
            copied_count += 1
            print(" ✅")
            
            # Mostrar progreso cada 20 pacientes
            if i % 20 == 0:
                print(f"    📊 Progreso: {i}/{len(existing_patients)} - {copied_count} exitosos")
            
        except Exception as e:
            failed_count += 1
            print(f" ❌ Error: {e}")
    
    # 6. Verificar resultado
    print(f"\n📊 RESULTADO:")
    print(f"✅ Copiados exitosamente: {copied_count}")
    print(f"❌ Fallidos: {failed_count}")
    
    if copied_count > 0:
        print(f"\n🎯 SIGUIENTE PASO:")
        print(f"1. Cambiar OUTPUT_FOLDER en tu script a:")
        print(f"   OUTPUT_FOLDER = r\"{new_folder}\"")
        print(f"2. Ejecutar preprocessing - saltará los ya copiados")
        print(f"3. Continuará procesando el resto en D: con espacio")
        
        # Crear archivo de configuración
        config_content = f'''# NUEVA CONFIGURACIÓN PARA DISCO D:

IMAGES_FOLDER = r"C:\\Users\\usuario\\Documents\\Mama_Mia\\datos\\images"
SEGMENTATIONS_FOLDER = r"C:\\Users\\usuario\\Documents\\Mama_Mia\\datos\\segmentations"  
SPLITS_CSV = r"C:\\Users\\usuario\\Documents\\Mama_Mia\\datos\\train_test_splits.csv"

# NUEVA RUTA EN DISCO D:
OUTPUT_FOLDER = r"{new_folder}"

print("✅ CONFIGURACIÓN ACTUALIZADA PARA DISCO D:")
print(f"📁 Output: {{OUTPUT_FOLDER}}")
print(f"💾 Pacientes ya copiados: {copied_count}")
'''
        
        with open("config_disco_D.py", "w", encoding="utf-8") as f:
            f.write(config_content)
        
        print(f"📝 Archivo creado: config_disco_D.py")
        
        return True
    else:
        print(f"❌ No se pudo copiar ningún archivo")
        return False

def verify_d_drive_space():
    """Verificar disponibilidad del disco D:"""
    
    print("💾 VERIFICANDO DISCO D:")
    print("=" * 30)
    
    try:
        if os.path.exists("D:\\"):
            total, used, free = shutil.disk_usage("D:\\")
            free_gb = free // (1024**3)
            total_gb = total // (1024**3)
            used_gb = used // (1024**3)
            
            print(f"📊 Disco D: estadísticas:")
            print(f"   Total: {total_gb} GB")
            print(f"   Usado: {used_gb} GB") 
            print(f"   Libre: {free_gb} GB")
            
            if free_gb > 100:
                print(f"✅ EXCELENTE - Mucho espacio disponible")
                return True
            elif free_gb > 50:
                print(f"✅ BUENO - Espacio suficiente")
                return True
            elif free_gb > 10:
                print(f"⚠️ LIMITADO - Poco espacio pero usable") 
                return True
            else:
                print(f"❌ INSUFICIENTE - Muy poco espacio")
                return False
        else:
            print(f"❌ Disco D: no existe o no está disponible")
            return False
            
    except Exception as e:
        print(f"❌ Error verificando disco D:: {e}")
        return False

if __name__ == "__main__":
    
    print("🔄 MIGRANDO PREPROCESSING A DISCO D:")
    print("=" * 50)
    
    # Verificar disco D:
    if verify_d_drive_space():
        
        # Proceder con la migración
        success = move_preprocessing_to_d_drive()
        
        if success:
            print(f"\n🎉 ¡MIGRACIÓN EXITOSA!")
            print(f"🚀 Ahora puedes continuar el preprocessing en D: sin problemas de espacio")
        else:
            print(f"\n❌ Error en la migración")
    else:
        print(f"\n❌ Disco D: no disponible o sin espacio suficiente")