from pathlib import Path
import SimpleITK as sitk

def get_case_name_from_nii_gz(file_path):
    '''
    Extrae nombre de caso de archivo .nii.gz
    case_000_0000.nii.gz → case_000
    '''
    # file_path.name = 'case_000_0000.nii.gz'
    name = file_path.name
    
    # Remover .nii.gz (no solo .gz)
    if name.endswith('.nii.gz'):
        base_name = name[:-7]  # Remover '.nii.gz' (7 caracteres)
    else:
        base_name = file_path.stem  # Fallback
    
    # Extraer caso base: case_000_0000 → case_000
    if base_name.endswith('_0000'):
        case_base = base_name[:-5]  # Remover '_0000'
        return case_base
    else:
        return None

def check_file_integrity(file_path):
    try:
        img = sitk.ReadImage(str(file_path))
        return True, "OK"
    except Exception as e:
        return False, str(e)

# Directorio de test
test_dir = Path(r"C:\nnUNet_raw\Dataset114_A2_PrePost1_Crops\imagesTs")
print(f"🔍 Procesando: {test_dir}")

# Buscar archivos _0000.nii.gz
channel0_files = list(test_dir.glob("*_0000.nii.gz"))
print(f"Archivos _0000.nii.gz encontrados: {len(channel0_files)}")

if len(channel0_files) == 0:
    print("❌ No se encontraron archivos _0000.nii.gz")
    exit()

# DEBUG: Mostrar extracción corregida
print(f"\\n🔍 DEBUG EXTRACCIÓN CORREGIDA:")
print("archivo.name → sin .nii.gz → caso_base")

cases = []
for i, f in enumerate(channel0_files[:5]):  # Primeros 5 para debug
    case_name = get_case_name_from_nii_gz(f)
    print(f"\\n{i+1}. {f.name}")
    print(f"   Sin .nii.gz: '{f.name[:-7]}'")
    print(f"   Caso extraído: '{case_name}'")
    
    if case_name:
        cases.append(case_name)

print(f"\\n📊 DEBUG RESULTADO:")
print(f"Casos extraídos en debug: {len(cases)}")

if len(cases) > 0:
    print(f"✅ EXTRACCIÓN FUNCIONA - PROCESANDO TODOS...")
    
    # Procesar TODOS los archivos
    all_cases = []
    for f in channel0_files:
        case_name = get_case_name_from_nii_gz(f)
        if case_name:
            all_cases.append(case_name)
    
    all_cases = sorted(list(set(all_cases)))
    print(f"\\nTotal casos únicos: {len(all_cases)}")
    
    # Verificar que existen ambos archivos por caso
    print(f"\\n🔍 Verificando existencia de archivos...")
    valid_cases = []
    missing_files = []
    
    for case in all_cases:
        file_pre = test_dir / f"{case}_0000.nii.gz"
        file_post = test_dir / f"{case}_0001.nii.gz"
        
        if file_pre.exists() and file_post.exists():
            valid_cases.append(case)
        else:
            missing_files.append(case)
    
    print(f"\\n📊 VERIFICACIÓN ARCHIVOS:")
    print(f"Casos con ambos archivos: {len(valid_cases)}")
    print(f"Casos con archivos faltantes: {len(missing_files)}")
    
    if len(missing_files) > 0:
        print(f"\\nPrimeros casos con archivos faltantes:")
        for case in missing_files[:5]:
            file_pre = test_dir / f"{case}_0000.nii.gz"
            file_post = test_dir / f"{case}_0001.nii.gz"
            print(f"  {case}: PRE={file_pre.exists()}, POST={file_post.exists()}")
    
    # Verificar integridad de archivos (una muestra)
    if len(valid_cases) > 0:
        print(f"\\n🔍 Verificando integridad de muestra...")
        
        sample_cases = valid_cases[:min(50, len(valid_cases))]  # Muestra de 50
        integrity_ok = []
        integrity_bad = []
        
        for case in sample_cases:
            file_pre = test_dir / f"{case}_0000.nii.gz"
            file_post = test_dir / f"{case}_0001.nii.gz"
            
            pre_ok, pre_msg = check_file_integrity(file_pre)
            post_ok, post_msg = check_file_integrity(file_post)
            
            if pre_ok and post_ok:
                integrity_ok.append(case)
            else:
                integrity_bad.append((case, pre_msg if not pre_ok else post_msg))
        
        print(f"\\n📊 INTEGRIDAD (muestra de {len(sample_cases)}):")
        print(f"Casos íntegros: {len(integrity_ok)}")
        print(f"Casos corruptos: {len(integrity_bad)}")
        
        if len(integrity_bad) > 0:
            print(f"\\nCasos corruptos encontrados:")
            for case, error in integrity_bad[:3]:
                print(f"  {case}: {error}")
        
        # Estimar casos válidos finales
        if len(sample_cases) > 0:
            integrity_rate = len(integrity_ok) / len(sample_cases)
            estimated_valid = int(len(valid_cases) * integrity_rate)
            print(f"\\n📈 ESTIMACIÓN FINAL:")
            print(f"Tasa de integridad: {integrity_rate:.2%}")
            print(f"Casos válidos estimados: {estimated_valid} de {len(valid_cases)}")
        
        # Guardar casos válidos (todos los que tienen archivos)
        with open("final_valid_cases.txt", "w") as f:
            for case in valid_cases:
                f.write(f"{case}\\n")
        
        print(f"\\n✅ Casos guardados en: final_valid_cases.txt")
        print(f"Total casos para predicción: {len(valid_cases)}")
        
        if len(valid_cases) >= 250:
            print(f"\\n🎉 SUFICIENTES CASOS PARA PREDICCIÓN")
            print(f"Listo para proceder con {len(valid_cases)} casos")
        else:
            print(f"\\n⚠️ Pocos casos válidos: {len(valid_cases)}")
    
else:
    print(f"\\n❌ EXTRACCIÓN SIGUE FALLANDO")
    
    # Debug adicional
    print(f"\\nDEBUG ADICIONAL:")
    for f in channel0_files[:3]:
        print(f"Archivo: {f.name}")
        print(f"  Termina en .nii.gz: {f.name.endswith('.nii.gz')}")
        print(f"  Sin .nii.gz: '{f.name[:-7] if f.name.endswith('.nii.gz') else 'N/A'}'")
        print(f"  Termina en _0000: {f.name[:-7].endswith('_0000') if f.name.endswith('.nii.gz') else False}")
