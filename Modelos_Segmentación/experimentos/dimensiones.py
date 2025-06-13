import nibabel as nib
from pathlib import Path

# Ruta a la carpeta con predicciones
ruta = Path(r"C:\nnUNet_raw\Dataset114_A2_PrePost1_Crops\labelsTs")

# Obtener lista de archivos .nii.gz
archivos = sorted(ruta.glob("*.nii.gz"))

# Comprobar si se encontraron archivos
if not archivos:
    print("No se encontraron archivos .nii.gz en la ruta especificada.")
else:
    print(f"Total de archivos encontrados: {len(archivos)}\n")

    # Mostrar dimensiones de cada archivo
    for archivo in archivos:
        try:
            img = nib.load(str(archivo))
            shape = img.shape
            print(f"{archivo.name}: {shape}")
        except Exception as e:
            print(f"Error al leer {archivo.name}: {e}")
