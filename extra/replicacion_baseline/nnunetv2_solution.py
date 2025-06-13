# ALTERNATIVA RAPIDA - Usar modelo nnUNet v1 compatible

print("ALTERNATIVA RAPIDA - MAMA-MIA")
print("=" * 40)

print("Tu modelo actual es nnUNet v2 (PyTorch .pth files)")
print("El nnUNet v1 instalado espera .pkl files")
print("")
print("OPCIONES:")
print("")
print("1. INSTALAR nnUNetv2 (RECOMENDADO):")
print("   pip install nnunetv2")
print("   Luego usar el script anterior")
print("")
print("2. USAR MODELO COMPATIBLE v1:")
print("   - Buscar pesos MAMA-MIA en formato nnUNet v1")
print("   - O convertir los pesos actuales")
print("")
print("3. IMPLEMENTACION MANUAL:")
print("   - Cargar checkpoints PyTorch directamente")
print("   - Implementar pipeline de inferencia custom")
print("")

# Verificar qué versión de nnUNet tienes
try:
    import nnunet
    print(f"nnUNet v1 instalado: {nnunet.__file__}")
except:
    print("nnUNet v1 no encontrado")

try:
    import nnunetv2
    print(f"nnUNetv2 instalado: {nnunetv2.__file__}")
except:
    print("nnUNetv2 NO instalado")

print("\nRECOMENDACION:")
print("pip install nnunetv2")
print("Es la solucion mas rapida para tu modelo actual")