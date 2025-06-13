# Resumen del Análisis de Segmentaciones

## Estadísticas Generales
- Total de casos analizados: 1506
- Casos con múltiples tumores (>3 en un slice): 800
- Volumen tumoral promedio (experto): 30398.97 píxeles
- Promedio de slices con tumor por paciente: 26.14
- Volumen tumoral promedio (automático): 32247.95 píxeles

## Casos Destacados

### Slices con mayor número de tumores
- Paciente duke_543, Slice 76: 188 tumores
- Paciente duke_543, Slice 77: 181 tumores
- Paciente duke_543, Slice 78: 178 tumores
- Paciente duke_543, Slice 74: 173 tumores
- Paciente duke_543, Slice 80: 173 tumores

### Pacientes con mayor volumen tumoral
- Paciente ispy2_559021: 802064.00 píxeles
- Paciente ispy2_928815: 682026.00 píxeles
- Paciente ispy2_728182: 669373.00 píxeles
- Paciente duke_535: 641823.00 píxeles
- Paciente ispy2_357017: 545914.00 píxeles

## Visualizaciones generadas
Se han generado las siguientes visualizaciones en la carpeta 'resultados_analisis/imagenes/':
- Casos con múltiples tumores en un solo slice
- Slices con mayor área tumoral por paciente
- Casos límite con alto número de tumores

## Interpretación
- Los casos con múltiples tumores pequeños representan un desafío para la segmentación automática.
- Se observa variabilidad en el número de slices afectados por paciente.
- La comparación entre experto y automático muestra diferencias significativas en algunos casos.
