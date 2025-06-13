# =====================================
# EVALUACIÓN FINAL REAL - A2 STRATEGY
# =====================================

Write-Host "🚀 EVALUACIÓN EN TU TEST SET REAL" -ForegroundColor Green

# Ubicaciones exactas
$datasetPath = "C:\nnUNet_raw\Dataset114_A2_PrePost1_Crops"
$testImages = "$datasetPath\imagesTs"
$testLabels = "$datasetPath\labelsTs"
$predictions = ".\final_test_predictions_A2"
$datasetJson = "$datasetPath\dataset.json"

Write-Host "
📁 CONFIGURACIÓN:" -ForegroundColor Yellow
Write-Host "   Dataset: Dataset114 A2 Strategy" -ForegroundColor Gray
Write-Host "   Test images: $testImages" -ForegroundColor Gray
Write-Host "   Test labels: $testLabels" -ForegroundColor Gray
Write-Host "   Predictions: $predictions" -ForegroundColor Gray

# Verificar casos de test
$testCount = (Get-ChildItem "$testImages" -Filter "*_0000.nii.gz").Count
Write-Host "
📊 Casos de test: $testCount" -ForegroundColor Cyan

# Crear carpeta de predicciones
New-Item -ItemType Directory -Path $predictions -Force | Out-Null

Write-Host "
🎯 GENERANDO PREDICCIONES..." -ForegroundColor Yellow
Write-Host "⏱️ Esto tomará varios minutos..."

# Generar predicciones en test set
nnUNetv2_predict -d 114 -i $testImages -o $predictions -c 3d_fullres -f 0

Write-Host "
✅ PREDICCIONES COMPLETADAS" -ForegroundColor Green

Write-Host "
📈 EVALUANDO CONTRA GROUND TRUTH..." -ForegroundColor Yellow

# Evaluar contra ground truth real
nnUNetv2_evaluate_folder -djfile $datasetJson -pf $predictions -gf $testLabels

Write-Host "
🎯 ESTE ES TU DICE REAL EN TEST" -ForegroundColor Red
Write-Host "🔄 Comparar con validación: 0.8060" -ForegroundColor Yellow

Write-Host "
📊 INTERPRETACIÓN:" -ForegroundColor Green
Write-Host "• Si test ≈ 0.806 (±0.02): ✅ Excelente generalización" -ForegroundColor Gray
Write-Host "• Si test > 0.806: 🎉 Modelo mejoró en test" -ForegroundColor Gray  
Write-Host "• Si test < 0.78: ⚠️ Posible overfitting" -ForegroundColor Gray
