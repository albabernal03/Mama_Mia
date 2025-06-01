# LANZAR TODOS LOS EXPERIMENTOS EN PARALELO
# Cada experimento corre en un proceso separado

Write-Host "🚀 LANZANDO TODOS LOS EXPERIMENTOS EN PARALELO..." -ForegroundColor Green

$BASE_DIR = "..\datos"
$NUM_CASES = 1506

# LANZAR CADA EXPERIMENTO COMO TRABAJO EN SEGUNDO PLANO
Write-Host "Lanzando A1 Baseline..." -ForegroundColor Yellow
Start-Job -Name "A1_Baseline" -ScriptBlock {
    Set-Location $using:PWD
    python experiment_a1.py --base_dir $using:BASE_DIR --num_cases $using:NUM_CASES
}

Write-Host "Lanzando A2 Post-Only..." -ForegroundColor Yellow
Start-Job -Name "A2_PostOnly" -ScriptBlock {
    Set-Location $using:PWD
    python experiment_a2.py --base_dir $using:BASE_DIR --num_cases $using:NUM_CASES
}

Write-Host "Lanzando A3 Multi-Channel..." -ForegroundColor Yellow
Start-Job -Name "A3_MultiChannel" -ScriptBlock {
    Set-Location $using:PWD
    python experiment_a3.py --base_dir $using:BASE_DIR --num_cases $using:NUM_CASES
}

Write-Host "Lanzando B1 Optimal..." -ForegroundColor Yellow
Start-Job -Name "B1_Optimal" -ScriptBlock {
    Set-Location $using:PWD
    python experiment_b1.py --base_dir $using:BASE_DIR --num_cases $using:NUM_CASES
}

Write-Host "Lanzando A4 Temporal..." -ForegroundColor Yellow
Start-Job -Name "A4_Temporal" -ScriptBlock {
    Set-Location $using:PWD
    python experiment_a4.py --base_dir $using:BASE_DIR --num_cases $using:NUM_CASES
}

# MOSTRAR TRABAJOS ACTIVOS
Write-Host "`n📊 TRABAJOS EN EJECUCIÓN:" -ForegroundColor Green
Get-Job

Write-Host "`n⏱️ MONITOREANDO PROGRESO..." -ForegroundColor Cyan
Write-Host "Usar: Get-Job para ver estado" -ForegroundColor White
Write-Host "Usar: Receive-Job -Name 'A1_Baseline' para ver output" -ForegroundColor White
Write-Host "Usar: Wait-Job * para esperar que terminen todos" -ForegroundColor White

# LOOP DE MONITOREO
do {
    Start-Sleep 30
    $jobs = Get-Job
    $completed = ($jobs | Where-Object {$_.State -eq "Completed"}).Count
    $running = ($jobs | Where-Object {$_.State -eq "Running"}).Count
    $failed = ($jobs | Where-Object {$_.State -eq "Failed"}).Count
    
    Write-Host "$(Get-Date -Format 'HH:mm:ss') | Completados: $completed | Ejecutando: $running | Fallidos: $failed" -ForegroundColor Cyan
    
} while ($running -gt 0)

Write-Host "`n🎉 TODOS LOS EXPERIMENTOS COMPLETADOS!" -ForegroundColor Green

# MOSTRAR RESULTADOS
Write-Host "`n📋 RESULTADOS:" -ForegroundColor Yellow
foreach ($job in Get-Job) {
    Write-Host "`n--- $($job.Name) ---" -ForegroundColor White
    Receive-Job $job
}

# LIMPIAR TRABAJOS
Remove-Job *

Write-Host "`n🚀 LISTO PARA ENTRENAR:" -ForegroundColor Green
Write-Host "nnUNetv2_plan_and_preprocess -d 101 && nnUNetv2_train 101 3d_fullres 0" -ForegroundColor Cyan
Write-Host "nnUNetv2_plan_and_preprocess -d 102 && nnUNetv2_train 102 3d_fullres 0" -ForegroundColor Cyan
Write-Host "nnUNetv2_plan_and_preprocess -d 103 && nnUNetv2_train 103 3d_fullres 0" -ForegroundColor Cyan
Write-Host "nnUNetv2_plan_and_preprocess -d 104 && nnUNetv2_train 104 3d_fullres 0" -ForegroundColor Cyan
Write-Host "nnUNetv2_plan_and_preprocess -d 105 && nnUNetv2_train 105 3d_fullres 0" -ForegroundColor Cyan