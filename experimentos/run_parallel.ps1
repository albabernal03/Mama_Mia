# SCRIPT CORREGIDO - COPIAR TODO ESTO
# LANZAR TODOS LOS EXPERIMENTOS EN PARALELO

Write-Host "LANZANDO TODOS LOS EXPERIMENTOS EN PARALELO..." -ForegroundColor Green

$BASE_DIR = "..\datos"
$NUM_CASES = 1506

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


Write-Host "Lanzando A4 Temporal..." -ForegroundColor Yellow
Start-Job -Name "A4_Temporal" -ScriptBlock {
    Set-Location $using:PWD
    python experiment_a4.py --base_dir $using:BASE_DIR --num_cases $using:NUM_CASES
}

Write-Host "TRABAJOS EN EJECUCION:" -ForegroundColor Green
Get-Job

Write-Host "Usar Get-Job para monitorear progreso" -ForegroundColor White