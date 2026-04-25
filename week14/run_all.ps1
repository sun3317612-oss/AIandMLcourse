# Week 14 모든 예제 자동 실행 스크립트 (PowerShell)
# All PINN examples runner

$scripts = @(
    "01_basic_pinn.py",
    "02_heat_equation_1d.py",
    "03_wave_equation_1d.py",
    "04_heat_equation_2d.py",
    "05_burgers_equation.py",
    "06_wave_equation_2d.py",
    "07_complex_boundary.py"
)

Write-Host "=" * 70
Write-Host "Week 14: PINN 예제 전체 실행"
Write-Host "=" * 70
Write-Host "`n총 $($scripts.Count)개의 스크립트를 순차적으로 실행합니다.`n"

$results = @()
$totalStart = Get-Date

for ($i = 0; $i -lt $scripts.Count; $i++) {
    $script = $scripts[$i]
    $num = $i + 1
    
    Write-Host "`n[$num/$($scripts.Count)] $script 실행 중..." -ForegroundColor Cyan
    Write-Host ("-" * 70)
    
    $start = Get-Date
    
    try {
        & uv run $script
        $exitCode = $LASTEXITCODE
        
        $elapsed = (Get-Date) - $start
        
        if ($exitCode -eq 0) {
            Write-Host "`n✅ $script 완료 (실행 시간: $($elapsed.TotalSeconds.ToString('0.0'))초)" -ForegroundColor Green
            $results += [PSCustomObject]@{
                Script = $script
                Success = $true
                Time = $elapsed.TotalSeconds
            }
        } else {
            Write-Host "`n❌ $script 실패 (오류 코드: $exitCode)" -ForegroundColor Red
            $results += [PSCustomObject]@{
                Script = $script
                Success = $false
                Time = $elapsed.TotalSeconds
            }
        }
    }
    catch {
        $elapsed = (Get-Date) - $start
        Write-Host "`n❌ $script 실행 중 예외 발생: $_" -ForegroundColor Red
        $results += [PSCustomObject]@{
            Script = $script
            Success = $false
            Time = $elapsed.TotalSeconds
        }
    }
    
    Write-Host ("-" * 70)
}

$totalElapsed = (Get-Date) - $totalStart

# 최종 결과 요약
Write-Host "`n" + ("=" * 70)
Write-Host "실행 결과 요약"
Write-Host ("=" * 70)

$successCount = ($results | Where-Object { $_.Success }).Count

foreach ($result in $results) {
    $status = if ($result.Success) { "✅" } else { "❌" }
    $color = if ($result.Success) { "Green" } else { "Red" }
    Write-Host "$status $($result.Script.PadRight(30)) ($($result.Time.ToString('0.0'))초)" -ForegroundColor $color
}

Write-Host "`n" + ("=" * 70)
Write-Host "성공: $successCount/$($scripts.Count)"
Write-Host "총 실행 시간: $($totalElapsed.TotalSeconds.ToString('0.0'))초 ($($totalElapsed.TotalMinutes.ToString('0.1'))분)"
Write-Host ("=" * 70)

if ($successCount -eq $scripts.Count) {
    Write-Host "`n🎉 모든 예제가 성공적으로 실행되었습니다!" -ForegroundColor Green
} else {
    Write-Host "`n⚠️  $($scripts.Count - $successCount)개 예제에서 오류가 발생했습니다." -ForegroundColor Yellow
}
