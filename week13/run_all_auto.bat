@echo off
chcp 65001 > nul
echo ======================================================================
echo Week 13: PINN 기초 이론 (ODE 편) - 자동 실행 (중단 없음)
echo ======================================================================
echo.
echo 총 6개의 프로그램을 자동으로 실행합니다.
echo 예상 소요 시간: 약 20-30분
echo.
timeout /t 5

echo.
echo [1/6] 01_simple_ode.py 실행 중...
uv run python 01_simple_ode.py
if %errorlevel% neq 0 (
    echo ERROR: 01_simple_ode.py 실행 실패
    exit /b %errorlevel%
)
echo [1/6] 완료

echo.
echo [2/6] 02_harmonic_oscillator.py 실행 중...
uv run python 02_harmonic_oscillator.py
if %errorlevel% neq 0 (
    echo ERROR: 02_harmonic_oscillator.py 실행 실패
    exit /b %errorlevel%
)
echo [2/6] 완료

echo.
echo [3/6] 03_damped_oscillator.py 실행 중...
uv run python 03_damped_oscillator.py
if %errorlevel% neq 0 (
    echo ERROR: 03_damped_oscillator.py 실행 실패
    exit /b %errorlevel%
)
echo [3/6] 완료

echo.
echo [4/6] 04_boundary_value_problem.py 실행 중...
uv run python 04_boundary_value_problem.py
if %errorlevel% neq 0 (
    echo ERROR: 04_boundary_value_problem.py 실행 실패
    exit /b %errorlevel%
)
echo [4/6] 완료

echo.
echo [5/6] 05_lorenz_system.py 실행 중...
uv run python 05_lorenz_system.py
if %errorlevel% neq 0 (
    echo ERROR: 05_lorenz_system.py 실행 실패
    exit /b %errorlevel%
)
echo [5/6] 완료

echo.
echo [6/6] 06_comparison_frameworks.py 실행 중...
uv run python 06_comparison_frameworks.py
if %errorlevel% neq 0 (
    echo ERROR: 06_comparison_frameworks.py 실행 실패
    exit /b %errorlevel%
)
echo [6/6] 완료

echo.
echo ======================================================================
echo 모든 프로그램 실행 완료!
echo ======================================================================
echo.
echo 생성된 파일:
dir outputs\*.png /b 2>nul
echo.
echo 총 실행 시간: 약 %time%
echo.
pause

