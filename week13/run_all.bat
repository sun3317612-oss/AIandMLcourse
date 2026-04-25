@echo off
chcp 65001 > nul
echo ======================================================================
echo Week 13: PINN 기초 이론 (ODE 편) - 전체 프로그램 실행
echo ======================================================================
echo.
echo 총 6개의 프로그램을 순차적으로 실행합니다.
echo 각 프로그램은 3-5분 정도 소요됩니다.
echo.
echo 출력 파일은 outputs 디렉토리에 저장됩니다.
echo.
pause

echo.
echo [1/6] 01_simple_ode.py - 단순 ODE (TensorFlow)
echo ======================================================================
uv run python 01_simple_ode.py
if %errorlevel% neq 0 (
    echo ERROR: 01_simple_ode.py 실행 실패
    pause
    exit /b %errorlevel%
)
echo.
echo 완료! outputs/01_*.png 생성됨
echo.
pause

echo.
echo [2/6] 02_harmonic_oscillator.py - 단진자 (PyTorch)
echo ======================================================================
uv run python 02_harmonic_oscillator.py
if %errorlevel% neq 0 (
    echo ERROR: 02_harmonic_oscillator.py 실행 실패
    pause
    exit /b %errorlevel%
)
echo.
echo 완료! outputs/02_*.png 생성됨
echo.
pause

echo.
echo [3/6] 03_damped_oscillator.py - 감쇠 진동자 (TensorFlow)
echo ======================================================================
uv run python 03_damped_oscillator.py
if %errorlevel% neq 0 (
    echo ERROR: 03_damped_oscillator.py 실행 실패
    pause
    exit /b %errorlevel%
)
echo.
echo 완료! outputs/03_*.png 생성됨
echo.
pause

echo.
echo [4/6] 04_boundary_value_problem.py - 경계값 문제 (PyTorch)
echo ======================================================================
uv run python 04_boundary_value_problem.py
if %errorlevel% neq 0 (
    echo ERROR: 04_boundary_value_problem.py 실행 실패
    pause
    exit /b %errorlevel%
)
echo.
echo 완료! outputs/04_*.png 생성됨
echo.
pause

echo.
echo [5/6] 05_lorenz_system.py - 로렌츠 시스템 (PyTorch)
echo ======================================================================
uv run python 05_lorenz_system.py
if %errorlevel% neq 0 (
    echo ERROR: 05_lorenz_system.py 실행 실패
    pause
    exit /b %errorlevel%
)
echo.
echo 완료! outputs/05_*.png 생성됨
echo.
pause

echo.
echo [6/6] 06_comparison_frameworks.py - 프레임워크 비교
echo ======================================================================
uv run python 06_comparison_frameworks.py
if %errorlevel% neq 0 (
    echo ERROR: 06_comparison_frameworks.py 실행 실패
    pause
    exit /b %errorlevel%
)
echo.
echo 완료! outputs/06_*.png 생성됨
echo.

echo.
echo ======================================================================
echo 모든 프로그램 실행 완료!
echo ======================================================================
echo.
echo 생성된 파일 목록:
echo.
dir outputs\*.png /b
echo.
echo outputs 디렉토리에서 결과 이미지를 확인하세요.
echo.
pause

