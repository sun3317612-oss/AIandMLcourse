@echo off
REM Week 10 전자기학 시뮬레이션 - 모든 프로그램 실행
REM Windows Batch Script

echo ======================================================================
echo Week 10: 전자기학 시뮬레이션 - 전체 실행
echo ======================================================================
echo.

echo outputs 디렉토리 확인 중...
if not exist outputs mkdir outputs
echo [OK] outputs 디렉토리 준비 완료
echo.

echo ======================================================================
echo 01. Electric Field Basics (단일 점전하)
echo ======================================================================
python 01_electric_field_basics.py
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] 01번 프로그램 실행 실패!
    pause
    exit /b 1
)
echo.

echo ======================================================================
echo 02. Electric Potential (전위와 등전위선)
echo ======================================================================
python 02_electric_potential.py
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] 02번 프로그램 실행 실패!
    pause
    exit /b 1
)
echo.

echo ======================================================================
echo 03. Electric Field Lines (전기력선)
echo ======================================================================
python 03_electric_field_lines.py
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] 03번 프로그램 실행 실패!
    pause
    exit /b 1
)
echo.

echo ======================================================================
echo 04. Magnetic Field Basics (직선 전류의 자기장)
echo ======================================================================
python 04_magnetic_field_basics.py
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] 04번 프로그램 실행 실패!
    pause
    exit /b 1
)
echo.

echo ======================================================================
echo 05. Lorentz Force (로렌츠 힘과 입자 운동)
echo ======================================================================
python 05_lorentz_force.py
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] 05번 프로그램 실행 실패!
    pause
    exit /b 1
)
echo.

echo ======================================================================
echo 06. Maxwell 1D (1D 파동 방정식)
echo ======================================================================
python 06_maxwell_1d.py
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] 06번 프로그램 실행 실패!
    pause
    exit /b 1
)
echo.

echo ======================================================================
echo 07. Maxwell 2D (2D 파동 방정식 - 시간이 걸릴 수 있습니다)
echo ======================================================================
python 07_maxwell_2d.py
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] 07번 프로그램 실행 실패!
    pause
    exit /b 1
)
echo.

echo ======================================================================
echo 08. Multiple Charges (다중 점전하)
echo ======================================================================
python 08_multiple_charges.py
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] 08번 프로그램 실행 실패!
    pause
    exit /b 1
)
echo.

echo ======================================================================
echo 09. EM Wave Animation (전자기파 애니메이션 - 60 프레임)
echo ======================================================================
python 09_em_wave_animation.py
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] 09번 프로그램 실행 실패!
    pause
    exit /b 1
)
echo.

echo ======================================================================
echo 10. Conductor Potential (도체 전위 분포 - 라플라스 방정식)
echo ======================================================================
python 10_conductor_potential.py
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] 10번 프로그램 실행 실패!
    pause
    exit /b 1
)
echo.

echo ======================================================================
echo 모든 프로그램 실행 완료!
echo ======================================================================
echo.
echo 생성된 파일은 outputs/ 디렉토리에서 확인하실 수 있습니다.
echo.
echo 생성된 파일 목록:
dir /b outputs
echo.

pause

