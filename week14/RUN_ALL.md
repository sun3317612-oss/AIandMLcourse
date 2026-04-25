# Week 14 자동 실행 방법

## 모든 예제 한 번에 실행하기

### 방법 1: Python 스크립트 사용 (권장)

```bash
uv run run_all.py
```

### 방법 2: PowerShell 스크립트 사용 (Windows)

```powershell
.\run_all.ps1
```

## 개별 예제 실행

```bash
cd week14
uv run 01_basic_pinn.py       # 기본 ODE
uv run 02_heat_equation_1d.py # 1D Heat Equation
uv run 03_wave_equation_1d.py # 1D Wave Equation
uv run 04_heat_equation_2d.py # 2D Heat Equation
uv run 05_burgers_equation.py # Burgers Equation
uv run 06_wave_equation_2d.py # 2D Wave Equation
uv run 07_complex_boundary.py # Complex Boundary
```

## 주의사항

- 전체 실행 시간: 약 30-60분 예상
- GPU가 있으면 더 빠르게 실행됩니다
- 결과 이미지는 `outputs/` 폴더에 저장됩니다
- 메모리가 부족하면 일부 예제가 실패할 수 있습니다

## 출력 확인

모든 스크립트 실행 후 `outputs/` 폴더에서 결과 이미지를 확인하세요:

- `01_basic_pinn.png`
- `02_heat_equation_1d.png`
- `03_wave_equation_1d.png`
- `04_heat_equation_2d.png`
- `05_burgers_equation.png`
- `06_wave_equation_2d.png`
- `07_complex_boundary.png`
