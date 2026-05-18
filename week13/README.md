# Week 13: PINN 기초 이론 (ODE 편)

## 개요

Physics-Informed Neural Networks (PINNs)를 사용하여 상미분방정식(ODE)을 풀어보는 실습입니다.

## 프로그램 목록

1. **`01_simple_ode.py`** - 단순 ODE (TensorFlow)
   - dy/dt = -y, y(0) = 1
   - 가장 기본적인 PINN 예제
   
2. **`02_harmonic_oscillator.py`** - 단순 조화 진동자 (TensorFlow)
   - d²y/dt² + ω²y = 0 (ω = 2)
   - 선형 조화 진동자
   
3. **`03_damped_oscillator.py`** - 감쇠 진동자 (TensorFlow)
   - d²y/dt² + 2ζω·dy/dt + ω²y = 0
   - Under/Critical/Over-damped 비교
   
4. **`04_boundary_value_problem.py`** - 경계값 문제 (TensorFlow)
   - d²y/dx² = -x, y(0)=0, y(1)=0
   - BVP vs IVP
   
5. **`05_lorenz_system.py`** - 로렌츠 시스템 (TensorFlow)
   - 3변수 coupled ODE system
   - 혼돈 역학 (Chaotic dynamics)
   
6. **`06_comparison_frameworks.py`** - 방법 비교
   - PINN (TensorFlow) vs Traditional (RK4)
   - 성능/정확도 비교

## 실행 방법

### 방법 1: 개별 실행

각 프로그램을 개별적으로 실행:

```batch
uv run python 01_simple_ode.py
uv run python 02_harmonic_oscillator.py
...
```

### 방법 2: 순차 실행 (단계별 확인)

각 프로그램 실행 후 pause하여 결과 확인:

```batch
run_all.bat
```

- 각 프로그램 실행 후 일시 정지
- Enter 키를 눌러 다음 프로그램 진행
- 중간에 에러 발생 시 자동 중단

### 방법 3: 자동 실행 (중단 없음)

모든 프로그램을 자동으로 연속 실행:

```batch
run_all_auto.bat
```

- 중간에 pause 없이 자동 실행
- 예상 소요 시간: 20-30분
- 에러 발생 시에만 중단

## 출력 결과

모든 결과는 `outputs/` 디렉토리에 PNG 형식으로 저장됩니다:

```
outputs/
├── 01_simple_ode_pinn.png
├── 01_ode_residual.png
├── 02_harmonic_oscillator.png
├── 02_comparison_table.png
├── 03_damped_oscillator.png
├── 03_error_analysis.png
├── 03_damping_analysis.png
├── 04_boundary_value_problem.png
├── 05_lorenz_3d.png
├── 05_lorenz_analysis.png
├── 06_method_comparison.png
└── 06_recommendations.png
```

## 실행 시간

- 각 프로그램: 3-5분
- 전체 실행: 20-30분
- TensorFlow가 CPU/GPU를 자동으로 선택

## 의존성

필요한 패키지는 프로젝트 루트의 `pyproject.toml`에 정의되어 있습니다:

- `tensorflow` - TensorFlow 기반 PINN (모든 프로그램)
- `numpy` - 수치 계산
- `scipy` - RK4 비교용
- `matplotlib` - 시각화
- `psutil` - 메모리 사용량 측정 (06)

## 강의 자료

자세한 설명은 `week13.md`를 참조하세요:

- PINN 기본 개념
- Automatic Differentiation
- Loss 함수 설계
- 프로그램별 상세 해설
- PINN vs Traditional 방법 비교
- 실습 과제 및 FAQ

## 문제 해결

### matplotlib 한글 폰트 오류

프로그램에 한글 폰트 자동 선택 기능이 포함되어 있습니다:
- Windows: Malgun Gothic, Gulim, Batang
- Mac: AppleGothic

### 메모리 부족

프로그램이 메모리 부족으로 중단될 경우:
1. 다른 프로그램 종료
2. Epochs 수를 줄이기 (예: 10000 → 5000)
3. Collocation points 줄이기 (예: 200 → 100)

### 학습이 느린 경우

- TensorFlow는 CPU/GPU 자동 선택
- Epochs를 줄여서 빠른 테스트 가능
- 첫 실행 시 그래프 컴파일로 인해 느릴 수 있음

### Loss가 감소하지 않는 경우

1. Learning rate 조정 (0.001 → 0.0001)
2. IC/BC 가중치 증가 (100 → 1000)
3. 네트워크 크기 증가 (32 → 64 뉴런)

## 참고 사항

- 모든 프로그램은 CLI 기반으로 작동
- `plt.show()`가 포함되어 있어 그래프 창이 표시됨
- 그래프 창을 닫으면 다음으로 진행
- 그래프를 보지 않으려면 `plt.show()` 주석 처리

## 다음 단계

Week 13을 마친 후:
1. `week13.md`로 개념 복습
2. 파라미터를 변경하며 실험
3. Week 14: PINN PDE 편으로 진행

## 문의

강의 관련 문의사항은 담당 교수님께 연락하세요.

---

**전산물리: Neural Network와 물리 시뮬레이션**  
Week 13: PINN 기초 이론 (ODE 편)

