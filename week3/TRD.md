# TRD (Technical Requirements Document)
# Week3 신경망 기초 탐색기 (Neural Networks Interactive Explorer)

---

## 1. 기술 스택 (Tech Stack)

| 구분 | 기술 | 버전 | 용도 |
|------|------|------|------|
| 언어 | Python | 3.12+ | 전체 구현 |
| GUI | PySide6 | 6.x | 윈도우/위젯/레이아웃 |
| 수치 계산 | NumPy | 1.26+ | 신경망 행렬 연산 |
| 시각화 | Matplotlib | 3.8+ | 그래프 렌더링 |
| 패키지 관리 | uv | - | 의존성 관리 |

---

## 2. 아키텍처 (Architecture)

### 2.1 모듈 구조

```
week3/
├── week3_neural_explorer.py   # 메인 실행 파일 (단일 파일 구조)
│
│   내부 클래스:
│   ├── [신경망 구현]
│   │   ├── Perceptron         # 퍼셉트론 클래스
│   │   └── MLP                # 다층 퍼셉트론 클래스
│   │
│   ├── [활성화 함수]
│   │   ├── sigmoid(x)
│   │   ├── sigmoid_derivative(x)
│   │   ├── tanh(x)
│   │   ├── relu(x)
│   │   └── leaky_relu(x, alpha)
│   │
│   ├── [GUI 탭 위젯]
│   │   ├── PerceptronTab       # 탭 1
│   │   ├── ActivationFunctionsTab  # 탭 2
│   │   ├── ForwardPropTab      # 탭 3
│   │   ├── MLPTab              # 탭 4
│   │   └── UniversalApproxTab  # 탭 5
│   │
│   └── MainWindow             # 메인 윈도우 (QTabWidget)
│
├── PRD.md
└── TRD.md
```

### 2.2 UI 구조 패턴

```
MainWindow (QMainWindow)
└── QTabWidget
    ├── Tab1: PerceptronTab (QWidget)
    │   ├── QHBoxLayout
    │   │   ├── control_panel (QGroupBox) — 좌측 250px
    │   │   │   ├── QComboBox (게이트 선택)
    │   │   │   ├── QDoubleSpinBox (학습률)
    │   │   │   ├── QSpinBox (에포크)
    │   │   │   ├── QPushButton (학습 시작)
    │   │   │   └── QTextEdit (결과 출력)
    │   │   └── FigureCanvas — 우측 (Matplotlib)
    ├── Tab2: ActivationFunctionsTab
    ├── Tab3: ForwardPropTab
    ├── Tab4: MLPTab
    └── Tab5: UniversalApproxTab
```

---

## 3. 알고리즘 명세 (Algorithm Specification)

### 3.1 퍼셉트론 (Perceptron)

```
클래스: Perceptron
초기화:
  - weights = randn(input_size) × 0.1
  - bias    = randn() × 0.1
  - lr      = learning_rate

활성화 함수:
  activation(x) = 1  if x >= 0
                  0  otherwise  (계단 함수)

예측:
  predict(inputs):
    summation = dot(inputs, weights) + bias
    return activation(summation)

학습 (퍼셉트론 학습 규칙):
  for each (input, label):
    error = label - predict(input)
    weights += lr × error × input
    bias    += lr × error
```

### 3.2 다층 퍼셉트론 (MLP)

```
클래스: MLP (2층 구조)
파라미터:
  - W1: (input_size, hidden_size)  Xavier 초기화
  - b1: (1, hidden_size)           영벡터
  - W2: (hidden_size, output_size) Xavier 초기화
  - b2: (1, output_size)           영벡터

Xavier 초기화:
  W = randn(n_in, n_out) × sqrt(2 / n_in)

순전파 (Forward Propagation):
  z1 = X @ W1 + b1
  a1 = sigmoid(z1)
  z2 = a1 @ W2 + b2
  a2 = sigmoid(z2)
  return a2

손실 함수 (MSE):
  L = mean((a2 - y)²)

역전파 (Backpropagation):
  m = 배치 크기

  출력층:
    dz2 = a2 - y
    dW2 = (1/m) × a1ᵀ @ dz2
    db2 = (1/m) × sum(dz2)

  은닉층 (Chain Rule):
    da1 = dz2 @ W2ᵀ
    dz1 = da1 ⊙ sigmoid'(z1)
    dW1 = (1/m) × Xᵀ @ dz1
    db1 = (1/m) × sum(dz1)

  가중치 업데이트:
    W -= lr × dW
    b -= lr × db
```

### 3.3 만능 근사 네트워크 (Universal Approximation)

```
구조: 단일 은닉층 네트워크 (1→N→1)
활성화: Sigmoid (은닉층), 선형 (출력층)

순전파:
  z1 = X_train @ W1 + b1      # (200, N)
  a1 = sigmoid(z1)
  y_pred = a1 @ W2 + b2       # 선형 출력

역전파:
  dz2 = (y_pred - y_train) / m
  dW2 = a1ᵀ @ dz2
  da1 = dz2 @ W2ᵀ
  dz1 = da1 × sigmoid'(z1)
  dW1 = X_trainᵀ @ dz1

비교 실험:
  뉴런 수: [3, 10, 50]
  공통 시드: np.random.seed(42)
```

---

## 4. Matplotlib + PySide6 통합 방법

```python
# 백엔드 설정 (import pyplot 전에 설정)
import matplotlib
matplotlib.use('QtAgg')

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# 캔버스를 QWidget에 삽입
self.figure = Figure(figsize=(10, 5))
self.canvas = FigureCanvas(self.figure)
layout.addWidget(self.canvas)

# 그래프 업데이트
self.figure.clear()
ax = self.figure.add_subplot(111)
ax.plot(...)
self.figure.tight_layout()
self.canvas.draw()
```

---

## 5. 한글 폰트 설정

```python
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt

def set_korean_font():
    font_list = [f.name for f in fm.fontManager.ttflist]
    for font in ['Malgun Gothic', 'Gulim', 'Batang', 'Dotum', 'NanumGothic', 'AppleGothic']:
        if font in font_list:
            plt.rcParams['font.family'] = font
            break
    plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지
```

---

## 6. 의존성 설정 (pyproject.toml)

```toml
[project]
dependencies = [
    "pyside6>=6.6.0",      # 추가 필요
    "numpy>=1.26.0",
    "matplotlib>=3.8.0",
    ...
]
```

설치 명령:
```bash
uv add pyside6
```

실행 명령:
```bash
uv run week3/week3_neural_explorer.py
```

---

## 7. 파라미터 범위 및 기본값

| 탭 | 파라미터 | 기본값 | 범위 |
|----|---------|--------|------|
| 퍼셉트론 | 학습률 | 0.1 | 0.001 ~ 1.0 |
| 퍼셉트론 | 에포크 | 100 | 10 ~ 10,000 |
| MLP | 은닉 뉴런 | 4 | 2 ~ 20 |
| MLP | 학습률 | 0.5 | 0.01 ~ 2.0 |
| MLP | 에포크 | 10,000 | 100 ~ 50,000 |
| 만능근사 | 에포크 | 5,000 | 1,000 ~ 20,000 |
| 만능근사 | 학습률 | 0.01 | 0.001 ~ 0.1 |
| 순전파 | x1, x2 | 0.5, 0.8 | -2.0 ~ 2.0 |
| 순전파 | 은닉 크기 | 3 | 2 ~ 8 |

---

## 8. 시각화 명세

| 탭 | 플롯 | 타입 | 설명 |
|----|------|------|------|
| 퍼셉트론 | 결정 경계 | contourf | 배경 영역 채색 + 데이터 포인트 |
| 활성화 함수 | 함수 비교 | line | 선택된 함수들 오버레이 |
| 활성화 함수 | 미분 비교 | line | Gradient 비교 |
| 순전파 | 레이어 값 | bar | z1, a1 비교 |
| MLP | Loss 곡선 | line (log) | 학습 진행 |
| MLP | 결정 경계 | contourf (RdYlBu) | 확률 출력 시각화 |
| MLP | 은닉층 활성화 | imshow (viridis) | 히트맵 |
| 만능 근사 | 목표 vs 예측 | line | 3개 서브플롯 비교 |

---

## 9. 에러 처리

- 학습 발산 방지: sigmoid 입력 clip (-500, 500)
- Xavier 초기화로 gradient vanishing/exploding 방지
- 랜덤 시드 고정 (seed=42): 만능 근사 탭에서 재현성 보장

---

작성자: 이태영 (물리학과)
작성일: 2026-03-31
과목: AI와 인공지능 (부산대 물리학과)
