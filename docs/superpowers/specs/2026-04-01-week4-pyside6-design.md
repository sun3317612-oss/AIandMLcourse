# Week 4 PySide6 Neural Explorer — Design Spec

**Date:** 2026-04-01
**File:** `week4/week4_explorer.py`
**Status:** Approved

---

## 1. 목적

TensorFlow/Keras 기반의 4개 Week 4 Lab을 하나의 PySide6 데스크탑 앱으로 통합한다.
사용자가 파라미터를 직접 조절하고 학습을 실행하며, 실시간으로 loss 곡선과 예측 결과를 확인할 수 있는 **인터랙티브 트레이닝 GUI + 파라미터 실험 도구**다.

---

## 2. 범위

- **포함**: 4개 Lab 모두 (1D 근사, 포물선, 과적합, 진자)
- **제외**: 기존 `01~04.py` 스크립트 수정 없음 (독립 유지)
- **신규 파일**: `week4/week4_explorer.py` 단일 파일

---

## 3. 아키텍처

### 3.1 클래스 구조

```
MainWindow (QMainWindow)
├── QTabWidget
│   ├── Lab1Widget  — 1D 함수 근사
│   ├── Lab2Widget  — 포물선 운동
│   ├── Lab3Widget  — 과적합 데모
│   └── Lab4Widget  — 진자 주기
│
└── 공통 LabWidget 레이아웃
    ├── 왼쪽 ParamPanel (QWidget, 고정 너비 260px)
    │   ├── 파라미터 입력 위젯들
    │   └── 실행/중지 버튼 (QPushButton)
    └── 오른쪽 ChartPanel (QWidget)
        ├── Loss 캔버스 (FigureCanvasQTAgg)
        └── 예측 캔버스 (FigureCanvasQTAgg)
```

### 3.2 스레딩 모델

TensorFlow 학습은 메인 스레드를 블로킹하므로 `QThread`로 분리한다.

```
[메인 스레드 — UI]              [TrainingThread — QThread]
  [실행] 클릭
      │
      └──→ TrainingThread(params).start()
                    │
                    └──→ model.fit(
                             callbacks=[LambdaCallback(
                               on_epoch_end=lambda e,l: self.epoch_done.emit(e, l)
                             )]
                         )
                                   │  PyQt Signal (thread-safe)
      ←────────────────────────────┘
      │  epoch_done(epoch, logs)
      └──→ 차트 업데이트 (matplotlib axes.cla() + redraw)
```

**중지 버튼:** `TrainingThread._stop = True` 플래그 세팅 →
Keras `on_epoch_end` 콜백에서 `model.stop_training = True` 로 조기 종료.

### 3.3 Signal 정의 (TrainingThread)

```python
epoch_done  = pyqtSignal(int, dict)   # (epoch, logs)
train_done  = pyqtSignal(object)      # trained keras model
train_error = pyqtSignal(str)         # error message
```

---

## 4. 네비게이션

상단 `QTabWidget` 탭 4개:

| 탭 인덱스 | 이름 | 아이콘 |
|-----------|------|--------|
| 0 | 1D 함수 근사 | 📐 |
| 1 | 포물선 운동  | 🎯 |
| 2 | 과적합 데모  | 📊 |
| 3 | 진자 주기    | 🔬 |

탭 전환은 학습 중에도 허용. 각 Lab은 독립 `TrainingThread`를 가진다.

---

## 5. Lab별 파라미터 & 차트

### Lab 1 — 1D 함수 근사

**파라미터:**
- 함수 선택: `sin(x)` / `cos(x)+0.5sin(2x)` / `x·sin(x)` (QComboBox)
- Hidden Layers: 텍스트 입력 `[128, 128, 64]` (QLineEdit)
- Epochs: 슬라이더 100 ~ 5000, 기본값 3000
- Learning Rate: `0.01` / `0.001` / `0.0001` (QComboBox)
- Activation: `tanh` / `relu` (QComboBox)

**차트:**
- 상단: Train/Val Loss 곡선 (실시간, epoch마다 업데이트)
- 하단: 실제값(녹색) vs 예측값(주황 점선) 오버레이 (학습 완료 후 렌더)

---

### Lab 2 — 포물선 운동

**파라미터:**
- Hidden Layers (QLineEdit), Epochs 슬라이더, LR (QComboBox)
- 테스트 초기속력 v₀: 슬라이더 10 ~ 50 m/s, 기본값 30
- 테스트 발사각 θ: 슬라이더 10° ~ 80°, 기본값 45°

**차트:**
- 상단: Train/Val Loss 곡선 (실시간)
- 하단: 예측 궤적 vs 물리 공식 궤적 (x-y 평면, 학습 완료 후)

---

### Lab 3 — 과적합 데모

**파라미터:**
- Epochs: 슬라이더 50 ~ 500, 기본값 200
- 노이즈 수준: 슬라이더 0.1 ~ 1.0, 기본값 0.3

**동작:** [모두 실행] 버튼을 누르면 Underfit → Good Fit → Overfit 순서로 3개 모델을 순차 학습.
각 모델 학습 중 현재 모델명을 상태바에 표시.

**차트:**
- 상단: 현재 학습 중인 모델의 Train/Val Loss 곡선 (실시간, 모델 전환 시 초기화)
- 하단: 3개 모델 예측 곡선 비교 (모두 완료 후 렌더 — Underfit 빨강, Good 초록, Overfit 파랑)

---

### Lab 4 — 진자 주기

**파라미터:**
- Hidden Layers (QLineEdit), Epochs 슬라이더, LR (QComboBox)
- 테스트 길이 L: `0.5m` / `1.0m` / `2.0m` (QComboBox)
- 각도 범위: 슬라이더 최대 θ₀ 5° ~ 80°, 기본값 80°

**차트:**
- 상단: Train/Val Loss 곡선 (실시간)
- 하단: 주기 T vs 각도 θ₀ — 예측값 vs 이론값(타원적분) 비교 (학습 완료 후)

---

## 6. 데이터 흐름

```
1. 사용자가 파라미터 입력
2. [실행] 클릭
3. 입력값 검증
   - Hidden Layers: 정규식 파싱 → 실패 시 빨간 테두리 + 상태바 오류
   - Epochs/LR: 범위 자동 클램프
4. TrainingThread 생성 → .start()
5. [실행] 버튼 → [중지]로 변경, 파라미터 입력 비활성화
6. 매 epoch: epoch_done 시그널 → Loss 차트 실시간 업데이트
7. 학습 완료: train_done 시그널 → 예측 차트 렌더링
8. [중지] → [실행]으로 복구, 입력 재활성화
```

---

## 7. 엣지 케이스 처리

| 상황 | 처리 방법 |
|------|-----------|
| 잘못된 레이어 입력 (`[abc]`, 빈 값) | QLineEdit 빨간 테두리 + 상태바 오류 메시지, 학습 미시작 |
| 학습 중 탭 전환 | 허용 — 각 Lab은 독립 TrainingThread |
| 학습 중 앱 종료 | `closeEvent`에서 모든 Thread에 stop flag 세팅 후 `.wait()` |
| 학습 중 [실행] 재클릭 방지 | 학습 중 버튼 비활성화 |
| TF 학습 중 예외 발생 | `train_error` 시그널 → 상태바에 오류 표시 |

---

## 8. 의존성

기존 `pyproject.toml`에 추가 필요:

```toml
dependencies = [
  "tensorflow",
  "numpy",
  "matplotlib",
  "PySide6",       # 신규 추가
]
```

---

## 9. 파일 구조 (변경 후)

```
week4/
├── week4_explorer.py      ← 신규 (이 스펙의 구현 대상)
├── 01perfect1d.py         ← 기존 유지
├── 02projectile.py        ← 기존 유지
├── 03overfitting.py       ← 기존 유지
├── 04pendulum.py          ← 기존 유지
└── outputs/               ← 기존 유지
```

---

## 10. 성공 기준

- 4개 Lab 모두 앱 내에서 학습 실행 가능
- 학습 중 Loss 곡선이 실시간 업데이트됨 (매 epoch마다)
- 학습 완료 후 예측 차트가 정확히 렌더링됨
- 학습 중 UI가 멈추지 않음 (메인 스레드 블로킹 없음)
- 잘못된 입력에 대한 명확한 오류 표시
