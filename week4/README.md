# Week 4 Neural Explorer

PySide6 기반 인터랙티브 Neural Network 트레이닝 GUI.  
4개의 물리 Lab을 하나의 데스크탑 앱에서 실행하고, 파라미터를 바꿔가며 학습 결과를 실시간으로 확인할 수 있다.

---

## 개발 방식

### Claude Code + Superpowers 워크플로우

이 프로그램은 Claude Code CLI와 **Superpowers 스킬 시스템**을 사용해 구현했다.

```
brainstorming → design spec → writing-plans → 9개 Task 순차 구현
```

1. **`superpowers:brainstorming`** — 요구사항 탐색
   - "4개 Lab을 GUI로 통합한다"는 아이디어에서 시작
   - 인터랙티브 트레이닝 + 실시간 차트 + 파라미터 실험 도구로 구체화

2. **`superpowers:writing-plans`** — 설계 명세 작성
   - 설계 문서: `docs/superpowers/specs/2026-04-01-week4-pyside6-design.md`
   - 구현 계획: `docs/superpowers/plans/2026-04-01-week4-pyside6-explorer.md`
   - 클래스 구조, 스레딩 모델, 각 Lab 데이터 형식을 문서화 후 승인

3. **`superpowers:subagent-driven-development`** — Task 단위 구현
   - Task 1~9로 분할하여 순차 구현 및 커밋

| Task | 내용 |
|------|------|
| 1 | `parse_layer_string` 유틸 + 스켈레톤 |
| 2 | `TrainingThread` 베이스 클래스 |
| 3 | UI 헬퍼 + Lab1 데이터/모델/Thread |
| 4 | `Lab1Widget` UI |
| 5 | Lab2 포물선 운동 |
| 6 | Lab3 과적합 데모 |
| 7 | Lab4 진자 주기 |
| 8 | `MainWindow` + `main()` |
| 9 | 전체 테스트 14개 통과 + 최종 커밋 |

---

### PySide6 아키텍처

#### 클래스 구조

```
MainWindow (QMainWindow)
└── QTabWidget
    ├── Lab1Widget  — 1D 함수 근사
    ├── Lab2Widget  — 포물선 운동
    ├── Lab3Widget  — 과적합 데모
    └── Lab4Widget  — 진자 주기

각 LabWidget
├── 왼쪽: ParamPanel (고정 너비 260px)
│   ├── 파라미터 입력 위젯
│   └── 실행 / 중지 버튼
└── 오른쪽: ChartPanel
    ├── Loss 캔버스 (FigureCanvasQTAgg)
    └── 예측 캔버스 (FigureCanvasQTAgg)
```

#### QThread 분리 — TensorFlow 블로킹 문제 해결

TensorFlow `model.fit()`은 완료될 때까지 스레드를 점유한다.  
메인 스레드에서 직접 실행하면 UI가 멈추기 때문에 `QThread`로 분리했다.

```
[메인 스레드 — UI]          [TrainingThread — QThread]
  [실행] 클릭
      └──→ thread.start()
                └──→ model.fit(callbacks=[LambdaCallback])
                              │  epoch 완료마다
                              └──→ epoch_done.emit(epoch, logs)
                                         │
                              ←──────────┘ (Qt Signal, thread-safe)
              _on_epoch()  ←──  실시간 Loss 차트 갱신
              _on_done()   ←──  학습 완료 후 예측 차트 표시
```

**Signal 종류:**
- `epoch_done(int, dict)` — 매 epoch마다 loss 값 전달
- `train_done(object)` — 학습된 Keras 모델 전달
- `train_error(str)` — 오류 메시지 전달

---

## 파일 구조

```
week4/
├── README.md               # 이 파일
├── week4_explorer.py       # PySide6 GUI 앱 (단일 파일)
├── test_week4_explorer.py  # 유닛 테스트 (14개)
├── week4.md                # 개별 스크립트(01~04) 상세 설명
├── 01perfect1d.py          # Lab1 단독 실행
├── 02projectile.py         # Lab2 단독 실행
├── 03overfitting.py        # Lab3 단독 실행
├── 04pendulum.py           # Lab4 단독 실행
└── outputs/                # 단독 스크립트 실행 시 그래프 저장
```

---

## 실행 방법

### 환경 설정

```bash
# 프로젝트 루트에서
uv sync
```

### 실행

```bash
uv run week4/week4_explorer.py
```

또는 가상환경을 직접 활성화한 경우:

```bash
.venv\Scripts\activate      # Windows
python week4/week4_explorer.py
```

### 의존성

| 패키지 | 용도 |
|--------|------|
| `pyside6` | GUI 프레임워크 |
| `tensorflow` | Keras 모델 학습 |
| `matplotlib` | 차트 (QtAgg 백엔드) |
| `numpy` | 데이터 생성 |

---

## Lab별 사용 방법

### Lab 1 — 1D 함수 근사

Neural Network가 수학 함수를 얼마나 잘 근사하는지 실험한다.

| 파라미터 | 설명 | 기본값 |
|---------|------|--------|
| 함수 선택 | `sin(x)` / `cos(x)+0.5sin(2x)` / `x·sin(x)` | `sin(x)` |
| Hidden Layers | `[뉴런수, ...]` 형식으로 입력 | `[128, 128, 64]` |
| Epochs | 슬라이더 (100 ~ 5000) | 3000 |
| Learning Rate | 0.01 / 0.001 / 0.0001 | 0.01 |
| Activation | `tanh` / `relu` | `tanh` |

학습이 끝나면 오른쪽에 **Loss 곡선**과 **실제값 vs 예측값** 비교 그래프가 표시된다.

> **팁:** 복잡한 함수일수록 레이어를 깊게, `tanh`가 `relu`보다 함수 근사에 유리한 경우가 많다.

---

### Lab 2 — 포물선 운동

포물선 운동 궤적 `(v₀, θ, t) → (x, y)`를 학습한다.  
학습 후 원하는 초기 조건으로 궤적을 예측해 물리 공식과 비교한다.

| 파라미터 | 설명 | 기본값 |
|---------|------|--------|
| Hidden Layers | `[뉴런수, ...]` 형식 | `[128, 64, 32]` |
| Epochs | 슬라이더 (100 ~ 2000) | 500 |
| Learning Rate | 0.01 / 0.001 / 0.0001 | 0.001 |
| 테스트 v₀ (m/s) | 슬라이더 (10 ~ 50) | 30 |
| 테스트 θ (°) | 슬라이더 (10 ~ 80) | 45 |

학습 완료 후 설정한 `v₀`와 `θ`로 예측 궤적과 물리 공식 궤적을 겹쳐 표시한다.

> **팁:** θ=45°가 최대 사거리. 예측값이 물리 공식에 얼마나 근접하는지 확인해보자.

---

### Lab 3 — 과적합 데모

3가지 모델(과소적합 / 적절 / 과적합)을 동시에 학습해 비교한다.  
파라미터 조작 없이 **실행** 버튼만 누르면 된다.

| 모델 | 구조 | 특징 |
|------|------|------|
| Underfit | `[4]` | 너무 단순 — 패턴 미학습 |
| Good Fit | `[32, 16]` + Dropout | 일반화 성능 최적 |
| Overfit | `[256, 128, 64, 32]` | 노이즈까지 학습 |

| 파라미터 | 설명 | 기본값 |
|---------|------|--------|
| Epochs | 슬라이더 (50 ~ 500) | 200 |
| Noise Level | 슬라이더 (0.0 ~ 1.0) | 0.3 |

오른쪽 차트에서 **Train Loss vs Val Loss 분기** 시점이 과적합의 핵심 지표다.

---

### Lab 4 — 진자 주기

진자 길이 `L`과 초기 각도 `θ₀`로 주기 `T`를 예측한다.  
작은 각도 근사(`T = 2π√(L/g)`)와 큰 각도에서의 비선형성을 학습한다.

| 파라미터 | 설명 | 기본값 |
|---------|------|--------|
| Hidden Layers | `[뉴런수, ...]` 형식 | `[64, 32, 16]` |
| Epochs | 슬라이더 (100 ~ 3000) | 1000 |
| Learning Rate | 0.01 / 0.001 / 0.0001 | 0.001 |
| 테스트 길이 L (m) | 슬라이더 (0.1 ~ 3.0) | 1.0 |

학습 완료 후 여러 길이에 대한 **주기 vs 각도** 그래프가 표시된다.

> **팁:** L을 4배 늘리면 T가 약 2배 증가한다 (제곱근 관계). Neural Network가 이 비선형 관계를 학습하는지 확인해보자.

---

## 공통 조작

- **실행** — 학습 시작. 학습 중에는 파라미터 입력이 잠긴다.
- **중지** — 학습 중 언제든 중단 가능. 현재 epoch까지의 결과가 차트에 남는다.
- **하단 상태바** — 현재 epoch와 loss 값이 실시간으로 표시된다.
- **탭 전환** — 학습 중에도 다른 Lab 탭으로 이동할 수 있다.

Hidden Layers 입력 형식: `[128, 64, 32]` (대괄호 + 쉼표 구분, 양의 정수만)
