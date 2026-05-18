# 전산물리: AI와 물리학의 만남
## Computational Physics: From Neural Networks to Physics-Informed AI

| 항목 | 내용 |
|------|------|
| 개설 | 부산대학교 물리학과 |
| 대상 | 2학년 / 2026년 1학기 |
| 시간 | 주 3시간 (강의 2시간 + 실습 1시간) |
| 도구 | Python, Claude AI / Cursor, uv |
| 저장소 | [github.com/BogKim2/AIandMLcourse](https://github.com/BogKim2/AIandMLcourse) |

본 저장소는 강의 **실습 코드**, **주차별 가이드(`weekN.md`)**, **강의 계획·문서**를 담습니다. 이론·일정·평가의 전체 그림은 루트의 [`course.md`](course.md)를, 주제별 **디렉터리 구조와 학습 루트**는 [`directory.md`](directory.md)를 함께 보는 것을 권장합니다.

**더 긴 저장소 가이드(확장판)** 는 루트의 [`README.md`](README.md) 를 참고하십시오. 본 파일은 **핵심만 요약**한 단순 안내입니다.

---

## 목차

1. [빠른 시작](#빠른-시작)
2. [저장소 루트(파일·설정)](#저장소-루트파일설정)
3. [Part I: Week 1–7 — 딥러닝·LLM 기초](#part-i-week-17--딥러닝llm-기초)
4. [Part II: Week 9–12 — 전산 물리 시뮬레이션](#part-ii-week-912--전산-물리-시뮬레이션)
5. [Part III: Week 13–14 — PINN](#part-iii-week-1314--pinn)
6. [`plan/` — 강의계획서(LaTeX)](#plan--강의계획서latex)
7. [`docs/` — 문서·메타·PDCA 스냅샷](#docs--문서메타pdca-스냅샷)
8. [실행 팁·과제·문제해결](#실행-팁과제문제해결)
9. [참고 문헌(요약)](#참고-문헌요약)

---

## 빠른 시작

```powershell
# 저장소 clone 후 프로젝트 루트에서
cd aicoursework

# uv로 가상환경 + 의존성 (pyproject.toml, uv.lock 기준)
uv venv
uv sync

# 예: Week 1 실습
cd week1
uv run python 00_hello_world.py
uv run python 01_hello_nn.py
```

- **필수 Python**: `>= 3.12` (`.python-version` 참고)
- **의존성**: [`pyproject.toml`](pyproject.toml) — `tensorflow`, `numpy`, `matplotlib`, `scipy`, `seaborn` 등
- **코딩·플롯 공통 규칙**: [`.cursorrules`](.cursorrules) (한글 폰트, scatter 마커 규칙, 가중치 초기화 등)

---

## 저장소 루트(파일·설정)

| 경로 | 역할 |
|------|------|
| `pyproject.toml` / `uv.lock` | 패키지 선언 및 잠금 버전 |
| `.python-version` | 권장 Python 3.12 |
| `.gitignore` | `__pycache__`, `.venv`, `.claude/` 등 제외 |
| `course.md` | **강의 전체 개요** — 16주 일정, 교재, 학습목표(긴 문서) |
| `directory.md` | **주차별 폴더 구조**·난이도 표·체크리스트·실행 예시 |
| `README.md` | **확장판** 저장소 가이드(상세) |
| `README_simple.md` | 본 문서 — **요약** 안내 |
| `main.py` | 프로젝트 엔트리 샘플 (`Hello from aicoursework!`) |
| `AI&MLhw.xlsx` | 과제·출제 관련 스프레드시트(강의 운용용) |
| `.cursorrules` | Cursor/에이전트용 프로젝트 규칙 |
| `week1` … `week14` | 주차별 실습(아래 절에서 상세) |
| `plan/` | LaTeX 강의계획서 원본·PDF |
| `docs/` | 분석 문서, PDCA/에이전트 메타 JSON 등 |

> **Week 8** 은 이 저장소에 **별도 `week8/` 폴더가 없습니다.** [`course.md`](course.md) 기준으로 중간고사·LLM 코딩 입문 주간으로 안내됩니다.

---

## Part I: Week 1–7 — 딥러닝·LLM 기초

### `week1/` — 강의 소개 및 환경 설정

- **요지**: `uv`·Git·Cursor, 첫 TensorFlow NN, 수치적 피팅과 NN 비교.
- **핵심 스크립트**
  - `00_hello_world.py` — 환경·TensorFlow·플롯 점검
  - `01_hello_nn.py` — 선형 모델 SGD (예: `y = 2x - 1`)
  - `02_polynomial_fitting.py` — 다항식/과적합 직관
- **기타**
  - `week1.md`, `hw1.md` — 학습 가이드·과제 안내
  - `guides/` — 설치·환경 PDF 가이드, `generate_pdfs.py`로 생성·보강
  - `outputs/` — `training_loss.png`, `model_fit.png`, `02_numerical_fitting.png` 등

---

### `week2/` — 머신러닝 기초

- **스크립트**: `01_linear_regression_spring.py` (훅의 법칙), `02_unsupervised_clustering.py`, `03_data_preprocessing.py`, `04_gradient_descent_vis.py`
- **보충 `ex/`**: `01_spring_scipy.py`, `04_optimization_scipy.py` (SciPy 최적화)
- **문서**: `week2.md`
- **outputs/**: 실습·보충 예제 그래프 (`spring_fitting.png`, `ex_*.png` 등)

---

### `week3/` — 신경망 기초(퍼셉트론·MLP·근사)

- **스크립트**: `01_perceptron.py` ~ `05_universal_approximation.py`, 유틸 `check_fonts.py`
- **문서**: `week3.md`
- **outputs/**: 퍼셉트론, 활성화, 순전파, MLP 학습, UAT 시각화 PNG

---

### `week4/` — 물리 데이터로 학습

- **스크립트**: `01perfect1d.py` (1D 함수 근사), `02projectile.py` (포물선), `03overfitting.py`, `04pendulum.py` (진자·RK4 연계)
- **문서**: `week4.md`
- **outputs/**: 1D 근사, 극한 함수 테스트, 네트워크 크기 비교 등 (저장소에 일부 그래프 포함)

---

### `week5/` — 딥러닝 핵심 기법

- **스크립트**: `01_regularization.py`, `02_overfitting_underfitting.py`, `03_data_augmentation.py`, `04_transfer_learning.py`, `05_mnist_cnn.py`
- **문서**: `week5.md`
- **outputs/**: 정규화/과적합, 증강 예시, transfer 요약 `txt`, MNIST 결과 PNG

---

### `week6/` — Transformer·Attention

- **스크립트**: `01_attention_basics.py` ~ `05_sequence_modeling.py`
- **자동 실행**: `run.bat` (Windows에서 순차 실행)
- **문서**: `week6.md`
- **outputs/**: 어텐션 가중치, 멀티헤드, 위치인코딩, 트랜스포머 데이터플로우, 시퀀스 실험 등 (다수 PNG)

---

### `week7/` — LLM 개론(토큰·GPT/BERT·파인튜닝)

- **스크립트**: `01_tokens_and_embeddings.py`, `02_gpt_bert_architectures.py`, `03_pretraining_finetuning.py`, `04_claude_api_simple.py` (API 개념·시뮬)
- **자동 실행**: `run.bat`
- **문서**: `week7.md`
- **outputs/**: 토크나이징, 임베딩, 아키텍처 비교, 파이프라인, 프롬프트/물리응용 시각자료

---

## Part II: Week 9–12 — 전산 물리 시뮬레이션

### `week9/` — 고전 역학

- **스크립트**: `01euler_rk4.py`, `02planetary.py`, `03chaotic_pendulum.py`, `04lagrangian_hamiltonian.py`
- **확장 `ex/`**: `01three_body.py`, `02llm_optimization.py`; `ex/outputs/`에 궤적·보존 분석 그래프, `README.md`
- **문서**: `week9.md`
- **outputs/**: 수치해법 오차, 케플러/행성, 이중진자·상공간, 라그랑지/해밀토니 비교

---

### `week10/` — 전자기학(정전기·Maxwell·FDTD류 시연)

- **스크립트**: `01_electric_field_basics.py` ~ `10_conductor_potential.py`, `09_em_wave_animation.py` (GIF 생성)
- **자동 실행**: `run.bat`
- **문서**: `week10.md`
- **outputs/**: 전하·전위·전기력선, 자기장, 로렌츠, 1D/2D 파동, 복수 전하, `09_em_wave.gif` 등

---

### `week11/` — 양자역학(우물·터널링·수소)

- **메인**: `01schrodinger.py`, `02wavefunction.py`, `03tunneling.py`, `04wells_oscillator.py`
- **`coursematerial/`**: 수소·해밀토니 행렬 시각화 `01visualizehamiltonian.py`, `02hydrogenatom.py`, `03hydrogenatom_spherical.py` 및 분석용 PNG/TXT
- **문서**: `week11.md`
- **outputs/**: 무한/유한 우물, 조화진동자, 파동·터널·비교 Plots (다수)

---

### `week12/` — 통계물리·Monte Carlo·Ising

- **스크립트**: `01_random_walk.py` ~ `08_ising_2d_advanced.py`
- **문서**: `week12.md`
- **참고**: 이 주차는 저장소에 **미리 생성된 `outputs/`가 없을 수 있으며**, 실행 시 Plots·데이터가 생깁니다.

---

## Part III: Week 13–14 — PINN

### `week13/` — PINN으로 ODE

- **스크립트**: `01_simple_ode.py`, `02_harmonic_oscillator.py`, `03_damped_oscillator.py`, `04_boundary_value_problem.py`, `05_lorenz_system.py`, `06_comparison_frameworks.py` (TensorFlow + RK4/`odeint` 비교; `06` 은 `psutil` 이 필요할 수 있음)
- **문서**: `week13.md`, `README.md` (주차 내부 실행 안내)
- **배치**: `run_all.bat`, `run_all_auto.bat`
- **outputs/**: PINN 해·잔차, 감쇠·로렌츠 3D 등

---

### `week14/` — PINN으로 PDE

- **스크립트**: `01_basic_pinn.py` ~ `07_complex_boundary.py` (열·파동·Burgers·복잡 경계)
- **일괄 실행**: `run_all.py`, `run_all.ps1`, 설명 `RUN_ALL.md`
- **문서**: `week14.md`
- **outputs/**: 1D/2D 열·파동, Burgers, 경계 PNG

---

## `plan/` — 강의계획서(LaTeX)

| 종류 | 설명 |
|------|------|
| `plan.tex` / `plan.pdf` | 국문(또는 기본) 강의계획 PDF |
| `plan_eng.tex` / `plan_eng.pdf` | 영문版 소스·PDF |
| `*.aux`, `*.log`, `*.out`, `*.fls`, `*.fdb_latexmk`, `*.synctex.gz` | LaTeX·latexmk **부산물**(편집·동기화 시 재생성됨) |

수정은 `.tex`만 편집한 뒤 로컬에서 `latexmk` 또는 사용 중인 TeX 환경으로 PDF를 빌드하면 됩니다.

---

## `docs/` — 문서·메타·PDCA 스냅샷

| 경로 | 설명 |
|------|------|
| `03-analysis/aicoursework-docs.analysis.md` | 문서/프로젝트 **분석 메모** |
| `.bkit-memory.json` | bkit/에이전트 관련 로컬 메타(도구마다 갱신) |
| `.pdca-status.json` | PDCA 진행 상태 |
| `.pdca-snapshots/snapshot-*.json` | PDCA 스냅샷(타임스탬프별) |

> 팀 협업이나 **제출용으로 깔끔한 리포지토리**만 남기고 싶다면, 위 메타·스냅샷을 `.gitignore`에 넣는 것을 검토할 수 있습니다(현재는 예제·이력용으로 포함될 수 있음).

---

## 실행 팁·과제·문제해결

- **한 주를 통째로 돌릴 때**: `week6`, `week7`, `week10`은 `run.bat`; `week14`는 `uv run python run_all.py` 또는 `run_all.ps1`.
- **GPU**: TensorFlow `tf.config.list_physical_devices('GPU')` 등으로 확인.
- **한글 그래프 깨짐**: Windows `Malgun Gothic`, macOS `AppleGothic` 등 — `.cursorrules`의 폰트 설정과 `week3/check_fonts.py` 참고.
- **의존성 누락**: `uv sync` 후에도 모듈 오류가 나면 `uv pip install …`로 개별 설치. 상세·예외는 [`README.md`](README.md) 환경 절.

**과제·평가·16주 전체 일정**은 [`course.md`](course.md)의 표와 본문을 기준으로 합니다.

---

## 주차·폴더·대표 스크립트 대응표

| 주차 | 폴더 | 주제(한 줄) | 대표·핵심 파일 |
|------|------|-------------|----------------|
| 1 | `week1/` | 환경·첫 NN | `00_hello_world.py`, `01_hello_nn.py`, `02_polynomial_fitting.py`, `guides/*.pdf` |
| 2 | `week2/` | ML기초·군집 | `01_linear_regression_spring.py` ~ `04_gradient_descent_vis.py`, `ex/01_spring_scipy.py` |
| 3 | `week3/` | 퍼셉트론·MLP | `01_perceptron.py` ~ `05_universal_approximation.py` |
| 4 | `week4/` | 물리 데이터 NN | `01perfect1d.py`, `02projectile.py`, `03overfitting.py`, `04pendulum.py` |
| 5 | `week5/` | 정규화·CNN | `01_regularization.py` ~ `05_mnist_cnn.py` |
| 6 | `week6/` | Attention·Transformer | `01_attention_basics.py` ~ `05_sequence_modeling.py`, `run.bat` |
| 7 | `week7/` | LLM 개론 | `01_tokens_and_embeddings.py` ~ `04_claude_api_simple.py`, `run.bat` |
| 8 | *(폴더 없음)* | 중간·LLM 입문 | [`course.md`](course.md) 일정만 참고 |
| 9 | `week9/` | 고전역학 ODE | `01euler_rk4.py` ~ `04lagrangian_hamiltonian.py`, `ex/01three_body.py` |
| 10 | `week10/` | 전자기·파동 | `01_electric_field_basics.py` ~ `10_conductor_potential.py`, `run.bat` |
| 11 | `week11/` | 양자(우물·수소) | `01schrodinger.py` ~ `04wells_oscillator.py`, `coursematerial/*.py` |
| 12 | `week12/` | MC·Ising | `01_random_walk.py` ~ `08_ising_2d_advanced.py` |
| 13 | `week13/` | PINN·ODE | `01_simple_ode.py` ~ `06_comparison_frameworks.py`, `run_all*.bat` |
| 14 | `week14/` | PINN·PDE | `01_basic_pinn.py` ~ `07_complex_boundary.py`, `run_all.py` |

---

## 참고 문헌(요약)

- **MIT 6.S191** — Introduction to Deep Learning
- **Vaswani et al. (2017)** — *Attention Is All You Need*
- **Raissi et al. (2019)** — Physics-Informed Neural Networks
- **교재**: Goodfellow et al. *Deep Learning*; Newman *Computational Physics*

더 긴 인용·링크·체크리스트는 [`directory.md`](directory.md) 후반을 참고하십시오.

---

*README_simple: 저장소 요약 가이드. **파일 단위·환경·평가·FAQ 상세**는 [`README.md`](README.md) 를 보십시오. 강의 정책·일정·평가 비율의 정본은 `course.md`입니다.*
