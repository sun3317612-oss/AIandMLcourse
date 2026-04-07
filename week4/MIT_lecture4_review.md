# MIT 6.S191 Lecture 4 — Deep Generative Modeling 요약

> 강의: A. Amini, A.P. Amini — MIT 6.S191 Introduction to Deep Learning  
> 파일: `6S191_MIT_DeepLearning_L4.pdf`

---

## 1. 생성 모델이란?

지금까지의 모델: **입력 X → 레이블 Y** (분류, 회귀)  
생성 모델의 목표: **데이터의 분포 P(X) 자체를 학습**해서 새로운 샘플을 생성

> "데이터를 외우는 게 아니라, 데이터가 어디서 왔는지를 이해한다"

---

## 2. Latent Variable (잠재 변수)

데이터 X 뒤에 숨어있는 저차원 표현 **z**를 가정

```
z (잠재 공간, 저차원) ──→ Decoder ──→ X (고차원 데이터)
```

핵심 아이디어: 복잡한 데이터도 저차원 잠재 공간에서 구조화된 표현을 가진다.

---

## 3. Autoencoder (오토인코더)

```
X ──→ Encoder ──→ z (병목) ──→ Decoder ──→ X̂
         손실: ||X - X̂||²  (재구성 오차)
```

- 압축 + 복원을 동시에 학습
- 문제: z 공간이 불연속적 → 새로운 샘플 생성 불가

---

## 4. Variational Autoencoder (VAE)

오토인코더의 한계를 확률적 방식으로 극복

### 구조

```
X ──→ Encoder ──→ [μ, σ²]
                      ↓ (reparameterization: z = μ + σ·ε, ε~N(0,1))
                      z ──→ Decoder ──→ X̂
```

### 손실 함수 (ELBO)

```
L = E[log p(X|z)] - D_KL(q(z|X) || p(z))
    └── 재구성 손실 ──┘   └──── 정규화 항 ────┘
```

| 항 | 역할 |
|----|------|
| 재구성 손실 | X를 잘 복원하도록 |
| KL Divergence | z가 N(0,1)에서 멀어지지 않도록 |

### Reparameterization Trick

`z`를 직접 샘플링하면 역전파 불가 → `z = μ + σ·ε` 으로 변환  
ε만 랜덤, μ·σ는 학습 가능 → 그래디언트가 흐른다

### 결과

- z 공간이 연속적이고 정규분포 형태
- 임의의 z를 샘플링해서 새로운 데이터 생성 가능
- 잠재 공간에서 보간(interpolation) 가능

---

## 5. Generative Adversarial Network (GAN)

**두 네트워크의 경쟁으로 생성 품질을 높인다**

```
z (랜덤 노이즈) ──→ Generator G ──→ Fake X
                                         ↓
Real X ─────────────────────────→ Discriminator D ──→ Real/Fake 판별
```

### 목적 함수 (Minimax)

```
min_G max_D  E[log D(x)] + E[log(1 - D(G(z)))]
```

- **D 입장**: 진짜는 1, 가짜는 0으로 판별 → 최대화
- **G 입장**: D를 속여서 1을 받아야 함 → 최소화

### 학습 과정

```
Step 1: D 학습  — 진짜/가짜 구분 능력 향상
Step 2: G 학습  — D를 속이는 이미지 생성
         (반복)
```

### 주요 문제점

| 문제 | 설명 |
|------|------|
| Mode Collapse | G가 몇 가지 패턴만 반복 생성 |
| 학습 불안정 | G와 D 균형 맞추기 어려움 |
| 평가 어려움 | Loss가 낮다고 좋은 이미지가 아님 |

---

## 6. VAE vs GAN 비교

| | VAE | GAN |
|--|-----|-----|
| 학습 방식 | 확률적 최대우도 (ELBO) | 적대적 학습 |
| 생성 품질 | 다소 흐림 (blurry) | 선명하고 현실적 |
| 잠재 공간 | 구조화됨, 해석 가능 | 덜 구조화됨 |
| 학습 안정성 | 안정적 | 불안정할 수 있음 |
| 다양성 | 높음 | Mode Collapse 위험 |

---

## 7. 응용 사례

- 얼굴 생성 / 이미지 합성 (StyleGAN)
- 데이터 증강 (Data Augmentation)
- 이미지-이미지 변환 (pix2pix, CycleGAN)
- 신약 후보 분자 구조 생성
- 음성 합성

---

> **핵심 한 줄 요약:**  
> VAE는 *확률 분포*로 잠재 공간을 정규화해 안정적 생성을 하고,  
> GAN은 *경쟁 구조*로 현실적인 샘플을 만들어낸다.
