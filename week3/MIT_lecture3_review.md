# MIT 6.S191 Lecture 3 시청 보고서

## 기본 정보

| 항목 | 내용 |
|------|------|
| 이름 | 이태영 |
| 학번 | 202312150 |
| 강의명 | AI와 인공지능 |
| GitHub | [sun3317612-oss](https://github.com/sun3317612-oss) |
| 참고 강의 | MIT 6.S191 Introduction to Deep Learning - Lecture 3: Deep Computer Vision |
| 강의 일자 | January 6, 2026 (Alexander Amini) |

---

## 1. 강의 개요

이번 강의의 핵심 질문은 한 문장으로 요약된다.

> *"To know what is where by looking."*

컴퓨터 비전(Computer Vision)이란, 이미지로부터 세상에 무엇이 있는지, 어디에 있는지, 어떤 일이 일어나고 있는지를 발견하고 예측하는 분야다. 강의는 컴퓨터가 이미지를 어떻게 인식하는지부터 시작해, CNN(Convolutional Neural Network)의 작동 원리, 그리고 실제 응용까지 체계적으로 다룬다.

---

## 2. 컴퓨터 비전의 영향 (Rise and Impact)

컴퓨터 비전은 이미 다양한 분야에서 실질적인 영향을 미치고 있다.

| 분야 | 예시 |
|------|------|
| 로보틱스 | Boston Dynamics의 보행 로봇 |
| 자율주행 | 카메라 기반 차선 인식 및 내비게이션 |
| 의학·생물학 | 유방암, COVID-19 X-ray, 피부암 이미지 진단 |
| 접근성 | 시각 장애인을 위한 주변 환경 인식 |
| 모바일 컴퓨팅 | 스마트폰 사진 자동 태깅 |
| 얼굴 인식 | 얼굴 랜드마크 추출 및 3D 메시 구성 |

특히 의학 분야에서는 딥러닝 기반 모델이 유방암, 피부암 등의 진단에서 전문의 수준에 근접하는 성능을 보이고 있다는 점이 인상적이었다.

---

## 3. 컴퓨터가 이미지를 보는 방식 (What Computers "See")

### 이미지 = 숫자 행렬

컴퓨터에게 이미지는 단순히 **숫자로 구성된 행렬**이다.

- 각 픽셀은 0~255 사이의 정수값
- 흑백 이미지: `H × W` 행렬
- 컬러(RGB) 이미지: `H × W × 3` 텐서
  - 예: 1080p 컬러 이미지 → `1080 × 1080 × 3`

즉, 사람이 "얼굴"이라고 보는 것을 컴퓨터는 수백만 개의 숫자 배열로 인식한다.

### 컴퓨터 비전의 주요 태스크

| 태스크 | 출력 형태 | 예시 |
|--------|-----------|------|
| 분류 (Classification) | 클래스 레이블 (확률 벡터) | Lincoln: 0.8, Washington: 0.1, ... |
| 회귀 (Regression) | 연속적인 수치값 | 바운딩 박스 좌표 (x, y, w, h) |

---

## 4. 수동 특징 추출의 한계 (Manual Feature Extraction)

CNN이 등장하기 전, 전통적인 방법은 **수동 특징 추출(Manual Feature Extraction)**이었다.

**기존 방식의 흐름:**
```
도메인 지식 → 특징 정의 → 특징 검출하여 분류
```

예를 들어 얼굴 인식이라면 "코, 눈, 입"을, 차량 인식이라면 "바퀴, 번호판, 헤드라이트"를 사람이 직접 정의해야 했다.

**문제점:**

- **시점 변화(Viewpoint variation)**: 같은 물체도 각도에 따라 다르게 보임
- **크기 변화(Scale variation)**: 물체가 크거나 작을 수 있음
- **변형(Deformation)**: 물체의 형태가 달라질 수 있음
- **가림(Occlusion)**: 물체의 일부가 다른 물체에 가려질 수 있음
- **배경 잡음(Background clutter)**: 배경이 복잡할 경우 구분이 어려움
- **클래스 내 변화(Intra-class variation)**: 같은 클래스 내에서도 외형 차이가 큼

이처럼 수동 특징 추출은 **확장성이 없고 취약하다.** 이것이 데이터로부터 직접 특징을 학습하는 방법이 필요한 이유다.

---

## 5. 계층적 특징 학습 (Learning Feature Representations)

핵심 질문: **데이터에서 직접 특징의 계층 구조를 학습할 수 있을까?**

딥러닝은 이 질문에 "yes"라고 답한다. 계층이 깊어질수록 더 추상적인 특징이 학습된다.

| 계층 수준 | 학습되는 특징 |
|-----------|--------------|
| 저수준 (Conv Layer 1) | 엣지, 어두운 점, 방향선 |
| 중간 수준 (Conv Layer 2) | 눈, 귀, 코 등 부분 패턴 |
| 고수준 (Conv Layer 3) | 얼굴 구조, 전체 형태 |

이는 사람의 시각 피질이 계층적으로 정보를 처리하는 방식과 유사하다.

---

## 6. 완전 연결 신경망의 문제 (Fully Connected NN)

이미지에 일반적인 완전 연결(Fully Connected) 신경망을 그대로 적용하면 두 가지 문제가 생긴다.

1. **공간 정보 손실**: 2D 이미지를 1D 벡터로 펼치면 픽셀 간 위치 관계가 사라진다.
2. **파라미터 폭발**: 모든 픽셀이 모든 뉴런과 연결되므로 파라미터 수가 엄청나게 많아진다.

**해결 방향: 공간 구조 활용**

이미지의 공간적 구조를 유지하면서, **입력의 패치(patch)를 hidden layer의 뉴런과 연결**하는 방식이 필요하다. 즉, 각 뉴런이 이미지 전체가 아닌 **국소 영역(local region)**만 "본다."

---

## 7. 합성곱 연산 (The Convolution Operation)

### 원리

5×5 이미지에 3×3 필터를 적용하는 예:

1. 3×3 필터를 이미지 위에서 **슬라이딩**
2. 겹치는 영역과 필터의 원소별 곱(element-wise multiply)을 계산
3. 모두 더해 feature map의 한 값을 산출

```
이미지 (5×5)  ⊗  필터 (3×3)  =  Feature Map (3×3)
```

이 연산을 통해 이미지 전체에 걸쳐 **동일한 필터**가 공유 적용된다(파라미터 공유).

### 필터에 따른 결과 차이

다양한 필터를 적용하면 같은 이미지에서 서로 다른 특징이 추출된다.

| 필터 | 효과 |
|------|------|
| 샤프닝 필터 | 이미지 선명하게 |
| 엣지 검출 필터 | 경계선 강조 |
| 강한 엣지 검출 필터 | 더 강한 경계선 강조 |

딥러닝 CNN에서는 이 필터의 가중치를 **학습을 통해** 자동으로 찾는다.

---

## 8. 합성곱 레이어의 구조 (Convolutional Layer: Local Connectivity)

하나의 합성곱 뉴런 (p, q)의 계산:

$$\sum_{i=1}^{4}\sum_{j=1}^{4} w_{ij} \cdot x_{i+p,\, j+q} + b$$

이 연산은 세 단계로 구성된다:

1. **가중치 창(window) 적용**: 필터를 로컬 패치에 슬라이딩
2. **선형 결합 계산**: 가중합 계산
3. **비선형 활성화 함수 적용**: 비선형성 도입

```python
# TensorFlow
tf.keras.layers.Conv2D(filters, filter_size, activation='relu')

# PyTorch
torch.nn.Conv2d(in_channels, out_channels, kernel_size)
```

---

## 9. 비선형성 도입: ReLU

합성곱 연산 후에는 반드시 **비선형 활성화 함수**를 적용한다. 실세계 데이터는 선형이 아니기 때문이다.

가장 많이 쓰이는 함수는 **ReLU (Rectified Linear Unit)**:

$$g(z) = \max(0,\, z)$$

음수 값을 모두 0으로 만드는 **픽셀 단위(pixel-by-pixel)** 연산이다.

```python
tf.keras.layers.ReLU()
torch.nn.ReLU()
```

---

## 10. 풀링 (Pooling)

풀링은 특징 맵의 **해상도를 줄이면서 중요한 정보를 보존**하는 연산이다.

**Max Pooling 예시**: 4×4 입력에 2×2 필터, stride=2 적용 → 2×2 출력

- 각 2×2 영역에서 **최댓값**만 선택

**효과:**
1. **차원 축소 (Reduced dimensionality)**: 계산량 감소
2. **공간 불변성 (Spatial invariance)**: 특징의 정확한 위치보다 존재 여부에 집중

```python
tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=2)
torch.nn.MaxPool2d(kernel_size=(2,2), stride=2)
```

---

## 11. CNN 분류 아키텍처 전체 구조

CNN을 이용한 이미지 분류의 전체 파이프라인:

```
INPUT → [Conv + ReLU] → [Pooling] → [Conv + ReLU] → [Pooling]
       ←────────────── Feature Learning ──────────────→
                                          → Flatten → FC → Softmax
                                         ←── Classification ───→
```

각 단계의 역할:
1. **합성곱**: 이미지에서 특징 학습
2. **비선형 활성화**: 실세계 비선형성 처리
3. **풀링**: 차원 축소 및 공간 불변성 확보
4. **Flatten + FC + Softmax**: 최종 클래스 확률 출력

### TensorFlow 구현 예시

```python
import tensorflow as tf

def generate_model():
    model = tf.keras.Sequential([
        # 첫 번째 합성곱 블록
        tf.keras.layers.Conv2D(32, filter_size=3, activation='relu'),
        tf.keras.layers.MaxPool2D(pool_size=2, strides=2),

        # 두 번째 합성곱 블록
        tf.keras.layers.Conv2D(64, filter_size=3, activation='relu'),
        tf.keras.layers.MaxPool2D(pool_size=2, strides=2),

        # 완전 연결 분류기
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')  # 10개 클래스
    ])
    return model
```

---

## 12. CNN의 다양한 응용

같은 CNN 특징 학습 백본(backbone)을 다양한 태스크에 활용할 수 있다.

| 태스크 | 출력 | 비고 |
|--------|------|------|
| 분류 (Classification) | 클래스 레이블 | Softmax |
| 객체 검출 (Object Detection) | (x, y, w, h) + 클래스 | Bounding box |
| 의미론적 분할 (Semantic Segmentation) | 픽셀별 클래스 | FCN |
| 확률적 제어 (Probabilistic Control) | 제어 명령 분포 | 자율주행 |

---

## 13. 객체 검출 (Object Detection)

### 문제 정의

- **분류만 할 때**: 이미지에서 클래스 레이블 출력
- **검출까지 할 때**: 클래스 레이블 + 위치 `(x, y, w, h)` 동시 출력
- **다중 객체 검출**: 여러 객체의 바운딩 박스 리스트 출력

### R-CNN (Region-based CNN)

```
1. 입력 이미지
2. 약 2,000개의 지역 후보(Region Proposals) 추출
3. 각 지역에 CNN 적용하여 특징 계산
4. 지역 분류
```

**문제점:**
- **느리다**: 2,000개 지역을 각각 CNN에 통과시켜야 하므로 추론 속도가 매우 느림
- **취약하다**: 지역 후보 추출이 수동으로 정의되어 있어 유연성이 부족

---

## 14. 의미론적 분할 (Semantic Segmentation)

**Fully Convolutional Network (FCN)**을 사용한다.

- 모든 레이어가 합성곱으로만 구성
- **다운샘플링**: 저해상도 고수준 특징 추출
- **업샘플링(Transposed Convolution)**: 원래 해상도로 복원, 픽셀별 레이블 예측

```
입력 (3 × H × W) → 다운샘플링 → 업샘플링 → 예측 (H × W)
```

```python
tf.keras.layers.Conv2DTranspose   # TensorFlow
torch.nn.ConvTranspose2d          # PyTorch
```

---

## 15. 응용 사례: 자율주행 (End-to-End Autonomous Navigation)

강의에서 소개한 MIT 연구팀의 자율주행 시스템:

- **입력**: 카메라 원시 영상(Raw Perception I) + 조잡한 지도(Coarse Maps M, 예: GPS)
- **출력**: 조향 각도 등의 확률적 제어 명령

손실 함수:
$$L = -\log P(\theta \,|\, I, M)$$

- **핵심**: 어떠한 인간 레이블링이나 어노테이션 없이 **완전 엔드-투-엔드(end-to-end)** 학습
- 각 입력 소스(카메라 여러 각도, GPS 지도)에 독립적인 합성곱 서브네트워크를 적용한 뒤 concat하여 최종 제어 명령 예측

---

## 16. 핵심 개념 요약

| 개념 | 설명 |
|------|------|
| 합성곱(Convolution) | 슬라이딩 필터로 로컬 특징 추출, 파라미터 공유 |
| ReLU | 비선형 활성화: g(z) = max(0, z) |
| 풀링(Pooling) | 차원 축소 + 공간 불변성 확보 |
| 특징 계층성 | 저수준(엣지) → 중간 수준(부위) → 고수준(구조) |
| FCN | 다운샘플링 + 업샘플링으로 픽셀 단위 예측 |
| R-CNN | 지역 후보 기반 객체 검출 (느리고 취약함) |
| End-to-End | 원시 입력에서 최종 출력까지 일괄 학습 |

---

## 17. 느낀 점 및 질문

### 느낀 점

완전 연결 신경망에서 CNN으로의 전환이 단순히 성능 향상이 아니라, **이미지의 구조적 성질을 네트워크 설계에 반영한 결과**라는 점이 인상 깊었다. 파라미터 공유와 국소 연결이라는 두 가지 아이디어만으로 이미지 처리에서 압도적인 효율을 얻는다는 것이 놀라웠다.

또한 같은 CNN 백본 위에 분류, 검출, 분할, 자율주행 제어까지 다양한 태스크를 얹을 수 있다는 점에서, **특징 학습(feature learning)이 얼마나 범용적인지** 다시 한번 확인할 수 있었다.

### 스스로 생각해본 질문

1. **Max Pooling 외에 다른 다운샘플링 방법은?** → Stride가 큰 합성곱(Strided Convolution)을 사용하면 풀링 없이도 해상도를 줄일 수 있다고 생각했는데, 강의 마지막에 "How else can we downsample?"이라는 질문이 나왔다. 정확히 이 점을 다음에 탐구해보고 싶다.

2. **R-CNN의 느린 속도 문제는 어떻게 해결되었나?** → Fast R-CNN, Faster R-CNN, YOLO 등의 후속 연구에서 어떤 방식으로 개선되었는지 공부해볼 필요가 있다.

3. **자율주행 end-to-end 모델이 GPS 지도 없이도 작동할 수 있을까?** → 모델 입력에서 지도 정보를 제거했을 때 성능이 얼마나 떨어지는지 궁금하다.
