export type Chapter = {
  slug: string
  title: string
  description: string
  free: boolean
  image?: string
  content: string
}

export const chapters: Chapter[] = [
  {
    slug: "regularization",
    title: "1. Regularization (규제)",
    description: "L1/L2, Dropout, Batch Normalization으로 과적합을 막는 기법",
    free: true,
    image: "/images/week5/01_regularization_plot.png",
    content: `
## 개념

모델이 훈련 데이터에 과도하게 맞춰지는 **과적합(Overfitting)**을 막기 위한 기법들입니다.

### L1 / L2 Regularization

가중치(Weight)의 크기에 페널티를 부여하여 모델을 단순하게 만듭니다.

- **L1**: 가중치의 절대값 합을 손실 함수에 추가 → 일부 가중치를 0으로 만들어 희소성(Sparsity) 확보
- **L2**: 가중치의 제곱 합을 손실 함수에 추가 → 가중치를 전반적으로 작게 유지

### Dropout

학습 시 무작위로 일부 뉴런을 꺼버려서(0으로 만들어) 특정 뉴런에 대한 의존을 방지합니다.

### Batch Normalization

각 층의 입력을 평균 0, 분산 1로 정규화하여 학습을 안정화하고 속도를 높입니다.

\`\`\`python
model = tf.keras.Sequential([
    layers.Dense(64, activation='relu',
                 kernel_regularizer=regularizers.l2(0.001)),
    layers.Dropout(0.5),
    layers.BatchNormalization(),
    layers.Dense(1)
])
\`\`\`

## 결과 해석

- **None**: 규제 없이 훈련하면 Validation Loss가 다시 증가 (과적합)
- **Dropout/L2**: 과적합 억제 → Validation Loss가 안정적으로 유지
    `,
  },
  {
    slug: "overfitting",
    title: "2. Overfitting vs Underfitting",
    description: "모델 복잡도와 일반화 성능의 관계를 시각적으로 이해",
    free: true,
    image: "/images/week5/02_overfitting_underfitting.png",
    content: `
## 개념

| 상태 | 설명 |
|------|------|
| **Underfitting** | 모델이 너무 단순 → 데이터 패턴을 제대로 학습 못함 |
| **Overfitting** | 모델이 너무 복잡 → 훈련 데이터의 노이즈까지 학습 |
| **Balanced** | 적절한 복잡도 → 일반화 성능 우수 |

## Loss Curve로 진단하기

\`\`\`python
history = model.fit(X_train, y_train,
                    validation_split=0.2,
                    epochs=100)

# Train Loss vs Validation Loss 비교
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Validation')
\`\`\`

## 결과 해석

- **Overfit 모델**: Train Loss ↓ 낮음, Val Loss ↑ 높음 (격차 큼)
- **Underfit 모델**: Train Loss도 높고, Val Loss도 높음
- **Balanced 모델**: 두 Loss 모두 수렴하며 격차 작음
    `,
  },
  {
    slug: "data-augmentation",
    title: "3. Data Augmentation (데이터 증강)",
    description: "데이터 부족 문제를 해결하는 이미지 증강 기법",
    free: false,
    image: "/images/week5/03_augmentation_examples.png",
    content: `
## 개념

데이터가 부족할 때, 기존 이미지를 변형(회전, 뒤집기, 확대/축소 등)하여 데이터의 다양성을 늘립니다.
모델이 위치나 각도 변화에 강건(Robust)해지도록 돕습니다.

## Keras로 구현

\`\`\`python
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
    layers.RandomTranslation(0.1, 0.1),
])

# 모델 파이프라인에 통합
model = tf.keras.Sequential([
    data_augmentation,
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(10, activation='softmax'),
])
\`\`\`

## 언제 사용하나?

- 훈련 데이터가 수백~수천 장 이하일 때
- 이미지 분류, 객체 탐지 등 컴퓨터 비전 태스크
- Transfer Learning과 함께 사용 시 효과 극대화
    `,
  },
  {
    slug: "transfer-learning",
    title: "4. Transfer Learning (전이 학습)",
    description: "ImageNet 사전 학습 모델을 활용해 적은 데이터로 높은 성능 달성",
    free: false,
    image: undefined,
    content: `
## 개념

이미 대량의 데이터(ImageNet 1.2M장)로 학습된 모델의 지식을 가져와,
내가 가진 적은 데이터의 문제 해결에 활용하는 방법입니다.

| 방법 | 설명 |
|------|------|
| **Feature Extraction** | Pre-trained 모델의 Conv Base를 고정(Freeze), 분류기만 새로 학습 |
| **Fine-tuning** | Pre-trained 모델의 상위 층 일부도 미세하게 같이 학습 |

## MobileNetV2로 구현

\`\`\`python
# 사전 학습된 모델 불러오기 (분류기 제외)
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(160, 160, 3),
    include_top=False,
    weights='imagenet'
)

# Feature Extraction: Base 동결
base_model.trainable = False

# 새 분류기 추가
model = tf.keras.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes, activation='softmax'),
])

print(f"학습 가능 파라미터: {model.trainable_variables.__len__()}")
\`\`\`

## 핵심 인사이트

- ImageNet의 저수준 특징(에지, 텍스처)은 대부분의 이미지 태스크에 공통으로 유용
- 데이터가 100장뿐이어도 ResNet50 수준의 성능을 낼 수 있는 이유
    `,
  },
  {
    slug: "cnn-mnist",
    title: "5. CNN 실습 — MNIST 손글씨 인식",
    description: "Conv2D + MaxPooling2D로 99% 정확도 달성하는 CNN 구현",
    free: false,
    image: "/images/week5/05_mnist_cnn_result.png",
    content: `
## CNN 구조

**CNN (Convolutional Neural Network)**은 이미지 처리에 특화된 딥러닝 구조입니다.

| 레이어 | 역할 |
|--------|------|
| **Conv2D** | 이미지의 특징(에지, 패턴) 추출 |
| **MaxPooling2D** | 이미지 크기 축소 + 중요 특징 보존 |
| **Flatten** | 2D 특징 맵 → 1D 벡터 변환 |
| **Dense** | 최종 분류 수행 |

## MNIST CNN 구현

\`\`\`python
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax'),
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(train_images, train_labels,
                    epochs=10, validation_split=0.1)
\`\`\`

## 결과

- 학습 정확도: **~99%**
- 검증 정확도: **~99%**
- Dense-only 모델 대비 파라미터 수 감소 + 정확도 향상
    `,
  },
]

export function getChapter(slug: string): Chapter | undefined {
  return chapters.find((c) => c.slug === slug)
}

export function getPrevNext(slug: string): {
  prev: Chapter | null
  next: Chapter | null
} {
  const idx = chapters.findIndex((c) => c.slug === slug)
  return {
    prev: idx > 0 ? chapters[idx - 1] : null,
    next: idx < chapters.length - 1 ? chapters[idx + 1] : null,
  }
}
