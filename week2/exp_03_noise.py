import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

# 스스로 해보기 3: 노이즈 크기를 바꿔보기
# 원본: scale=1.5 → 실험: scale=5.0 (노이즈가 매우 크면 어떻게 되는가?)
output_dir = 'outputs'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

weights = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
true_lengths = 2 * weights + 10
plot_weights = np.linspace(0, 10, 100)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for ax, scale in zip(axes, [1.5, 5.0]):
    np.random.seed(42)
    noise = np.random.normal(loc=0.0, scale=scale, size=len(weights))
    measured_lengths = true_lengths + noise

    tf.random.set_seed(42)
    model = tf.keras.Sequential([tf.keras.layers.Dense(units=1, input_shape=[1])])
    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
                  loss='mean_squared_error')
    model.fit(weights, measured_lengths, epochs=500, verbose=0)

    w = float(model.layers[0].get_weights()[0][0])
    b = float(model.layers[0].get_weights()[1][0])
    pred = model.predict(plot_weights.reshape(-1, 1), verbose=0)

    ax.scatter(weights, measured_lengths, color='blue', label='측정 데이터 (노이즈 포함)')
    ax.plot(weights, true_lengths, 'g--', label='진짜 법칙 (y=2x+10)')
    ax.plot(plot_weights, pred, 'r-', label=f'AI 예측 (y={w:.2f}x+{b:.2f})')
    ax.set_title(f'노이즈 scale = {scale}')
    ax.set_xlabel('Weight (kg)')
    ax.set_ylabel('Length (cm)')
    ax.legend()
    ax.grid(True)
    print(f"[scale={scale}] 학습 결과: 길이 = {w:.2f} * 무게 + {b:.2f}")

plt.suptitle('[실험 3] 노이즈 비교: scale=1.5 vs scale=5.0', fontsize=14, fontweight='bold')
plt.tight_layout()
save_path = os.path.join(output_dir, 'exp_03_noise.png')
plt.savefig(save_path)
print(f"\n저장: {save_path}")
print("관찰: 노이즈가 클수록 데이터가 분산되고, AI 예측이 진짜 법칙에서 더 멀어짐!")
