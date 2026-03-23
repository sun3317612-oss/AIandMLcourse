import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

# 스스로 해보기 2: epochs를 바꿔보기
# 원본: 500 → 실험: 100 (적게 학습하면 어떻게 되는가?)
output_dir = 'outputs'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

weights = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
true_lengths = 2 * weights + 10
np.random.seed(42)
noise = np.random.normal(loc=0.0, scale=1.5, size=len(weights))
measured_lengths = true_lengths + noise

plot_weights = np.linspace(0, 15, 100)
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for ax, epochs in zip(axes, [100, 500]):
    tf.random.set_seed(42)
    model = tf.keras.Sequential([tf.keras.layers.Dense(units=1, input_shape=[1])])
    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
                  loss='mean_squared_error')
    model.fit(weights, measured_lengths, epochs=epochs, verbose=0)

    w = float(model.layers[0].get_weights()[0][0])
    b = float(model.layers[0].get_weights()[1][0])
    pred = model.predict(plot_weights.reshape(-1, 1), verbose=0)

    ax.scatter(weights, measured_lengths, color='blue', label='측정 데이터')
    ax.plot(weights, true_lengths, 'g--', label='진짜 법칙 (y=2x+10)')
    ax.plot(plot_weights, pred, 'r-', label=f'AI 예측 (y={w:.2f}x+{b:.2f})')
    ax.set_title(f'epochs = {epochs}')
    ax.set_xlabel('Weight (kg)')
    ax.set_ylabel('Length (cm)')
    ax.legend()
    ax.grid(True)
    print(f"[epochs={epochs}] 학습 결과: 길이 = {w:.2f} * 무게 + {b:.2f}")

plt.suptitle('[실험 2] Epochs 비교: 100 vs 500', fontsize=14, fontweight='bold')
plt.tight_layout()
save_path = os.path.join(output_dir, 'exp_02_epochs.png')
plt.savefig(save_path)
print(f"\n저장: {save_path}")
print("관찰: epochs=100이면 학습이 덜 돼서 진짜 법칙과 차이가 더 큼!")
