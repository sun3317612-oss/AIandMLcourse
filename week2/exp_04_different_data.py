import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

# 스스로 해보기 4: 다른 데이터로 실험하기
# 고무줄 실험: 초기 길이 3cm, 1kg당 8cm 늘어남 (용수철보다 훨씬 잘 늘어남)
# 식: Length = 8 * Weight + 3
output_dir = 'outputs'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 새로운 데이터: 고무줄 (k=8, b=3)
weights = np.array([0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0], dtype=float)
true_lengths = 8 * weights + 3  # 진짜 공식

np.random.seed(7)
noise = np.random.normal(loc=0.0, scale=1.0, size=len(weights))
measured_lengths = true_lengths + noise

print("[고무줄 데이터]")
print("무게(kg):", weights)
print("측정된 길이(cm):", np.round(measured_lengths, 2))

# 모델 학습
tf.random.set_seed(42)
model = tf.keras.Sequential([tf.keras.layers.Dense(units=1, input_shape=[1])])
model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
              loss='mean_squared_error')
model.fit(weights, measured_lengths, epochs=500, verbose=0)

w = float(model.layers[0].get_weights()[0][0])
b = float(model.layers[0].get_weights()[1][0])
print(f"\n[학습 결과]")
print(f"예측된 식: 길이 = {w:.2f} * 무게 + {b:.2f}")
print(f"실제 식  : 길이 = 8.00 * 무게 + 3.00")

# 시각화: 용수철(원본)과 고무줄(실험) 나란히 비교
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 왼쪽: 원본 용수철 (Week2 Lab4 결과)
w_spring = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
np.random.seed(42)
noisy_spring = 2 * w_spring + 10 + np.random.normal(0, 1.5, len(w_spring))
axes[0].scatter(w_spring, noisy_spring, color='blue', label='측정 데이터')
axes[0].plot(w_spring, 2 * w_spring + 10, 'g--', label='진짜 법칙 (y=2x+10)')
axes[0].set_title('원본: 용수철 (k=2, b=10)')
axes[0].set_xlabel('Weight (kg)')
axes[0].set_ylabel('Length (cm)')
axes[0].legend()
axes[0].grid(True)

# 오른쪽: 새로운 고무줄
plot_w = np.linspace(0, 5, 100)
pred = model.predict(plot_w.reshape(-1, 1), verbose=0)
axes[1].scatter(weights, measured_lengths, color='orange', label='측정 데이터')
axes[1].plot(weights, true_lengths, 'g--', label='진짜 법칙 (y=8x+3)')
axes[1].plot(plot_w, pred, 'r-', label=f'AI 예측 (y={w:.2f}x+{b:.2f})')
axes[1].set_title('실험: 고무줄 (k=8, b=3)')
axes[1].set_xlabel('Weight (kg)')
axes[1].set_ylabel('Length (cm)')
axes[1].legend()
axes[1].grid(True)

plt.suptitle('[실험 4] 다른 데이터: 용수철 vs 고무줄', fontsize=14, fontweight='bold')
plt.tight_layout()
save_path = os.path.join(output_dir, 'exp_04_different_data.png')
plt.savefig(save_path)
print(f"\n저장: {save_path}")
print("관찰: 완전히 다른 k, b 값을 가진 데이터도 AI가 올바르게 학습함!")
