"""
01. PINN 기본 개념 (Physics-Informed Neural Networks)
간단한 ODE를 PINN으로 풀기

문제: du/dt = -u, u(0) = 1
해: u(t) = e^(-t)
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os

# 한글 폰트 설정
def set_korean_font():
    font_list = [f.name for f in fm.fontManager.ttflist]
    if 'Malgun Gothic' in font_list:
        plt.rcParams['font.family'] = 'Malgun Gothic'
    elif 'Gulim' in font_list:
        plt.rcParams['font.family'] = 'Gulim'
    elif 'Batang' in font_list:
        plt.rcParams['font.family'] = 'Batang'
    elif 'AppleGothic' in font_list:
        plt.rcParams['font.family'] = 'AppleGothic'
    plt.rcParams['axes.unicode_minus'] = False

set_korean_font()

# 출력 디렉토리
output_dir = 'outputs'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print("="*70)
print("01. PINN 기본 개념 - 간단한 ODE 풀기")
print("="*70)

# 신경망 모델 정의
def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(20, activation='tanh', input_shape=(1,)),
        tf.keras.layers.Dense(20, activation='tanh'),
        tf.keras.layers.Dense(1)
    ])
    return model

model = build_model()

# 물리 법칙을 포함한 손실 함수
@tf.function
def physics_loss(t):
    with tf.GradientTape() as tape:
        tape.watch(t)
        u = model(t)
    du_dt = tape.gradient(u, t)
    
    # 방정식: du/dt + u = 0
    pde_residual = du_dt + u
    return tf.reduce_mean(tf.square(pde_residual))

# 초기 조건 손실
@tf.function
def initial_loss(t_init, u_init):
    u_pred = model(t_init)
    return tf.reduce_mean(tf.square(u_pred - u_init))

# 전체 손실 함수
@tf.function
def total_loss(t_physics, t_init, u_init):
    loss_phys = physics_loss(t_physics)
    loss_init = initial_loss(t_init, u_init)
    return loss_phys + 10.0 * loss_init  # 초기조건에 더 큰 가중치

# 훈련 데이터 생성
n_train = 100
t_train = np.linspace(0, 3, n_train).reshape(-1, 1).astype(np.float32)
t_init = np.array([[0.0]], dtype=np.float32)
u_init = np.array([[1.0]], dtype=np.float32)

# Optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

# 훈련
print("\n훈련 시작...")
epochs = 3000
loss_history = []

for epoch in range(epochs):
    with tf.GradientTape() as tape:
        loss = total_loss(tf.constant(t_train), t_init, u_init)
    
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    loss_history.append(loss.numpy())
    
    if (epoch + 1) % 300 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.numpy():.6f}")

print(f"최종 Loss: {loss_history[-1]:.6f}")

# 예측
t_test = np.linspace(0, 3, 200).reshape(-1, 1).astype(np.float32)
u_pred = model(t_test).numpy()
u_true = np.exp(-t_test)

# 시각화
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 1. 해 비교
axes[0].plot(t_test, u_true, 'b-', linewidth=2, label='True: u = e^(-t)')
axes[0].plot(t_test, u_pred, 'r--', linewidth=2, label='PINN 예측')
axes[0].scatter([0], [1], c='green', s=100, zorder=5, label='초기조건: u(0)=1')
axes[0].set_xlabel('t', fontsize=12, fontweight='bold')
axes[0].set_ylabel('u(t)', fontsize=12, fontweight='bold')
axes[0].set_title('PINN Solution vs True Solution', fontsize=13, fontweight='bold')
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)

# 2. 손실 곡선
axes[1].semilogy(loss_history, 'purple', linewidth=2)
axes[1].set_xlabel('Epoch', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Loss (log scale)', fontsize=12, fontweight='bold')
axes[1].set_title('Training Loss', fontsize=13, fontweight='bold')
axes[1].grid(True, alpha=0.3)

plt.suptitle('PINN으로 ODE 풀기: du/dt = -u', fontsize=15, fontweight='bold')
plt.tight_layout()

output_path = os.path.join(output_dir, '01_basic_pinn.png')
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\n그래프 저장: {output_path}")

# 오차 분석
mse = np.mean((u_pred - u_true)**2)
rel_error = np.mean(np.abs((u_pred - u_true) / u_true)) * 100

print(f"\n성능:")
print(f"  MSE: {mse:.6e}")
print(f"  상대 오차: {rel_error:.2f}%")
