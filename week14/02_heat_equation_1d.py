"""
02. 1D Heat Equation (열전도 방정식)
∂u/∂t = α ∂²u/∂x²

초기조건: u(x, 0) = sin(πx)
경계조건: u(0, t) = u(1, t) = 0
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

output_dir = 'outputs'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print("="*70)
print("02. 1D Heat Equation with PINN")
print("="*70)

# 물리 상수
alpha = 0.01  # 열확산계수

# 신경망 모델
def build_model():
    return tf.keras.Sequential([
        tf.keras.layers.Dense(32, activation='tanh', input_shape=(2,)),
        tf.keras.layers.Dense(32, activation='tanh'),
        tf.keras.layers.Dense(32, activation='tanh'),
        tf.keras.layers.Dense(1)
    ])

model = build_model()

# PDE 잔차 계산
@tf.function
def pde_residual(x, t):
    with tf.GradientTape(persistent=True) as tape:
        tape.watch([x, t])
        xt = tf.concat([x, t], axis=1)
        u = model(xt)
        
        # 1차 미분
        du_dt = tape.gradient(u, t)
        du_dx = tape.gradient(u, x)
    
    # 2차 미분
    d2u_dx2 = tape.gradient(du_dx, x)
    del tape
    
    # Heat equation: ∂u/∂t - α ∂²u/∂x² = 0
    residual = du_dt - alpha * d2u_dx2
    return residual

# 손실 함수들
@tf.function
def physics_loss(x_phys, t_phys):
    residual = pde_residual(x_phys, t_phys)
    return tf.reduce_mean(tf.square(residual))

@tf.function
def initial_condition_loss(x_ic, t_ic):
    xt = tf.concat([x_ic, t_ic], axis=1)
    u_pred = model(xt)
    u_true = tf.sin(np.pi * x_ic)  # u(x, 0) = sin(πx)
    return tf.reduce_mean(tf.square(u_pred - u_true))

@tf.function
def boundary_condition_loss(x_bc, t_bc):
    xt = tf.concat([x_bc, t_bc], axis=1)
    u_pred = model(xt)
    u_true = tf.zeros_like(u_pred)  # u(0, t) = u(1, t) = 0
    return tf.reduce_mean(tf.square(u_pred - u_true))

# 훈련 데이터 생성
n_train = 2000
x_train = np.random.uniform(0, 1, (n_train, 1)).astype(np.float32)
t_train = np.random.uniform(0, 1, (n_train, 1)).astype(np.float32)

# 초기조건 (t=0)
n_ic = 50
x_ic = np.linspace(0, 1, n_ic).reshape(-1, 1).astype(np.float32)
t_ic = np.zeros((n_ic, 1), dtype=np.float32)

# 경계조건 (x=0, x=1)
n_bc = 50
t_bc = np.linspace(0, 1, n_bc).reshape(-1, 1).astype(np.float32)
x_bc_left = np.zeros((n_bc, 1), dtype=np.float32)
x_bc_right = np.ones((n_bc, 1), dtype=np.float32)
x_bc = np.vstack([x_bc_left, x_bc_right])
t_bc = np.vstack([t_bc, t_bc])

# Optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

# 훈련
print("\n훈련 시작...")
epochs = 5000
loss_history = []

for epoch in range(epochs):
    with tf.GradientTape() as tape:
        loss_pde = physics_loss(tf.constant(x_train), tf.constant(t_train))
        loss_ic = initial_condition_loss(tf.constant(x_ic), tf.constant(t_ic))
        loss_bc = boundary_condition_loss(tf.constant(x_bc), tf.constant(t_bc))
        
        total_loss = loss_pde + 10.0 * loss_ic + 10.0 * loss_bc
    
    gradients = tape.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    loss_history.append(total_loss.numpy())
    
    if (epoch + 1) % 500 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss.numpy():.6f}, "
              f"PDE: {loss_pde.numpy():.6f}, IC: {loss_ic.numpy():.6f}, BC: {loss_bc.numpy():.6f}")

print(f"최종 Loss: {loss_history[-1]:.6f}")

# 예측 및 시각화
x_test = np.linspace(0, 1, 100)
t_test = np.linspace(0, 1, 100)
X, T = np.meshgrid(x_test, t_test)

xt_test = np.column_stack([X.ravel(), T.ravel()]).astype(np.float32)
u_pred = model(xt_test).numpy().reshape(X.shape)

# 해석해
u_true = np.sin(np.pi * X) * np.exp(-alpha * (np.pi**2) * T)

# 시각화
fig = plt.figure(figsize=(16, 10))

# 1. PINN 예측
ax1 = fig.add_subplot(2, 3, 1, projection='3d')
surf1 = ax1.plot_surface(X, T, u_pred, cmap='viridis', alpha=0.8)
ax1.set_xlabel('x', fontweight='bold')
ax1.set_ylabel('t', fontweight='bold')
ax1.set_zlabel('u(x,t)', fontweight='bold')
ax1.set_title('PINN Solution', fontweight='bold')
fig.colorbar(surf1, ax=ax1, shrink=0.5)

# 2. 해석해
ax2 = fig.add_subplot(2, 3, 2, projection='3d')
surf2 = ax2.plot_surface(X, T, u_true, cmap='viridis', alpha=0.8)
ax2.set_xlabel('x', fontweight='bold')
ax2.set_ylabel('t', fontweight='bold')
ax2.set_zlabel('u(x,t)', fontweight='bold')
ax2.set_title('Analytical Solution', fontweight='bold')
fig.colorbar(surf2, ax=ax2, shrink=0.5)

# 3. 오차
ax3 = fig.add_subplot(2, 3, 3, projection='3d')
error = np.abs(u_pred - u_true)
surf3 = ax3.plot_surface(X, T, error, cmap='hot', alpha=0.8)
ax3.set_xlabel('x', fontweight='bold')
ax3.set_ylabel('t', fontweight='bold')
ax3.set_zlabel('Error', fontweight='bold')
ax3.set_title(f'Absolute Error (Max: {error.max():.4f})', fontweight='bold')
fig.colorbar(surf3, ax=ax3, shrink=0.5)

# 4. 시간별 스냅샷
ax4 = fig.add_subplot(2, 3, 4)
for t_snap in [0.0, 0.25, 0.5, 0.75, 1.0]:
    idx = int(t_snap * 99)
    ax4.plot(x_test, u_pred[idx, :], label=f't={t_snap:.2f} (PINN)', linestyle='--')
    ax4.plot(x_test, u_true[idx, :], label=f't={t_snap:.2f} (True)', linestyle='-', alpha=0.7)
ax4.set_xlabel('x', fontweight='bold')
ax4.set_ylabel('u(x,t)', fontweight='bold')
ax4.set_title('Time Snapshots', fontweight='bold')
ax4.legend(fontsize=8, ncol=2)
ax4.grid(True, alpha=0.3)

# 5. 손실 곡선
ax5 = fig.add_subplot(2, 3, 5)
ax5.semilogy(loss_history, 'purple', linewidth=2)
ax5.set_xlabel('Epoch', fontweight='bold')
ax5.set_ylabel('Loss (log scale)', fontweight='bold')
ax5.set_title('Training Loss', fontweight='bold')
ax5.grid(True, alpha=0.3)

# 6. 오차 히트맵
ax6 = fig.add_subplot(2, 3, 6)
im = ax6.imshow(error, extent=[0, 1, 0, 1], origin='lower', aspect='auto', cmap='hot')
ax6.set_xlabel('x', fontweight='bold')
ax6.set_ylabel('t', fontweight='bold')
ax6.set_title('Error Heatmap', fontweight='bold')
fig.colorbar(im, ax=ax6)

plt.suptitle('1D Heat Equation: ∂u/∂t = α ∂²u/∂x²', fontsize=16, fontweight='bold')
plt.tight_layout()

output_path = os.path.join(output_dir, '02_heat_equation_1d.png')
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\n그래프 저장: {output_path}")

# 성능 평가
mse = np.mean((u_pred - u_true)**2)
rel_error = np.mean(np.abs((u_pred - u_true) / (u_true + 1e-8))) * 100

print(f"\n성능:")
print(f"  MSE: {mse:.6e}")
print(f"  평균 상대 오차: {rel_error:.2f}%")
print(f"  최대 절대 오차: {error.max():.6f}")
