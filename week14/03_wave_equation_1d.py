"""
03. 1D Wave Equation (파동 방정식)
∂²u/∂t² = c² ∂²u/∂x²

초기조건: u(x, 0) = sin(πx), ∂u/∂t(x, 0) = 0
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
print("03. 1D Wave Equation with PINN")
print("="*70)

# 물리 상수
c = 1.0  # 파동 속도

# 신경망 모델
def build_model():
    return tf.keras.Sequential([
        tf.keras.layers.Dense(50, activation='tanh', input_shape=(2,)),
        tf.keras.layers.Dense(50, activation='tanh'),
        tf.keras.layers.Dense(50, activation='tanh'),
        tf.keras.layers.Dense(1)
    ])

model = build_model()

# PDE 잔차
@tf.function
def pde_residual(x, t):
    with tf.GradientTape(persistent=True) as tape2:
        tape2.watch([x, t])
        with tf.GradientTape(persistent=True) as tape1:
            tape1.watch([x, t])
            xt = tf.concat([x, t], axis=1)
            u = model(xt)
        
        du_dt = tape1.gradient(u, t)
        du_dx = tape1.gradient(u, x)
        del tape1
    
    d2u_dt2 = tape2.gradient(du_dt, t)
    d2u_dx2 = tape2.gradient(du_dx, x)
    del tape2
    
    # Wave equation: ∂²u/∂t² - c² ∂²u/∂x² = 0
    residual = d2u_dt2 - c**2 * d2u_dx2
    return residual

# 손실 함수
@tf.function
def physics_loss(x_phys, t_phys):
    residual = pde_residual(x_phys, t_phys)
    return tf.reduce_mean(tf.square(residual))

@tf.function
def initial_position_loss(x_ic, t_ic):
    xt = tf.concat([x_ic, t_ic], axis=1)
    u_pred = model(xt)
    u_true = tf.sin(np.pi * x_ic)  # u(x, 0) = sin(πx)
    return tf.reduce_mean(tf.square(u_pred - u_true))

@tf.function
def initial_velocity_loss(x_ic, t_ic):
    with tf.GradientTape() as tape:
        tape.watch(t_ic)
        xt = tf.concat([x_ic, t_ic], axis=1)
        u = model(xt)
    du_dt = tape.gradient(u, t_ic)
    # ∂u/∂t(x, 0) = 0
    return tf.reduce_mean(tf.square(du_dt))

@tf.function
def boundary_loss(x_bc, t_bc):
    xt = tf.concat([x_bc, t_bc], axis=1)
    u_pred = model(xt)
    return tf.reduce_mean(tf.square(u_pred))

# 훈련 데이터
n_train = 3000
x_train = np.random.uniform(0, 1, (n_train, 1)).astype(np.float32)
t_train = np.random.uniform(0, 2, (n_train, 1)).astype(np.float32)

# 초기조건
n_ic = 100
x_ic = np.linspace(0, 1, n_ic).reshape(-1, 1).astype(np.float32)
t_ic = np.zeros((n_ic, 1), dtype=np.float32)

# 경계조건
n_bc = 100
t_bc = np.linspace(0, 2, n_bc).reshape(-1, 1).astype(np.float32)
x_bc_left = np.zeros((n_bc, 1), dtype=np.float32)
x_bc_right = np.ones((n_bc, 1), dtype=np.float32)
x_bc = np.vstack([x_bc_left, x_bc_right])
t_bc = np.vstack([t_bc, t_bc])

# Optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

# 훈련
print("\n훈련 시작...")
epochs = 8000
loss_history = []

for epoch in range(epochs):
    with tf.GradientTape() as tape:
        loss_pde = physics_loss(tf.constant(x_train), tf.constant(t_train))
        loss_ic_pos = initial_position_loss(tf.constant(x_ic), tf.constant(t_ic))
        loss_ic_vel = initial_velocity_loss(tf.constant(x_ic), tf.constant(t_ic))
        loss_bc = boundary_loss(tf.constant(x_bc), tf.constant(t_bc))
        
        total_loss = loss_pde + 20.0 * loss_ic_pos + 20.0 * loss_ic_vel + 10.0 * loss_bc
    
    gradients = tape.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    loss_history.append(total_loss.numpy())
    
    if (epoch + 1) % 1000 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss.numpy():.6f}")

print(f"최종 Loss: {loss_history[-1]:.6f}")

# 예측
x_test = np.linspace(0, 1, 100)
t_test = np.linspace(0, 2, 100)
X, T = np.meshgrid(x_test, t_test)

xt_test = np.column_stack([X.ravel(), T.ravel()]).astype(np.float32)
u_pred = model(xt_test).numpy().reshape(X.shape)

# 해석해
u_true = np.sin(np.pi * X) * np.cos(np.pi * c * T)

# 시각화
fig = plt.figure(figsize=(16, 10))

# 1. PINN 예측
ax1 = fig.add_subplot(2, 3, 1, projection='3d')
surf1 = ax1.plot_surface(X, T, u_pred, cmap='seismic', alpha=0.8)
ax1.set_xlabel('x', fontweight='bold')
ax1.set_ylabel('t', fontweight='bold')
ax1.set_zlabel('u(x,t)', fontweight='bold')
ax1.set_title('PINN Solution', fontweight='bold')
fig.colorbar(surf1, ax=ax1, shrink=0.5)

# 2. 해석해
ax2 = fig.add_subplot(2, 3, 2, projection='3d')
surf2 = ax2.plot_surface(X, T, u_true, cmap='seismic', alpha=0.8)
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
for t_snap in [0.0, 0.5, 1.0, 1.5, 2.0]:
    idx = int(t_snap / 2 * 99)
    ax4.plot(x_test, u_pred[idx, :], '--', label=f't={t_snap:.1f} (PINN)')
    ax4.plot(x_test, u_true[idx, :], '-', alpha=0.7, label=f't={t_snap:.1f} (True)')
ax4.set_xlabel('x', fontweight='bold')
ax4.set_ylabel('u(x,t)', fontweight='bold')
ax4.set_title('Time Snapshots', fontweight='bold')
ax4.legend(fontsize=7, ncol=2)
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
im = ax6.imshow(error, extent=[0, 1, 0, 2], origin='lower', aspect='auto', cmap='hot')
ax6.set_xlabel('x', fontweight='bold')
ax6.set_ylabel('t', fontweight='bold')
ax6.set_title('Error Heatmap', fontweight='bold')
fig.colorbar(im, ax=ax6)

plt.suptitle('1D Wave Equation: ∂²u/∂t² = c² ∂²u/∂x²', fontsize=16, fontweight='bold')
plt.tight_layout()

output_path = os.path.join(output_dir, '03_wave_equation_1d.png')
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\n그래프 저장: {output_path}")

# 성능 평가
mse = np.mean((u_pred - u_true)**2)
rel_error = np.mean(np.abs((u_pred - u_true) / (np.abs(u_true) + 1e-8))) * 100

print(f"\n성능:")
print(f"  MSE: {mse:.6e}")
print(f"  평균 상대 오차: {rel_error:.2f}%")
print(f"  최대 절대 오차: {error.max():.6f}")
