"""
04. 2D Heat Equation (2차원 열전도 방정식)
∂u/∂t = α (∂²u/∂x² + ∂²u/∂y²)

초기조건: u(x, y, 0) = sin(πx) sin(πy)
경계조건: u = 0 on all boundaries
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
print("04. 2D Heat Equation with PINN")
print("="*70)

# 물리 상수
alpha = 0.01

# 신경망 모델 (3개 입력: x, y, t)
def build_model():
    return tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='tanh', input_shape=(3,)),
        tf.keras.layers.Dense(64, activation='tanh'),
        tf.keras.layers.Dense(64, activation='tanh'),
        tf.keras.layers.Dense(64, activation='tanh'),
        tf.keras.layers.Dense(1)
    ])

model = build_model()

# PDE 잔차
@tf.function
def pde_residual(x, y, t):
    with tf.GradientTape(persistent=True) as tape:
        tape.watch([x, y, t])
        xyt = tf.concat([x, y, t], axis=1)
        u = model(xyt)
        
        du_dt = tape.gradient(u, t)
        du_dx = tape.gradient(u, x)
        du_dy = tape.gradient(u, y)
    
    d2u_dx2 = tape.gradient(du_dx, x)
    d2u_dy2 = tape.gradient(du_dy, y)
    del tape
    
    # 2D Heat equation: ∂u/∂t - α(∂²u/∂x² + ∂²u/∂y²) = 0
    residual = du_dt - alpha * (d2u_dx2 + d2u_dy2)
    return residual

# 손실 함수
@tf.function
def physics_loss(x_phys, y_phys, t_phys):
    residual = pde_residual(x_phys, y_phys, t_phys)
    return tf.reduce_mean(tf.square(residual))

@tf.function
def initial_condition_loss(x_ic, y_ic, t_ic):
    xyt = tf.concat([x_ic, y_ic, t_ic], axis=1)
    u_pred = model(xyt)
    u_true = tf.sin(np.pi * x_ic) * tf.sin(np.pi * y_ic)
    return tf.reduce_mean(tf.square(u_pred - u_true))

@tf.function
def boundary_condition_loss(x_bc, y_bc, t_bc):
    xyt = tf.concat([x_bc, y_bc, t_bc], axis=1)
    u_pred = model(xyt)
    return tf.reduce_mean(tf.square(u_pred))

# 훈련 데이터 생성
n_train = 5000
x_train = np.random.uniform(0, 1, (n_train, 1)).astype(np.float32)
y_train = np.random.uniform(0, 1, (n_train, 1)).astype(np.float32)
t_train = np.random.uniform(0, 0.5, (n_train, 1)).astype(np.float32)

# 초기조건 (t=0)
n_ic = 50
x_ic, y_ic = np.meshgrid(np.linspace(0, 1, n_ic), np.linspace(0, 1, n_ic))
x_ic = x_ic.reshape(-1, 1).astype(np.float32)
y_ic = y_ic.reshape(-1, 1).astype(np.float32)
t_ic = np.zeros_like(x_ic)

# 경계조건
n_bc = 50
t_bc_sample = np.linspace(0, 0.5, n_bc).astype(np.float32)

# x=0 경계
x_bc1 = np.zeros((n_bc * n_bc, 1), dtype=np.float32)
y_bc1 = np.tile(np.linspace(0, 1, n_bc), n_bc).reshape(-1, 1).astype(np.float32)
t_bc1 = np.repeat(t_bc_sample, n_bc).reshape(-1, 1)

# x=1 경계
x_bc2 = np.ones((n_bc * n_bc, 1), dtype=np.float32)
y_bc2 = y_bc1.copy()
t_bc2 = t_bc1.copy()

# y=0 경계
y_bc3 = np.zeros((n_bc * n_bc, 1), dtype=np.float32)
x_bc3 = np.tile(np.linspace(0, 1, n_bc), n_bc).reshape(-1, 1).astype(np.float32)
t_bc3 = np.repeat(t_bc_sample, n_bc).reshape(-1, 1)

# y=1 경계
y_bc4 = np.ones((n_bc * n_bc, 1), dtype=np.float32)
x_bc4 = x_bc3.copy()
t_bc4 = t_bc3.copy()

x_bc = np.vstack([x_bc1, x_bc2, x_bc3, x_bc4])
y_bc = np.vstack([y_bc1, y_bc2, y_bc3, y_bc4])
t_bc = np.vstack([t_bc1, t_bc2, t_bc3, t_bc4])

# Optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

# 훈련
print("\n훈련 시작...")
epochs = 6000
loss_history = []

for epoch in range(epochs):
    with tf.GradientTape() as tape:
        loss_pde = physics_loss(
            tf.constant(x_train), tf.constant(y_train), tf.constant(t_train))
        loss_ic = initial_condition_loss(
            tf.constant(x_ic), tf.constant(y_ic), tf.constant(t_ic))
        loss_bc = boundary_condition_loss(
            tf.constant(x_bc), tf.constant(y_bc), tf.constant(t_bc))
        
        total_loss = loss_pde + 10.0 * loss_ic + 10.0 * loss_bc
    
    gradients = tape.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    loss_history.append(total_loss.numpy())
    
    if (epoch + 1) % 1000 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss.numpy():.6f}")

print(f"최종 Loss: {loss_history[-1]:.6f}")

# 예측 및 시각화
n_test = 50
x_plot, y_plot = np.meshgrid(np.linspace(0, 1, n_test), np.linspace(0, 1, n_test))
t_snapshots = [0.0, 0.1, 0.3, 0.5]

fig = plt.figure(figsize=(16, 12))

for idx, t_snap in enumerate(t_snapshots):
    x_flat = x_plot.ravel().reshape(-1, 1).astype(np.float32)
    y_flat = y_plot.ravel().reshape(-1, 1).astype(np.float32)
    t_flat = np.full_like(x_flat, t_snap)
    
    xyt_test = np.column_stack([x_flat, y_flat, t_flat]).astype(np.float32)
    u_pred = model(xyt_test).numpy().reshape(x_plot.shape)
    
    # 해석해
    u_true = np.sin(np.pi * x_plot) * np.sin(np.pi * y_plot) * \
             np.exp(-alpha * 2 * (np.pi**2) * t_snap)
    
    error = np.abs(u_pred - u_true)
    
    # PINN 예측
    ax = fig.add_subplot(4, 3, 3*idx + 1)
    im = ax.contourf(x_plot, y_plot, u_pred, levels=20, cmap='hot')
    ax.set_xlabel('x', fontweight='bold')
    ax.set_ylabel('y', fontweight='bold')
    ax.set_title(f'PINN: t={t_snap:.1f}', fontweight='bold')
    ax.set_aspect('equal')
    plt.colorbar(im, ax=ax)
    
    # 해석해
    ax = fig.add_subplot(4, 3, 3*idx + 2)
    im = ax.contourf(x_plot, y_plot, u_true, levels=20, cmap='hot')
    ax.set_xlabel('x', fontweight='bold')
    ax.set_ylabel('y', fontweight='bold')
    ax.set_title(f'True: t={t_snap:.1f}', fontweight='bold')
    ax.set_aspect('equal')
    plt.colorbar(im, ax=ax)
    
    # 오차
    ax = fig.add_subplot(4, 3, 3*idx + 3)
    im = ax.contourf(x_plot, y_plot, error, levels=20, cmap='viridis')
    ax.set_xlabel('x', fontweight='bold')
    ax.set_ylabel('y', fontweight='bold')
    ax.set_title(f'Error: t={t_snap:.1f} (Max: {error.max():.4f})', fontweight='bold')
    ax.set_aspect('equal')
    plt.colorbar(im, ax=ax)

plt.suptitle('2D Heat Equation: ∂u/∂t = α(∂²u/∂x² + ∂²u/∂y²)', 
             fontsize=16, fontweight='bold')
plt.tight_layout()

output_path = os.path.join(output_dir, '04_heat_equation_2d.png')
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\n그래프 저장: {output_path}")

print(f"\n성능: 최종 t={t_snapshots[-1]}에서")
print(f"  MSE: {np.mean((u_pred - u_true)**2):.6e}")
print(f"  최대 절대 오차: {error.max():.6f}")
