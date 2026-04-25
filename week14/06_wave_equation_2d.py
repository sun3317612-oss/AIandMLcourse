"""
06. 2D Wave Equation (2차원 파동 방정식)
∂²u/∂t² = c² (∂²u/∂x² + ∂²u/∂y²)

초기조건: u(x, y, 0) = sin(πx)sin(πy), ∂u/∂t(x, y, 0) = 0
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
print("06. 2D Wave Equation with PINN")
print("="*70)

# 물리 상수
c = 1.0  # 파동 속도

# 신경망 모델
def build_model():
    return tf.keras.Sequential([
        tf.keras.layers.Dense(70, activation='tanh', input_shape=(3,)),
        tf.keras.layers.Dense(70, activation='tanh'),
        tf.keras.layers.Dense(70, activation='tanh'),
        tf.keras.layers.Dense(70, activation='tanh'),
        tf.keras.layers.Dense(1)
    ])

model = build_model()

# PDE 잔차
@tf.function
def pde_residual(x, y, t):
    with tf.GradientTape(persistent=True) as tape2:
        tape2.watch([x, y, t])
        with tf.GradientTape(persistent=True) as tape1:
            tape1.watch([x, y, t])
            xyt = tf.concat([x, y, t], axis=1)
            u = model(xyt)
        
        du_dt = tape1.gradient(u, t)
        du_dx = tape1.gradient(u, x)
        du_dy = tape1.gradient(u, y)
        del tape1
    
    d2u_dt2 = tape2.gradient(du_dt, t)
    d2u_dx2 = tape2.gradient(du_dx, x)
    d2u_dy2 = tape2.gradient(du_dy, y)
    del tape2
    
    # 2D Wave equation: ∂²u/∂t² - c²(∂²u/∂x² + ∂²u/∂y²) = 0
    residual = d2u_dt2 - c**2 * (d2u_dx2 + d2u_dy2)
    return residual

# 손실 함수
@tf.function
def physics_loss(x_phys, y_phys, t_phys):
    residual = pde_residual(x_phys, y_phys, t_phys)
    return tf.reduce_mean(tf.square(residual))

@tf.function
def initial_position_loss(x_ic, y_ic, t_ic):
    xyt = tf.concat([x_ic, y_ic, t_ic], axis=1)
    u_pred = model(xyt)
    u_true = tf.sin(np.pi * x_ic) * tf.sin(np.pi * y_ic)
    return tf.reduce_mean(tf.square(u_pred - u_true))

@tf.function
def initial_velocity_loss(x_ic, y_ic, t_ic):
    with tf.GradientTape() as tape:
        tape.watch(t_ic)
        xyt = tf.concat([x_ic, y_ic, t_ic], axis=1)
        u = model(xyt)
    du_dt = tape.gradient(u, t_ic)
    return tf.reduce_mean(tf.square(du_dt))

@tf.function
def boundary_loss(x_bc, y_bc, t_bc):
    xyt = tf.concat([x_bc, y_bc, t_bc], axis=1)
    u_pred = model(xyt)
    return tf.reduce_mean(tf.square(u_pred))

# 훈련 데이터
n_train = 6000
x_train = np.random.uniform(0, 1, (n_train, 1)).astype(np.float32)
y_train = np.random.uniform(0, 1, (n_train, 1)).astype(np.float32)
t_train = np.random.uniform(0, 1, (n_train, 1)).astype(np.float32)

# 초기조건
n_ic = 40
x_ic_grid, y_ic_grid = np.meshgrid(np.linspace(0, 1, n_ic), np.linspace(0, 1, n_ic))
x_ic = x_ic_grid.reshape(-1, 1).astype(np.float32)
y_ic = y_ic_grid.reshape(-1, 1).astype(np.float32)
t_ic = np.zeros_like(x_ic)

# 경계조건
n_bc_side = 30
t_bc_vals = np.linspace(0, 1, n_bc_side).astype(np.float32)

# 4개 경계 생성
bc_points = []
for edge in ['x0', 'x1', 'y0', 'y1']:
    for t_val in t_bc_vals:
        if edge == 'x0':
            bc_points.append([0.0, np.random.uniform(0, 1), t_val])
        elif edge == 'x1':
            bc_points.append([1.0, np.random.uniform(0, 1), t_val])
        elif edge == 'y0':
            bc_points.append([np.random.uniform(0, 1), 0.0, t_val])
        else:  # y1
            bc_points.append([np.random.uniform(0, 1), 1.0, t_val])

bc_points = np.array(bc_points, dtype=np.float32)
x_bc = bc_points[:, 0:1]
y_bc = bc_points[:, 1:2]
t_bc = bc_points[:, 2:3]

# Optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

# 훈련
print("\n훈련 시작...")
epochs = 8000
loss_history = []

for epoch in range(epochs):
    with tf.GradientTape() as tape:
        loss_pde = physics_loss(
            tf.constant(x_train), tf.constant(y_train), tf.constant(t_train))
        loss_ic_pos = initial_position_loss(
            tf.constant(x_ic), tf.constant(y_ic), tf.constant(t_ic))
        loss_ic_vel = initial_velocity_loss(
            tf.constant(x_ic), tf.constant(y_ic), tf.constant(t_ic))
        loss_bc = boundary_loss(
            tf.constant(x_bc), tf.constant(y_bc), tf.constant(t_bc))
        
        total_loss = loss_pde + 20.0 * loss_ic_pos + 20.0 * loss_ic_vel + 10.0 * loss_bc
    
    gradients = tape.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    loss_history.append(total_loss.numpy())
    
    if (epoch + 1) % 1000 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss.numpy():.6f}")

print(f"최종 Loss: {loss_history[-1]:.6f}")

# 예측 및 시각화
n_plot = 40
x_plot, y_plot = np.meshgrid(np.linspace(0, 1, n_plot), np.linspace(0, 1, n_plot))
t_snapshots = [0.0, 0.25, 0.5, 0.75]

fig = plt.figure(figsize=(16, 12))

for idx, t_snap in enumerate(t_snapshots):
    x_flat = x_plot.ravel().reshape(-1, 1).astype(np.float32)
    y_flat = y_plot.ravel().reshape(-1, 1).astype(np.float32)
    t_flat = np.full_like(x_flat, t_snap)
    
    u_pred = model(np.column_stack([x_flat, y_flat, t_flat])).numpy().reshape(x_plot.shape)
    
    # 해석해
    u_true = np.sin(np.pi * x_plot) * np.sin(np.pi * y_plot) * \
             np.cos(np.pi * c * np.sqrt(2) * t_snap)
    
    # PINN (3D)
    ax = fig.add_subplot(4, 3, 3*idx + 1, projection='3d')
    surf = ax.plot_surface(x_plot, y_plot, u_pred, cmap='seismic', alpha=0.8)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('u')
    ax.set_title(f'PINN: t={t_snap:.2f}', fontweight='bold')
    ax.set_zlim([-1.5, 1.5])
    
    # True (3D)
    ax = fig.add_subplot(4, 3, 3*idx + 2, projection='3d')
    surf = ax.plot_surface(x_plot, y_plot, u_true, cmap='seismic', alpha=0.8)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('u')
    ax.set_title(f'True: t={t_snap:.2f}', fontweight='bold')
    ax.set_zlim([-1.5, 1.5])
    
    # 오차 (히트맵)
    error = np.abs(u_pred - u_true)
    ax = fig.add_subplot(4, 3, 3*idx + 3)
    im = ax.imshow(error, extent=[0, 1, 0, 1], origin='lower', aspect='auto', cmap='hot')
    ax.set_xlabel('x', fontweight='bold')
    ax.set_ylabel('y', fontweight='bold')
    ax.set_title(f'Error: t={t_snap:.2f} (Max: {error.max():.3f})', fontweight='bold')
    plt.colorbar(im, ax=ax)

plt.suptitle('2D Wave Equation: ∂²u/∂t² = c²(∂²u/∂x² + ∂²u/∂y²)', 
             fontsize=16, fontweight='bold')
plt.tight_layout()

output_path = os.path.join(output_dir, '06_wave_equation_2d.png')
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\n그래프 저장: {output_path}")

print(f"\n성능:")
print(f"  최종 Loss: {loss_history[-1]:.6e}")
