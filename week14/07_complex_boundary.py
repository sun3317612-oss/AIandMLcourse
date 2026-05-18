"""
07. Complex Boundary Conditions (복잡한 경계조건)
2D Heat Equation with:
- Dirichlet boundary (u = value)
- Neumann boundary (∂u/∂n = 0)
- Mixed boundary conditions

실용적인 예: L자 모양 영역에서의 열전도
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
print("07. Complex Boundary Conditions with PINN")
print("="*70)

# 물리 상수
alpha = 0.01

# 영역 정의: L자 모양 (0≤x≤1, 0≤y≤1에서 오른쪽 위 (0.5≤x≤1, 0.5≤y≤1) 제외)
def is_in_domain(x, y):
    """L자 영역 확인"""
    in_domain = (x >= 0) & (x <= 1) & (y >= 0) & (y <= 1)
    excluded = (x >= 0.5) & (x <= 1) & (y >= 0.5) & (y <= 1)
    return in_domain & ~excluded

# 신경망 모델
def build_model():
    return tf.keras.Sequential([
        tf.keras.layers.Dense(80, activation='tanh', input_shape=(3,)),
        tf.keras.layers.Dense(80, activation='tanh'),
        tf.keras.layers.Dense(80, activation='tanh'),
        tf.keras.layers.Dense(80, activation='tanh'),
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
    # 초기 온도: 중앙에 뜨거운 점
    u_true = tf.exp(-50.0 * ((x_ic - 0.25)**2 + (y_ic - 0.25)**2))
    return tf.reduce_mean(tf.square(u_pred - u_true))

@tf.function
def dirichlet_bc_loss(x_bc, y_bc, t_bc, u_bc):
    """Dirichlet BC: u = value"""
    xyt = tf.concat([x_bc, y_bc, t_bc], axis=1)
    u_pred = model(xyt)
    return tf.reduce_mean(tf.square(u_pred - u_bc))

@tf.function
def neumann_bc_loss(x_bc, y_bc, t_bc, nx, ny):
    """Neumann BC: ∂u/∂n = 0 (단열 경계)"""
    with tf.GradientTape(persistent=True) as tape:
        tape.watch([x_bc, y_bc])
        xyt = tf.concat([x_bc, y_bc, t_bc], axis=1)
        u = model(xyt)
    
    du_dx = tape.gradient(u, x_bc)
    du_dy = tape.gradient(u, y_bc)
    del tape  # persistent tape는 사용 후 삭제
    
    # 법선 방향 미분: ∂u/∂n = ∇u · n
    du_dn = du_dx * nx + du_dy * ny
    return tf.reduce_mean(tf.square(du_dn))

# 훈련 데이터 생성
print("\n훈련 데이터 생성 중...")

# 물리 법칙 적용 영역 (L자 내부)
n_train = 5000
x_cand = np.random.uniform(0, 1, (n_train * 2, 1))
y_cand = np.random.uniform(0, 1, (n_train * 2, 1))
mask = is_in_domain(x_cand, y_cand).flatten()
x_train = x_cand[mask][:n_train].astype(np.float32)
y_train = y_cand[mask][:n_train].astype(np.float32)
t_train = np.random.uniform(0, 0.5, (n_train, 1)).astype(np.float32)

# 초기조건
n_ic = 50
x_ic_cand, y_ic_cand = np.meshgrid(np.linspace(0, 1, n_ic), np.linspace(0, 1, n_ic))
mask_ic = is_in_domain(x_ic_cand, y_ic_cand)
x_ic = x_ic_cand[mask_ic].reshape(-1, 1).astype(np.float32)
y_ic = y_ic_cand[mask_ic].reshape(-1, 1).astype(np.float32)
t_ic = np.zeros_like(x_ic)

# Dirichlet BC: 왼쪽 경계 (x=0)에서 u=0 (고정 온도)
n_bc = 30
y_dir = np.linspace(0, 0.5, n_bc).reshape(-1, 1).astype(np.float32)
x_dir = np.zeros_like(y_dir)
t_dir = np.random.uniform(0, 0.5, (n_bc, 1)).astype(np.float32)
u_dir = np.zeros_like(x_dir)

# Neumann BC: 아래쪽 경계 (y=0)에서 ∂u/∂n=0 (단열)
x_neu = np.linspace(0, 0.5, n_bc).reshape(-1, 1).astype(np.float32)
y_neu = np.zeros_like(x_neu)
t_neu = np.random.uniform(0, 0.5, (n_bc, 1)).astype(np.float32)
nx_neu = np.zeros_like(x_neu)  # 법선 방향: (0, -1)
ny_neu = -np.ones_like(y_neu)

# Optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

# 훈련
print("\n훈련 시작...")
epochs = 7000
loss_history = []

for epoch in range(epochs):
    with tf.GradientTape() as tape:
        loss_pde = physics_loss(
            tf.constant(x_train), tf.constant(y_train), tf.constant(t_train))
        loss_ic = initial_condition_loss(
            tf.constant(x_ic), tf.constant(y_ic), tf.constant(t_ic))
        loss_dir = dirichlet_bc_loss(
            tf.constant(x_dir), tf.constant(y_dir), tf.constant(t_dir), tf.constant(u_dir))
        loss_neu = neumann_bc_loss(
            tf.constant(x_neu), tf.constant(y_neu), tf.constant(t_neu),
            tf.constant(nx_neu), tf.constant(ny_neu))
        
        total_loss = loss_pde + 15.0 * loss_ic + 10.0 * loss_dir + 10.0 * loss_neu
    
    gradients = tape.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    loss_history.append(total_loss.numpy())
    
    if (epoch + 1) % 1000 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss.numpy():.6f}")

print(f"최종 Loss: {loss_history[-1]:.6f}")

# 예측 및 시각화
n_plot = 60
x_plot, y_plot = np.meshgrid(np.linspace(0, 1, n_plot), np.linspace(0, 1, n_plot))
t_snapshots = [0.0, 0.1, 0.3, 0.5]

fig = plt.figure(figsize=(16, 10))

for idx, t_snap in enumerate(t_snapshots):
    x_flat = x_plot.ravel().reshape(-1, 1).astype(np.float32)
    y_flat = y_plot.ravel().reshape(-1, 1).astype(np.float32)
    t_flat = np.full_like(x_flat, t_snap)
    
    u_pred = model(np.column_stack([x_flat, y_flat, t_flat])).numpy().reshape(x_plot.shape)
    
    # L자 밖 영역 마스킹
    mask_plot = ~is_in_domain(x_plot, y_plot)
    u_pred[mask_plot] = np.nan
    
    # 2D 플롯
    ax = fig.add_subplot(2, 4, idx + 1)
    im = ax.contourf(x_plot, y_plot, u_pred, levels=20, cmap='hot')
    ax.set_xlabel('x', fontweight='bold')
    ax.set_ylabel('y', fontweight='bold')
    ax.set_title(f't = {t_snap:.1f}s', fontweight='bold')
    ax.set_aspect('equal')
    plt.colorbar(im, ax=ax, label='Temperature')
    
    # 경계 표시
    ax.plot([0, 0], [0, 0.5], 'b-', linewidth=3, label='Dirichlet (u=0)')
    ax.plot([0, 0.5], [0, 0], 'g-', linewidth=3, label='Neumann (∂u/∂n=0)')
    if idx == 0:
        ax.legend(fontsize=8)
    
    # 3D 플롯
    ax = fig.add_subplot(2, 4, idx + 5, projection='3d')
    surf = ax.plot_surface(x_plot, y_plot, u_pred, cmap='hot', alpha=0.9)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('u (Temperature)')
    ax.set_title(f't = {t_snap:.1f}s (3D)', fontweight='bold')
    ax.set_zlim([0, 1])

plt.suptitle('L-Shaped Domain with Mixed Boundary Conditions', 
             fontsize=16, fontweight='bold')
plt.tight_layout()

output_path = os.path.join(output_dir, '07_complex_boundary.png')
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\n그래프 저장: {output_path}")

print(f"\n요약:")
print(f"  - L자 모양 영역 (복잡한 기하학)")
print(f"  - Dirichlet BC (x=0): 고정 온도 u=0")
print(f"  - Neumann BC (y=0): 단열 경계 ∂u/∂n=0")
print(f"  - 초기 조건: 중앙에 열원")
