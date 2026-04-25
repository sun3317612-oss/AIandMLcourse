"""
05. Burgers Equation (버거스 방정식)
∂u/∂t + u ∂u/∂x = ν ∂²u/∂x²

비선형 PDE의 대표적인 예제
초기조건: u(x, 0) = -sin(πx)
경계조건: u(-1, t) = u(1, t) = 0
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
print("05. Burgers Equation with PINN")
print("="*70)

# 물리 상수
nu = 0.01 / np.pi  # 점성 계수

# 신경망 모델
def build_model():
    return tf.keras.Sequential([
        tf.keras.layers.Dense(40, activation='tanh', input_shape=(2,)),
        tf.keras.layers.Dense(40, activation='tanh'),
        tf.keras.layers.Dense(40, activation='tanh'),
        tf.keras.layers.Dense(40, activation='tanh'),
        tf.keras.layers.Dense(1)
    ])

model = build_model()

# PDE 잔차 (비선형!)
@tf.function
def pde_residual(x, t):
    with tf.GradientTape(persistent=True) as tape:
        tape.watch([x, t])
        xt = tf.concat([x, t], axis=1)
        u = model(xt)
        
        du_dt = tape.gradient(u, t)
        du_dx = tape.gradient(u, x)
    
    d2u_dx2 = tape.gradient(du_dx, x)
    del tape
    
    # Burgers equation: ∂u/∂t + u ∂u/∂x - ν ∂²u/∂x² = 0
    # 비선형 항: u ∂u/∂x
    residual = du_dt + u * du_dx - nu * d2u_dx2
    return residual

# 손실 함수
@tf.function
def physics_loss(x_phys, t_phys):
    residual = pde_residual(x_phys, t_phys)
    return tf.reduce_mean(tf.square(residual))

@tf.function
def initial_condition_loss(x_ic, t_ic):
    xt = tf.concat([x_ic, t_ic], axis=1)
    u_pred = model(xt)
    u_true = -tf.sin(np.pi * x_ic)  # u(x, 0) = -sin(πx)
    return tf.reduce_mean(tf.square(u_pred - u_true))

@tf.function
def boundary_condition_loss(x_bc, t_bc):
    xt = tf.concat([x_bc, t_bc], axis=1)
    u_pred = model(xt)
    return tf.reduce_mean(tf.square(u_pred))

# 훈련 데이터
n_train = 4000
x_train = np.random.uniform(-1, 1, (n_train, 1)).astype(np.float32)
t_train = np.random.uniform(0, 1, (n_train, 1)).astype(np.float32)

# 초기조건
n_ic = 100
x_ic = np.linspace(-1, 1, n_ic).reshape(-1, 1).astype(np.float32)
t_ic = np.zeros((n_ic, 1), dtype=np.float32)

# 경계조건
n_bc = 100
t_bc = np.linspace(0, 1, n_bc).reshape(-1, 1).astype(np.float32)
x_bc_left = np.full((n_bc, 1), -1.0, dtype=np.float32)
x_bc_right = np.full((n_bc, 1), 1.0, dtype=np.float32)
x_bc = np.vstack([x_bc_left, x_bc_right])
t_bc = np.vstack([t_bc, t_bc])

# Optimizer (비선형이므로 학습률 조정)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

# 훈련
print("\n훈련 시작 (비선형 PDE이므로 시간이 더 걸릴 수 있습니다)...")
epochs = 10000
loss_history = []

for epoch in range(epochs):
    with tf.GradientTape() as tape:
        loss_pde = physics_loss(tf.constant(x_train), tf.constant(t_train))
        loss_ic = initial_condition_loss(tf.constant(x_ic), tf.constant(t_ic))
        loss_bc = boundary_condition_loss(tf.constant(x_bc), tf.constant(t_bc))
        
        total_loss = loss_pde + 20.0 * loss_ic + 10.0 * loss_bc
    
    gradients = tape.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    loss_history.append(total_loss.numpy())
    
    if (epoch + 1) % 1000 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss.numpy():.6f}")

print(f"최종 Loss: {loss_history[-1]:.6f}")

# 예측
x_test = np.linspace(-1, 1, 200)
t_test = np.linspace(0, 1, 200)
X, T = np.meshgrid(x_test, t_test)

xt_test = np.column_stack([X.ravel(), T.ravel()]).astype(np.float32)
u_pred = model(xt_test).numpy().reshape(X.shape)

# 시각화
fig = plt.figure(figsize=(16, 10))

# 1. PINN 예측 (3D)
ax1 = fig.add_subplot(2, 3, 1, projection='3d')
surf1 = ax1.plot_surface(X, T, u_pred, cmap='coolwarm', alpha=0.8)
ax1.set_xlabel('x', fontweight='bold')
ax1.set_ylabel('t', fontweight='bold')
ax1.set_zlabel('u(x,t)', fontweight='bold')
ax1.set_title('PINN Solution', fontweight='bold')
fig.colorbar(surf1, ax=ax1, shrink=0.5)

# 2. 히트맵
ax2 = fig.add_subplot(2, 3, 2)
im = ax2.imshow(u_pred, extent=[-1, 1, 0, 1], origin='lower', aspect='auto', cmap='coolwarm')
ax2.set_xlabel('x', fontweight='bold')
ax2.set_ylabel('t', fontweight='bold')
ax2.set_title('Solution Heatmap', fontweight='bold')
fig.colorbar(im, ax=ax2)

# 3. 시간별 스냅샷
ax3 = fig.add_subplot(2, 3, 3)
for t_snap in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
    idx = int(t_snap * 199)
    ax3.plot(x_test, u_pred[idx, :], label=f't={t_snap:.1f}', linewidth=2)
ax3.set_xlabel('x', fontweight='bold')
ax3.set_ylabel('u(x,t)', fontweight='bold')
ax3.set_title('Time Evolution', fontweight='bold')
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3)

# 4. 초기조건 vs 예측
ax4 = fig.add_subplot(2, 3, 4)
x_ic_plot = np.linspace(-1, 1, 100)
u_ic_true = -np.sin(np.pi * x_ic_plot)
u_ic_pred = model(np.column_stack([x_ic_plot, np.zeros_like(x_ic_plot)]).astype(np.float32)).numpy()
ax4.plot(x_ic_plot, u_ic_true, 'b-', linewidth=2, label='True IC')
ax4.plot(x_ic_plot, u_ic_pred, 'r--', linewidth=2, label='PINN at t=0')
ax4.set_xlabel('x', fontweight='bold')
ax4.set_ylabel('u(x,0)', fontweight='bold')
ax4.set_title('Initial Condition Check', fontweight='bold')
ax4.legend(fontsize=10)
ax4.grid(True, alpha=0.3)

# 5. 손실 곡선
ax5 = fig.add_subplot(2, 3, 5)
ax5.semilogy(loss_history, 'purple', linewidth=2)
ax5.set_xlabel('Epoch', fontweight='bold')
ax5.set_ylabel('Loss (log scale)', fontweight='bold')
ax5.set_title('Training Loss', fontweight='bold')
ax5.grid(True, alpha=0.3)

# 6. x=0에서의 시간 변화
ax6 = fig.add_subplot(2, 3, 6)
x_idx = 100  # x=0 근처
ax6.plot(t_test, u_pred[:, x_idx], 'g-', linewidth=2, label='u(0, t)')
ax6.set_xlabel('t', fontweight='bold')
ax6.set_ylabel('u(0,t)', fontweight='bold')
ax6.set_title('Time Evolution at x=0', fontweight='bold')
ax6.legend(fontsize=10)
ax6.grid(True, alpha=0.3)

plt.suptitle('Burgers Equation: ∂u/∂t + u∂u/∂x = ν∂²u/∂x² (Nonlinear PDE)', 
             fontsize=16, fontweight='bold')
plt.tight_layout()

output_path = os.path.join(output_dir, '05_burgers_equation.png')
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\n그래프 저장: {output_path}")

print(f"\n성능:")
print(f"  최종 Loss: {loss_history[-1]:.6e}")
print(f"  비선형 항 (u∂u/∂x)으로 인해 복잡한 동역학이 나타남")
