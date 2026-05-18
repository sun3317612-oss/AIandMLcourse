"""
05. Lorenz System with PINN (TensorFlow)
로렌츠 시스템 - 혼돈 역학을 PINN으로 풀기

Problem: Lorenz equations (coupled ODEs)
dx/dt = σ(y - x)
dy/dt = x(ρ - z) - y  
dz/dt = xy - βz

Parameters: σ=10, ρ=28, β=8/3
Initial conditions: (x₀, y₀, z₀) = (1, 1, 1)

학습 목표:
1. Coupled ODE systems
2. Chaotic dynamics
3. 3D trajectory visualization
4. Long-time integration challenges
"""

import numpy as np
import tensorflow as tf
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.font_manager as fm
from matplotlib.gridspec import GridSpec
import os
import time

# 출력 디렉토리 확인
output_dir = 'outputs'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print("="*70)
print("Lorenz System with PINN (TensorFlow)")
print("="*70)
print("Chaotic dynamics: σ=10, ρ=28, β=8/3")
print("Initial conditions: (1, 1, 1)")
print("="*70)

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

# ============================================================================
# Parameters
# ============================================================================

sigma = 10.0
rho = 28.0
beta = 8.0 / 3.0

t_min, t_max = 0.0, 20.0
x0, y0, z0 = 1.0, 1.0, 1.0

# ============================================================================
# PINN Model for 3 outputs
# ============================================================================

def create_lorenz_pinn():
    """PINN for Lorenz system - outputs (x, y, z)"""
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='tanh', input_shape=(1,)),
        tf.keras.layers.Dense(64, activation='tanh'),
        tf.keras.layers.Dense(64, activation='tanh'),
        tf.keras.layers.Dense(3, activation=None)  # 3 outputs: x, y, z
    ])
    return model

def compute_derivatives(model, t):
    """각 변수에 대한 시간 미분 계산"""
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(t)
        xyz = model(t, training=True)
        x = xyz[:, 0:1]
        y = xyz[:, 1:2]
        z = xyz[:, 2:3]
    
    dx_dt = tape.gradient(x, t)
    dy_dt = tape.gradient(y, t)
    dz_dt = tape.gradient(z, t)
    
    del tape
    return xyz, dx_dt, dy_dt, dz_dt

# ============================================================================
# Loss Functions
# ============================================================================

def compute_physics_loss(model, t_collocation, sigma, rho, beta):
    """
    Physics loss for Lorenz system:
    dx/dt = σ(y - x)
    dy/dt = x(ρ - z) - y
    dz/dt = xy - βz
    """
    xyz, dx_dt, dy_dt, dz_dt = compute_derivatives(model, t_collocation)
    
    x = xyz[:, 0:1]
    y = xyz[:, 1:2]
    z = xyz[:, 2:3]
    
    # Lorenz equations residuals
    residual_x = dx_dt - sigma * (y - x)
    residual_y = dy_dt - (x * (rho - z) - y)
    residual_z = dz_dt - (x * y - beta * z)
    
    physics_loss = (tf.reduce_mean(tf.square(residual_x)) + 
                    tf.reduce_mean(tf.square(residual_y)) +
                    tf.reduce_mean(tf.square(residual_z)))
    
    return physics_loss

def compute_initial_condition_loss(model, t_initial, x0, y0, z0):
    """Initial conditions: x(0)=x0, y(0)=y0, z(0)=z0"""
    xyz = model(t_initial, training=True)
    
    ic_loss = (tf.reduce_mean(tf.square(xyz[:, 0:1] - x0)) +
               tf.reduce_mean(tf.square(xyz[:, 1:2] - y0)) +
               tf.reduce_mean(tf.square(xyz[:, 2:3] - z0)))
    
    return ic_loss

def train_step(model, optimizer, t_collocation, t_initial, sigma, rho, beta, x0, y0, z0):
    """Training step"""
    with tf.GradientTape() as tape:
        physics_loss = compute_physics_loss(model, t_collocation, sigma, rho, beta)
        ic_loss = compute_initial_condition_loss(model, t_initial, x0, y0, z0)
        
        total_loss = physics_loss + 100.0 * ic_loss
    
    gradients = tape.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    return total_loss, physics_loss, ic_loss

# ============================================================================
# RK4 Solution for Comparison
# ============================================================================

print("\n1. Computing RK4 solution...")

def lorenz_ode(state, t, sigma, rho, beta):
    """Lorenz system for scipy.integrate"""
    x, y, z = state
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z
    return [dxdt, dydt, dzdt]

t_rk4 = np.linspace(t_min, t_max, 2000)
initial_state = [x0, y0, z0]
solution_rk4 = odeint(lorenz_ode, initial_state, t_rk4, 
                       args=(sigma, rho, beta))

print(f"   RK4 solution computed with {len(t_rk4)} points")

# ============================================================================
# Training PINN
# ============================================================================

print("\n2. Training PINN...")

n_collocation = 200
t_collocation = tf.convert_to_tensor(
    np.linspace(t_min, t_max, n_collocation).reshape(-1, 1),
    dtype=tf.float32
)
t_initial = tf.convert_to_tensor([[t_min]], dtype=tf.float32)

model = create_lorenz_pinn()
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

epochs = 20000
print_every = 3000

start_time = time.time()

history = {'total': [], 'physics': [], 'ic': []}

for epoch in range(epochs):
    total_loss, physics_loss, ic_loss = train_step(
        model, optimizer, t_collocation, t_initial, sigma, rho, beta, x0, y0, z0
    )
    
    history['total'].append(float(total_loss.numpy()))
    history['physics'].append(float(physics_loss.numpy()))
    history['ic'].append(float(ic_loss.numpy()))
    
    if (epoch + 1) % print_every == 0:
        print(f"   Epoch {epoch+1}/{epochs}: Loss={total_loss.numpy():.6f}")

elapsed = time.time() - start_time
print(f"\n   Training completed in {elapsed:.2f} seconds")

# ============================================================================
# Evaluation
# ============================================================================

print("\n3. Evaluating PINN solution...")

t_test = np.linspace(t_min, t_max, 1000)
t_test_tf = tf.convert_to_tensor(t_test.reshape(-1, 1), dtype=tf.float32)

xyz_pinn = model(t_test_tf, training=False).numpy()

x_pinn = xyz_pinn[:, 0]
y_pinn = xyz_pinn[:, 1]
z_pinn = xyz_pinn[:, 2]

# Interpolate RK4 for comparison
x_rk4_interp = np.interp(t_test, t_rk4, solution_rk4[:, 0])
y_rk4_interp = np.interp(t_test, t_rk4, solution_rk4[:, 1])
z_rk4_interp = np.interp(t_test, t_rk4, solution_rk4[:, 2])

# Error
error_x = np.sqrt(np.mean((x_pinn - x_rk4_interp)**2))
error_y = np.sqrt(np.mean((y_pinn - y_rk4_interp)**2))
error_z = np.sqrt(np.mean((z_pinn - z_rk4_interp)**2))

print(f"   L2 error - x: {error_x:.4f}, y: {error_y:.4f}, z: {error_z:.4f}")

# ============================================================================
# Visualization
# ============================================================================

print("\n4. Creating visualizations...")

# 3D Lorenz attractor
fig = plt.figure(figsize=(18, 6))

# (a) PINN
ax1 = fig.add_subplot(131, projection='3d')
ax1.plot(x_pinn, y_pinn, z_pinn, 'r-', linewidth=0.8, alpha=0.8)
ax1.scatter([x0], [y0], [z0], c='green', s=100, marker='o')
ax1.set_xlabel('X', fontsize=10)
ax1.set_ylabel('Y', fontsize=10)
ax1.set_zlabel('Z', fontsize=10)
ax1.set_title('(a) PINN Solution', fontsize=12, weight='bold')
ax1.view_init(elev=20, azim=45)

# (b) RK4
ax2 = fig.add_subplot(132, projection='3d')
ax2.plot(solution_rk4[:, 0], solution_rk4[:, 1], solution_rk4[:, 2], 
         'b-', linewidth=0.8, alpha=0.8)
ax2.scatter([x0], [y0], [z0], c='green', s=100, marker='o')
ax2.set_xlabel('X', fontsize=10)
ax2.set_ylabel('Y', fontsize=10)
ax2.set_zlabel('Z', fontsize=10)
ax2.set_title('(b) RK4 Solution', fontsize=12, weight='bold')
ax2.view_init(elev=20, azim=45)

# (c) Overlay
ax3 = fig.add_subplot(133, projection='3d')
ax3.plot(x_pinn, y_pinn, z_pinn, 'r-', linewidth=0.8, alpha=0.6, label='PINN')
ax3.plot(solution_rk4[:, 0], solution_rk4[:, 1], solution_rk4[:, 2], 
         'b--', linewidth=0.8, alpha=0.6, label='RK4')
ax3.scatter([x0], [y0], [z0], c='green', s=100, marker='o', label='IC')
ax3.set_xlabel('X', fontsize=10)
ax3.set_ylabel('Y', fontsize=10)
ax3.set_zlabel('Z', fontsize=10)
ax3.set_title('(c) Comparison', fontsize=12, weight='bold')
ax3.legend(fontsize=9)
ax3.view_init(elev=20, azim=45)

plt.tight_layout()
plt.savefig(f'{output_dir}/05_lorenz_3d.png', dpi=150, bbox_inches='tight')
print(f"   Saved: {output_dir}/05_lorenz_3d.png")

# Time series and phase space
fig2 = plt.figure(figsize=(15, 10))
gs = GridSpec(3, 2, figure=fig2, hspace=0.35, wspace=0.3)

# Time series
for idx, (var_pinn, var_rk4, name) in enumerate([
    (x_pinn, x_rk4_interp, 'X'),
    (y_pinn, y_rk4_interp, 'Y'),
    (z_pinn, z_rk4_interp, 'Z')
]):
    ax = fig2.add_subplot(gs[idx, 0])
    ax.plot(t_test, var_rk4, 'b-', linewidth=1.5, label='RK4', alpha=0.7)
    ax.plot(t_test, var_pinn, 'r--', linewidth=1.5, label='PINN')
    ax.set_xlabel('Time t', fontsize=11)
    ax.set_ylabel(name, fontsize=11)
    ax.set_title(f'({chr(97+idx)}) {name}(t) Time Series', fontsize=12, weight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

# Phase space projections
projections = [
    (x_pinn, y_pinn, x_rk4_interp, y_rk4_interp, 'X', 'Y'),
    (y_pinn, z_pinn, y_rk4_interp, z_rk4_interp, 'Y', 'Z'),
    (x_pinn, z_pinn, x_rk4_interp, z_rk4_interp, 'X', 'Z')
]

for idx, (v1_pinn, v2_pinn, v1_rk4, v2_rk4, name1, name2) in enumerate(projections):
    ax = fig2.add_subplot(gs[idx, 1])
    ax.plot(v1_rk4, v2_rk4, 'b-', linewidth=0.8, alpha=0.5, label='RK4')
    ax.plot(v1_pinn, v2_pinn, 'r-', linewidth=0.8, alpha=0.5, label='PINN')
    ax.scatter([x0 if name1=='X' else (y0 if name1=='Y' else z0)],
               [y0 if name2=='Y' else z0],
               c='green', s=100, marker='o', zorder=5)
    ax.set_xlabel(name1, fontsize=11)
    ax.set_ylabel(name2, fontsize=11)
    ax.set_title(f'({chr(100+idx)}) {name1}-{name2} Phase Space', 
                 fontsize=12, weight='bold')
    if idx == 0:
        ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

plt.savefig(f'{output_dir}/05_lorenz_analysis.png', dpi=150, bbox_inches='tight')
print(f"   Saved: {output_dir}/05_lorenz_analysis.png")

# ============================================================================
# Summary
# ============================================================================

print("\n" + "="*70)
print("Summary:")
print("="*70)
print(f"1. Lorenz System: σ={sigma}, ρ={rho}, β={beta:.3f}")
print(f"   Chaotic attractor with butterfly shape")
print(f"\n2. PINN Implementation:")
print(f"   - Network outputs 3 variables: (x, y, z)")
print(f"   - 3 coupled ODEs enforced by physics loss")
print(f"   - Training: {epochs} epochs in {elapsed:.1f}s")
print(f"\n3. Accuracy vs RK4:")
print(f"   - L2 error X: {error_x:.4f}")
print(f"   - L2 error Y: {error_y:.4f}")
print(f"   - L2 error Z: {error_z:.4f}")
print(f"\n4. Chaotic Dynamics:")
print(f"   - Sensitive to initial conditions")
print(f"   - Long-term prediction challenging")
print(f"   - PINN captures overall structure")
print(f"\n5. Challenges:")
print(f"   - Chaos makes long-time integration difficult")
print(f"   - May need more training for better accuracy")
print(f"   - Butterfly effect limits predictability")
print("="*70)

plt.show()
