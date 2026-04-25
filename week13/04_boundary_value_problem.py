"""
04. Boundary Value Problem with PINN (TensorFlow)
경계값 문제를 PINN으로 풀기 - TensorFlow 사용

Problem: d²y/dx² = -x
Boundary conditions: y(0) = 0, y(1) = 0
Analytical solution: y(x) = -(x³ - x)/6

BVP vs IVP:
- IVP: 초기 조건 (한 점에서 시작)
- BVP: 경계 조건 (양 끝점 조건)

학습 목표:
1. TensorFlow로 PINN 구현
2. Boundary Value Problem
3. Two-point boundary conditions
4. Comparison with Finite Difference
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.gridspec import GridSpec
import os
import time

# 출력 디렉토리 확인
output_dir = 'outputs'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print("="*70)
print("Boundary Value Problem with PINN (TensorFlow)")
print("="*70)
print("Problem: d²y/dx² = -x")
print("Boundary conditions: y(0) = 0, y(1) = 0")
print("Analytical: y(x) = -(x³ - x)/6")
print("="*70)

# 한글 폰트 설정
def set_korean_font():
    """한글 폰트를 설정합니다."""
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
# PINN Model in TensorFlow
# ============================================================================

def create_pinn_model():
    """PINN 모델 생성"""
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(32, activation='tanh', input_shape=(1,)),
        tf.keras.layers.Dense(32, activation='tanh'),
        tf.keras.layers.Dense(32, activation='tanh'),
        tf.keras.layers.Dense(1, activation=None)
    ])
    return model

def compute_derivatives(model, x):
    """1차 및 2차 미분 계산"""
    with tf.GradientTape(persistent=True) as tape2:
        tape2.watch(x)
        with tf.GradientTape(persistent=True) as tape1:
            tape1.watch(x)
            y = model(x, training=True)
        dy_dx = tape1.gradient(y, x)
    d2y_dx2 = tape2.gradient(dy_dx, x)
    
    del tape1, tape2
    return y, dy_dx, d2y_dx2

# ============================================================================
# Loss Functions
# ============================================================================

def compute_physics_loss(model, x_collocation):
    """
    Physics loss: d²y/dx² - (-x) = 0
    즉, d²y/dx² = -x
    """
    y, dy_dx, d2y_dx2 = compute_derivatives(model, x_collocation)
    
    # Residual: d²y/dx² + x = 0
    residual = d2y_dx2 + x_collocation
    
    physics_loss = tf.reduce_mean(tf.square(residual))
    
    return physics_loss

def compute_boundary_loss(model, x_left, x_right, y_left=0.0, y_right=0.0):
    """
    Boundary conditions loss:
    - y(0) = 0
    - y(1) = 0
    """
    y_pred_left = model(x_left, training=True)
    y_pred_right = model(x_right, training=True)
    
    bc_left_loss = tf.reduce_mean(tf.square(y_pred_left - y_left))
    bc_right_loss = tf.reduce_mean(tf.square(y_pred_right - y_right))
    
    boundary_loss = bc_left_loss + bc_right_loss
    
    return boundary_loss

def train_step(model, optimizer, x_collocation, x_left, x_right, w_physics, w_boundary):
    """Training step"""
    with tf.GradientTape() as tape:
        # Compute losses
        physics_loss = compute_physics_loss(model, x_collocation)
        boundary_loss = compute_boundary_loss(model, x_left, x_right)
        
        total_loss = w_physics * physics_loss + w_boundary * boundary_loss
    
    gradients = tape.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    return total_loss, physics_loss, boundary_loss

# ============================================================================
# Training
# ============================================================================

print("\n1. Setting up PINN...")

# Domain
x_min, x_max = 0.0, 1.0
n_collocation = 100

# Collocation points
x_collocation = tf.convert_to_tensor(
    np.linspace(x_min, x_max, n_collocation).reshape(-1, 1),
    dtype=tf.float32
)

# Boundary points
x_left = tf.convert_to_tensor([[x_min]], dtype=tf.float32)
x_right = tf.convert_to_tensor([[x_max]], dtype=tf.float32)

# Create model
model = create_pinn_model()
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

print(f"   Collocation points: {n_collocation}")
print(f"   Domain: [{x_min}, {x_max}]")
print(f"   Model parameters: {sum([tf.size(v).numpy() for v in model.trainable_variables])}")

# Training
print("\n2. Training PINN...")
start_time = time.time()

epochs = 15000
print_every = 2000

w_physics = 1.0
w_boundary = 100.0

history = {'total': [], 'physics': [], 'boundary': []}

for epoch in range(epochs):
    total_loss, physics_loss, boundary_loss = train_step(
        model, optimizer, x_collocation, x_left, x_right, w_physics, w_boundary
    )
    
    # Record history
    history['total'].append(float(total_loss.numpy()))
    history['physics'].append(float(physics_loss.numpy()))
    history['boundary'].append(float(boundary_loss.numpy()))
    
    if (epoch + 1) % print_every == 0:
        print(f"   Epoch {epoch+1}/{epochs}: "
              f"Total={total_loss.numpy():.6f}, "
              f"Physics={physics_loss.numpy():.6f}, "
              f"Boundary={boundary_loss.numpy():.6f}")

elapsed = time.time() - start_time
print(f"\n   Training completed in {elapsed:.2f} seconds.")

# ============================================================================
# Finite Difference Method for Comparison
# ============================================================================

print("\n3. Computing Finite Difference solution...")

def solve_bvp_finite_difference(n_points=101):
    """
    Finite Difference Method으로 BVP 풀기
    d²y/dx² = -x, y(0)=0, y(1)=0
    """
    x = np.linspace(0, 1, n_points)
    dx = x[1] - x[0]
    
    # Coefficient matrix for d²y/dx²
    A = np.zeros((n_points, n_points))
    b = np.zeros(n_points)
    
    # Boundary conditions
    A[0, 0] = 1.0
    b[0] = 0.0  # y(0) = 0
    
    A[-1, -1] = 1.0
    b[-1] = 0.0  # y(1) = 0
    
    # Interior points: d²y/dx² = (y_{i+1} - 2y_i + y_{i-1})/dx²
    for i in range(1, n_points - 1):
        A[i, i-1] = 1.0
        A[i, i] = -2.0
        A[i, i+1] = 1.0
        b[i] = -x[i] * dx**2  # RHS: -x
    
    # Solve linear system
    y = np.linalg.solve(A, b)
    
    return x, y

x_fd, y_fd = solve_bvp_finite_difference(n_points=101)

# ============================================================================
# Evaluation
# ============================================================================

print("\n4. Evaluating solutions...")

# Test points
x_test = np.linspace(x_min, x_max, 200)
x_test_tf = tf.convert_to_tensor(x_test.reshape(-1, 1), dtype=tf.float32)

# PINN prediction
y_pinn = model(x_test_tf, training=False).numpy()

# Analytical solution
y_exact = -(x_test**3 - x_test) / 6

# Error analysis
l2_error_pinn = np.sqrt(np.mean((y_pinn.flatten() - y_exact)**2))
l2_error_fd = np.sqrt(np.mean((y_fd - np.interp(x_fd, x_test, y_exact))**2))

max_error_pinn = np.max(np.abs(y_pinn.flatten() - y_exact))

print(f"   PINN L2 error: {l2_error_pinn:.6f}")
print(f"   FD L2 error: {l2_error_fd:.6f}")
print(f"   PINN Max error: {max_error_pinn:.6f}")

# ============================================================================
# Visualization
# ============================================================================

print("\n5. Creating visualizations...")

fig = plt.figure(figsize=(15, 10))
gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

# (a) Training history
ax1 = fig.add_subplot(gs[0, 0])
epochs_arr = np.arange(1, epochs + 1)
ax1.semilogy(epochs_arr, history['total'], 'b-', linewidth=2, label='Total')
ax1.semilogy(epochs_arr, history['physics'], 'g--', linewidth=1.5, label='Physics')
ax1.semilogy(epochs_arr, history['boundary'], 'r--', linewidth=1.5, label='Boundary')
ax1.set_xlabel('Epoch', fontsize=11)
ax1.set_ylabel('Loss (log scale)', fontsize=11)
ax1.set_title('(a) Training History (TensorFlow)', fontsize=12, weight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3, which='both')

# (b) Solution comparison
ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(x_test, y_exact, 'b-', linewidth=2.5, label='Exact: -(x³-x)/6')
ax2.plot(x_test, y_pinn, 'r--', linewidth=2, label='PINN (TensorFlow)')
ax2.plot(x_fd, y_fd, 'g:', linewidth=2, label='Finite Difference')
ax2.scatter([0, 1], [0, 0], c='orange', s=100, marker='o', zorder=5, 
            label='Boundary conditions')
ax2.set_xlabel('x', fontsize=11)
ax2.set_ylabel('y(x)', fontsize=11)
ax2.set_title('(b) Solution Comparison', fontsize=12, weight='bold')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

# (c) Error comparison
ax3 = fig.add_subplot(gs[1, 0])
error_pinn = np.abs(y_pinn.flatten() - y_exact)
ax3.plot(x_test, error_pinn, 'r-', linewidth=2, label='PINN error')
ax3.fill_between(x_test, 0, error_pinn, alpha=0.3, color='red')
ax3.set_xlabel('x', fontsize=11)
ax3.set_ylabel('|y_pred - y_exact|', fontsize=11)
ax3.set_title(f'(c) PINN Error (L2={l2_error_pinn:.6f})', fontsize=12, weight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)

# (d) Second derivative verification
ax4 = fig.add_subplot(gs[1, 1])

# Compute d²y/dx² from PINN
y_test, dy_dx, d2y_dx2 = compute_derivatives(model, x_test_tf)
d2y_dx2_np = d2y_dx2.numpy()

# Should equal -x
ax4.plot(x_test, d2y_dx2_np, 'r-', linewidth=2, label='d²y/dx² (PINN)')
ax4.plot(x_test, -x_test, 'b--', linewidth=2, label='-x (exact)')
ax4.set_xlabel('x', fontsize=11)
ax4.set_ylabel('d²y/dx²', fontsize=11)
ax4.set_title('(d) Second Derivative Verification', fontsize=12, weight='bold')
ax4.legend(fontsize=10)
ax4.grid(True, alpha=0.3)

plt.savefig(f'{output_dir}/04_boundary_value_problem.png', dpi=150, bbox_inches='tight')
print(f"   Saved: {output_dir}/04_boundary_value_problem.png")

# ============================================================================
# Summary
# ============================================================================

print("\n" + "="*70)
print("Summary:")
print("="*70)
print(f"1. Problem: d²y/dx² = -x, y(0)=0, y(1)=0")
print(f"   Solution: y(x) = -(x³ - x)/6")
print(f"\n2. TensorFlow Implementation:")
print(f"   - Model: Sequential with 3 hidden layers")
print(f"   - Autograd: tf.GradientTape")
print(f"   - Device: Automatic GPU/CPU selection")
print(f"\n3. Accuracy:")
print(f"   - PINN L2 error: {l2_error_pinn:.6f}")
print(f"   - FD L2 error: {l2_error_fd:.6f}")
print(f"   - PINN is comparable to traditional method")
print(f"\n4. BVP vs IVP:")
print(f"   - BVP: Boundary conditions at multiple points")
print(f"   - IVP: Initial conditions at one point")
print(f"   - Both handled by PINN with appropriate loss")
print(f"\n5. TensorFlow Advantages:")
print(f"   - Production-ready deployment")
print(f"   - Automatic device management")
print(f"   - Powerful optimization tools")
print("="*70)

plt.show()
