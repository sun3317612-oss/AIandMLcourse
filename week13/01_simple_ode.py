"""
01. Simple ODE with PINN (TensorFlow)
Physics-Informed Neural Network로 가장 간단한 ODE 풀기 - TensorFlow 사용

Problem: dy/dt = -y, y(0) = 1
Analytical solution: y(x) = exp(-x)

PINN의 기본 개념:
- Neural network가 해 함수 y(x)를 근사
- Physics loss: ODE residual (dy/dx + y)^2
- Initial condition loss: (y(0) - 1)^2
- Automatic differentiation으로 dy/dx 계산

학습 목표:
1. PINN의 기본 구조 이해
2. TensorFlow GradientTape 사용법
3. Physics-informed loss function
4. Training process 관찰
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
print("Simple ODE with Physics-Informed Neural Network")
print("="*70)
print("Problem: dy/dx = -y, y(0) = 1")
print("Analytical: y(x) = exp(-x)")
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
# PINN Model Definition
# ============================================================================

def create_pinn_model():
    """
    PINN을 위한 Neural Network 생성
    
    Architecture:
    - Input: x (1D)
    - Hidden layers: 3 layers × 32 neurons
    - Activation: tanh (smooth derivatives)
    - Output: y(x)
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(32, activation='tanh', input_shape=(1,)),
        tf.keras.layers.Dense(32, activation='tanh'),
        tf.keras.layers.Dense(32, activation='tanh'),
        tf.keras.layers.Dense(1, activation=None)  # Linear output
    ])
    return model

# ============================================================================
# Physics-Informed Loss Function
# ============================================================================

def compute_physics_loss(model, x_collocation):
    """
    Physics loss를 계산합니다: (dy/dx + y)^2
    
    ODE: dy/dx = -y
    Residual: dy/dx + y = 0
    
    Parameters:
    -----------
    model : tf.keras.Model
        PINN 모델
    x_collocation : tf.Tensor
        Collocation points (물리 법칙을 만족해야 하는 점들)
    
    Returns:
    --------
    physics_loss : float
        Physics residual의 MSE
    """
    with tf.GradientTape() as tape:
        tape.watch(x_collocation)
        y_pred = model(x_collocation, training=True)
    
    # Automatic differentiation: dy/dx
    dy_dx = tape.gradient(y_pred, x_collocation)
    
    # ODE residual: dy/dx - (-y) = dy/dx + y
    residual = dy_dx + y_pred
    
    # Mean squared error
    physics_loss = tf.reduce_mean(tf.square(residual))
    
    return physics_loss

def compute_initial_condition_loss(model, x_initial, y_initial):
    """
    Initial condition loss를 계산합니다: (y(0) - 1)^2
    
    Parameters:
    -----------
    model : tf.keras.Model
        PINN 모델
    x_initial : tf.Tensor
        Initial point (x=0)
    y_initial : tf.Tensor
        Initial value (y=1)
    
    Returns:
    --------
    ic_loss : float
        Initial condition MSE
    """
    y_pred = model(x_initial, training=True)
    ic_loss = tf.reduce_mean(tf.square(y_pred - y_initial))
    return ic_loss

@tf.function
def train_step(model, optimizer, x_collocation, x_initial, y_initial, w_physics=1.0, w_ic=100.0):
    """
    하나의 training step을 수행합니다.
    
    Total Loss = w_physics * L_physics + w_ic * L_IC
    
    Parameters:
    -----------
    w_physics : float
        Physics loss weight
    w_ic : float
        Initial condition loss weight (usually larger)
    
    Returns:
    --------
    total_loss, physics_loss, ic_loss
    """
    with tf.GradientTape() as tape:
        # Physics loss
        physics_loss = compute_physics_loss(model, x_collocation)
        
        # Initial condition loss
        ic_loss = compute_initial_condition_loss(model, x_initial, y_initial)
        
        # Total loss
        total_loss = w_physics * physics_loss + w_ic * ic_loss
    
    # Compute gradients and update weights
    gradients = tape.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    return total_loss, physics_loss, ic_loss

# ============================================================================
# Training
# ============================================================================

print("\n1. Setting up PINN...")

# Domain: x ∈ [0, 5]
x_min, x_max = 0.0, 5.0
n_collocation = 100

# Collocation points (물리 법칙을 만족해야 하는 점들)
x_collocation = tf.convert_to_tensor(
    np.linspace(x_min, x_max, n_collocation).reshape(-1, 1),
    dtype=tf.float32
)

# Initial condition: y(0) = 1
x_initial = tf.convert_to_tensor([[0.0]], dtype=tf.float32)
y_initial = tf.convert_to_tensor([[1.0]], dtype=tf.float32)

# Create model
model = create_pinn_model()
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

print(f"   Collocation points: {n_collocation}")
print(f"   Domain: [{x_min}, {x_max}]")
print(f"   Model parameters: {model.count_params()}")

# Training
print("\n2. Training PINN...")
start_time = time.time()

epochs = 10000
print_every = 1000

history = {
    'total_loss': [],
    'physics_loss': [],
    'ic_loss': []
}

for epoch in range(epochs):
    total_loss, physics_loss, ic_loss = train_step(
        model, optimizer, x_collocation, x_initial, y_initial,
        w_physics=1.0, w_ic=100.0
    )
    
    history['total_loss'].append(float(total_loss))
    history['physics_loss'].append(float(physics_loss))
    history['ic_loss'].append(float(ic_loss))
    
    if (epoch + 1) % print_every == 0:
        print(f"   Epoch {epoch+1}/{epochs}: "
              f"Total={total_loss:.6f}, "
              f"Physics={physics_loss:.6f}, "
              f"IC={ic_loss:.6f}")

elapsed = time.time() - start_time
print(f"\n   Training completed in {elapsed:.2f} seconds.")

# ============================================================================
# Evaluation and Comparison
# ============================================================================

print("\n3. Evaluating PINN solution...")

# Test points
x_test = np.linspace(x_min, x_max, 200).reshape(-1, 1)
x_test_tf = tf.convert_to_tensor(x_test, dtype=tf.float32)

# PINN prediction
y_pinn = model(x_test_tf, training=False).numpy()

# Analytical solution
y_exact = np.exp(-x_test)

# Error analysis
abs_error = np.abs(y_pinn - y_exact)
rel_error = abs_error / (np.abs(y_exact) + 1e-10)

l2_error = np.sqrt(np.mean((y_pinn - y_exact)**2))
max_error = np.max(abs_error)

print(f"   L2 error: {l2_error:.6f}")
print(f"   Max absolute error: {max_error:.6f}")
print(f"   Mean relative error: {np.mean(rel_error):.6f}")

# ============================================================================
# Visualization
# ============================================================================

print("\n4. Creating visualizations...")

fig = plt.figure(figsize=(15, 10))
gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

# (a) Training history
ax1 = fig.add_subplot(gs[0, 0])
epochs_arr = np.arange(1, epochs + 1)
ax1.semilogy(epochs_arr, history['total_loss'], 'b-', linewidth=2, label='Total Loss')
ax1.semilogy(epochs_arr, history['physics_loss'], 'g--', linewidth=1.5, label='Physics Loss')
ax1.semilogy(epochs_arr, history['ic_loss'], 'r--', linewidth=1.5, label='IC Loss')
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Loss (log scale)', fontsize=12)
ax1.set_title('(a) Training History', fontsize=13, weight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3, which='both')

# (b) Solution comparison
ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(x_test, y_exact, 'b-', linewidth=2.5, label='Exact: y=exp(-x)')
ax2.plot(x_test, y_pinn, 'r--', linewidth=2, label='PINN prediction')
ax2.scatter([0], [1], c='green', s=100, marker='o', zorder=5, 
            label='Initial condition')
ax2.set_xlabel('x', fontsize=12)
ax2.set_ylabel('y(x)', fontsize=12)
ax2.set_title('(b) Solution Comparison', fontsize=13, weight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

# (c) Absolute error
ax3 = fig.add_subplot(gs[1, 0])
ax3.plot(x_test, abs_error, 'r-', linewidth=2)
ax3.fill_between(x_test.flatten(), 0, abs_error.flatten(), alpha=0.3, color='red')
ax3.set_xlabel('x', fontsize=12)
ax3.set_ylabel('|y_PINN - y_exact|', fontsize=12)
ax3.set_title(f'(c) Absolute Error (L2={l2_error:.6f})', fontsize=13, weight='bold')
ax3.grid(True, alpha=0.3)

# (d) Pointwise comparison table
ax4 = fig.add_subplot(gs[1, 1])
ax4.axis('off')

# Select some points for table
x_samples = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
table_data = [['x', 'PINN', 'Exact', 'Error']]

for x_val in x_samples:
    idx = np.argmin(np.abs(x_test.flatten() - x_val))
    y_p = y_pinn[idx, 0]
    y_e = y_exact[idx, 0]
    err = abs(y_p - y_e)
    table_data.append([f'{x_val:.1f}', f'{y_p:.6f}', f'{y_e:.6f}', f'{err:.2e}'])

table = ax4.table(cellText=table_data, cellLoc='center', loc='center',
                  colWidths=[0.2, 0.3, 0.3, 0.2])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2)

# Header formatting
for i in range(4):
    table[(0, i)].set_facecolor('#4CAF50')
    table[(0, i)].set_text_props(weight='bold', color='white')

ax4.set_title('(d) Pointwise Comparison', fontsize=13, weight='bold', pad=20)

plt.savefig(f'{output_dir}/01_simple_ode_pinn.png', dpi=150, bbox_inches='tight')
print(f"   Saved: {output_dir}/01_simple_ode_pinn.png")

# ============================================================================
# Verify ODE Residual
# ============================================================================

print("\n5. Verifying ODE residual...")

# Compute dy/dx using automatic differentiation
with tf.GradientTape() as tape:
    tape.watch(x_test_tf)
    y_pred_tf = model(x_test_tf, training=False)

dy_dx_pinn = tape.gradient(y_pred_tf, x_test_tf).numpy()

# ODE residual: dy/dx + y
residual = dy_dx_pinn + y_pinn

fig2, axes = plt.subplots(1, 3, figsize=(15, 4))

# (a) y(x)
axes[0].plot(x_test, y_pinn, 'b-', linewidth=2, label='y(x)')
axes[0].set_xlabel('x', fontsize=11)
axes[0].set_ylabel('y', fontsize=11)
axes[0].set_title('(a) Solution y(x)', fontsize=12, weight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# (b) dy/dx
axes[1].plot(x_test, dy_dx_pinn, 'g-', linewidth=2, label='dy/dx (PINN)')
axes[1].plot(x_test, -y_exact, 'r--', linewidth=1.5, label='-y (exact)')
axes[1].set_xlabel('x', fontsize=11)
axes[1].set_ylabel('dy/dx', fontsize=11)
axes[1].set_title('(b) First Derivative', fontsize=12, weight='bold')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# (c) ODE residual
axes[2].plot(x_test, residual, 'm-', linewidth=2)
axes[2].axhline(y=0, color='k', linestyle='--', alpha=0.5)
axes[2].set_xlabel('x', fontsize=11)
axes[2].set_ylabel('dy/dx + y', fontsize=11)
axes[2].set_title('(c) ODE Residual (should be ~ 0)', fontsize=12, weight='bold')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{output_dir}/01_ode_residual.png', dpi=150, bbox_inches='tight')
print(f"   Saved: {output_dir}/01_ode_residual.png")

print(f"   Mean |residual|: {np.mean(np.abs(residual)):.6f}")
print(f"   Max |residual|: {np.max(np.abs(residual)):.6f}")

# ============================================================================
# Summary
# ============================================================================

print("\n" + "="*70)
print("Summary:")
print("="*70)
print(f"1. PINN Architecture:")
print(f"   - Input: x (1D)")
print(f"   - Hidden: 3 layers × 32 neurons (tanh)")
print(f"   - Output: y(x)")
print(f"   - Total parameters: {model.count_params()}")
print(f"\n2. Training:")
print(f"   - Epochs: {epochs}")
print(f"   - Time: {elapsed:.2f} seconds")
print(f"   - Final loss: {history['total_loss'][-1]:.6e}")
print(f"\n3. Accuracy:")
print(f"   - L2 error: {l2_error:.6f}")
print(f"   - Max error: {max_error:.6f}")
print(f"   - Mean ODE residual: {np.mean(np.abs(residual)):.6e}")
print(f"\n4. Key Concepts:")
print(f"   - Physics loss enforces ODE: dy/dx = -y")
print(f"   - IC loss enforces y(0) = 1")
print(f"   - Automatic differentiation computes dy/dx")
print(f"   - No mesh needed, only collocation points!")
print("="*70)

plt.show()

