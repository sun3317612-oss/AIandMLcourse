"""
02. Harmonic Oscillator with PINN (TensorFlow)
단순 조화 진동자를 PINN으로 풀기 - TensorFlow 사용

Problem: d²y/dt² + ω²y = 0  (ω = 2)
Initial conditions: y(0) = 1, dy/dt(0) = 0
Analytical solution: y(t) = cos(ωt)

PINN 확장:
- 2차 미분 방정식
- 2개의 초기 조건
- 에너지 보존 검증
- RK4와 비교

학습 목표:
1. Higher-order derivatives in PINN
2. Multiple initial conditions
3. Physics conservation laws
4. Comparison with traditional methods
"""

import numpy as np
import tensorflow as tf
from scipy.integrate import odeint
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
print("Harmonic Oscillator with Physics-Informed Neural Network")
print("="*70)
print("Problem: d²y/dt² + ω²y = 0")
print("Initial conditions: y(0) = 1, dy/dt(0) = 0")
print("Analytical: y(t) = cos(ωt)")
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
# Parameters
# ============================================================================

omega = 2.0  # Angular frequency
t_min, t_max = 0.0, 10.0
n_collocation = 150

print(f"\nParameters: ω = {omega}")

# ============================================================================
# PINN Model
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

# ============================================================================
# Physics-Informed Loss Functions
# ============================================================================

def compute_derivatives(model, t):
    """
    1차 및 2차 미분 계산
    
    Returns:
    --------
    y, dy_dt, d2y_dt2
    """
    with tf.GradientTape(persistent=True) as tape2:
        tape2.watch(t)
        with tf.GradientTape(persistent=True) as tape1:
            tape1.watch(t)
            y = model(t, training=True)
        dy_dt = tape1.gradient(y, t)
    d2y_dt2 = tape2.gradient(dy_dt, t)
    
    del tape1
    del tape2
    
    return y, dy_dt, d2y_dt2

def compute_physics_loss(model, t_collocation, omega):
    """
    Physics loss: d²y/dt² + ω²y = 0
    """
    y, dy_dt, d2y_dt2 = compute_derivatives(model, t_collocation)
    
    # ODE residual
    residual = d2y_dt2 + omega**2 * y
    
    physics_loss = tf.reduce_mean(tf.square(residual))
    return physics_loss

def compute_initial_conditions_loss(model, t_initial):
    """
    Initial conditions loss:
    - y(0) = 1
    - dy/dt(0) = 0
    """
    y, dy_dt, _ = compute_derivatives(model, t_initial)
    
    # y(0) = 1
    ic1_loss = tf.reduce_mean(tf.square(y - 1.0))
    
    # dy/dt(0) = 0
    ic2_loss = tf.reduce_mean(tf.square(dy_dt - 0.0))
    
    return ic1_loss, ic2_loss

def train_step(model, optimizer, t_collocation, t_initial, omega,
               w_physics=1.0, w_ic1=100.0, w_ic2=100.0):
    """Training step"""
    with tf.GradientTape() as tape:
        physics_loss = compute_physics_loss(model, t_collocation, omega)
        ic1_loss, ic2_loss = compute_initial_conditions_loss(model, t_initial)
        
        total_loss = (w_physics * physics_loss + 
                      w_ic1 * ic1_loss + 
                      w_ic2 * ic2_loss)
    
    gradients = tape.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    return total_loss, physics_loss, ic1_loss, ic2_loss

# ============================================================================
# Training
# ============================================================================

print("\n1. Training PINN...")

# Collocation points
t_collocation = tf.convert_to_tensor(
    np.linspace(t_min, t_max, n_collocation).reshape(-1, 1),
    dtype=tf.float32
)

# Initial point
t_initial = tf.convert_to_tensor([[0.0]], dtype=tf.float32)

# Create model
model = create_pinn_model()
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

# Training
start_time = time.time()
epochs = 15000
print_every = 2000

history = {'total': [], 'physics': [], 'ic1': [], 'ic2': []}

for epoch in range(epochs):
    total_loss, physics_loss, ic1_loss, ic2_loss = train_step(
        model, optimizer, t_collocation, t_initial, omega
    )
    
    history['total'].append(float(total_loss))
    history['physics'].append(float(physics_loss))
    history['ic1'].append(float(ic1_loss))
    history['ic2'].append(float(ic2_loss))
    
    if (epoch + 1) % print_every == 0:
        print(f"   Epoch {epoch+1}/{epochs}: "
              f"Total={total_loss:.6f}, "
              f"Physics={physics_loss:.6f}, "
              f"IC1={ic1_loss:.6f}, "
              f"IC2={ic2_loss:.6f}")

elapsed = time.time() - start_time
print(f"\n   Training completed in {elapsed:.2f} seconds.")

# ============================================================================
# RK4 Solution for Comparison
# ============================================================================

print("\n2. Computing RK4 solution for comparison...")

def harmonic_oscillator_ode(state, t, omega):
    """ODE for scipy.integrate.odeint"""
    y, v = state
    dydt = v
    dvdt = -omega**2 * y
    return [dydt, dvdt]

t_rk4 = np.linspace(t_min, t_max, 500)
initial_state = [1.0, 0.0]  # [y(0), dy/dt(0)]
solution_rk4 = odeint(harmonic_oscillator_ode, initial_state, t_rk4, args=(omega,))
y_rk4 = solution_rk4[:, 0]

# ============================================================================
# Evaluation
# ============================================================================

print("\n3. Evaluating solutions...")

# Test points
t_test = np.linspace(t_min, t_max, 300).reshape(-1, 1)
t_test_tf = tf.convert_to_tensor(t_test, dtype=tf.float32)

# PINN prediction
y_pinn = model(t_test_tf, training=False).numpy()

# Compute derivatives for energy
y_test, dy_dt_test, _ = compute_derivatives(model, t_test_tf)
dy_dt_pinn = dy_dt_test.numpy()

# Analytical solution
y_exact = np.cos(omega * t_test)
dy_dt_exact = -omega * np.sin(omega * t_test)

# Energy (should be conserved)
# E = (1/2)(dy/dt)² + (1/2)ω²y²
energy_pinn = 0.5 * dy_dt_pinn**2 + 0.5 * omega**2 * y_pinn**2
energy_exact = 0.5 * dy_dt_exact**2 + 0.5 * omega**2 * y_exact**2

# Error analysis
l2_error = np.sqrt(np.mean((y_pinn - y_exact)**2))
max_error = np.max(np.abs(y_pinn - y_exact))

print(f"   L2 error: {l2_error:.6f}")
print(f"   Max error: {max_error:.6f}")
print(f"   Initial energy: {energy_exact[0,0]:.6f}")
print(f"   Energy variation (PINN): {np.std(energy_pinn):.6f}")
print(f"   Energy variation (Exact): {np.std(energy_exact):.6f}")

# ============================================================================
# Visualization
# ============================================================================

print("\n4. Creating visualizations...")

fig = plt.figure(figsize=(15, 12))
gs = GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.3)

# (a) Training history
ax1 = fig.add_subplot(gs[0, 0])
epochs_arr = np.arange(1, epochs + 1)
ax1.semilogy(epochs_arr, history['total'], 'b-', linewidth=2, label='Total')
ax1.semilogy(epochs_arr, history['physics'], 'g--', linewidth=1.5, label='Physics')
ax1.semilogy(epochs_arr, history['ic1'], 'r--', linewidth=1.5, label='IC: y(0)=1')
ax1.semilogy(epochs_arr, history['ic2'], 'm--', linewidth=1.5, label='IC: dy/dt(0)=0')
ax1.set_xlabel('Epoch', fontsize=11)
ax1.set_ylabel('Loss (log scale)', fontsize=11)
ax1.set_title('(a) Training History', fontsize=12, weight='bold')
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3, which='both')

# (b) Solution comparison
ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(t_test, y_exact, 'b-', linewidth=2.5, label='Exact: cos(ωt)')
ax2.plot(t_test, y_pinn, 'r--', linewidth=2, label='PINN')
ax2.plot(t_rk4, y_rk4, 'g:', linewidth=1.5, label='RK4')
ax2.scatter([0], [1], c='orange', s=100, marker='o', zorder=5, label='IC')
ax2.set_xlabel('Time t', fontsize=11)
ax2.set_ylabel('y(t)', fontsize=11)
ax2.set_title(f'(b) Solution Comparison (ω={omega})', fontsize=12, weight='bold')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

# (c) Phase space portrait
ax3 = fig.add_subplot(gs[1, 0])
ax3.plot(y_exact, dy_dt_exact, 'b-', linewidth=2.5, label='Exact')
ax3.plot(y_pinn, dy_dt_pinn, 'r--', linewidth=2, label='PINN')
ax3.scatter([1], [0], c='orange', s=100, marker='o', zorder=5, label='IC')
ax3.set_xlabel('y', fontsize=11)
ax3.set_ylabel('dy/dt', fontsize=11)
ax3.set_title('(c) Phase Space Portrait', fontsize=12, weight='bold')
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3)
ax3.axis('equal')

# (d) Energy conservation
ax4 = fig.add_subplot(gs[1, 1])
ax4.plot(t_test, energy_pinn, 'r-', linewidth=2, label='PINN')
ax4.plot(t_test, energy_exact, 'b--', linewidth=2, label='Exact')
ax4.axhline(y=0.5, color='k', linestyle=':', alpha=0.5, label='E=0.5')
ax4.set_xlabel('Time t', fontsize=11)
ax4.set_ylabel('Total Energy', fontsize=11)
ax4.set_title('(d) Energy Conservation', fontsize=12, weight='bold')
ax4.legend(fontsize=9)
ax4.grid(True, alpha=0.3)

# (e) Error over time
ax5 = fig.add_subplot(gs[2, 0])
error_time = np.abs(y_pinn - y_exact)
ax5.plot(t_test, error_time, 'r-', linewidth=2)
ax5.fill_between(t_test.flatten(), 0, error_time.flatten(), alpha=0.3, color='red')
ax5.set_xlabel('Time t', fontsize=11)
ax5.set_ylabel('|y_PINN - y_exact|', fontsize=11)
ax5.set_title(f'(e) Absolute Error (L2={l2_error:.6f})', fontsize=12, weight='bold')
ax5.grid(True, alpha=0.3)

# (f) Multiple frequencies
ax6 = fig.add_subplot(gs[2, 1])
omega_list = [1.0, 2.0, 5.0]
colors = ['blue', 'green', 'red']

for omega_i, color in zip(omega_list, colors):
    t_plot = np.linspace(0, 10, 200)
    y_plot = np.cos(omega_i * t_plot)
    ax6.plot(t_plot, y_plot, color=color, linewidth=2, label=f'ω={omega_i}')

ax6.set_xlabel('Time t', fontsize=11)
ax6.set_ylabel('y(t)', fontsize=11)
ax6.set_title('(f) Different Frequencies', fontsize=12, weight='bold')
ax6.legend(fontsize=9)
ax6.grid(True, alpha=0.3)

plt.savefig(f'{output_dir}/02_harmonic_oscillator.png', dpi=150, bbox_inches='tight')
print(f"   Saved: {output_dir}/02_harmonic_oscillator.png")

# ============================================================================
# Detailed Comparison Table
# ============================================================================

fig2, ax = plt.subplots(figsize=(12, 6))
ax.axis('off')

# Sample points for table
t_samples = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).reshape(-1, 1)
t_samples_tf = tf.convert_to_tensor(t_samples, dtype=tf.float32)

y_samples_pinn = model(t_samples_tf, training=False).numpy()
y_samples_exact = np.cos(omega * t_samples)

# Interpolate RK4 for comparison
y_samples_rk4 = np.interp(t_samples.flatten(), t_rk4, y_rk4).reshape(-1, 1)

table_data = [['Time t', 'PINN', 'RK4', 'Exact', 'Error (PINN)']]

for i in range(len(t_samples)):
    t_val = t_samples[i, 0]
    y_p = y_samples_pinn[i, 0]
    y_r = y_samples_rk4[i, 0]
    y_e = y_samples_exact[i, 0]
    err = abs(y_p - y_e)
    table_data.append([
        f'{t_val:.1f}',
        f'{y_p:.6f}',
        f'{y_r:.6f}',
        f'{y_e:.6f}',
        f'{err:.2e}'
    ])

table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                 colWidths=[0.15, 0.22, 0.22, 0.22, 0.19])
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 2.5)

# Header formatting
for i in range(5):
    table[(0, i)].set_facecolor('#2196F3')
    table[(0, i)].set_text_props(weight='bold', color='white')

ax.set_title('Solution Comparison Table', fontsize=14, weight='bold', pad=20)

plt.savefig(f'{output_dir}/02_comparison_table.png', dpi=150, bbox_inches='tight')
print(f"   Saved: {output_dir}/02_comparison_table.png")

# ============================================================================
# Summary
# ============================================================================

print("\n" + "="*70)
print("Summary:")
print("="*70)
print(f"1. Problem: d²y/dt² + ω²y = 0, ω = {omega}")
print(f"   Initial conditions: y(0) = 1, dy/dt(0) = 0")
print(f"\n2. PINN Training:")
print(f"   - Epochs: {epochs}")
print(f"   - Time: {elapsed:.2f} seconds")
print(f"   - Final loss: {history['total'][-1]:.6e}")
print(f"\n3. Accuracy:")
print(f"   - L2 error vs exact: {l2_error:.6f}")
print(f"   - Max error: {max_error:.6f}")
print(f"\n4. Energy Conservation:")
print(f"   - Exact: E = {energy_exact[0,0]:.6f} (constant)")
print(f"   - PINN: E = {np.mean(energy_pinn):.6f} ± {np.std(energy_pinn):.6f}")
print(f"   - Relative variation: {np.std(energy_pinn)/np.mean(energy_pinn)*100:.3f}%")
print(f"\n5. Key Insights:")
print(f"   - PINN learns periodic solution accurately")
print(f"   - Energy nearly conserved (physics-consistent)")
print(f"   - Second derivative computed via automatic differentiation")
print(f"   - Two initial conditions enforced simultaneously")
print("="*70)

plt.show()

