"""
03. Damped Oscillator with PINN
감쇠 조화 진동자를 PINN으로 풀기

Problem: d²y/dt² + 2ζω·dy/dt + ω²y = 0
Initial conditions: y(0) = 1, dy/dt(0) = 0

Three regimes:
- Under-damped: ζ < 1 (진동하며 감쇠)
- Critically damped: ζ = 1 (가장 빠른 감쇠)
- Over-damped: ζ > 1 (느린 감쇠, 진동 없음)

학습 목표:
1. Parameter-dependent ODEs
2. Multiple solution regimes
3. Damping physics
4. PINN's flexibility with different behaviors
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
print("Damped Harmonic Oscillator with PINN")
print("="*70)
print("Problem: d²y/dt² + 2ζω·dy/dt + ω²y = 0")
print("Three regimes: ζ < 1 (under), ζ = 1 (critical), ζ > 1 (over)")
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
# Analytical Solutions
# ============================================================================

def analytical_solution(t, zeta, omega):
    """
    감쇠 조화 진동자의 해석해
    
    Parameters:
    -----------
    t : array
        Time
    zeta : float
        Damping ratio
    omega : float
        Natural frequency
    
    Returns:
    --------
    y : array
        Solution
    """
    if zeta < 1:  # Under-damped
        omega_d = omega * np.sqrt(1 - zeta**2)
        y = np.exp(-zeta * omega * t) * np.cos(omega_d * t)
    elif zeta == 1:  # Critically damped
        y = (1 + omega * t) * np.exp(-omega * t)
    else:  # Over-damped
        r1 = -omega * (zeta + np.sqrt(zeta**2 - 1))
        r2 = -omega * (zeta - np.sqrt(zeta**2 - 1))
        A = (r2 - 0) / (r2 - r1)
        B = (0 - r1) / (r2 - r1)
        y = A * np.exp(r1 * t) + B * np.exp(r2 * t)
    
    return y

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

def compute_derivatives(model, t):
    """1차 및 2차 미분 계산"""
    with tf.GradientTape(persistent=True) as tape2:
        tape2.watch(t)
        with tf.GradientTape(persistent=True) as tape1:
            tape1.watch(t)
            y = model(t, training=True)
        dy_dt = tape1.gradient(y, t)
    d2y_dt2 = tape2.gradient(dy_dt, t)
    
    del tape1, tape2
    return y, dy_dt, d2y_dt2

def compute_physics_loss(model, t_collocation, zeta, omega):
    """
    Physics loss: d²y/dt² + 2ζω·dy/dt + ω²y = 0
    """
    y, dy_dt, d2y_dt2 = compute_derivatives(model, t_collocation)
    
    residual = d2y_dt2 + 2*zeta*omega*dy_dt + omega**2*y
    physics_loss = tf.reduce_mean(tf.square(residual))
    
    return physics_loss

def compute_ic_loss(model, t_initial):
    """Initial conditions: y(0)=1, dy/dt(0)=0"""
    y, dy_dt, _ = compute_derivatives(model, t_initial)
    
    ic1_loss = tf.reduce_mean(tf.square(y - 1.0))
    ic2_loss = tf.reduce_mean(tf.square(dy_dt - 0.0))
    
    return ic1_loss, ic2_loss

def train_step(model, optimizer, t_collocation, t_initial, zeta, omega):
    """Training step"""
    with tf.GradientTape() as tape:
        physics_loss = compute_physics_loss(model, t_collocation, zeta, omega)
        ic1_loss, ic2_loss = compute_ic_loss(model, t_initial)
        
        total_loss = physics_loss + 100.0*ic1_loss + 100.0*ic2_loss
    
    gradients = tape.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    return total_loss, physics_loss, ic1_loss, ic2_loss

# ============================================================================
# Simulate Multiple Damping Ratios
# ============================================================================

omega = 2.0
zeta_values = [0.1, 0.3, 1.0, 2.0]  # Under, Under, Critical, Over
damping_names = ['Under-damped (ζ=0.1)', 'Under-damped (ζ=0.3)', 
                 'Critically damped (ζ=1.0)', 'Over-damped (ζ=2.0)']

t_min, t_max = 0.0, 10.0
n_collocation = 150

results = {}

print(f"\nω = {omega}")

for zeta, name in zip(zeta_values, damping_names):
    print(f"\n{'='*70}")
    print(f"Training for {name}")
    print(f"{'='*70}")
    
    # Prepare data
    t_collocation = tf.convert_to_tensor(
        np.linspace(t_min, t_max, n_collocation).reshape(-1, 1),
        dtype=tf.float32
    )
    t_initial = tf.convert_to_tensor([[0.0]], dtype=tf.float32)
    
    # Create and train model
    model = create_pinn_model()
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    
    epochs = 10000
    print_every = 2000
    
    start_time = time.time()
    
    for epoch in range(epochs):
        total_loss, physics_loss, ic1_loss, ic2_loss = train_step(
            model, optimizer, t_collocation, t_initial, zeta, omega
        )
        
        if (epoch + 1) % print_every == 0:
            print(f"   Epoch {epoch+1}/{epochs}: Loss={total_loss:.6f}")
    
    elapsed = time.time() - start_time
    print(f"   Training completed in {elapsed:.2f} seconds.")
    
    # Evaluate
    t_test = np.linspace(t_min, t_max, 300).reshape(-1, 1)
    t_test_tf = tf.convert_to_tensor(t_test, dtype=tf.float32)
    
    y_pinn = model(t_test_tf, training=False).numpy()
    y_exact = analytical_solution(t_test.flatten(), zeta, omega).reshape(-1, 1)
    
    l2_error = np.sqrt(np.mean((y_pinn - y_exact)**2))
    print(f"   L2 error: {l2_error:.6f}")
    
    results[zeta] = {
        'model': model,
        't_test': t_test,
        'y_pinn': y_pinn,
        'y_exact': y_exact,
        'l2_error': l2_error,
        'name': name
    }

# ============================================================================
# Visualization
# ============================================================================

print("\nCreating visualizations...")

fig = plt.figure(figsize=(15, 12))
gs = GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.3)

colors = ['blue', 'green', 'red', 'purple']

# (a-d) Individual solutions
for idx, (zeta, color) in enumerate(zip(zeta_values, colors)):
    row = idx // 2
    col = idx % 2
    ax = fig.add_subplot(gs[row, col])
    
    result = results[zeta]
    t_test = result['t_test']
    y_pinn = result['y_pinn']
    y_exact = result['y_exact']
    
    ax.plot(t_test, y_exact, '-', color=color, linewidth=2.5, 
            label='Exact', alpha=0.7)
    ax.plot(t_test, y_pinn, '--', color='black', linewidth=2, 
            label='PINN')
    
    # Envelope for under-damped
    if zeta < 1:
        envelope = np.exp(-zeta * omega * t_test)
        ax.plot(t_test, envelope, ':', color='gray', linewidth=1.5, 
                label='Envelope', alpha=0.7)
        ax.plot(t_test, -envelope, ':', color='gray', linewidth=1.5, alpha=0.7)
    
    ax.scatter([0], [1], c='orange', s=100, marker='o', zorder=5, label='IC')
    ax.set_xlabel('Time t', fontsize=11)
    ax.set_ylabel('y(t)', fontsize=11)
    ax.set_title(f'({chr(97+idx)}) {result["name"]}\n'
                 f'L2 error={result["l2_error"]:.6f}',
                 fontsize=12, weight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([-1.2, 1.5])

# (e) All solutions comparison
ax5 = fig.add_subplot(gs[2, :])
for zeta, color in zip(zeta_values, colors):
    result = results[zeta]
    ax5.plot(result['t_test'], result['y_exact'], '-', 
             color=color, linewidth=2.5, label=result['name'])

ax5.axhline(y=0, color='k', linestyle='--', alpha=0.3)
ax5.set_xlabel('Time t', fontsize=12)
ax5.set_ylabel('y(t)', fontsize=12)
ax5.set_title('(e) All Damping Regimes Comparison', fontsize=13, weight='bold')
ax5.legend(fontsize=10, loc='upper right')
ax5.grid(True, alpha=0.3)

plt.savefig(f'{output_dir}/03_damped_oscillator.png', dpi=150, bbox_inches='tight')
print(f"Saved: {output_dir}/03_damped_oscillator.png")

# ============================================================================
# Error Analysis
# ============================================================================

fig2, axes = plt.subplots(2, 2, figsize=(15, 10))
axes = axes.flatten()

for idx, (zeta, color) in enumerate(zip(zeta_values, colors)):
    result = results[zeta]
    error = np.abs(result['y_pinn'] - result['y_exact'])
    
    axes[idx].plot(result['t_test'], error, color=color, linewidth=2)
    axes[idx].fill_between(result['t_test'].flatten(), 0, error.flatten(), 
                            alpha=0.3, color=color)
    axes[idx].set_xlabel('Time t', fontsize=11)
    axes[idx].set_ylabel('|y_PINN - y_exact|', fontsize=11)
    axes[idx].set_title(f'{result["name"]}\nL2={result["l2_error"]:.6f}', 
                        fontsize=12, weight='bold')
    axes[idx].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{output_dir}/03_error_analysis.png', dpi=150, bbox_inches='tight')
print(f"Saved: {output_dir}/03_error_analysis.png")

# ============================================================================
# Damping Effect Analysis
# ============================================================================

print("\nDamping Effect Analysis:")

fig3, axes = plt.subplots(1, 2, figsize=(15, 5))

# (a) Peak decay
zeta_range = np.linspace(0.01, 0.99, 30)
omega_d_range = omega * np.sqrt(1 - zeta_range**2)
period_range = 2*np.pi / omega_d_range

axes[0].plot(zeta_range, period_range, 'b-', linewidth=2)
axes[0].set_xlabel('Damping Ratio ζ', fontsize=12)
axes[0].set_ylabel('Period', fontsize=12)
axes[0].set_title('(a) Period vs Damping (Under-damped)', fontsize=13, weight='bold')
axes[0].grid(True, alpha=0.3)

# (b) Decay time constant
tau_range = 1 / (zeta_range * omega)
axes[1].plot(zeta_range, tau_range, 'r-', linewidth=2)
axes[1].set_xlabel('Damping Ratio ζ', fontsize=12)
axes[1].set_ylabel('Decay Time τ = 1/(ζω)', fontsize=12)
axes[1].set_title('(b) Decay Rate vs Damping', fontsize=13, weight='bold')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{output_dir}/03_damping_analysis.png', dpi=150, bbox_inches='tight')
print(f"Saved: {output_dir}/03_damping_analysis.png")

# ============================================================================
# Summary
# ============================================================================

print("\n" + "="*70)
print("Summary:")
print("="*70)
print(f"Problem: d²y/dt² + 2ζω·dy/dt + ω²y = 0, ω={omega}")
print("\nResults for different damping ratios:")

for zeta in zeta_values:
    result = results[zeta]
    regime = "Under" if zeta < 1 else ("Critical" if zeta == 1 else "Over")
    print(f"   ζ={zeta}: {regime}-damped, L2 error={result['l2_error']:.6f}")

print("\nKey Insights:")
print("   - Under-damped (ζ<1): Oscillates with exponential decay")
print("   - Critically damped (ζ=1): Fastest return to equilibrium")
print("   - Over-damped (ζ>1): Slow return, no oscillation")
print("   - PINN successfully captures all three regimes")
print("   - Same network architecture adapts to different physics")
print("="*70)

plt.show()

