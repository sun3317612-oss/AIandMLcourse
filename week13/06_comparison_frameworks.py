"""
06. PINN vs Traditional Methods Comparison
PINN (TensorFlow)과 전통적 방법 비교 연구

Same problem solved with:
1. PINN (TensorFlow)
2. Traditional RK4

Test problem: Van der Pol oscillator
d²y/dt² - μ(1-y²)·dy/dt + y = 0
μ = 1.0 (nonlinearity parameter)
Initial: y(0) = 2, dy/dt(0) = 0

Performance metrics:
- Accuracy
- Training time
- Memory usage
- Code complexity

학습 목표:
1. PINN vs Traditional comparison
2. When to use which
3. Performance trade-offs
4. Best practices
"""

import numpy as np
import time
import psutil
import os

# TensorFlow
import tensorflow as tf

# Traditional
from scipy.integrate import odeint

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.gridspec import GridSpec

# 출력 디렉토리
output_dir = 'outputs'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print("="*70)
print("Method Comparison: PINN (TensorFlow) vs Traditional (RK4)")
print("="*70)
print("Problem: Van der Pol oscillator, μ=1.0")
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
# Problem Setup
# ============================================================================

mu = 1.0  # Nonlinearity parameter
t_min, t_max = 0.0, 20.0
y0, v0 = 2.0, 0.0

n_collocation = 150
epochs = 10000

# ============================================================================
# Method 1: Traditional RK4
# ============================================================================

print("\n1. Traditional RK4 Solution...")

def van_der_pol_ode(state, t, mu):
    """Van der Pol oscillator"""
    y, v = state
    dydt = v
    dvdt = mu * (1 - y**2) * v - y
    return [dydt, dvdt]

start_time = time.time()
memory_before = psutil.Process().memory_info().rss / 1024**2

t_rk4 = np.linspace(t_min, t_max, 1000)
solution_rk4 = odeint(van_der_pol_ode, [y0, v0], t_rk4, args=(mu,))
y_rk4 = solution_rk4[:, 0]

rk4_time = time.time() - start_time
memory_after = psutil.Process().memory_info().rss / 1024**2
rk4_memory = memory_after - memory_before

print(f"   Time: {rk4_time:.4f} seconds")
print(f"   Memory: {rk4_memory:.2f} MB")

# ============================================================================
# Method 2: PINN with TensorFlow
# ============================================================================

print("\n2. PINN with TensorFlow...")

def create_tf_model():
    """TensorFlow model"""
    return tf.keras.Sequential([
        tf.keras.layers.Dense(32, activation='tanh', input_shape=(1,)),
        tf.keras.layers.Dense(32, activation='tanh'),
        tf.keras.layers.Dense(32, activation='tanh'),
        tf.keras.layers.Dense(1)
    ])

def compute_tf_derivatives(model, t):
    """Compute derivatives in TensorFlow"""
    with tf.GradientTape(persistent=True) as tape2:
        tape2.watch(t)
        with tf.GradientTape(persistent=True) as tape1:
            tape1.watch(t)
            y = model(t, training=True)
        dy_dt = tape1.gradient(y, t)
    d2y_dt2 = tape2.gradient(dy_dt, t)
    del tape1, tape2
    return y, dy_dt, d2y_dt2

def train_step_tf(model, optimizer, t_coll, t_init, mu):
    """TensorFlow training step"""
    with tf.GradientTape() as tape:
        # Physics loss: d²y/dt² - μ(1-y²)·dy/dt + y = 0
        y, dy_dt, d2y_dt2 = compute_tf_derivatives(model, t_coll)
        residual = d2y_dt2 - mu * (1 - y**2) * dy_dt + y
        physics_loss = tf.reduce_mean(tf.square(residual))
        
        # IC loss
        y_ic, dy_dt_ic, _ = compute_tf_derivatives(model, t_init)
        ic_loss = tf.reduce_mean(tf.square(y_ic - y0)) + \
                  tf.reduce_mean(tf.square(dy_dt_ic - v0))
        
        total_loss = physics_loss + 100.0 * ic_loss
    
    gradients = tape.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return total_loss

# Train TensorFlow model
t_coll_tf = tf.convert_to_tensor(
    np.linspace(t_min, t_max, n_collocation).reshape(-1, 1), dtype=tf.float32
)
t_init_tf = tf.convert_to_tensor([[t_min]], dtype=tf.float32)

model_tf = create_tf_model()
optimizer_tf = tf.keras.optimizers.Adam(learning_rate=0.001)

start_time = time.time()
memory_before = psutil.Process().memory_info().rss / 1024**2

for epoch in range(epochs):
    loss = train_step_tf(model_tf, optimizer_tf, t_coll_tf, t_init_tf, mu)
    if (epoch + 1) % 2000 == 0:
        print(f"   Epoch {epoch+1}/{epochs}: Loss={loss:.6f}")

tf_train_time = time.time() - start_time
memory_after = psutil.Process().memory_info().rss / 1024**2
tf_memory = memory_after - memory_before

print(f"   Training time: {tf_train_time:.4f} seconds")
print(f"   Memory: {tf_memory:.2f} MB")

# Evaluate TensorFlow
t_test = np.linspace(t_min, t_max, 500).reshape(-1, 1)
t_test_tf = tf.convert_to_tensor(t_test, dtype=tf.float32)

start_time = time.time()
y_tf = model_tf(t_test_tf, training=False).numpy()
tf_inference_time = time.time() - start_time

y_rk4_interp = np.interp(t_test.flatten(), t_rk4, y_rk4).reshape(-1, 1)
tf_error = np.sqrt(np.mean((y_tf - y_rk4_interp)**2))

print(f"   Inference time: {tf_inference_time:.4f} seconds")
print(f"   L2 error vs RK4: {tf_error:.6f}")

# ============================================================================
# Visualization
# ============================================================================

print("\n3. Creating comparison visualizations...")

fig = plt.figure(figsize=(15, 12))
gs = GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.3)

# (a) Solution comparison
ax1 = fig.add_subplot(gs[0, :])
ax1.plot(t_rk4, y_rk4, 'k-', linewidth=2.5, label='RK4 (Reference)', alpha=0.8)
ax1.plot(t_test, y_tf, 'b--', linewidth=2, label='TensorFlow PINN')
ax1.scatter([0], [y0], c='green', s=100, marker='o', zorder=5, label='IC')
ax1.set_xlabel('Time t', fontsize=12)
ax1.set_ylabel('y(t)', fontsize=12)
ax1.set_title('(a) Van der Pol Oscillator Solutions', fontsize=13, weight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# (b) Error comparison
ax2 = fig.add_subplot(gs[1, 0])
error_tf = np.abs(y_tf - y_rk4_interp)

ax2.plot(t_test, error_tf, 'b-', linewidth=2, label=f'TensorFlow (L2={tf_error:.6f})')
ax2.fill_between(t_test.flatten(), 0, error_tf.flatten(), alpha=0.3, color='blue')
ax2.set_xlabel('Time t', fontsize=11)
ax2.set_ylabel('|y_PINN - y_RK4|', fontsize=11)
ax2.set_title('(b) PINN Error vs RK4', fontsize=12, weight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

# (c) Training time comparison
ax3 = fig.add_subplot(gs[1, 1])
methods = ['RK4', 'TensorFlow\nPINN']
times = [rk4_time, tf_train_time]
colors = ['gray', 'blue']

bars = ax3.bar(methods, times, color=colors, alpha=0.7, edgecolor='black')
for bar, time_val in zip(bars, times):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height,
             f'{time_val:.2f}s', ha='center', va='bottom', fontsize=10)

ax3.set_ylabel('Time (seconds)', fontsize=11)
ax3.set_title('(c) Computation Time', fontsize=12, weight='bold')
ax3.grid(True, alpha=0.3, axis='y')

# (d) Memory usage comparison
ax4 = fig.add_subplot(gs[2, 0])
memory_vals = [rk4_memory, tf_memory]

bars = ax4.bar(methods, memory_vals, color=colors, alpha=0.7, edgecolor='black')
for bar, mem_val in zip(bars, memory_vals):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height,
             f'{mem_val:.1f}MB', ha='center', va='bottom', fontsize=10)

ax4.set_ylabel('Memory (MB)', fontsize=11)
ax4.set_title('(d) Memory Usage', fontsize=12, weight='bold')
ax4.grid(True, alpha=0.3, axis='y')

# (e) Performance summary table
ax5 = fig.add_subplot(gs[2, 1])
ax5.axis('off')

summary_data = [
    ['Metric', 'RK4', 'TF PINN'],
    ['Training (s)', f'{rk4_time:.3f}', f'{tf_train_time:.3f}'],
    ['Inference (s)', f'{rk4_time:.4f}', f'{tf_inference_time:.4f}'],
    ['Memory (MB)', f'{rk4_memory:.1f}', f'{tf_memory:.1f}'],
    ['L2 Error', '0 (ref)', f'{tf_error:.6f}'],
    ['Code Lines', '~10', '~80']
]

table = ax5.table(cellText=summary_data, cellLoc='center', loc='center',
                  colWidths=[0.4, 0.3, 0.3])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2.5)

# Header
for i in range(3):
    table[(0, i)].set_facecolor('#4CAF50')
    table[(0, i)].set_text_props(weight='bold', color='white')

ax5.set_title('(e) Performance Summary', fontsize=12, weight='bold', pad=20)

plt.savefig(f'{output_dir}/06_method_comparison.png', dpi=150, bbox_inches='tight')
print(f"   Saved: {output_dir}/06_method_comparison.png")

# ============================================================================
# Recommendations
# ============================================================================

fig2, ax = plt.subplots(figsize=(14, 8))
ax.axis('off')

recommendations = [
    ['Use Case', 'Recommendation', 'Reason'],
    ['Simple forward ODE', 'RK4/scipy', 'Fast, accurate, proven'],
    ['Inverse problems', 'PINN', 'Can estimate parameters from data'],
    ['Sparse/noisy data', 'PINN', 'Physics regularization helps'],
    ['Real-time simulation', 'RK4/scipy', 'Much faster execution'],
    ['Complex boundary conditions', 'PINN', 'Flexible constraint handling'],
    ['Stiff ODEs', 'Specialized solver', 'Use LSODA, BDF methods'],
    ['Parameter sensitivity', 'PINN', 'Can learn from partial observations'],
    ['Production deployment', 'RK4/scipy', 'Reliable, fast, well-tested'],
    ['Research/exploration', 'PINN', 'Novel approach, flexible'],
    ['High accuracy needed', 'RK4/scipy', 'Guaranteed convergence']
]

table = ax.table(cellText=recommendations, cellLoc='left', loc='center',
                 colWidths=[0.25, 0.25, 0.5])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2.2)

for i in range(3):
    table[(0, i)].set_facecolor('#2196F3')
    table[(0, i)].set_text_props(weight='bold', color='white')

ax.set_title('When to Use Which Method?', fontsize=15, weight='bold', pad=20)

plt.savefig(f'{output_dir}/06_recommendations.png', dpi=150, bbox_inches='tight')
print(f"   Saved: {output_dir}/06_recommendations.png")

# ============================================================================
# Summary
# ============================================================================

print("\n" + "="*70)
print("Comparison Summary:")
print("="*70)
print(f"\n1. Traditional RK4:")
print(f"   - Fastest: {rk4_time:.3f}s")
print(f"   - Most accurate (reference)")
print(f"   - Simplest code (~10 lines)")
print(f"   - Best for: Standard ODEs, production")
print(f"\n2. TensorFlow PINN:")
print(f"   - Training: {tf_train_time:.3f}s")
print(f"   - Error: {tf_error:.6f}")
print(f"   - Good for: Inverse problems, sparse data")
print(f"   - Trade-off: Slower but more flexible")
print(f"\n3. When to use PINN:")
print(f"   + Inverse problems (parameter estimation)")
print(f"   + Sparse/noisy data")
print(f"   + Complex boundary conditions")
print(f"   + Need smooth analytical-like solution")
print(f"   + Learning from partial observations")
print(f"\n4. When to use Traditional:")
print(f"   + Simple forward problem")
print(f"   + Need speed and efficiency")
print(f"   + Guaranteed accuracy required")
print(f"   + Well-established, reliable methods")
print(f"   + Production deployment")
print("\n5. Key Insight:")
print("   PINN is not a replacement but a complement to traditional methods.")
print("   Use PINN when physics constraints can regularize sparse data,")
print("   or when solving inverse problems. Use traditional methods for")
print("   standard forward problems where speed and accuracy are critical.")
print("="*70)

plt.show()
