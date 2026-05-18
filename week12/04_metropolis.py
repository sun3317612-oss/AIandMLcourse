"""
04. Metropolis-Hastings Algorithm
Metropolis-Hastings 알고리즘

Monte Carlo 시뮬레이션의 핵심 알고리즘을 자세히 다룹니다:
- Detailed Balance 원리
- Acceptance Ratio 분석
- 에너지 지형 탐색
- 열평형 도달 과정

학습 목표:
1. Metropolis 알고리즘의 작동 원리
2. 상세 균형 조건 (Detailed Balance)
3. Acceptance Rate와 효율성
4. 평형화 과정 이해
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.gridspec import GridSpec
import os

# 출력 디렉토리 확인
output_dir = 'outputs'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print("="*70)
print("Metropolis-Hastings Algorithm")
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
# Example 1: 1D Potential Well (Simple Example)
# ============================================================================

print("\n1. Metropolis Algorithm on 1D Potential...")

def potential_1d(x):
    """
    1D 이중 우물 포텐셜 (double-well potential)
    V(x) = (x^2 - 1)^2
    
    두 개의 최소값: x = -1, x = 1
    에너지 장벽: x = 0
    """
    return (x**2 - 1)**2

def metropolis_1d(x, T, step_size=0.5):
    """
    1D에서 Metropolis 스텝을 수행합니다.
    
    Parameters:
    -----------
    x : float
        현재 위치
    T : float
        온도
    step_size : float
        제안 이동 거리
    
    Returns:
    --------
    x_new : float
        새로운 위치
    accepted : bool
        제안이 수락되었는지 여부
    """
    # 새로운 위치 제안
    x_trial = x + np.random.uniform(-step_size, step_size)
    
    # 에너지 변화
    dE = potential_1d(x_trial) - potential_1d(x)
    
    # Metropolis 기준
    if dE < 0 or np.random.random() < np.exp(-dE / T):
        return x_trial, True
    else:
        return x, False

# 다양한 온도에서 시뮬레이션
temperatures = [0.05, 0.2, 0.5, 1.0]
n_steps = 10000
x_range = np.linspace(-2, 2, 200)

fig1 = plt.figure(figsize=(15, 10))
gs = GridSpec(2, 2, figure=fig1, hspace=0.3, wspace=0.3)

for idx, T in enumerate(temperatures):
    print(f"   T = {T:.2f}...")
    
    # 시뮬레이션
    x = 0.0  # 시작 위치 (에너지 장벽)
    trajectory = np.zeros(n_steps)
    acceptance_count = 0
    
    for step in range(n_steps):
        x, accepted = metropolis_1d(x, T, step_size=0.3)
        trajectory[step] = x
        if accepted:
            acceptance_count += 1
    
    acceptance_rate = acceptance_count / n_steps
    
    ax = fig1.add_subplot(gs[idx // 2, idx % 2])
    
    # 히스토그램 (분포)
    counts, bins = np.histogram(trajectory[1000:], bins=50, density=True)
    ax.bar(bins[:-1], counts, width=np.diff(bins), alpha=0.6, 
           color='skyblue', edgecolor='black', label='Sampled distribution')
    
    # 포텐셜 에너지
    ax_pot = ax.twinx()
    V_vals = potential_1d(x_range)
    ax_pot.plot(x_range, V_vals, 'r-', linewidth=2.5, label='Potential V(x)')
    
    # 볼츠만 분포 (이론)
    boltzmann = np.exp(-V_vals / T)
    boltzmann /= np.trapz(boltzmann, x_range)
    ax.plot(x_range, boltzmann, 'g--', linewidth=2, label='Boltzmann dist')
    
    ax.set_xlabel('Position x', fontsize=11)
    ax.set_ylabel('Probability Density', fontsize=11)
    ax_pot.set_ylabel('Potential V(x)', fontsize=11, color='red')
    ax_pot.tick_params(axis='y', labelcolor='red')
    
    ax.set_title(f'T = {T:.2f}, Acceptance = {acceptance_rate:.1%}', 
                 fontsize=12, weight='bold')
    ax.legend(loc='upper left', fontsize=9)
    ax_pot.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    print(f"      Acceptance rate: {acceptance_rate:.3f}")
    print(f"      Mean position: {np.mean(trajectory[1000:]):.4f}")
    print(f"      Std position: {np.std(trajectory[1000:]):.4f}")

plt.savefig(f'{output_dir}/04_metropolis_potential.png', dpi=150, bbox_inches='tight')
print(f"\n   Saved: {output_dir}/04_metropolis_potential.png")

# ============================================================================
# Example 2: Acceptance Rate vs Step Size
# ============================================================================

print("\n2. Analyzing Acceptance Rate vs Step Size...")

T = 0.5
step_sizes = np.logspace(-2, 1, 15)
acceptance_rates = []
efficiencies = []
n_steps_test = 5000

for step_size in step_sizes:
    x = 0.0
    acceptance_count = 0
    positions = []
    
    for step in range(n_steps_test):
        x, accepted = metropolis_1d(x, T, step_size)
        if accepted:
            acceptance_count += 1
        positions.append(x)
    
    acceptance_rate = acceptance_count / n_steps_test
    acceptance_rates.append(acceptance_rate)
    
    # 효율성: 수락률과 탐색 범위의 곱
    # 이상적으로는 acceptance ~ 0.5 정도가 좋음
    positions = np.array(positions)
    std_pos = np.std(positions[1000:])
    efficiency = acceptance_rate * std_pos
    efficiencies.append(efficiency)

fig2 = plt.figure(figsize=(15, 5))

# (a) Acceptance rate vs step size
ax1 = fig2.add_subplot(1, 3, 1)
ax1.semilogx(step_sizes, acceptance_rates, 'bo-', markersize=6, linewidth=2)
ax1.axhline(y=0.5, color='r', linestyle='--', linewidth=1.5, 
            label='Optimal ~ 0.5')
ax1.set_xlabel('Step Size', fontsize=12)
ax1.set_ylabel('Acceptance Rate', fontsize=12)
ax1.set_title('(a) Acceptance Rate vs Step Size', fontsize=13, weight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

print(f"   Optimal step size (acceptance ~ 0.5):")
opt_idx = np.argmin(np.abs(np.array(acceptance_rates) - 0.5))
print(f"      Step size: {step_sizes[opt_idx]:.4f}")
print(f"      Acceptance: {acceptance_rates[opt_idx]:.3f}")

# (b) Efficiency vs step size
ax2 = fig2.add_subplot(1, 3, 2)
ax2.semilogx(step_sizes, efficiencies, 'go-', markersize=6, linewidth=2)
opt_eff_idx = np.argmax(efficiencies)
ax2.axvline(x=step_sizes[opt_eff_idx], color='r', linestyle='--', linewidth=1.5,
            label=f'Max at {step_sizes[opt_eff_idx]:.3f}')
ax2.set_xlabel('Step Size', fontsize=12)
ax2.set_ylabel('Efficiency (arb. units)', fontsize=12)
ax2.set_title('(b) Sampling Efficiency vs Step Size', fontsize=13, weight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

print(f"   Maximum efficiency:")
print(f"      Step size: {step_sizes[opt_eff_idx]:.4f}")
print(f"      Efficiency: {efficiencies[opt_eff_idx]:.4f}")

# (c) Example trajectories
ax3 = fig2.add_subplot(1, 3, 3)

for step_size, color, label in [(0.05, 'b', 'Small step (0.05)'), 
                                  (0.3, 'g', 'Medium step (0.3)'),
                                  (2.0, 'r', 'Large step (2.0)')]:
    x = 0.0
    trajectory = []
    for step in range(500):
        x, _ = metropolis_1d(x, T, step_size)
        trajectory.append(x)
    ax3.plot(trajectory, color=color, alpha=0.7, linewidth=1, label=label)

ax3.set_xlabel('Monte Carlo Step', fontsize=12)
ax3.set_ylabel('Position x', fontsize=12)
ax3.set_title(f'(c) Example Trajectories (T={T})', fontsize=13, weight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{output_dir}/04_metropolis_acceptance.png', dpi=150, bbox_inches='tight')
print(f"\n   Saved: {output_dir}/04_metropolis_acceptance.png")

# ============================================================================
# Example 3: Equilibration Process
# ============================================================================

print("\n3. Studying Equilibration Process...")

T = 0.3
n_steps = 5000
initial_positions = [-2.0, -1.0, 0.0, 1.0, 2.0]

fig3 = plt.figure(figsize=(15, 10))
gs = GridSpec(2, 2, figure=fig3, hspace=0.3, wspace=0.3)

# (a) Trajectories from different starting points
ax1 = fig3.add_subplot(gs[0, :])

colors = ['blue', 'green', 'orange', 'red', 'purple']
all_trajectories = []

for x0, color in zip(initial_positions, colors):
    x = x0
    trajectory = [x]
    
    for step in range(n_steps):
        x, _ = metropolis_1d(x, T, step_size=0.3)
        trajectory.append(x)
    
    all_trajectories.append(trajectory)
    ax1.plot(trajectory, color=color, alpha=0.6, linewidth=1.5, label=f'x0={x0:.1f}')

# 평균 위치 (모든 궤적)
mean_trajectory = np.mean(all_trajectories, axis=0)
ax1.plot(mean_trajectory, 'k-', linewidth=2.5, label='Mean')

ax1.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, linewidth=1)
ax1.axhline(y=-1.0, color='r', linestyle='--', alpha=0.5, linewidth=1)
ax1.set_xlabel('Monte Carlo Step', fontsize=12)
ax1.set_ylabel('Position x', fontsize=12)
ax1.set_title(f'(a) Equilibration from Different Starting Points (T={T})', 
              fontsize=13, weight='bold')
ax1.legend(fontsize=10, ncol=6, loc='upper right')
ax1.grid(True, alpha=0.3)

# (b) Energy vs time
ax2 = fig3.add_subplot(gs[1, 0])

for traj, color in zip(all_trajectories, colors):
    energies = [potential_1d(x) for x in traj]
    ax2.plot(energies, color=color, alpha=0.6, linewidth=1)

# 평균 에너지
mean_energy = np.mean([[potential_1d(x) for x in traj] for traj in all_trajectories], axis=0)
ax2.plot(mean_energy, 'k-', linewidth=2.5, label='Mean energy')

# 평형 에너지 (이론)
x_samples = np.linspace(-2, 2, 1000)
V_samples = potential_1d(x_samples)
boltzmann_weights = np.exp(-V_samples / T)
equilibrium_energy = np.sum(V_samples * boltzmann_weights) / np.sum(boltzmann_weights)
ax2.axhline(y=equilibrium_energy, color='r', linestyle='--', linewidth=2,
            label=f'Equilibrium E={equilibrium_energy:.3f}')

ax2.set_xlabel('Monte Carlo Step', fontsize=12)
ax2.set_ylabel('Potential Energy', fontsize=12)
ax2.set_title('(b) Energy Evolution', fontsize=13, weight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

# (c) Running average convergence
ax3 = fig3.add_subplot(gs[1, 1])

for traj, color in zip(all_trajectories, colors):
    running_mean = np.cumsum(traj) / np.arange(1, len(traj) + 1)
    ax3.plot(running_mean, color=color, alpha=0.6, linewidth=1.5)

ax3.axhline(y=0, color='k', linestyle='--', alpha=0.5)
ax3.set_xlabel('Monte Carlo Step', fontsize=12)
ax3.set_ylabel('Running Mean Position', fontsize=12)
ax3.set_title('(c) Convergence of Running Average', fontsize=13, weight='bold')
ax3.grid(True, alpha=0.3)
ax3.set_ylim([-1.5, 1.5])

plt.savefig(f'{output_dir}/04_metropolis_equilibration.png', dpi=150, bbox_inches='tight')
print(f"\n   Saved: {output_dir}/04_metropolis_equilibration.png")

print(f"   Equilibration analysis:")
print(f"      Theoretical equilibrium energy: {equilibrium_energy:.6f}")
print(f"      Final mean energy (all trajectories): {mean_energy[-1]:.6f}")

# ============================================================================
# Example 4: Detailed Balance Verification
# ============================================================================

print("\n4. Verifying Detailed Balance...")

T = 0.5
n_bins = 20
n_steps_db = 50000

# Long simulation
x = 0.0
trajectory = []
for step in range(n_steps_db):
    x, _ = metropolis_1d(x, T, step_size=0.3)
    if step > 5000:  # Skip equilibration
        trajectory.append(x)

# Histogram (실험적 분포)
trajectory = np.array(trajectory)
counts, bins = np.histogram(trajectory, bins=n_bins, density=True)
bin_centers = (bins[:-1] + bins[1:]) / 2

# 볼츠만 분포 (이론적 평형 분포)
x_theory = np.linspace(-2, 2, 500)
V_theory = potential_1d(x_theory)
boltzmann_theory = np.exp(-V_theory / T)
Z = np.trapz(boltzmann_theory, x_theory)  # Partition function
boltzmann_theory /= Z

fig4 = plt.figure(figsize=(15, 5))

# (a) Distribution comparison
ax1 = fig4.add_subplot(1, 3, 1)
ax1.bar(bin_centers, counts, width=np.diff(bins)[0], alpha=0.6, 
        color='skyblue', edgecolor='black', label='Simulation')
ax1.plot(x_theory, boltzmann_theory, 'r-', linewidth=2.5, label='Boltzmann dist')
ax1.set_xlabel('Position x', fontsize=12)
ax1.set_ylabel('Probability Density', fontsize=12)
ax1.set_title(f'(a) Equilibrium Distribution (T={T})', fontsize=13, weight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Chi-squared test
# 이론 분포에서 예상되는 카운트
expected_counts = []
for i in range(len(bins)-1):
    mask = (x_theory >= bins[i]) & (x_theory < bins[i+1])
    if np.sum(mask) > 0:
        expected = np.mean(boltzmann_theory[mask]) * len(trajectory) * np.diff(bins)[0]
        expected_counts.append(expected)
    else:
        expected_counts.append(0)

expected_counts = np.array(expected_counts)
observed_counts = counts * len(trajectory) * np.diff(bins)[0]

# (b) Observed vs Expected
ax2 = fig4.add_subplot(1, 3, 2)
ax2.scatter(expected_counts, observed_counts, s=50, alpha=0.7)
max_count = max(expected_counts.max(), observed_counts.max())
ax2.plot([0, max_count], [0, max_count], 'r--', linewidth=2, label='Perfect agreement')
ax2.set_xlabel('Expected Counts (Boltzmann)', fontsize=12)
ax2.set_ylabel('Observed Counts (Simulation)', fontsize=12)
ax2.set_title('(b) Detailed Balance Check', fontsize=13, weight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

# (c) Free energy profile
ax3 = fig4.add_subplot(1, 3, 3)

# 실험적 자유 에너지: F(x) = -kT * ln[P(x)]
# 노이즈 방지를 위해 smoothing
from scipy.ndimage import gaussian_filter1d
counts_smooth = gaussian_filter1d(counts, sigma=1.5)
counts_smooth[counts_smooth < 1e-10] = 1e-10
free_energy_sim = -T * np.log(counts_smooth)
free_energy_sim -= free_energy_sim.min()  # 최소값을 0으로

# 이론적 자유 에너지
free_energy_theory = V_theory
free_energy_theory -= free_energy_theory.min()

ax3.plot(bin_centers, free_energy_sim, 'bo-', markersize=4, linewidth=1.5, 
         label='From simulation')
ax3.plot(x_theory, free_energy_theory, 'r-', linewidth=2, label='True potential')
ax3.set_xlabel('Position x', fontsize=12)
ax3.set_ylabel('Free Energy F(x)', fontsize=12)
ax3.set_title('(c) Free Energy Profile', fontsize=13, weight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{output_dir}/04_metropolis_detailed_balance.png', dpi=150, bbox_inches='tight')
print(f"\n   Saved: {output_dir}/04_metropolis_detailed_balance.png")

# ============================================================================
# Summary
# ============================================================================

print("\n" + "="*70)
print("Key Results:")
print("="*70)
print(f"1. Metropolis Algorithm:")
print(f"   - Generates samples from Boltzmann distribution")
print(f"   - Always accept if dE < 0")
print(f"   - Accept with probability exp(-dE/T) if dE > 0")
print(f"\n2. Acceptance Rate:")
print(f"   - Too small steps: high acceptance but slow exploration")
print(f"   - Too large steps: low acceptance, many rejections")
print(f"   - Optimal: ~50% acceptance rate")
print(f"\n3. Equilibration:")
print(f"   - System needs time to reach equilibrium")
print(f"   - Discard initial 'burn-in' period")
print(f"   - Independent of initial condition after equilibration")
print(f"\n4. Detailed Balance:")
print(f"   - Simulation produces correct Boltzmann distribution")
print(f"   - Satisfies P(i->j) * P(i) = P(j->i) * P(j)")
print(f"   - Free energy can be extracted from distribution")
print("="*70)

plt.show()

