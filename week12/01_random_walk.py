"""
01. Random Walk Simulation
랜덤 워크 시뮬레이션

Monte Carlo 방법의 가장 기본적인 예제입니다:
- 1D와 2D 랜덤 워크
- 중심극한정리 검증
- 확산과 브라운 운동의 기초

학습 목표:
1. 무작위 과정의 통계적 성질 이해
2. Monte Carlo 방법의 기본 원리
3. 확산 현상과의 연결
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
print("Random Walk Simulation - Monte Carlo Basics")
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
# 1D Random Walk
# ============================================================================

def random_walk_1d(n_steps):
    """
    1D 랜덤 워크를 수행합니다.
    
    Parameters:
    -----------
    n_steps : int
        걸음 수
    
    Returns:
    --------
    positions : array
        각 시간에서의 위치
    """
    # 각 스텝에서 +1 또는 -1로 이동
    steps = np.random.choice([-1, 1], size=n_steps)
    positions = np.concatenate([[0], np.cumsum(steps)])
    return positions

def simulate_multiple_walks_1d(n_walks, n_steps):
    """
    여러 개의 1D 랜덤 워크를 시뮬레이션합니다.
    
    Parameters:
    -----------
    n_walks : int
        워크의 개수
    n_steps : int
        각 워크의 스텝 수
    
    Returns:
    --------
    all_positions : array (n_walks, n_steps+1)
        모든 워크의 위치 데이터
    """
    all_positions = np.zeros((n_walks, n_steps + 1))
    for i in range(n_walks):
        all_positions[i] = random_walk_1d(n_steps)
    return all_positions

# ============================================================================
# 2D Random Walk
# ============================================================================

def random_walk_2d(n_steps):
    """
    2D 랜덤 워크를 수행합니다.
    
    Parameters:
    -----------
    n_steps : int
        걸음 수
    
    Returns:
    --------
    x, y : arrays
        x, y 좌표
    """
    # 각 스텝에서 상하좌우 중 하나로 이동
    angles = np.random.uniform(0, 2*np.pi, size=n_steps)
    dx = np.cos(angles)
    dy = np.sin(angles)
    
    x = np.concatenate([[0], np.cumsum(dx)])
    y = np.concatenate([[0], np.cumsum(dy)])
    
    return x, y

def simulate_multiple_walks_2d(n_walks, n_steps):
    """
    여러 개의 2D 랜덤 워크를 시뮬레이션합니다.
    
    Parameters:
    -----------
    n_walks : int
        워크의 개수
    n_steps : int
        각 워크의 스텝 수
    
    Returns:
    --------
    all_x, all_y : arrays (n_walks, n_steps+1)
        모든 워크의 x, y 좌표
    """
    all_x = np.zeros((n_walks, n_steps + 1))
    all_y = np.zeros((n_walks, n_steps + 1))
    
    for i in range(n_walks):
        x, y = random_walk_2d(n_steps)
        all_x[i] = x
        all_y[i] = y
    
    return all_x, all_y

# ============================================================================
# Simulation 1: 1D Random Walk Examples
# ============================================================================

print("\n1. Simulating 1D Random Walks...")
n_steps = 1000
n_walks_display = 5

# 몇 개의 랜덤 워크 시뮬레이션
walks_1d = simulate_multiple_walks_1d(n_walks_display, n_steps)

fig = plt.figure(figsize=(15, 10))
gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

# (a) 개별 워크 궤적
ax1 = fig.add_subplot(gs[0, 0])
time_steps = np.arange(n_steps + 1)
for i in range(n_walks_display):
    ax1.plot(time_steps, walks_1d[i], alpha=0.7, linewidth=1.5, label=f'Walk {i+1}')
ax1.set_xlabel('Time Step', fontsize=12)
ax1.set_ylabel('Position', fontsize=12)
ax1.set_title('(a) 1D Random Walk Trajectories', fontsize=13, weight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend()
ax1.axhline(y=0, color='k', linestyle='--', alpha=0.3)

# (b) 최종 위치 분포 (중심극한정리)
ax2 = fig.add_subplot(gs[0, 1])
n_walks_stat = 10000
walks_stat = simulate_multiple_walks_1d(n_walks_stat, n_steps)
final_positions = walks_stat[:, -1]

counts, bins, patches = ax2.hist(final_positions, bins=50, density=True, 
                                   alpha=0.7, color='blue', edgecolor='black')

# 이론적 가우시안 분포 (중심극한정리)
mean_theory = 0
std_theory = np.sqrt(n_steps)
x_theory = np.linspace(final_positions.min(), final_positions.max(), 100)
gaussian = (1/(std_theory * np.sqrt(2*np.pi))) * np.exp(-0.5*((x_theory - mean_theory)/std_theory)**2)
ax2.plot(x_theory, gaussian, 'r-', linewidth=2, label=f'Gaussian (sigma={std_theory:.1f})')

ax2.set_xlabel('Final Position', fontsize=12)
ax2.set_ylabel('Probability Density', fontsize=12)
ax2.set_title(f'(b) Final Position Distribution (N={n_walks_stat} walks)', 
              fontsize=13, weight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

print(f"   - Mean of final positions: {np.mean(final_positions):.2f} (theory: 0)")
print(f"   - Std of final positions: {np.std(final_positions):.2f} (theory: {std_theory:.2f})")

# (c) Mean Square Displacement (MSD)
ax3 = fig.add_subplot(gs[1, 0])
n_walks_msd = 1000
walks_msd = simulate_multiple_walks_1d(n_walks_msd, n_steps)

# MSD 계산: <x^2(t)>
msd = np.mean(walks_msd**2, axis=0)
msd_theory = np.arange(n_steps + 1)  # 이론값: t에 비례

ax3.plot(time_steps, msd, 'b-', linewidth=2, label='Simulation')
ax3.plot(time_steps, msd_theory, 'r--', linewidth=2, label='Theory: t')
ax3.set_xlabel('Time Step', fontsize=12)
ax3.set_ylabel('Mean Square Displacement', fontsize=12)
ax3.set_title('(c) Mean Square Displacement vs Time', fontsize=13, weight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)

print(f"   - MSD slope (simulation): {msd[-1]/n_steps:.3f}")
print(f"   - MSD slope (theory): 1.000")

# (d) Diffusion constant estimation
ax4 = fig.add_subplot(gs[1, 1])
# 확산 상수: D = MSD / (2t)
# 1D에서: <x^2> = 2Dt, 따라서 D = <x^2>/(2t)
diffusion_const = msd[1:] / (2 * time_steps[1:])
ax4.plot(time_steps[1:], diffusion_const, 'g-', linewidth=2)
ax4.axhline(y=0.5, color='r', linestyle='--', linewidth=2, 
            label='Theory: D=0.5 (step^2/time)')
ax4.set_xlabel('Time Step', fontsize=12)
ax4.set_ylabel('Diffusion Constant D', fontsize=12)
ax4.set_title('(d) Diffusion Constant Estimation', fontsize=13, weight='bold')
ax4.set_ylim([0.3, 0.7])
ax4.legend()
ax4.grid(True, alpha=0.3)

avg_D = np.mean(diffusion_const[100:])  # 초기 구간 제외
print(f"   - Average D (t>100): {avg_D:.4f} (theory: 0.5)")

plt.savefig(f'{output_dir}/01_random_walk_1d.png', dpi=150, bbox_inches='tight')
print(f"\n   Saved: {output_dir}/01_random_walk_1d.png")

# ============================================================================
# Simulation 2: 2D Random Walk
# ============================================================================

print("\n2. Simulating 2D Random Walks...")
n_steps_2d = 1000
n_walks_2d = 10

all_x, all_y = simulate_multiple_walks_2d(n_walks_2d, n_steps_2d)

fig2 = plt.figure(figsize=(15, 5))

# (a) 2D 궤적
ax1 = fig2.add_subplot(1, 3, 1)
for i in range(n_walks_2d):
    ax1.plot(all_x[i], all_y[i], alpha=0.6, linewidth=1)
    ax1.plot(all_x[i, 0], all_y[i, 0], 'go', markersize=8, label='Start' if i==0 else '')
    ax1.plot(all_x[i, -1], all_y[i, -1], 'ro', markersize=8, label='End' if i==0 else '')

ax1.set_xlabel('X Position', fontsize=12)
ax1.set_ylabel('Y Position', fontsize=12)
ax1.set_title('(a) 2D Random Walk Trajectories', fontsize=13, weight='bold')
ax1.grid(True, alpha=0.3)
ax1.axis('equal')
ax1.legend()

# (b) 최종 위치의 2D 분포
ax2 = fig2.add_subplot(1, 3, 2)
n_walks_2d_stat = 5000
all_x_stat, all_y_stat = simulate_multiple_walks_2d(n_walks_2d_stat, n_steps_2d)
final_x = all_x_stat[:, -1]
final_y = all_y_stat[:, -1]

h = ax2.hist2d(final_x, final_y, bins=40, cmap='hot', density=True)
plt.colorbar(h[3], ax=ax2, label='Probability Density')

# 이론적 원형 윤곽선 (2D 가우시안의 표준편차)
theta = np.linspace(0, 2*np.pi, 100)
for n_sigma in [1, 2, 3]:
    r = n_sigma * np.sqrt(n_steps_2d)
    ax2.plot(r*np.cos(theta), r*np.sin(theta), 'b--', linewidth=1.5, 
             alpha=0.7, label=f'{n_sigma}sigma' if n_sigma==1 else '')

ax2.set_xlabel('X Position', fontsize=12)
ax2.set_ylabel('Y Position', fontsize=12)
ax2.set_title(f'(b) Final Position Distribution (N={n_walks_2d_stat})', 
              fontsize=13, weight='bold')
ax2.axis('equal')
ax2.legend()

print(f"   - Mean X: {np.mean(final_x):.2f}, Mean Y: {np.mean(final_y):.2f}")
print(f"   - Std X: {np.std(final_x):.2f}, Std Y: {np.std(final_y):.2f}")
print(f"   - Theory Std: {np.sqrt(n_steps_2d):.2f}")

# (c) 2D MSD
ax3 = fig2.add_subplot(1, 3, 3)
n_walks_2d_msd = 1000
all_x_msd, all_y_msd = simulate_multiple_walks_2d(n_walks_2d_msd, n_steps_2d)

# MSD 계산: <r^2(t)> = <x^2(t) + y^2(t)>
r_squared = all_x_msd**2 + all_y_msd**2
msd_2d = np.mean(r_squared, axis=0)
msd_2d_theory = 2 * np.arange(n_steps_2d + 1)  # 2D: <r^2> = 2Dt, D=1

time_steps_2d = np.arange(n_steps_2d + 1)
ax3.plot(time_steps_2d, msd_2d, 'b-', linewidth=2, label='Simulation')
ax3.plot(time_steps_2d, msd_2d_theory, 'r--', linewidth=2, label='Theory: 2t')
ax3.set_xlabel('Time Step', fontsize=12)
ax3.set_ylabel('Mean Square Displacement', fontsize=12)
ax3.set_title('(c) 2D MSD vs Time', fontsize=13, weight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)

print(f"   - MSD slope (simulation): {msd_2d[-1]/n_steps_2d:.3f}")
print(f"   - MSD slope (theory): 2.000")

plt.tight_layout()
plt.savefig(f'{output_dir}/01_random_walk_2d.png', dpi=150, bbox_inches='tight')
print(f"   Saved: {output_dir}/01_random_walk_2d.png")

# ============================================================================
# Summary
# ============================================================================

print("\n" + "="*70)
print("Key Results:")
print("="*70)
print(f"1. 1D Random Walk:")
print(f"   - Position follows Gaussian distribution (Central Limit Theorem)")
print(f"   - MSD ~ t (linear relationship)")
print(f"   - Diffusion constant D ~ 0.5")
print(f"\n2. 2D Random Walk:")
print(f"   - Radial distribution follows 2D Gaussian")
print(f"   - MSD ~ 2t (linear, factor of 2 for 2D)")
print(f"   - Related to Brownian motion and diffusion")
print(f"\n3. Monte Carlo Principle:")
print(f"   - Random sampling creates statistical ensemble")
print(f"   - Average over many realizations gives physical properties")
print(f"   - Foundation for Metropolis algorithm (next programs)")
print("="*70)

plt.show()

