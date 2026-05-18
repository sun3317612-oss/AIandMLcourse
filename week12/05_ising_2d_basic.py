"""
05. 2D Ising Model - Basic Simulation
2D 이징 모델 - 기본 시뮬레이션

드디어 상전이를 보여주는 2D 이징 모델입니다:
- 2D 격자 위의 스핀 시스템
- 온도에 따른 자발적 자화
- 임계 온도 근처의 행동
- 스핀 배열 시각화

학습 목표:
1. 2D 이징 모델의 기본 구현
2. 상전이 현상 관찰
3. 임계 온도 Tc ~ 2.269 확인
4. 스핀 배열의 시각화
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
print("2D Ising Model - Basic Simulation")
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
# 2D Ising Model Functions
# ============================================================================

def initialize_spins_2d(L, state='random'):
    """
    2D 스핀 배열을 초기화합니다.
    
    Parameters:
    -----------
    L : int
        격자 크기 (L x L)
    state : str
        'random', 'up', 'down'
    
    Returns:
    --------
    spins : array (L, L)
        스핀 배열
    """
    if state == 'random':
        return np.random.choice([-1, 1], size=(L, L))
    elif state == 'up':
        return np.ones((L, L), dtype=int)
    elif state == 'down':
        return -np.ones((L, L), dtype=int)

def calculate_energy_2d(spins, J=1.0):
    """
    2D 이징 모델의 에너지를 계산합니다.
    
    H = -J * sum_{<i,j>} s_i * s_j
    
    Parameters:
    -----------
    spins : array (L, L)
        스핀 배열
    J : float
        교환 상호작용
    
    Returns:
    --------
    energy : float
        총 에너지
    """
    # 주기적 경계 조건
    # 각 스핀은 4개의 이웃과 상호작용
    energy = 0.0
    
    # 오른쪽 이웃
    energy += -J * np.sum(spins * np.roll(spins, -1, axis=1))
    # 아래 이웃
    energy += -J * np.sum(spins * np.roll(spins, -1, axis=0))
    
    return energy

def calculate_magnetization_2d(spins):
    """
    자화를 계산합니다.
    
    M = sum(s_i) / N
    
    Parameters:
    -----------
    spins : array (L, L)
        스핀 배열
    
    Returns:
    --------
    magnetization : float
        단위 스핀당 자화
    """
    return np.sum(spins) / spins.size

def metropolis_step_2d(spins, T, J=1.0):
    """
    Metropolis 알고리즘으로 한 스텝을 수행합니다.
    
    Parameters:
    -----------
    spins : array (L, L)
        현재 스핀 배열
    T : float
        온도
    J : float
        상호작용
    
    Returns:
    --------
    spins : array (L, L)
        업데이트된 스핀 배열
    accepted : bool
        스핀 플립이 수락되었는지
    """
    L = spins.shape[0]
    
    # 무작위로 한 스핀 선택
    i = np.random.randint(0, L)
    j = np.random.randint(0, L)
    
    # 이웃 스핀들과의 상호작용 (4개 이웃)
    neighbors_sum = (
        spins[(i+1)%L, j] + 
        spins[(i-1)%L, j] + 
        spins[i, (j+1)%L] + 
        spins[i, (j-1)%L]
    )
    
    # 에너지 변화
    dE = 2 * J * spins[i, j] * neighbors_sum
    
    # Metropolis 기준
    if dE < 0 or np.random.random() < np.exp(-dE / T):
        spins[i, j] *= -1
        return spins, True
    
    return spins, False

def monte_carlo_sweep_2d(spins, T, J=1.0):
    """
    Monte Carlo sweep: L^2번의 스핀 플립 시도
    
    Parameters:
    -----------
    spins : array (L, L)
        스핀 배열
    T : float
        온도
    J : float
        상호작용
    
    Returns:
    --------
    spins : array (L, L)
        업데이트된 스핀 배열
    acceptance_rate : float
        수락률
    """
    L = spins.shape[0]
    N = L * L
    accepted = 0
    
    for _ in range(N):
        spins, acc = metropolis_step_2d(spins, T, J)
        if acc:
            accepted += 1
    
    return spins, accepted / N

def simulate_2d_ising(L, T, J=1.0, n_sweeps=1000, n_equilibration=200):
    """
    2D 이징 모델을 시뮬레이션합니다.
    
    Parameters:
    -----------
    L : int
        격자 크기
    T : float
        온도
    J : float
        상호작용
    n_sweeps : int
        Monte Carlo sweep 횟수
    n_equilibration : int
        평형화 sweep 횟수
    
    Returns:
    --------
    magnetizations : array
        각 sweep에서의 자화
    energies : array
        각 sweep에서의 에너지
    spins_final : array
        최종 스핀 배열
    spin_snapshots : list
        중간 스냅샷들
    """
    spins = initialize_spins_2d(L, 'random')
    
    magnetizations = np.zeros(n_sweeps)
    energies = np.zeros(n_sweeps)
    spin_snapshots = []
    snapshot_steps = [0, n_sweeps//4, n_sweeps//2, 3*n_sweeps//4, n_sweeps-1]
    
    # 평형화
    print(f"   Equilibrating (T={T:.3f})...", end='', flush=True)
    for _ in range(n_equilibration):
        spins, _ = monte_carlo_sweep_2d(spins, T, J)
    print(" Done.")
    
    # 측정
    print(f"   Measuring...", end='', flush=True)
    for sweep in range(n_sweeps):
        spins, _ = monte_carlo_sweep_2d(spins, T, J)
        magnetizations[sweep] = abs(calculate_magnetization_2d(spins))  # 절대값
        energies[sweep] = calculate_energy_2d(spins, J) / (L * L)  # per spin
        
        if sweep in snapshot_steps:
            spin_snapshots.append(spins.copy())
        
        if (sweep + 1) % 200 == 0:
            print(f".", end='', flush=True)
    
    print(" Done.")
    
    return magnetizations, energies, spins, spin_snapshots

# ============================================================================
# Simulation 1: Different Temperatures
# ============================================================================

print("\n1. Simulating at Different Temperatures...")

L = 50
J = 1.0
n_sweeps = 1000
n_equilibration = 200

# 임계 온도 Tc ≈ 2.269 J/kB (Onsager solution)
Tc = 2.269

temperatures = [1.5, 2.0, Tc, 3.0]  # T < Tc, T ~ Tc, T > Tc

fig1 = plt.figure(figsize=(15, 12))
gs = GridSpec(4, 4, figure=fig1, hspace=0.4, wspace=0.4)

for idx, T in enumerate(temperatures):
    print(f"\n   Temperature T = {T:.3f} J/kB (Tc = {Tc:.3f})...")
    
    mag, energy, spins_final, snapshots = simulate_2d_ising(
        L, T, J, n_sweeps, n_equilibration
    )
    
    # (a) 최종 스핀 배열
    ax_spin = fig1.add_subplot(gs[idx, 0])
    im = ax_spin.imshow(spins_final, cmap='RdBu_r', vmin=-1, vmax=1, interpolation='nearest')
    ax_spin.set_title(f'T={T:.2f} Final Config', fontsize=11, weight='bold')
    ax_spin.axis('off')
    if idx == 0:
        plt.colorbar(im, ax=ax_spin, fraction=0.046, pad=0.04, label='Spin')
    
    # (b) 자화 시계열
    ax_mag = fig1.add_subplot(gs[idx, 1])
    ax_mag.plot(mag, 'b-', alpha=0.7, linewidth=0.8)
    ax_mag.axhline(y=np.mean(mag), color='r', linestyle='--', linewidth=2, 
                   label=f'Mean={np.mean(mag):.3f}')
    ax_mag.set_xlabel('MC Sweep', fontsize=10)
    ax_mag.set_ylabel('|M|', fontsize=10)
    ax_mag.set_title(f'Magnetization', fontsize=11, weight='bold')
    ax_mag.legend(fontsize=8)
    ax_mag.grid(True, alpha=0.3)
    
    # (c) 에너지 시계열
    ax_energy = fig1.add_subplot(gs[idx, 2])
    ax_energy.plot(energy, 'g-', alpha=0.7, linewidth=0.8)
    ax_energy.axhline(y=np.mean(energy), color='r', linestyle='--', linewidth=2,
                      label=f'Mean={np.mean(energy):.3f}')
    ax_energy.set_xlabel('MC Sweep', fontsize=10)
    ax_energy.set_ylabel('E per spin', fontsize=10)
    ax_energy.set_title(f'Energy', fontsize=11, weight='bold')
    ax_energy.legend(fontsize=8)
    ax_energy.grid(True, alpha=0.3)
    
    # (d) 자화 히스토그램
    ax_hist = fig1.add_subplot(gs[idx, 3])
    ax_hist.hist(mag, bins=30, density=True, alpha=0.7, color='skyblue', edgecolor='black')
    ax_hist.axvline(x=np.mean(mag), color='r', linestyle='--', linewidth=2)
    ax_hist.set_xlabel('|M|', fontsize=10)
    ax_hist.set_ylabel('Probability', fontsize=10)
    ax_hist.set_title(f'Distribution', fontsize=11, weight='bold')
    ax_hist.grid(True, alpha=0.3)
    
    print(f"      Mean |M| = {np.mean(mag):.4f} +/- {np.std(mag):.4f}")
    print(f"      Mean E/N = {np.mean(energy):.4f} +/- {np.std(energy):.4f}")

plt.savefig(f'{output_dir}/05_ising_2d_temperatures.png', dpi=150, bbox_inches='tight')
print(f"\n   Saved: {output_dir}/05_ising_2d_temperatures.png")

# ============================================================================
# Simulation 2: Evolution at Different Temperatures
# ============================================================================

print("\n2. Visualizing Evolution...")

L = 40
n_sweeps = 1000
snapshot_times = [0, 50, 200, 500, 900]

fig2 = plt.figure(figsize=(15, 10))
gs = GridSpec(3, 5, figure=fig2, hspace=0.3, wspace=0.3)

temps_to_show = [1.5, Tc, 3.5]

for t_idx, T in enumerate(temps_to_show):
    print(f"\n   Evolving at T = {T:.3f}...")
    
    spins = initialize_spins_2d(L, 'random')
    
    # 스냅샷 저장
    snapshots = []
    sweep_count = 0
    
    for sweep in range(n_sweeps):
        spins, _ = monte_carlo_sweep_2d(spins, T, J)
        
        if sweep in snapshot_times:
            snapshots.append(spins.copy())
    
    # 스냅샷 표시
    for s_idx, (snapshot, sweep_num) in enumerate(zip(snapshots, snapshot_times)):
        ax = fig2.add_subplot(gs[t_idx, s_idx])
        im = ax.imshow(snapshot, cmap='RdBu_r', vmin=-1, vmax=1, interpolation='nearest')
        
        if s_idx == 0:
            ax.set_ylabel(f'T={T:.2f}', fontsize=12, weight='bold')
        if t_idx == 0:
            ax.set_title(f'Sweep {sweep_num}', fontsize=11, weight='bold')
        
        ax.axis('off')
        
        # Magnetization 표시
        mag = calculate_magnetization_2d(snapshot)
        ax.text(0.5, -0.1, f'M={mag:.3f}', transform=ax.transAxes,
                ha='center', fontsize=9)

plt.savefig(f'{output_dir}/05_ising_2d_evolution.png', dpi=150, bbox_inches='tight')
print(f"\n   Saved: {output_dir}/05_ising_2d_evolution.png")

# ============================================================================
# Simulation 3: Cluster Formation Near Tc
# ============================================================================

print("\n3. Analyzing Cluster Formation Near Tc...")

L = 60
temps_cluster = [Tc - 0.3, Tc, Tc + 0.3]

fig3 = plt.figure(figsize=(15, 5))

for idx, T in enumerate(temps_cluster):
    print(f"\n   T = {T:.3f}...")
    
    # Long simulation for better statistics
    mag, energy, spins_final, _ = simulate_2d_ising(L, T, J, n_sweeps=1500, n_equilibration=300)
    
    ax = fig3.add_subplot(1, 3, idx+1)
    im = ax.imshow(spins_final, cmap='RdBu_r', vmin=-1, vmax=1, interpolation='nearest')
    
    mag_mean = np.mean(mag)
    mag_std = np.std(mag)
    
    if T < Tc:
        phase_str = "Ordered (T < Tc)"
    elif abs(T - Tc) < 0.1:
        phase_str = "Critical (T ~ Tc)"
    else:
        phase_str = "Disordered (T > Tc)"
    
    ax.set_title(f'T = {T:.3f} J/kB\n{phase_str}\nM = {mag_mean:.3f} +/- {mag_std:.3f}',
                 fontsize=12, weight='bold')
    ax.axis('off')
    
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Spin')
    
    print(f"      Phase: {phase_str}")
    print(f"      Magnetization: {mag_mean:.4f} +/- {mag_std:.4f}")

plt.tight_layout()
plt.savefig(f'{output_dir}/05_ising_2d_clusters.png', dpi=150, bbox_inches='tight')
print(f"\n   Saved: {output_dir}/05_ising_2d_clusters.png")

# ============================================================================
# Summary
# ============================================================================

print("\n" + "="*70)
print("Key Results:")
print("="*70)
print(f"1. Phase Transition in 2D Ising Model:")
print(f"   - T < Tc: Ordered phase, high magnetization")
print(f"   - T = Tc: Critical point, large fluctuations")
print(f"   - T > Tc: Disordered phase, M ~ 0")
print(f"\n2. Critical Temperature:")
print(f"   - Tc = {Tc:.3f} J/kB (Onsager's exact solution)")
print(f"   - 2D Ising DOES have phase transition (unlike 1D)")
print(f"   - Dimension matters!")
print(f"\n3. Spin Configurations:")
print(f"   - Low T: Large domains of aligned spins")
print(f"   - Near Tc: Clusters of all sizes (critical opalescence)")
print(f"   - High T: Random, no correlation")
print(f"\n4. Fluctuations:")
print(f"   - Maximum fluctuations near Tc")
print(f"   - Susceptibility diverges at Tc")
print(f"   - Related to correlation length")
print("="*70)

plt.show()

