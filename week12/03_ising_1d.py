"""
03. 1D Ising Model Simulation
1D 이징 모델 시뮬레이션

이징 모델의 가장 간단한 형태인 1D 버전을 다룹니다:
- 스핀 상호작용과 열역학
- Metropolis 알고리즘 도입
- 정확해와 비교
- 2D로 확장하기 전 준비

학습 목표:
1. 이징 모델의 기본 개념 이해
2. 상호작용하는 스핀 시스템
3. 온도에 따른 자화 변화
4. Monte Carlo 시뮬레이션 시작
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
print("1D Ising Model Simulation")
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
# 1D Ising Model Functions
# ============================================================================

def initialize_spins_1d(N, state='random'):
    """
    1D 스핀 배열을 초기화합니다.
    
    Parameters:
    -----------
    N : int
        스핀의 개수
    state : str
        'random': 무작위 배치
        'up': 모두 위 (+1)
        'down': 모두 아래 (-1)
    
    Returns:
    --------
    spins : array
        스핀 배열 (+1 또는 -1)
    """
    if state == 'random':
        return np.random.choice([-1, 1], size=N)
    elif state == 'up':
        return np.ones(N, dtype=int)
    elif state == 'down':
        return -np.ones(N, dtype=int)

def calculate_energy_1d(spins, J=1.0, h=0.0):
    """
    1D 이징 모델의 에너지를 계산합니다.
    
    H = -J * sum(s_i * s_{i+1}) - h * sum(s_i)
    
    Parameters:
    -----------
    spins : array
        스핀 배열
    J : float
        교환 상호작용 (J > 0: 강자성, J < 0: 반강자성)
    h : float
        외부 자기장
    
    Returns:
    --------
    energy : float
        총 에너지
    """
    N = len(spins)
    
    # 주기적 경계 조건 (periodic boundary condition)
    interaction_energy = -J * np.sum(spins * np.roll(spins, -1))
    field_energy = -h * np.sum(spins)
    
    return interaction_energy + field_energy

def calculate_magnetization_1d(spins):
    """
    자화(magnetization)를 계산합니다.
    
    M = sum(s_i) / N
    
    Parameters:
    -----------
    spins : array
        스핀 배열
    
    Returns:
    --------
    magnetization : float
        단위 스핀당 자화
    """
    return np.sum(spins) / len(spins)

def metropolis_step_1d(spins, T, J=1.0, h=0.0):
    """
    Metropolis 알고리즘으로 한 스텝을 수행합니다.
    
    Parameters:
    -----------
    spins : array
        현재 스핀 배열
    T : float
        온도 (kB = 1 단위)
    J, h : float
        상호작용 및 자기장
    
    Returns:
    --------
    spins : array
        업데이트된 스핀 배열
    accepted : bool
        스핀 플립이 수락되었는지 여부
    """
    N = len(spins)
    
    # 무작위로 한 스핀 선택
    i = np.random.randint(0, N)
    
    # 에너지 변화 계산 (스핀을 뒤집었을 때)
    # 이웃 스핀들과의 상호작용만 변화
    left = (i - 1) % N
    right = (i + 1) % N
    
    dE = 2 * J * spins[i] * (spins[left] + spins[right]) + 2 * h * spins[i]
    
    # Metropolis 기준
    if dE < 0 or np.random.random() < np.exp(-dE / T):
        spins[i] *= -1
        return spins, True
    
    return spins, False

def simulate_1d_ising(N, T, J=1.0, h=0.0, n_steps=10000, n_equilibration=2000):
    """
    1D 이징 모델을 시뮬레이션합니다.
    
    Parameters:
    -----------
    N : int
        스핀 개수
    T : float
        온도
    J, h : float
        상호작용 및 자기장
    n_steps : int
        Monte Carlo 스텝 수
    n_equilibration : int
        평형 도달을 위한 초기 스텝 수
    
    Returns:
    --------
    magnetizations : array
        각 스텝에서의 자화
    energies : array
        각 스텝에서의 에너지
    spins_final : array
        최종 스핀 배열
    """
    spins = initialize_spins_1d(N, 'random')
    
    magnetizations = np.zeros(n_steps)
    energies = np.zeros(n_steps)
    
    # 평형화 (equilibration)
    for _ in range(n_equilibration):
        spins, _ = metropolis_step_1d(spins, T, J, h)
    
    # 측정
    for step in range(n_steps):
        spins, _ = metropolis_step_1d(spins, T, J, h)
        magnetizations[step] = calculate_magnetization_1d(spins)
        energies[step] = calculate_energy_1d(spins, J, h) / N  # per spin
    
    return magnetizations, energies, spins

def exact_magnetization_1d(T, J=1.0, h=0.0):
    """
    1D 이징 모델의 정확한 자화를 계산합니다.
    
    h=0일 때, 1D 이징 모델은 모든 온도에서 M=0
    h!=0일 때는 다른 결과
    
    Parameters:
    -----------
    T : float
        온도
    J, h : float
        상호작용 및 자기장
    
    Returns:
    --------
    magnetization : float
        열역학적 자화
    """
    if h == 0:
        return 0.0
    else:
        # h != 0인 경우 근사해
        return np.tanh(h / T)

# ============================================================================
# Simulation 1: Basic 1D Ising Model
# ============================================================================

print("\n1. Simulating 1D Ising Model at Different Temperatures...")

N = 100
J = 1.0
h = 0.0
n_steps = 10000
temperatures = [0.5, 1.0, 2.0, 5.0]

fig1 = plt.figure(figsize=(15, 10))
gs = GridSpec(2, 2, figure=fig1, hspace=0.3, wspace=0.3)

for idx, T in enumerate(temperatures):
    print(f"   Simulating T = {T:.1f}...")
    
    mag, energy, spins_final = simulate_1d_ising(N, T, J, h, n_steps)
    
    ax = fig1.add_subplot(gs[idx // 2, idx % 2])
    
    # 스핀 배열 시각화 (위쪽)
    ax_spin = ax.inset_axes([0, 0.85, 1, 0.15])
    spin_colors = np.where(spins_final > 0, 1, -1)
    ax_spin.imshow([spin_colors], cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
    ax_spin.set_xticks([])
    ax_spin.set_yticks([])
    ax_spin.set_ylabel('Spins', fontsize=9)
    
    # 자화 시계열
    ax.plot(mag, 'b-', alpha=0.7, linewidth=0.5)
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    
    # 이동 평균
    window = 200
    mag_smooth = np.convolve(mag, np.ones(window)/window, mode='valid')
    ax.plot(np.arange(window-1, n_steps), mag_smooth, 'r-', linewidth=2, 
            label=f'Moving avg (w={window})')
    
    ax.set_xlabel('Monte Carlo Step', fontsize=11)
    ax.set_ylabel('Magnetization per Spin', fontsize=11)
    ax.set_title(f'T = {T:.1f} J/kB\nMean M = {np.mean(mag):.4f}, Std M = {np.std(mag):.4f}', 
                 fontsize=12, weight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    print(f"      Mean M = {np.mean(mag):.6f}, Std M = {np.std(mag):.6f}")

plt.savefig(f'{output_dir}/03_ising_1d_temperatures.png', dpi=150, bbox_inches='tight')
print(f"\n   Saved: {output_dir}/03_ising_1d_temperatures.png")

# ============================================================================
# Simulation 2: Temperature Scan
# ============================================================================

print("\n2. Temperature Scan...")

N = 100
J = 1.0
h = 0.0
temperatures = np.linspace(0.1, 5.0, 25)
n_steps = 5000

mean_mags = []
std_mags = []
mean_energies = []
specific_heats = []

for T in temperatures:
    mag, energy, _ = simulate_1d_ising(N, T, J, h, n_steps)
    
    mean_mags.append(np.abs(np.mean(mag)))  # 절대값 (대칭성 때문)
    std_mags.append(np.std(mag))
    mean_energies.append(np.mean(energy))
    
    # 비열 계산: C = (< E^2> - <E>^2) / (kB * T^2)
    energy_sq_mean = np.mean(energy**2)
    energy_mean_sq = np.mean(energy)**2
    C = (energy_sq_mean - energy_mean_sq) / (T**2)
    specific_heats.append(C)

fig2 = plt.figure(figsize=(15, 5))

# (a) Magnetization vs Temperature
ax1 = fig2.add_subplot(1, 3, 1)
ax1.errorbar(temperatures, mean_mags, yerr=std_mags, fmt='bo-', 
             capsize=3, markersize=4, linewidth=1.5, label='Simulation')

# 1D 이징 모델은 h=0일 때 모든 온도에서 M=0 (정확해)
ax1.axhline(y=0, color='r', linestyle='--', linewidth=2, label='Exact (M=0)')

ax1.set_xlabel('Temperature (J/kB)', fontsize=12)
ax1.set_ylabel('|Magnetization| per Spin', fontsize=12)
ax1.set_title('(a) Magnetization vs Temperature', fontsize=13, weight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# (b) Energy vs Temperature
ax2 = fig2.add_subplot(1, 3, 2)
ax2.plot(temperatures, mean_energies, 'go-', markersize=4, linewidth=1.5, label='Simulation')

# 정확해: E/N = -J * tanh(J/T)
exact_energies = -J * np.tanh(J / temperatures)
ax2.plot(temperatures, exact_energies, 'r--', linewidth=2, label='Exact solution')

ax2.set_xlabel('Temperature (J/kB)', fontsize=12)
ax2.set_ylabel('Energy per Spin', fontsize=12)
ax2.set_title('(b) Energy vs Temperature', fontsize=13, weight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

print(f"   Energy comparison at T=1.0:")
idx_t1 = np.argmin(np.abs(temperatures - 1.0))
print(f"      Simulation: {mean_energies[idx_t1]:.6f}")
print(f"      Exact: {exact_energies[idx_t1]:.6f}")

# (c) Specific Heat vs Temperature
ax3 = fig2.add_subplot(1, 3, 3)
ax3.plot(temperatures, specific_heats, 'mo-', markersize=4, linewidth=1.5, label='Simulation')

# 정확해: C/N = (J/T)^2 * sech^2(J/T)
exact_specific_heats = (J / temperatures)**2 / np.cosh(J / temperatures)**2
ax3.plot(temperatures, exact_specific_heats, 'r--', linewidth=2, label='Exact solution')

ax3.set_xlabel('Temperature (J/kB)', fontsize=12)
ax3.set_ylabel('Specific Heat per Spin', fontsize=12)
ax3.set_title('(c) Specific Heat vs Temperature', fontsize=13, weight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{output_dir}/03_ising_1d_thermodynamics.png', dpi=150, bbox_inches='tight')
print(f"\n   Saved: {output_dir}/03_ising_1d_thermodynamics.png")

# ============================================================================
# Simulation 3: Correlation Function
# ============================================================================

print("\n3. Calculating Spin Correlation Function...")

N = 200
T = 1.0
J = 1.0
h = 0.0
n_samples = 5000

# 상관 함수: C(r) = <s_i * s_{i+r}>
max_distance = N // 2
correlations = np.zeros(max_distance)

spins = initialize_spins_1d(N, 'random')

# 평형화
for _ in range(2000):
    spins, _ = metropolis_step_1d(spins, T, J, h)

# 샘플링
for _ in range(n_samples):
    spins, _ = metropolis_step_1d(spins, T, J, h)
    
    for r in range(max_distance):
        correlations[r] += np.mean(spins * np.roll(spins, r))

correlations /= n_samples

# 정규화: C(0) = 1
correlations /= correlations[0]

fig3 = plt.figure(figsize=(15, 5))

# (a) Correlation function (linear)
ax1 = fig3.add_subplot(1, 3, 1)
distances = np.arange(max_distance)
ax1.plot(distances, correlations, 'bo-', markersize=3, linewidth=1.5, label='Simulation')

# 이론적 상관 함수: C(r) = exp(-r/xi), xi = 1/ln[coth(J/T)]
xi = 1.0 / np.log(1.0 / np.tanh(J / T))
correlation_theory = np.exp(-distances / xi)
ax1.plot(distances, correlation_theory, 'r--', linewidth=2, 
         label=f'Theory: exp(-r/xi), xi={xi:.2f}')

ax1.set_xlabel('Distance r', fontsize=12)
ax1.set_ylabel('Correlation C(r)', fontsize=12)
ax1.set_title(f'(a) Spin Correlation Function (T={T} J/kB)', fontsize=13, weight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# (b) Correlation function (semi-log)
ax2 = fig3.add_subplot(1, 3, 2)
ax2.semilogy(distances[1:], np.abs(correlations[1:]), 'bo-', markersize=3, 
             linewidth=1.5, label='|Simulation|')
ax2.semilogy(distances[1:], correlation_theory[1:], 'r--', linewidth=2, label='Theory')

ax2.set_xlabel('Distance r', fontsize=12)
ax2.set_ylabel('|Correlation C(r)|', fontsize=12)
ax2.set_title('(b) Semi-log Plot', fontsize=13, weight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3, which='both')

print(f"   Correlation length:")
print(f"      Theory: xi = {xi:.3f}")
# 시뮬레이션에서 xi 추정 (C(r) ~ exp(-r/xi))
# log(C(r)) = -r/xi
valid_idx = (correlations > 0.01) & (distances > 0)
if np.sum(valid_idx) > 5:
    fit_distances = distances[valid_idx]
    fit_corr = np.log(correlations[valid_idx])
    slope, _ = np.polyfit(fit_distances[:10], fit_corr[:10], 1)
    xi_sim = -1.0 / slope
    print(f"      Simulation: xi = {xi_sim:.3f}")

# (c) Correlation length vs Temperature
ax3 = fig3.add_subplot(1, 3, 3)
temps_corr = np.linspace(0.5, 5.0, 20)
xi_values = 1.0 / np.log(1.0 / np.tanh(J / temps_corr))

ax3.plot(temps_corr, xi_values, 'r-', linewidth=2)
ax3.axhline(y=N/4, color='k', linestyle='--', alpha=0.5, 
            label=f'System size L/4 = {N/4:.0f}')

ax3.set_xlabel('Temperature (J/kB)', fontsize=12)
ax3.set_ylabel('Correlation Length xi', fontsize=12)
ax3.set_title('(c) Correlation Length vs Temperature', fontsize=13, weight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)
ax3.set_ylim([0, 20])

plt.tight_layout()
plt.savefig(f'{output_dir}/03_ising_1d_correlation.png', dpi=150, bbox_inches='tight')
print(f"\n   Saved: {output_dir}/03_ising_1d_correlation.png")

# ============================================================================
# Summary
# ============================================================================

print("\n" + "="*70)
print("Key Results:")
print("="*70)
print(f"1. 1D Ising Model Physics:")
print(f"   - NO phase transition at finite temperature (h=0)")
print(f"   - Magnetization M=0 for all T > 0")
print(f"   - Only T=0 has ordered state")
print(f"\n2. Thermodynamic Properties:")
print(f"   - Energy: E/N = -J * tanh(J/T)")
print(f"   - Specific heat: C/N = (J/T)^2 * sech^2(J/T)")
print(f"   - Matches exact solution very well")
print(f"\n3. Correlations:")
print(f"   - Exponential decay: C(r) ~ exp(-r/xi)")
print(f"   - Correlation length: xi = 1/ln[coth(J/T)]")
print(f"   - Finite correlation length at all T > 0")
print(f"\n4. Lesson for 2D:")
print(f"   - 2D Ising DOES have phase transition!")
print(f"   - Critical temperature Tc ~ 2.269 J/kB")
print(f"   - Dimension matters in statistical physics")
print("="*70)

plt.show()

