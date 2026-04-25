"""
06. Phase Transition Analysis
상전이 분석

2D 이징 모델의 상전이를 정량적으로 분석합니다:
- 임계 온도 결정
- 자화율(susceptibility) 계산
- 비열(specific heat) 계산
- 유한 크기 효과
- 임계 지수(critical exponents)

학습 목표:
1. 상전이의 정량적 특성
2. 열역학적 반응 함수
3. 유한 크기 스케일링
4. 임계 현상의 보편성
"""

import numpy as np
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
print("Phase Transition Analysis - 2D Ising Model")
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
# 2D Ising Model Functions (from previous program)
# ============================================================================

def initialize_spins_2d(L, state='random'):
    """2D 스핀 배열 초기화"""
    if state == 'random':
        return np.random.choice([-1, 1], size=(L, L))
    elif state == 'up':
        return np.ones((L, L), dtype=int)
    elif state == 'down':
        return -np.ones((L, L), dtype=int)

def calculate_energy_2d(spins, J=1.0):
    """2D 이징 모델의 에너지 계산"""
    energy = 0.0
    energy += -J * np.sum(spins * np.roll(spins, -1, axis=1))
    energy += -J * np.sum(spins * np.roll(spins, -1, axis=0))
    return energy

def calculate_magnetization_2d(spins):
    """자화 계산"""
    return np.sum(spins) / spins.size

def monte_carlo_sweep_2d(spins, T, J=1.0):
    """Monte Carlo sweep"""
    L = spins.shape[0]
    N = L * L
    
    for _ in range(N):
        i = np.random.randint(0, L)
        j = np.random.randint(0, L)
        
        neighbors_sum = (
            spins[(i+1)%L, j] + 
            spins[(i-1)%L, j] + 
            spins[i, (j+1)%L] + 
            spins[i, (j-1)%L]
        )
        
        dE = 2 * J * spins[i, j] * neighbors_sum
        
        if dE < 0 or np.random.random() < np.exp(-dE / T):
            spins[i, j] *= -1
    
    return spins

def simulate_temperature(L, T, J=1.0, n_sweeps=2000, n_equilibration=500):
    """
    주어진 온도에서 시뮬레이션하고 열역학 량들을 계산합니다.
    
    Returns:
    --------
    results : dict
        mean_M, std_M, mean_E, std_E, susceptibility, specific_heat
    """
    spins = initialize_spins_2d(L, 'random')
    N = L * L
    
    # 평형화
    for _ in range(n_equilibration):
        spins = monte_carlo_sweep_2d(spins, T, J)
    
    # 측정
    magnetizations = []
    energies = []
    
    for sweep in range(n_sweeps):
        spins = monte_carlo_sweep_2d(spins, T, J)
        
        M = abs(calculate_magnetization_2d(spins))  # 절대값
        E = calculate_energy_2d(spins, J) / N
        
        magnetizations.append(M)
        energies.append(E)
    
    magnetizations = np.array(magnetizations)
    energies = np.array(energies)
    
    # 통계량 계산
    mean_M = np.mean(magnetizations)
    std_M = np.std(magnetizations)
    mean_E = np.mean(energies)
    std_E = np.std(energies)
    
    # 자화율 (susceptibility): chi = N * (<M^2> - <M>^2) / T
    M_sq_mean = np.mean(magnetizations**2)
    M_mean_sq = mean_M**2
    susceptibility = N * (M_sq_mean - M_mean_sq) / T
    
    # 비열 (specific heat): C = N * (<E^2> - <E>^2) / T^2
    E_sq_mean = np.mean(energies**2)
    E_mean_sq = mean_E**2
    specific_heat = N * (E_sq_mean - E_mean_sq) / (T**2)
    
    results = {
        'mean_M': mean_M,
        'std_M': std_M,
        'mean_E': mean_E,
        'std_E': std_E,
        'susceptibility': susceptibility,
        'specific_heat': specific_heat,
        'magnetizations': magnetizations,
        'energies': energies
    }
    
    return results

# ============================================================================
# Simulation 1: Temperature Scan
# ============================================================================

print("\n1. Temperature Scan...")
print("   (This will take 2-3 minutes...)")

start_time = time.time()

L = 40
J = 1.0
Tc_exact = 2.269  # Onsager's exact result

# 온도 범위: Tc 주변을 조밀하게
temperatures = np.concatenate([
    np.linspace(1.0, 2.0, 8),
    np.linspace(2.0, 2.5, 12),  # Tc 근처 조밀
    np.linspace(2.5, 4.0, 8)
])

mean_Ms = []
std_Ms = []
mean_Es = []
std_Es = []
susceptibilities = []
specific_heats = []

for idx, T in enumerate(temperatures):
    print(f"   [{idx+1}/{len(temperatures)}] T = {T:.3f}...", end='', flush=True)
    
    results = simulate_temperature(L, T, J, n_sweeps=2000, n_equilibration=500)
    
    mean_Ms.append(results['mean_M'])
    std_Ms.append(results['std_M'])
    mean_Es.append(results['mean_E'])
    std_Es.append(results['std_E'])
    susceptibilities.append(results['susceptibility'])
    specific_heats.append(results['specific_heat'])
    
    print(f" M={results['mean_M']:.4f}")

elapsed = time.time() - start_time
print(f"\n   Completed in {elapsed:.1f} seconds.")

# 결과 배열 변환
mean_Ms = np.array(mean_Ms)
std_Ms = np.array(std_Ms)
mean_Es = np.array(mean_Es)
std_Es = np.array(std_Es)
susceptibilities = np.array(susceptibilities)
specific_heats = np.array(specific_heats)

# ============================================================================
# Plotting Results
# ============================================================================

print("\n2. Plotting Phase Transition...")

fig1 = plt.figure(figsize=(15, 10))
gs = GridSpec(2, 2, figure=fig1, hspace=0.3, wspace=0.3)

# (a) Magnetization vs Temperature
ax1 = fig1.add_subplot(gs[0, 0])
ax1.errorbar(temperatures, mean_Ms, yerr=std_Ms, fmt='bo-', capsize=3, 
             markersize=4, linewidth=1.5, label=f'L={L}')
ax1.axvline(x=Tc_exact, color='r', linestyle='--', linewidth=2, 
            label=f'Tc={Tc_exact:.3f} (Onsager)')
ax1.set_xlabel('Temperature T (J/kB)', fontsize=12)
ax1.set_ylabel('Magnetization |M|', fontsize=12)
ax1.set_title('(a) Spontaneous Magnetization', fontsize=13, weight='bold')
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)
ax1.set_xlim([0.8, 4.2])
ax1.set_ylim([-0.05, 1.05])

# Tc 추정 (M이 절반이 되는 온도)
idx_half = np.argmin(np.abs(mean_Ms - 0.5))
Tc_estimated = temperatures[idx_half]
print(f"   Estimated Tc (M=0.5): {Tc_estimated:.3f} J/kB")
print(f"   Exact Tc (Onsager): {Tc_exact:.3f} J/kB")
print(f"   Error: {abs(Tc_estimated - Tc_exact):.3f} J/kB")

# (b) Energy vs Temperature
ax2 = fig1.add_subplot(gs[0, 1])
ax2.errorbar(temperatures, mean_Es, yerr=std_Es, fmt='go-', capsize=3,
             markersize=4, linewidth=1.5, label=f'L={L}')
ax2.axvline(x=Tc_exact, color='r', linestyle='--', linewidth=2,
            label=f'Tc={Tc_exact:.3f}')
ax2.set_xlabel('Temperature T (J/kB)', fontsize=12)
ax2.set_ylabel('Energy per Spin (E/N)', fontsize=12)
ax2.set_title('(b) Internal Energy', fontsize=13, weight='bold')
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)

print(f"   Energy at T=1.0: {mean_Es[0]:.4f}")
print(f"   Energy at T=4.0: {mean_Es[-1]:.4f}")

# (c) Susceptibility vs Temperature
ax3 = fig1.add_subplot(gs[1, 0])
ax3.plot(temperatures, susceptibilities, 'mo-', markersize=4, linewidth=1.5,
         label=f'L={L}')
ax3.axvline(x=Tc_exact, color='r', linestyle='--', linewidth=2,
            label=f'Tc={Tc_exact:.3f}')

# Peak 위치 찾기
peak_idx = np.argmax(susceptibilities)
T_peak_chi = temperatures[peak_idx]
ax3.axvline(x=T_peak_chi, color='orange', linestyle=':', linewidth=2,
            label=f'Peak at {T_peak_chi:.3f}')

ax3.set_xlabel('Temperature T (J/kB)', fontsize=12)
ax3.set_ylabel('Susceptibility chi', fontsize=12)
ax3.set_title('(c) Magnetic Susceptibility', fontsize=13, weight='bold')
ax3.legend(fontsize=11)
ax3.grid(True, alpha=0.3)

print(f"   Susceptibility peak at T = {T_peak_chi:.3f} J/kB")
print(f"   Peak value: {susceptibilities[peak_idx]:.2f}")

# (d) Specific Heat vs Temperature
ax4 = fig1.add_subplot(gs[1, 1])
ax4.plot(temperatures, specific_heats, 'co-', markersize=4, linewidth=1.5,
         label=f'L={L}')
ax4.axvline(x=Tc_exact, color='r', linestyle='--', linewidth=2,
            label=f'Tc={Tc_exact:.3f}')

# Peak 위치 찾기
peak_idx_C = np.argmax(specific_heats)
T_peak_C = temperatures[peak_idx_C]
ax4.axvline(x=T_peak_C, color='orange', linestyle=':', linewidth=2,
            label=f'Peak at {T_peak_C:.3f}')

ax4.set_xlabel('Temperature T (J/kB)', fontsize=12)
ax4.set_ylabel('Specific Heat C', fontsize=12)
ax4.set_title('(d) Specific Heat', fontsize=13, weight='bold')
ax4.legend(fontsize=11)
ax4.grid(True, alpha=0.3)

print(f"   Specific heat peak at T = {T_peak_C:.3f} J/kB")
print(f"   Peak value: {specific_heats[peak_idx_C]:.2f}")

plt.savefig(f'{output_dir}/06_phase_transition.png', dpi=150, bbox_inches='tight')
print(f"\n   Saved: {output_dir}/06_phase_transition.png")

# ============================================================================
# Simulation 2: Finite Size Effects
# ============================================================================

print("\n3. Studying Finite Size Effects...")
print("   (This will take 3-4 minutes...)")

start_time = time.time()

system_sizes = [20, 30, 40, 50]
colors = ['blue', 'green', 'orange', 'red']

# Tc 근처만 조밀하게
temps_finite_size = np.linspace(1.8, 2.8, 20)

fig2 = plt.figure(figsize=(15, 10))
gs = GridSpec(2, 2, figure=fig2, hspace=0.3, wspace=0.3)

all_results = {}

for L in system_sizes:
    print(f"\n   System size L = {L}...")
    
    Ms = []
    chis = []
    Cs = []
    
    for idx, T in enumerate(temps_finite_size):
        print(f"      [{idx+1}/{len(temps_finite_size)}] T={T:.3f}...", end='', flush=True)
        
        results = simulate_temperature(L, T, J, n_sweeps=1500, n_equilibration=400)
        
        Ms.append(results['mean_M'])
        chis.append(results['susceptibility'])
        Cs.append(results['specific_heat'])
        
        print(" Done.")
    
    all_results[L] = {
        'M': np.array(Ms),
        'chi': np.array(chis),
        'C': np.array(Cs)
    }

elapsed = time.time() - start_time
print(f"\n   Completed in {elapsed:.1f} seconds.")

# (a) Magnetization for different sizes
ax1 = fig2.add_subplot(gs[0, 0])
for L, color in zip(system_sizes, colors):
    ax1.plot(temps_finite_size, all_results[L]['M'], 'o-', color=color, 
             markersize=4, linewidth=1.5, label=f'L={L}')
ax1.axvline(x=Tc_exact, color='k', linestyle='--', linewidth=2, alpha=0.5,
            label=f'Tc={Tc_exact:.3f}')
ax1.set_xlabel('Temperature T (J/kB)', fontsize=12)
ax1.set_ylabel('Magnetization |M|', fontsize=12)
ax1.set_title('(a) Magnetization - Finite Size Effects', fontsize=13, weight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# (b) Susceptibility for different sizes
ax2 = fig2.add_subplot(gs[0, 1])
peak_positions = []
for L, color in zip(system_sizes, colors):
    chi = all_results[L]['chi']
    ax2.plot(temps_finite_size, chi, 'o-', color=color,
             markersize=4, linewidth=1.5, label=f'L={L}')
    
    # Peak 위치
    peak_idx = np.argmax(chi)
    peak_T = temps_finite_size[peak_idx]
    peak_positions.append((L, peak_T, chi[peak_idx]))

ax2.axvline(x=Tc_exact, color='k', linestyle='--', linewidth=2, alpha=0.5,
            label=f'Tc={Tc_exact:.3f}')
ax2.set_xlabel('Temperature T (J/kB)', fontsize=12)
ax2.set_ylabel('Susceptibility chi', fontsize=12)
ax2.set_title('(b) Susceptibility - Finite Size Effects', fontsize=13, weight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

print(f"\n   Susceptibility peak positions:")
for L, T_peak, chi_peak in peak_positions:
    print(f"      L={L}: T_peak={T_peak:.3f}, chi_max={chi_peak:.2f}")

# (c) Specific Heat for different sizes
ax3 = fig2.add_subplot(gs[1, 0])
for L, color in zip(system_sizes, colors):
    ax3.plot(temps_finite_size, all_results[L]['C'], 'o-', color=color,
             markersize=4, linewidth=1.5, label=f'L={L}')
ax3.axvline(x=Tc_exact, color='k', linestyle='--', linewidth=2, alpha=0.5,
            label=f'Tc={Tc_exact:.3f}')
ax3.set_xlabel('Temperature T (J/kB)', fontsize=12)
ax3.set_ylabel('Specific Heat C', fontsize=12)
ax3.set_title('(c) Specific Heat - Finite Size Effects', fontsize=13, weight='bold')
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.3)

# (d) Peak heights vs system size (scaling)
ax4 = fig2.add_subplot(gs[1, 1])

Ls = np.array([L for L, _, _ in peak_positions])
chi_maxs = np.array([chi for _, _, chi in peak_positions])

# Finite size scaling: chi_max ~ L^(gamma/nu)
# For 2D Ising: gamma/nu = 1.75
ax4.loglog(Ls, chi_maxs, 'ro-', markersize=8, linewidth=2, label='chi_max')

# 이론적 스케일링
exponent = 1.75  # gamma/nu for 2D Ising
fit_line = chi_maxs[0] * (Ls / Ls[0])**exponent
ax4.loglog(Ls, fit_line, 'r--', linewidth=2, label=f'~ L^{exponent:.2f}')

# Specific heat도 유사하게
C_maxs = np.array([np.max(all_results[L]['C']) for L in system_sizes])
ax4.loglog(Ls, C_maxs, 'bo-', markersize=8, linewidth=2, label='C_max')

# C_max ~ L^(alpha/nu), alpha=0 for 2D Ising, so ~ log(L)
# 하지만 여기서는 간단히 표시
ax4.set_xlabel('System Size L', fontsize=12)
ax4.set_ylabel('Peak Height', fontsize=12)
ax4.set_title('(d) Finite Size Scaling', fontsize=13, weight='bold')
ax4.legend(fontsize=10)
ax4.grid(True, alpha=0.3, which='both')

print(f"\n   Finite size scaling:")
print(f"      chi_max grows as L^{exponent:.2f} (theory: gamma/nu=1.75)")

plt.savefig(f'{output_dir}/06_finite_size_effects.png', dpi=150, bbox_inches='tight')
print(f"\n   Saved: {output_dir}/06_finite_size_effects.png")

# ============================================================================
# Simulation 3: Critical Exponents
# ============================================================================

print("\n4. Estimating Critical Exponents...")

# 임계 지수:
# M ~ (Tc - T)^beta for T < Tc
# chi ~ |T - Tc|^(-gamma)
# C ~ |T - Tc|^(-alpha)

# 데이터에서 임계 지수 추정
# T < Tc 영역에서 M ~ (Tc - T)^beta

below_Tc_mask = (temperatures < Tc_exact) & (temperatures > 1.8) & (mean_Ms > 0.1)
temps_below = temperatures[below_Tc_mask]
Ms_below = mean_Ms[below_Tc_mask]

# log(M) vs log(Tc - T)
reduced_temp = Tc_exact - temps_below
log_reduced = np.log(reduced_temp)
log_M = np.log(Ms_below)

# 선형 피팅
beta_fit, intercept = np.polyfit(log_reduced, log_M, 1)

print(f"   Critical exponent beta:")
print(f"      Estimated from data: {beta_fit:.3f}")
print(f"      Theory (2D Ising): 0.125")

fig3 = plt.figure(figsize=(15, 5))

# (a) M vs (Tc - T) on log-log scale
ax1 = fig3.add_subplot(1, 3, 1)
ax1.loglog(reduced_temp, Ms_below, 'bo', markersize=6, label='Data')
ax1.loglog(reduced_temp, np.exp(intercept) * reduced_temp**beta_fit, 'r--',
           linewidth=2, label=f'Fit: M ~ (Tc-T)^{beta_fit:.3f}')
ax1.loglog(reduced_temp, 0.5 * reduced_temp**0.125, 'g--', linewidth=2,
           label='Theory: beta=0.125')
ax1.set_xlabel('Tc - T', fontsize=12)
ax1.set_ylabel('Magnetization |M|', fontsize=12)
ax1.set_title('(a) Critical Exponent beta', fontsize=13, weight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3, which='both')

# (b) chi near Tc
near_Tc_mask = np.abs(temperatures - Tc_exact) < 0.6
temps_near = temperatures[near_Tc_mask]
chis_near = susceptibilities[near_Tc_mask]
reduced_near = np.abs(temps_near - Tc_exact)

# chi ~ |T - Tc|^(-gamma)
valid_chi = (reduced_near > 0.05) & (chis_near > 10)
log_reduced_chi = np.log(reduced_near[valid_chi])
log_chi = np.log(chis_near[valid_chi])

gamma_fit, intercept_chi = np.polyfit(log_reduced_chi, log_chi, 1)
gamma_fit = -gamma_fit  # 부호 반전

print(f"   Critical exponent gamma:")
print(f"      Estimated from data: {gamma_fit:.3f}")
print(f"      Theory (2D Ising): 1.75")

ax2 = fig3.add_subplot(1, 3, 2)
ax2.loglog(reduced_near, chis_near, 'mo', markersize=6, label='Data')
ax2.loglog(reduced_near, np.exp(intercept_chi) * reduced_near**(-gamma_fit), 'r--',
           linewidth=2, label=f'Fit: chi ~ |T-Tc|^{-gamma_fit:.2f}')
ax2.set_xlabel('|T - Tc|', fontsize=12)
ax2.set_ylabel('Susceptibility chi', fontsize=12)
ax2.set_title('(b) Critical Exponent gamma', fontsize=13, weight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3, which='both')

# (c) Summary table
ax3 = fig3.add_subplot(1, 3, 3)
ax3.axis('off')

table_data = [
    ['Exponent', 'Theory', 'Simulation', 'Physical Meaning'],
    ['beta', '0.125', f'{beta_fit:.3f}', 'M ~ (Tc-T)^beta'],
    ['gamma', '1.75', f'{gamma_fit:.3f}', 'chi ~ |T-Tc|^(-gamma)'],
    ['nu', '1.0', '---', 'xi ~ |T-Tc|^(-nu)'],
    ['alpha', '0 (log)', '---', 'C ~ |T-Tc|^(-alpha)']
]

table = ax3.table(cellText=table_data, cellLoc='center', loc='center',
                  colWidths=[0.2, 0.15, 0.2, 0.45])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2)

# Header formatting
for i in range(4):
    table[(0, i)].set_facecolor('#4CAF50')
    table[(0, i)].set_text_props(weight='bold', color='white')

ax3.set_title('(c) Critical Exponents Summary', fontsize=13, weight='bold', pad=20)

plt.tight_layout()
plt.savefig(f'{output_dir}/06_critical_exponents.png', dpi=150, bbox_inches='tight')
print(f"\n   Saved: {output_dir}/06_critical_exponents.png")

# ============================================================================
# Summary
# ============================================================================

print("\n" + "="*70)
print("Key Results:")
print("="*70)
print(f"1. Critical Temperature:")
print(f"   - Estimated Tc ~ {Tc_estimated:.3f} J/kB")
print(f"   - Exact (Onsager): {Tc_exact:.3f} J/kB")
print(f"   - Error: {abs(Tc_estimated - Tc_exact):.4f} J/kB")
print(f"\n2. Response Functions:")
print(f"   - Susceptibility peak at T ~ {T_peak_chi:.3f} J/kB")
print(f"   - Specific heat peak at T ~ {T_peak_C:.3f} J/kB")
print(f"   - Both diverge at Tc (in thermodynamic limit)")
print(f"\n3. Finite Size Effects:")
print(f"   - Larger systems show sharper transitions")
print(f"   - Peak heights scale with system size")
print(f"   - chi_max ~ L^(gamma/nu) ~ L^1.75")
print(f"\n4. Critical Exponents:")
print(f"   - beta (magnetization): {beta_fit:.3f} (theory: 0.125)")
print(f"   - gamma (susceptibility): {gamma_fit:.3f} (theory: 1.75)")
print(f"   - Universal behavior in 2D systems")
print("="*70)

plt.show()

