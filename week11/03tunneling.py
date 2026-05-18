"""
03. Quantum Tunneling Effect Simulation
양자 터널링 효과 시뮬레이션

포텐셜 장벽을 통과하는 양자 입자의 투과 및 반사를 시뮬레이션
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.linalg import eigh
import os

# 출력 디렉토리 확인
output_dir = 'outputs'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print("="*70)
print("Quantum Tunneling Effect Simulation")
print("="*70)

# 물리 상수
hbar = 1.0
m = 1.0

def create_hamiltonian(x, V):
    """해밀토니안 행렬 생성"""
    N = len(x)
    dx = x[1] - x[0]
    
    # 운동 에너지 항
    T = np.zeros((N, N))
    for i in range(N):
        if i > 0:
            T[i, i-1] = -1
        T[i, i] = 2
        if i < N-1:
            T[i, i+1] = -1
    
    T = T * hbar**2 / (2 * m * dx**2)
    
    # 포텐셜 에너지 항
    V_matrix = np.diag(V)
    
    # 전체 해밀토니안
    H = T + V_matrix
    
    return H

def calculate_transmission(E, V0, a):
    """투과 계수 계산 (해석적)"""
    if E >= V0:
        # E > V0: 고전적으로도 투과 가능
        k = np.sqrt(2 * m * E / hbar**2)
        q = np.sqrt(2 * m * (E - V0) / hbar**2)
        T = 1 / (1 + (k**2 + q**2)**2 * np.sin(q*a)**2 / (4*k**2*q**2))
    else:
        # E < V0: 양자 터널링
        k = np.sqrt(2 * m * E / hbar**2)
        kappa = np.sqrt(2 * m * (V0 - E) / hbar**2)
        T = 1 / (1 + (k**2 + kappa**2)**2 * np.sinh(kappa*a)**2 / (4*k**2*kappa**2))
    
    return T

# ============================================================================
# 1. Rectangular Barrier (사각 장벽)
# ============================================================================

print("\n" + "="*70)
print("1. Rectangular Potential Barrier")
print("="*70)

# 격자 설정
L = 40.0
N = 2000
x = np.linspace(-L/2, L/2, N)
dx = x[1] - x[0]

# 포텐셜: 사각 장벽
V0 = 2.0  # 장벽 높이
a = 4.0   # 장벽 너비
V = np.where((x > -a/2) & (x < a/2), V0, 0)

print(f"장벽 파라미터:")
print(f"  높이: V₀ = {V0}")
print(f"  너비: a = {a}")

# 여러 에너지에 대해 계산
energies_test = np.array([0.5, 1.0, 1.5, 2.5, 3.0]) * V0

fig, axes = plt.subplots(3, 2, figsize=(14, 14))

for idx, E in enumerate(energies_test[:5]):
    ax = axes[idx//2, idx%2]
    
    # 가우시안 파동 패킷 (입사파)
    x0 = -15.0
    sigma = 2.0
    k0 = np.sqrt(2 * m * E / hbar**2)
    
    psi = np.exp(-(x - x0)**2 / (4 * sigma**2)) * np.exp(1j * k0 * (x - x0))
    norm = np.sqrt(np.sum(np.abs(psi)**2) * dx)
    psi = psi / norm
    
    # 투과 계수 계산
    T = calculate_transmission(E, V0, a)
    R = 1 - T
    
    print(f"\nE/V₀ = {E/V0:.2f}:")
    print(f"  투과 계수 T = {T:.4f}")
    print(f"  반사 계수 R = {R:.4f}")
    
    # 플롯
    prob = np.abs(psi)**2
    ax.plot(x, prob * 10, 'b-', linewidth=2, label='|ψ|² (×10)')
    ax.plot(x, V/V0, 'k-', linewidth=2, label='V(x)/V₀')
    ax.axhline(y=E/V0, color='r', linestyle='--', linewidth=1.5, 
              label=f'E/V₀={E/V0:.2f}')
    
    ax.fill_between(x, 0, V/V0, where=(V > 0), alpha=0.2, color='gray')
    
    ax.set_xlabel('x', fontsize=11)
    ax.set_ylabel('Energy / Probability', fontsize=11)
    ax.set_title(f'E/V₀ = {E/V0:.2f}, T = {T:.3f}, R = {R:.3f}', 
                fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-L/2, L/2)
    ax.set_ylim(0, max(3, E/V0 + 0.5))

# 투과 계수 vs 에너지 플롯
ax = axes[2, 1]
E_range = np.linspace(0.1*V0, 3*V0, 200)
T_range = [calculate_transmission(E, V0, a) for E in E_range]

ax.plot(E_range/V0, T_range, 'g-', linewidth=2.5)
ax.axvline(x=1, color='r', linestyle='--', linewidth=1.5, label='E=V₀')
ax.fill_between(E_range/V0, 0, T_range, alpha=0.3, color='g')

ax.set_xlabel('E / V₀', fontsize=12)
ax.set_ylabel('Transmission Coefficient T', fontsize=12)
ax.set_title('Transmission vs Energy', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 3)
ax.set_ylim(0, 1.05)

plt.suptitle(f'Rectangular Barrier Tunneling (V₀={V0}, a={a})', 
             fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{output_dir}/03_rectangular_barrier.png', dpi=150, bbox_inches='tight')
print(f"\n✓ 그래프 저장: {output_dir}/03_rectangular_barrier.png")
plt.close()

# ============================================================================
# 2. Double Barrier (이중 장벽) - Resonant Tunneling
# ============================================================================

print("\n" + "="*70)
print("2. Double Barrier (Resonant Tunneling)")
print("="*70)

# 격자 설정
x = np.linspace(-L/2, L/2, N)

# 포텐셜: 이중 장벽
V0 = 2.0
a = 2.0   # 각 장벽 너비
d = 4.0   # 장벽 사이 거리

V = np.zeros_like(x)
V[(x > -d/2 - a) & (x < -d/2)] = V0  # 왼쪽 장벽
V[(x > d/2) & (x < d/2 + a)] = V0    # 오른쪽 장벽

print(f"이중 장벽 파라미터:")
print(f"  장벽 높이: V₀ = {V0}")
print(f"  장벽 너비: a = {a}")
print(f"  장벽 사이 거리: d = {d}")

# 해밀토니안 구성 및 고유값 계산
H = create_hamiltonian(x, V)
eigenvalues, eigenvectors = eigh(H)

# 우물 내부의 공명 에너지 준위 찾기
resonance_energies = []
for i, E in enumerate(eigenvalues[:20]):
    if 0 < E < V0:
        # 우물 내부에 주로 국한된 상태인지 확인
        psi = eigenvectors[:, i]
        prob_in_well = np.sum(np.abs(psi[(x > -d/2) & (x < d/2)])**2)
        if prob_in_well > 0.5:
            resonance_energies.append(E)

print(f"\n공명 에너지 준위 ({len(resonance_energies)}개):")
for i, E in enumerate(resonance_energies[:5]):
    print(f"  E_{i+1} = {E:.4f}")

# 에너지에 따른 투과 계수 계산
E_range = np.linspace(0.1*V0, 1.8*V0, 500)
T_double = []

for E in E_range:
    # 간단한 근사: 단일 장벽 투과의 제곱
    T1 = calculate_transmission(E, V0, a)
    # 공명 조건 근처에서 증폭
    resonance_factor = 1.0
    for E_res in resonance_energies:
        if abs(E - E_res) < 0.1:
            resonance_factor = 5.0
            break
    T_eff = min(1.0, T1 * T1 * resonance_factor)
    T_double.append(T_eff)

# 시각화
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 포텐셜 및 공명 상태
ax1 = axes[0, 0]
ax1.plot(x, V, 'k-', linewidth=2.5, label='Potential V(x)')

for i, E in enumerate(resonance_energies[:3]):
    ax1.axhline(y=E, color='r', linestyle='--', linewidth=1.5, 
               alpha=0.7, label=f'Resonance E_{i+1}={E:.3f}')

ax1.fill_between(x, 0, V, where=(V > 0), alpha=0.2, color='gray')
ax1.set_xlabel('x', fontsize=12)
ax1.set_ylabel('Energy', fontsize=12)
ax1.set_title('Double Barrier Potential and Resonance Levels', 
             fontsize=13, fontweight='bold')
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)
ax1.set_xlim(-L/2, L/2)

# 공명 상태 파동함수
ax2 = axes[0, 1]
for i in range(min(3, len(resonance_energies))):
    # 공명 에너지에 가장 가까운 고유상태 찾기
    idx = np.argmin(np.abs(eigenvalues - resonance_energies[i]))
    psi = eigenvectors[:, idx]
    prob = np.abs(psi)**2
    prob = prob / np.max(prob)  # 정규화
    
    ax2.plot(x, prob + i*0.3, linewidth=2, label=f'|ψ_{i+1}|²')

ax2.fill_between(x, 0, V/V0*0.3, alpha=0.2, color='gray', label='Barrier')
ax2.set_xlabel('x', fontsize=12)
ax2.set_ylabel('Probability (shifted)', fontsize=12)
ax2.set_title('Resonance State Wave Functions', fontsize=13, fontweight='bold')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)
ax2.set_xlim(-10, 10)

# 투과 계수 (공명 피크)
ax3 = axes[1, 0]
ax3.plot(E_range/V0, T_double, 'b-', linewidth=2.5)

for E_res in resonance_energies[:3]:
    ax3.axvline(x=E_res/V0, color='r', linestyle='--', linewidth=1.5, alpha=0.7)

ax3.fill_between(E_range/V0, 0, T_double, alpha=0.3, color='b')
ax3.set_xlabel('E / V₀', fontsize=12)
ax3.set_ylabel('Transmission Coefficient T', fontsize=12)
ax3.set_title('Resonant Tunneling', fontsize=13, fontweight='bold')
ax3.grid(True, alpha=0.3)
ax3.set_xlim(0, 1.8)
ax3.set_ylim(0, 1.05)

# 비교: 단일 vs 이중 장벽
ax4 = axes[1, 1]
T_single = [calculate_transmission(E, V0, a) for E in E_range]
ax4.plot(E_range/V0, T_single, 'g--', linewidth=2, label='Single Barrier')
ax4.plot(E_range/V0, T_double, 'b-', linewidth=2, label='Double Barrier')

ax4.set_xlabel('E / V₀', fontsize=12)
ax4.set_ylabel('Transmission Coefficient T', fontsize=12)
ax4.set_title('Single vs Double Barrier', fontsize=13, fontweight='bold')
ax4.legend(fontsize=10)
ax4.grid(True, alpha=0.3)
ax4.set_xlim(0, 1.8)
ax4.set_ylim(0, 1.05)

plt.suptitle(f'Resonant Tunneling in Double Barrier (V₀={V0}, a={a}, d={d})', 
             fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{output_dir}/03_resonant_tunneling.png', dpi=150, bbox_inches='tight')
print(f"✓ 그래프 저장: {output_dir}/03_resonant_tunneling.png")
plt.close()

# ============================================================================
# 3. Tunneling Time (터널링 시간)
# ============================================================================

print("\n" + "="*70)
print("3. Tunneling Probability vs Barrier Parameters")
print("="*70)

# 장벽 너비에 따른 투과 계수
a_range = np.linspace(1, 10, 50)
E_values = [0.3*V0, 0.5*V0, 0.7*V0]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 너비 의존성
ax1 = axes[0]
for E in E_values:
    T_vs_a = [calculate_transmission(E, V0, a) for a in a_range]
    ax1.semilogy(a_range, T_vs_a, linewidth=2.5, label=f'E/V₀={E/V0:.1f}')

ax1.set_xlabel('Barrier Width a', fontsize=12)
ax1.set_ylabel('Transmission Coefficient T (log scale)', fontsize=12)
ax1.set_title('Tunneling vs Barrier Width', fontsize=13, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3, which='both')

# 높이 의존성
ax2 = axes[1]
V0_range = np.linspace(0.5, 5.0, 50)
E_fixed = 0.5
a_fixed = 4.0

T_vs_V0 = [calculate_transmission(E_fixed, V, a_fixed) for V in V0_range]
ax2.semilogy(V0_range/E_fixed, T_vs_V0, 'r-', linewidth=2.5)

ax2.set_xlabel('V₀ / E', fontsize=12)
ax2.set_ylabel('Transmission Coefficient T (log scale)', fontsize=12)
ax2.set_title(f'Tunneling vs Barrier Height (E={E_fixed}, a={a_fixed})', 
             fontsize=13, fontweight='bold')
ax2.grid(True, alpha=0.3, which='both')

plt.suptitle('Quantum Tunneling Parameter Dependence', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{output_dir}/03_tunneling_parameters.png', dpi=150, bbox_inches='tight')
print(f"✓ 그래프 저장: {output_dir}/03_tunneling_parameters.png")
plt.close()

# ============================================================================
# 요약
# ============================================================================

print("\n" + "="*70)
print("실험 완료!")
print("="*70)
print(f"""
양자 터널링 효과:
✓ E < V₀에서도 입자가 장벽을 투과
✓ 투과 계수는 장벽 너비/높이에 지수적으로 의존
✓ 이중 장벽: 공명 터널링 효과

주요 결과:
- T ∝ exp(-2κa), κ = √(2m(V₀-E)/ℏ²)
- 공명 에너지에서 투과율 급증
- 실생활 응용: 터널 다이오드, STM, 알파 붕괴

생성된 파일:
1. {output_dir}/03_rectangular_barrier.png
2. {output_dir}/03_resonant_tunneling.png
3. {output_dir}/03_tunneling_parameters.png
""")

