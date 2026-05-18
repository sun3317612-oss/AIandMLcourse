"""
04. Finite Well & Harmonic Oscillator - Detailed Analysis
유한 우물과 조화 진동자 상세 분석

양자역학의 대표적인 포텐셜에서의 파동함수와 에너지 준위
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.linalg import eigh
from scipy.special import hermite, factorial
import os

# 출력 디렉토리 확인
output_dir = 'outputs'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print("="*70)
print("Finite Well & Harmonic Oscillator Analysis")
print("="*70)

# 물리 상수
hbar = 1.0
m = 1.0

def create_hamiltonian(x, V):
    """해밀토니안 행렬 생성"""
    N = len(x)
    dx = x[1] - x[0]
    
    T = np.zeros((N, N))
    for i in range(N):
        if i > 0:
            T[i, i-1] = -1
        T[i, i] = 2
        if i < N-1:
            T[i, i+1] = -1
    
    T = T * hbar**2 / (2 * m * dx**2)
    V_matrix = np.diag(V)
    H = T + V_matrix
    
    return H

def hermite_function(x, n, omega):
    """Hermite 함수 (조화 진동자 고유함수)"""
    alpha = np.sqrt(m * omega / hbar)
    xi = alpha * x
    
    # 정규화 계수
    coeff = (alpha / np.sqrt(np.pi * 2**n * factorial(n)))**(1/2)
    
    # Hermite 다항식
    H_n = hermite(n)
    
    psi = coeff * H_n(xi) * np.exp(-xi**2 / 2)
    return psi

# ============================================================================
# 1. Finite Well - Depth Dependence
# ============================================================================

print("\n" + "="*70)
print("1. Finite Square Well - Depth Dependence")
print("="*70)

# 격자 설정
L = 30.0
N = 3000
x = np.linspace(-L/2, L/2, N)

# 우물 너비
a = 10.0

# 여러 깊이에 대해 계산
V0_values = [2.0, 5.0, 10.0, 20.0]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

for idx, V0 in enumerate(V0_values):
    ax = axes[idx//2, idx%2]
    
    # 포텐셜
    V = np.where(np.abs(x) < a/2, -V0, 0)
    
    # 해 구하기
    H = create_hamiltonian(x, V)
    eigenvalues, eigenvectors = eigh(H)
    
    # 속박 상태 찾기
    bound_states = eigenvalues < 0
    n_bound = np.sum(bound_states)
    
    print(f"\nV₀ = {V0}:")
    print(f"  속박 상태 개수: {n_bound}")
    print(f"  에너지: {eigenvalues[bound_states]}")
    
    # 포텐셜 플롯
    ax.plot(x, V, 'k-', linewidth=2, label='Potential')
    
    # 파동함수와 에너지 준위
    for i in range(min(n_bound, 5)):
        E = eigenvalues[i]
        psi = eigenvectors[:, i]
        dx = x[1] - x[0]
        norm = np.sqrt(np.sum(psi**2) * dx)
        psi = psi / norm
        
        # 파동함수를 에너지 준위에 표시
        psi_shifted = psi * 2 + E
        ax.plot(x, psi_shifted, linewidth=2, label=f'n={i+1}, E={E:.2f}')
        ax.axhline(y=E, color='gray', linestyle='--', alpha=0.3)
    
    ax.axhline(y=0, color='r', linestyle='--', linewidth=1)
    ax.set_xlabel('x', fontsize=11)
    ax.set_ylabel('Energy', fontsize=11)
    ax.set_title(f'V₀ = {V0}, {n_bound} bound states', 
                fontsize=12, fontweight='bold')
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-L/2, L/2)
    ax.set_ylim(-V0-2, 5)

plt.suptitle(f'Finite Square Well: Depth Dependence (a={a})', 
             fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{output_dir}/04_finite_well_depth.png', dpi=150, bbox_inches='tight')
print(f"\n✓ 그래프 저장: {output_dir}/04_finite_well_depth.png")
plt.close()

# ============================================================================
# 2. Finite Well - Width Dependence
# ============================================================================

print("\n" + "="*70)
print("2. Finite Square Well - Width Dependence")
print("="*70)

V0_fixed = 10.0
a_values = [4.0, 6.0, 8.0, 10.0]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

for idx, a in enumerate(a_values):
    ax = axes[idx//2, idx%2]
    
    # 포텐셜
    V = np.where(np.abs(x) < a/2, -V0_fixed, 0)
    
    # 해 구하기
    H = create_hamiltonian(x, V)
    eigenvalues, eigenvectors = eigh(H)
    
    # 속박 상태
    bound_states = eigenvalues < 0
    n_bound = np.sum(bound_states)
    
    print(f"\na = {a}:")
    print(f"  속박 상태 개수: {n_bound}")
    
    # 포텐셜 플롯
    ax.plot(x, V, 'k-', linewidth=2, label='Potential')
    
    # 파동함수
    for i in range(min(n_bound, 4)):
        E = eigenvalues[i]
        psi = eigenvectors[:, i]
        dx = x[1] - x[0]
        norm = np.sqrt(np.sum(psi**2) * dx)
        psi = psi / norm
        
        psi_shifted = psi * 2 + E
        ax.plot(x, psi_shifted, linewidth=2, label=f'n={i+1}')
        ax.axhline(y=E, color='gray', linestyle='--', alpha=0.3)
    
    ax.axhline(y=0, color='r', linestyle='--', linewidth=1)
    ax.set_xlabel('x', fontsize=11)
    ax.set_ylabel('Energy', fontsize=11)
    ax.set_title(f'a = {a}, {n_bound} bound states', 
                fontsize=12, fontweight='bold')
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-L/2, L/2)
    ax.set_ylim(-V0_fixed-2, 5)

plt.suptitle(f'Finite Square Well: Width Dependence (V₀={V0_fixed})', 
             fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{output_dir}/04_finite_well_width.png', dpi=150, bbox_inches='tight')
print(f"✓ 그래프 저장: {output_dir}/04_finite_well_width.png")
plt.close()

# ============================================================================
# 3. Harmonic Oscillator - Detailed Analysis
# ============================================================================

print("\n" + "="*70)
print("3. Harmonic Oscillator - Detailed Analysis")
print("="*70)

# 격자 설정
x = np.linspace(-10, 10, 2000)

# 조화 진동자 파라미터
omega = 1.0
V = 0.5 * m * omega**2 * x**2

# 해 구하기
H = create_hamiltonian(x, V)
eigenvalues, eigenvectors = eigh(H)

# 정규화
dx = x[1] - x[0]
for i in range(len(eigenvalues)):
    norm = np.sqrt(np.sum(eigenvectors[:, i]**2) * dx)
    eigenvectors[:, i] /= norm

n_states = 8

print(f"\n조화 진동자 (ω={omega}):")
print(f"해석적 해: E_n = ℏω(n + 1/2)")
print("\n비교:")
for n in range(n_states):
    E_numerical = eigenvalues[n]
    E_analytical = hbar * omega * (n + 0.5)
    error = abs(E_numerical - E_analytical)
    print(f"  n={n}: E_num={E_numerical:.6f}, E_ana={E_analytical:.6f}, "
          f"Error={error:.6f}")

# 시각화 1: 파동함수 및 에너지 준위
fig = plt.figure(figsize=(18, 12))
gs = GridSpec(3, 3, figure=fig)

# 전체 플롯
ax_main = fig.add_subplot(gs[0, :])
ax_main.plot(x, V, 'k-', linewidth=2.5, label='Potential V(x)')

for n in range(min(n_states, 6)):
    E = eigenvalues[n]
    psi = eigenvectors[:, n]
    
    # 해석적 해와 비교
    psi_analytical = hermite_function(x, n, omega)
    
    psi_shifted = psi * 3 + E
    ax_main.plot(x, psi_shifted, linewidth=2, label=f'n={n}')
    ax_main.axhline(y=E, color='gray', linestyle='--', alpha=0.3)

ax_main.set_xlabel('x', fontsize=12)
ax_main.set_ylabel('Energy / Wave function', fontsize=12)
ax_main.set_title('Harmonic Oscillator: Energy Levels and Wave Functions', 
                 fontsize=14, fontweight='bold')
ax_main.legend(fontsize=9, loc='upper right', ncol=2)
ax_main.grid(True, alpha=0.3)
ax_main.set_xlim(-8, 8)

# 개별 상태 상세 플롯
for i in range(6):
    if i < 6:
        ax = fig.add_subplot(gs[1 + i//3, i%3])
    else:
        break
    
    psi_num = eigenvectors[:, i]
    psi_ana = hermite_function(x, i, omega)
    prob_num = psi_num**2
    prob_ana = psi_ana**2
    
    ax.plot(x, psi_num, 'b-', linewidth=2, label='Numerical')
    ax.plot(x, psi_ana, 'r--', linewidth=2, label='Analytical')
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    
    ax.set_xlabel('x', fontsize=10)
    ax.set_ylabel('ψ(x)', fontsize=10)
    ax.set_title(f'n={i}, E={eigenvalues[i]:.4f}', fontsize=11, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-6, 6)

plt.suptitle('Quantum Harmonic Oscillator: Numerical vs Analytical', 
             fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{output_dir}/04_harmonic_oscillator_detailed.png', 
            dpi=150, bbox_inches='tight')
print(f"✓ 그래프 저장: {output_dir}/04_harmonic_oscillator_detailed.png")
plt.close()

# ============================================================================
# 4. Classical vs Quantum Probability
# ============================================================================

print("\n" + "="*70)
print("4. Classical vs Quantum Probability Distribution")
print("="*70)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

for i, n in enumerate([0, 1, 5, 10]):
    if i >= 4:
        break
    
    ax = axes[i//2, i%2]
    
    # 양자 확률
    if n < len(eigenvalues):
        E = eigenvalues[n]
        psi = eigenvectors[:, n]
        prob_quantum = psi**2
        
        # 고전 확률 (E = 1/2 m ω² x² → x_max = √(2E/mω²))
        x_max = np.sqrt(2 * E / (m * omega**2))
        
        # 고전적으로는 끝에서 속도가 0이므로 확률이 높음
        prob_classical = np.zeros_like(x)
        in_range = np.abs(x) <= x_max
        prob_classical[in_range] = 1 / (np.pi * np.sqrt(x_max**2 - x[in_range]**2 + 1e-10))
        prob_classical = prob_classical / np.sum(prob_classical * dx)  # 정규화
        
        ax.plot(x, prob_quantum, 'b-', linewidth=2.5, label='Quantum |ψ|²')
        ax.plot(x, prob_classical, 'r--', linewidth=2.5, label='Classical')
        ax.axvline(x=-x_max, color='g', linestyle=':', linewidth=1.5, alpha=0.7)
        ax.axvline(x=x_max, color='g', linestyle=':', linewidth=1.5, alpha=0.7, 
                  label=f'Classical turning point')
        
        ax.fill_between(x, 0, prob_quantum, alpha=0.2, color='b')
        
        ax.set_xlabel('x', fontsize=11)
        ax.set_ylabel('Probability Density', fontsize=11)
        ax.set_title(f'n={n}, E={E:.4f}', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-8, 8)

plt.suptitle('Quantum vs Classical Probability Distribution', 
             fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{output_dir}/04_classical_vs_quantum.png', dpi=150, bbox_inches='tight')
print(f"✓ 그래프 저장: {output_dir}/04_classical_vs_quantum.png")
plt.close()

# ============================================================================
# 5. Comparison: Finite Well vs Harmonic Oscillator
# ============================================================================

print("\n" + "="*70)
print("5. Comparison: Finite Well vs Harmonic Oscillator")
print("="*70)

# 비슷한 에너지 스케일로 설정
V0_comp = 10.0
a_comp = 5.0
omega_comp = 1.5

V_well = np.where(np.abs(x) < a_comp/2, -V0_comp, 0)
V_ho = 0.5 * m * omega_comp**2 * x**2 - V0_comp/2  # 중심을 맞춤

H_well = create_hamiltonian(x, V_well)
H_ho = create_hamiltonian(x, V_ho)

E_well, psi_well = eigh(H_well)
E_ho, psi_ho = eigh(H_ho)

# 정규화
for i in range(min(10, len(E_well))):
    psi_well[:, i] /= np.sqrt(np.sum(psi_well[:, i]**2) * dx)
    psi_ho[:, i] /= np.sqrt(np.sum(psi_ho[:, i]**2) * dx)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Finite Well
ax1 = axes[0]
ax1.plot(x, V_well, 'k-', linewidth=2.5, label='Potential')

bound_well = E_well < 0
for i in range(min(5, np.sum(bound_well))):
    E = E_well[i]
    psi = psi_well[:, i]
    psi_shifted = psi * 2 + E
    ax1.plot(x, psi_shifted, linewidth=2, label=f'n={i+1}')
    ax1.axhline(y=E, color='gray', linestyle='--', alpha=0.3)

ax1.set_xlabel('x', fontsize=12)
ax1.set_ylabel('Energy / Wave function', fontsize=12)
ax1.set_title('Finite Square Well', fontsize=13, fontweight='bold')
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)
ax1.set_xlim(-10, 10)

# Harmonic Oscillator
ax2 = axes[1]
ax2.plot(x, V_ho, 'k-', linewidth=2.5, label='Potential')

for i in range(5):
    E = E_ho[i]
    psi = psi_ho[:, i]
    psi_shifted = psi * 2 + E
    ax2.plot(x, psi_shifted, linewidth=2, label=f'n={i}')
    ax2.axhline(y=E, color='gray', linestyle='--', alpha=0.3)

ax2.set_xlabel('x', fontsize=12)
ax2.set_ylabel('Energy / Wave function', fontsize=12)
ax2.set_title('Harmonic Oscillator', fontsize=13, fontweight='bold')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)
ax2.set_xlim(-10, 10)

plt.suptitle('Comparison: Two Fundamental Potentials', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{output_dir}/04_comparison.png', dpi=150, bbox_inches='tight')
print(f"✓ 그래프 저장: {output_dir}/04_comparison.png")
plt.close()

# ============================================================================
# 요약
# ============================================================================

print("\n" + "="*70)
print("실험 완료!")
print("="*70)
print(f"""
유한 우물 & 조화 진동자 분석:
✓ Finite Well: 깊이/너비에 따른 속박 상태 변화
✓ Harmonic Oscillator: 등간격 에너지 준위
✓ 양자-고전 대응: 큰 n에서 고전적 분포에 근접

주요 특징:
- Finite Well: 유한개의 속박 상태
- Harmonic Oscillator: 무한개의 속박 상태, E_n = ℏω(n+1/2)
- 파동함수: 우함수(짝수 n) / 기함수(홀수 n)

생성된 파일:
1. {output_dir}/04_finite_well_depth.png
2. {output_dir}/04_finite_well_width.png
3. {output_dir}/04_harmonic_oscillator_detailed.png
4. {output_dir}/04_classical_vs_quantum.png
5. {output_dir}/04_comparison.png
""")

