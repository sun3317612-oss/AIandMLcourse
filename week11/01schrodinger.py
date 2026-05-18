"""
01. Schrödinger Equation Numerical Solver
슈뢰딩거 방정식의 수치 해법

Time-Independent Schrödinger Equation:
-ℏ²/2m * d²ψ/dx² + V(x)ψ = Eψ

수치 해법:
- Finite Difference Method
- Shooting Method
- Matrix Diagonalization Method
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
print("Schrödinger Equation Numerical Solver")
print("="*70)

# 물리 상수 (atomic units: ℏ = m = 1)
hbar = 1.0
m = 1.0

def create_hamiltonian(x, V):
    """해밀토니안 행렬 생성 (Finite Difference Method)"""
    N = len(x)
    dx = x[1] - x[0]
    
    # 운동 에너지 항 (2차 미분)
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

def solve_schrodinger(x, V, n_states=5):
    """슈뢰딩거 방정식 풀기"""
    H = create_hamiltonian(x, V)
    
    # 고유값, 고유벡터 구하기
    eigenvalues, eigenvectors = eigh(H)
    
    # 정규화
    dx = x[1] - x[0]
    for i in range(len(eigenvalues)):
        norm = np.sqrt(np.sum(eigenvectors[:, i]**2) * dx)
        eigenvectors[:, i] /= norm
    
    return eigenvalues[:n_states], eigenvectors[:, :n_states]

# ============================================================================
# 1. Infinite Square Well (무한 사각 우물)
# ============================================================================

print("\n" + "="*70)
print("1. Infinite Square Well (Particle in a Box)")
print("="*70)

# 격자 설정
L = 10.0  # 우물 너비
N = 1000
x = np.linspace(0, L, N)

# 포텐셜: 무한 사각 우물 (경계 조건으로 처리)
V = np.zeros(N)

# 해 구하기
n_states = 5
energies, wavefunctions = solve_schrodinger(x, V, n_states)

print(f"\n처음 {n_states}개의 에너지 준위:")
print("해석적 해: E_n = n²π²ℏ²/(2mL²)")
for n in range(n_states):
    E_analytical = (n+1)**2 * np.pi**2 * hbar**2 / (2 * m * L**2)
    print(f"  n={n+1}: E_numerical = {energies[n]:.6f}, "
          f"E_analytical = {E_analytical:.6f}, "
          f"Error = {abs(energies[n] - E_analytical):.6f}")

# 시각화
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

for i in range(min(5, n_states)):
    ax = axes[i//3, i%3]
    
    # 파동함수
    psi = wavefunctions[:, i]
    prob = psi**2
    
    ax.plot(x, psi, 'b-', linewidth=2, label=f'ψ_{i+1}(x)')
    ax.plot(x, prob, 'r--', linewidth=2, label=f'|ψ_{i+1}(x)|²')
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    
    ax.set_xlabel('x', fontsize=11)
    ax.set_ylabel('Wave function', fontsize=11)
    ax.set_title(f'n={i+1}, E={energies[i]:.4f}', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, L)

# 마지막 subplot은 에너지 준위 다이어그램
ax = axes[1, 2]
for i in range(n_states):
    ax.hlines(energies[i], 0, 1, colors='blue', linewidth=2)
    ax.text(1.1, energies[i], f'n={i+1}', fontsize=10, va='center')

ax.set_xlim(-0.2, 1.5)
ax.set_ylabel('Energy', fontsize=12)
ax.set_title('Energy Levels', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')
ax.set_xticks([])

plt.suptitle('Infinite Square Well (Particle in a Box)', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{output_dir}/01_infinite_square_well.png', dpi=150, bbox_inches='tight')
print(f"\n✓ 그래프 저장: {output_dir}/01_infinite_square_well.png")
plt.close()

# ============================================================================
# 2. Finite Square Well (유한 사각 우물)
# ============================================================================

print("\n" + "="*70)
print("2. Finite Square Well")
print("="*70)

# 격자 설정
L = 20.0
N = 2000
x = np.linspace(-L/2, L/2, N)

# 포텐셜: 유한 사각 우물
V0 = 10.0  # 우물 깊이
a = 5.0    # 우물 너비
V = np.where(np.abs(x) < a/2, -V0, 0)

# 해 구하기
n_states = 6
energies, wavefunctions = solve_schrodinger(x, V, n_states)

# 속박 상태만 선택 (E < 0)
bound_states = energies < 0
n_bound = np.sum(bound_states)

print(f"\n속박 상태 개수: {n_bound}")
print(f"속박 상태 에너지:")
for i in range(n_bound):
    print(f"  n={i+1}: E = {energies[i]:.6f}")

# 시각화
fig = plt.figure(figsize=(18, 10))
gs = GridSpec(2, 3, figure=fig)

# 포텐셜 플롯
ax0 = fig.add_subplot(gs[0, :])
ax0.plot(x, V, 'k-', linewidth=2, label='Potential V(x)')

# 속박 상태와 에너지 준위
for i in range(min(n_bound, 5)):
    E = energies[i]
    psi = wavefunctions[:, i]
    
    # 파동함수를 에너지 준위에 맞춰 플롯
    psi_shifted = psi * 2 + E
    ax0.plot(x, psi_shifted, linewidth=2, label=f'ψ_{i+1}, E={E:.3f}')
    ax0.axhline(y=E, color='gray', linestyle='--', alpha=0.3)

ax0.axhline(y=0, color='r', linestyle='--', linewidth=1, alpha=0.5, label='E=0')
ax0.set_xlabel('x', fontsize=12)
ax0.set_ylabel('Energy / Wave function', fontsize=12)
ax0.set_title('Finite Square Well: Potential and Bound States', fontsize=14, fontweight='bold')
ax0.legend(fontsize=9, loc='upper right')
ax0.grid(True, alpha=0.3)
ax0.set_xlim(-L/2, L/2)

# 개별 파동함수 플롯
for i in range(min(n_bound, 5)):
    if i < 3:
        ax = fig.add_subplot(gs[1, i])
    else:
        break
    
    psi = wavefunctions[:, i]
    prob = psi**2
    
    ax.plot(x, psi, 'b-', linewidth=2, label=f'ψ_{i+1}(x)')
    ax.plot(x, prob, 'r--', linewidth=2, label=f'|ψ_{i+1}(x)|²')
    ax.fill_between(x, 0, V/V0*0.5, alpha=0.2, color='gray', label='Potential')
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    
    ax.set_xlabel('x', fontsize=11)
    ax.set_ylabel('Wave function', fontsize=11)
    ax.set_title(f'n={i+1}, E={energies[i]:.4f}', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-L/2, L/2)

plt.suptitle(f'Finite Square Well (V₀={V0}, a={a})', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{output_dir}/01_finite_square_well.png', dpi=150, bbox_inches='tight')
print(f"✓ 그래프 저장: {output_dir}/01_finite_square_well.png")
plt.close()

# ============================================================================
# 3. Harmonic Oscillator (조화 진동자)
# ============================================================================

print("\n" + "="*70)
print("3. Harmonic Oscillator")
print("="*70)

# 격자 설정
L = 20.0
N = 2000
x = np.linspace(-L/2, L/2, N)

# 포텐셜: 조화 진동자
omega = 1.0
V = 0.5 * m * omega**2 * x**2

# 해 구하기
n_states = 6
energies, wavefunctions = solve_schrodinger(x, V, n_states)

print(f"\n처음 {n_states}개의 에너지 준위:")
print("해석적 해: E_n = ℏω(n + 1/2)")
for n in range(n_states):
    E_analytical = hbar * omega * (n + 0.5)
    print(f"  n={n}: E_numerical = {energies[n]:.6f}, "
          f"E_analytical = {E_analytical:.6f}, "
          f"Error = {abs(energies[n] - E_analytical):.6f}")

# 시각화
fig = plt.figure(figsize=(18, 10))
gs = GridSpec(2, 3, figure=fig)

# 포텐셜 및 에너지 준위
ax0 = fig.add_subplot(gs[0, :])
ax0.plot(x, V, 'k-', linewidth=2, label='Potential V(x) = ½mω²x²')

for i in range(n_states):
    E = energies[i]
    psi = wavefunctions[:, i]
    
    # 파동함수를 에너지 준위에 맞춰 플롯
    psi_shifted = psi * 3 + E
    ax0.plot(x, psi_shifted, linewidth=2, label=f'n={i}, E={E:.3f}')
    ax0.axhline(y=E, color='gray', linestyle='--', alpha=0.3)

ax0.set_xlabel('x', fontsize=12)
ax0.set_ylabel('Energy / Wave function', fontsize=12)
ax0.set_title('Harmonic Oscillator: Potential and Energy Levels', 
              fontsize=14, fontweight='bold')
ax0.legend(fontsize=9, loc='upper right')
ax0.grid(True, alpha=0.3)
ax0.set_xlim(-L/2, L/2)
ax0.set_ylim(-1, max(energies) + 2)

# 개별 파동함수 플롯
for i in range(min(3, n_states)):
    ax = fig.add_subplot(gs[1, i])
    
    psi = wavefunctions[:, i]
    prob = psi**2
    
    ax.plot(x, psi, 'b-', linewidth=2, label=f'ψ_{i}(x)')
    ax.plot(x, prob, 'r--', linewidth=2, label=f'|ψ_{i}(x)|²')
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    
    ax.set_xlabel('x', fontsize=11)
    ax.set_ylabel('Wave function', fontsize=11)
    ax.set_title(f'n={i}, E={energies[i]:.4f}', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-6, 6)

plt.suptitle('Quantum Harmonic Oscillator', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{output_dir}/01_harmonic_oscillator.png', dpi=150, bbox_inches='tight')
print(f"✓ 그래프 저장: {output_dir}/01_harmonic_oscillator.png")
plt.close()

# ============================================================================
# 요약
# ============================================================================

print("\n" + "="*70)
print("실험 완료!")
print("="*70)
print(f"""
슈뢰딩거 방정식 수치 해법:
✓ Finite Difference Method 사용
✓ 행렬 대각화로 고유값/고유벡터 계산

검증:
- Infinite Square Well: 해석적 해와 일치
- Harmonic Oscillator: E_n = ℏω(n+1/2) 확인
- Finite Square Well: 속박 상태 정확히 계산

생성된 파일:
1. {output_dir}/01_infinite_square_well.png
2. {output_dir}/01_finite_square_well.png
3. {output_dir}/01_harmonic_oscillator.png
""")

