"""
02. Wave Function Visualization
파동함수 시각화

다양한 양자 상태의 파동함수를 시각화하고
시간에 따른 진화를 애니메이션으로 표현
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.animation import FuncAnimation
import os

# 출력 디렉토리 확인
output_dir = 'outputs'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print("="*70)
print("Wave Function Visualization")
print("="*70)

# 물리 상수
hbar = 1.0
m = 1.0

# ============================================================================
# 1. Gaussian Wave Packet (가우시안 파동 패킷)
# ============================================================================

print("\n" + "="*70)
print("1. Gaussian Wave Packet")
print("="*70)

# 격자 설정
x = np.linspace(-20, 20, 1000)
dx = x[1] - x[0]

# 가우시안 파동 패킷 파라미터
x0 = 0.0      # 중심 위치
sigma = 2.0   # 너비
k0 = 2.0      # 평균 운동량 (ℏk0)

# 시간 t=0에서의 파동함수
psi_real = np.exp(-(x - x0)**2 / (4 * sigma**2)) * np.cos(k0 * x)
psi_imag = np.exp(-(x - x0)**2 / (4 * sigma**2)) * np.sin(k0 * x)
psi = psi_real + 1j * psi_imag

# 정규화
norm = np.sqrt(np.sum(np.abs(psi)**2) * dx)
psi = psi / norm

# 확률 밀도
prob = np.abs(psi)**2

print(f"파동 패킷 파라미터:")
print(f"  중심 위치: x0 = {x0}")
print(f"  너비: σ = {sigma}")
print(f"  평균 운동량: ℏk0 = {hbar * k0:.2f}")
print(f"  불확정성 원리: Δx·Δp ≈ {sigma * hbar * 1:.2f} (≥ ℏ/2 = {hbar/2:.2f})")

# 시각화
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 실수부
ax1 = axes[0, 0]
ax1.plot(x, psi_real, 'b-', linewidth=2)
ax1.axhline(y=0, color='k', linestyle='-', alpha=0.3)
ax1.set_xlabel('x', fontsize=12)
ax1.set_ylabel('Re[ψ(x)]', fontsize=12)
ax1.set_title('Real Part of Wave Function', fontsize=13, fontweight='bold')
ax1.grid(True, alpha=0.3)

# 허수부
ax2 = axes[0, 1]
ax2.plot(x, psi_imag, 'r-', linewidth=2)
ax2.axhline(y=0, color='k', linestyle='-', alpha=0.3)
ax2.set_xlabel('x', fontsize=12)
ax2.set_ylabel('Im[ψ(x)]', fontsize=12)
ax2.set_title('Imaginary Part of Wave Function', fontsize=13, fontweight='bold')
ax2.grid(True, alpha=0.3)

# 확률 밀도
ax3 = axes[1, 0]
ax3.plot(x, prob, 'g-', linewidth=2)
ax3.fill_between(x, 0, prob, alpha=0.3, color='g')
ax3.set_xlabel('x', fontsize=12)
ax3.set_ylabel('|ψ(x)|²', fontsize=12)
ax3.set_title('Probability Density', fontsize=13, fontweight='bold')
ax3.grid(True, alpha=0.3)

# 운동량 공간
ax4 = axes[1, 1]
# 푸리에 변환
psi_k = np.fft.fftshift(np.fft.fft(psi))
k = np.fft.fftshift(np.fft.fftfreq(len(x), dx)) * 2 * np.pi
prob_k = np.abs(psi_k)**2
prob_k = prob_k / np.max(prob_k)  # 정규화

ax4.plot(k, prob_k, 'm-', linewidth=2)
ax4.fill_between(k, 0, prob_k, alpha=0.3, color='m')
ax4.set_xlabel('k (momentum / ℏ)', fontsize=12)
ax4.set_ylabel('|ψ(k)|² (normalized)', fontsize=12)
ax4.set_title('Momentum Space', fontsize=13, fontweight='bold')
ax4.grid(True, alpha=0.3)
ax4.set_xlim(-5, 5)

plt.suptitle(f'Gaussian Wave Packet (x₀={x0}, σ={sigma}, k₀={k0})', 
             fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{output_dir}/02_gaussian_wave_packet.png', dpi=150, bbox_inches='tight')
print(f"\n✓ 그래프 저장: {output_dir}/02_gaussian_wave_packet.png")
plt.close()

# ============================================================================
# 2. Superposition of States (상태의 중첩)
# ============================================================================

print("\n" + "="*70)
print("2. Superposition of States (Particle in a Box)")
print("="*70)

# Infinite square well 고유함수
L = 10.0
x = np.linspace(0, L, 1000)

def psi_n(x, n, L):
    """Infinite square well의 n번째 고유함수"""
    return np.sqrt(2/L) * np.sin(n * np.pi * x / L)

# 여러 상태의 중첩
n_max = 5
coefficients = np.array([1, 0.5, 0.3, 0.2, 0.1])  # 각 상태의 계수
coefficients = coefficients / np.linalg.norm(coefficients)  # 정규화

print(f"중첩 상태: ψ = Σ c_n ψ_n")
for i in range(n_max):
    print(f"  c_{i+1} = {coefficients[i]:.3f}")

# 중첩된 파동함수
psi_superposition = np.zeros_like(x)
for n in range(1, n_max + 1):
    psi_superposition += coefficients[n-1] * psi_n(x, n, L)

prob_superposition = psi_superposition**2

# 시각화
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# 개별 고유함수
for i in range(n_max):
    ax = axes[i//3, i%3]
    psi = psi_n(x, i+1, L)
    prob = psi**2
    
    ax.plot(x, psi, 'b-', linewidth=2, label=f'ψ_{i+1}')
    ax.plot(x, prob, 'r--', linewidth=2, label=f'|ψ_{i+1}|²')
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    
    ax.set_xlabel('x', fontsize=11)
    ax.set_ylabel('Wave function', fontsize=11)
    ax.set_title(f'n={i+1}, coefficient={coefficients[i]:.3f}', 
                fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, L)

# 중첩 상태
ax = axes[1, 2]
ax.plot(x, psi_superposition, 'g-', linewidth=2.5, label='Superposition ψ')
ax.plot(x, prob_superposition, 'm--', linewidth=2.5, label='|ψ|²')
ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)

ax.set_xlabel('x', fontsize=11)
ax.set_ylabel('Wave function', fontsize=11)
ax.set_title('Superposition State', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, L)

plt.suptitle('Superposition of Quantum States', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{output_dir}/02_superposition_states.png', dpi=150, bbox_inches='tight')
print(f"✓ 그래프 저장: {output_dir}/02_superposition_states.png")
plt.close()

# ============================================================================
# 3. 3D Hydrogen Atom Orbitals (수소 원자 오비탈)
# ============================================================================

print("\n" + "="*70)
print("3. Hydrogen Atom Orbitals (2D Cross-sections)")
print("="*70)

# Bohr radius
a0 = 1.0

def R_nl(r, n, l):
    """동경 파동함수 (간단한 형태)"""
    if n == 1 and l == 0:  # 1s
        return 2 * (1/a0)**(3/2) * np.exp(-r/a0)
    elif n == 2 and l == 0:  # 2s
        return (1/(2*np.sqrt(2))) * (1/a0)**(3/2) * (2 - r/a0) * np.exp(-r/(2*a0))
    elif n == 2 and l == 1:  # 2p
        return (1/(2*np.sqrt(6))) * (1/a0)**(3/2) * (r/a0) * np.exp(-r/(2*a0))
    elif n == 3 and l == 0:  # 3s
        return (1/(9*np.sqrt(3))) * (1/a0)**(3/2) * (6 - 6*r/a0 + (r/a0)**2) * np.exp(-r/(3*a0))
    else:
        return np.zeros_like(r)

# 2D 격자 (xz 평면)
x = np.linspace(-10*a0, 10*a0, 200)
z = np.linspace(-10*a0, 10*a0, 200)
X, Z = np.meshgrid(x, z)
R = np.sqrt(X**2 + Z**2)

# 오비탈 계산
orbitals = {
    '1s': R_nl(R, 1, 0),
    '2s': R_nl(R, 2, 0),
    '2p': R_nl(R, 2, 1) * Z / (R + 1e-10),  # p_z 오비탈
    '3s': R_nl(R, 3, 0)
}

print("수소 원자 오비탈 (2D 단면):")
for name in orbitals.keys():
    print(f"  {name} 오비탈")

# 시각화
fig, axes = plt.subplots(2, 4, figsize=(18, 9))

for idx, (name, psi) in enumerate(orbitals.items()):
    prob = psi**2
    
    # 행과 열 인덱스 계산 (2행 x 4열)
    row = idx // 2  # 0, 0, 1, 1
    col_offset = (idx % 2) * 2  # 0, 2, 0, 2
    
    # 파동함수
    ax1 = axes[row, col_offset]
    im1 = ax1.contourf(X/a0, Z/a0, psi, levels=20, cmap='RdBu')
    ax1.set_xlabel('x / a₀', fontsize=10)
    ax1.set_ylabel('z / a₀', fontsize=10)
    ax1.set_title(f'{name} Wave Function', fontsize=11, fontweight='bold')
    ax1.set_aspect('equal')
    plt.colorbar(im1, ax=ax1)
    
    # 확률 밀도
    ax2 = axes[row, col_offset + 1]
    im2 = ax2.contourf(X/a0, Z/a0, prob, levels=20, cmap='viridis')
    ax2.set_xlabel('x / a₀', fontsize=10)
    ax2.set_ylabel('z / a₀', fontsize=10)
    ax2.set_title(f'{name} |ψ|²', fontsize=11, fontweight='bold')
    ax2.set_aspect('equal')
    plt.colorbar(im2, ax=ax2)

plt.suptitle('Hydrogen Atom Orbitals', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{output_dir}/02_hydrogen_orbitals.png', dpi=150, bbox_inches='tight')
print(f"✓ 그래프 저장: {output_dir}/02_hydrogen_orbitals.png")
plt.close()

# ============================================================================
# 요약
# ============================================================================

print("\n" + "="*70)
print("실험 완료!")
print("="*70)
print(f"""
파동함수 시각화:
✓ Gaussian Wave Packet: 위치-운동량 표현
✓ Superposition States: 양자 중첩 원리
✓ Hydrogen Orbitals: 2D 단면 및 확률 밀도

물리적 의미:
- |ψ|²: 확률 밀도
- 불확정성 원리: Δx·Δp ≥ ℏ/2
- 중첩 원리: 양자 간섭 효과

생성된 파일:
1. {output_dir}/02_gaussian_wave_packet.png
2. {output_dir}/02_superposition_states.png
3. {output_dir}/02_hydrogen_orbitals.png
""")

