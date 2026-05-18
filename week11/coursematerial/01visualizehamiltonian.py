"""
Finite Well 포텐셜의 해밀토니안 행렬 분석
Hamiltonian Matrix Structure and Eigenvalue Analysis
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns

print("="*80)
print("Finite Well 포텐셜의 해밀토니안 행렬 분석")
print("="*80)

# 물리 상수 (원자 단위)
hbar = 1.0
m = 1.0

# 포텐셜 파라미터
V0 = 10.0  # 우물 깊이 (에너지 단위)

# ============================================================================
# 해밀토니안 행렬 생성
# ============================================================================

def create_hamiltonian(x, V):
    """해밀토니안 행렬 생성"""
    N = len(x)
    dx = x[1] - x[0]
    
    # 운동 에너지 항 (2차 미분을 행렬로 표현)
    T = np.zeros((N, N))
    for i in range(N):
        if i > 0:
            T[i, i-1] = -1  # 왼쪽 이웃
        T[i, i] = 2         # 자기 자신
        if i < N-1:
            T[i, i+1] = -1  # 오른쪽 이웃
    
    T = T * hbar**2 / (2 * m * dx**2)
    
    # 포텐셜 에너지 항 (대각 행렬)
    V_matrix = np.diag(V)
    
    # 전체 해밀토니안
    H = T + V_matrix
    
    return H, T, V_matrix

def finite_well_potential(x, V0, x_min=-1.0, x_max=1.0):
    """
    Finite Well 포텐셜
    x_min ~ x_max 구간에서 -V0의 포텐셜 (우물)
    """
    V = np.zeros_like(x)
    well_region = (x >= x_min) & (x <= x_max)
    V[well_region] = -V0
    return V

# ============================================================================
# 다양한 그리드 크기로 비교
# ============================================================================

print("\n[해밀토니안 행렬 크기 비교]")
print("-"*80)

grid_configs = [
    {"x_range": (-5, 5), "dx": 0.5, "label": "Coarse (dx=0.5)"},
    {"x_range": (-5, 5), "dx": 0.1, "label": "Medium (dx=0.1)"},
    {"x_range": (-5, 5), "dx": 0.05, "label": "Fine (dx=0.05)"},
]

hamiltonians = []
x_grids = []
potentials = []

for config in grid_configs:
    x_min, x_max = config["x_range"]
    dx = config["dx"]
    
    # 공간 그리드 생성
    x = np.arange(x_min, x_max + dx, dx)
    N = len(x)
    
    # 포텐셜 생성
    V = finite_well_potential(x, V0)
    
    # 해밀토니안 생성
    H, T, V_matrix = create_hamiltonian(x, V)
    
    hamiltonians.append((H, T, V_matrix))
    x_grids.append(x)
    potentials.append(V)
    
    # 행렬 정보 출력
    print(f"\n{config['label']}")
    print(f"  공간 범위: [{x_min}, {x_max}]")
    print(f"  그리드 간격 (dx): {dx}")
    print(f"  그리드 포인트 개수 (N): {N}")
    print(f"  해밀토니안 행렬 크기: {N} × {N}")
    print(f"  총 행렬 원소 개수: {N*N:,}")
    print(f"  메모리 사용량: {N*N*8/1024:.2f} KB (float64)")
    
    # 행렬 통계
    print(f"\n  해밀토니안 행렬 통계:")
    print(f"    최소값: {H.min():.4f}")
    print(f"    최대값: {H.max():.4f}")
    print(f"    대각 원소 범위: [{np.diag(H).min():.4f}, {np.diag(H).max():.4f}]")
    
    # 0이 아닌 원소 비율 (sparsity)
    non_zero = np.count_nonzero(H)
    sparsity = (1 - non_zero / (N*N)) * 100
    print(f"    0이 아닌 원소: {non_zero:,} ({100-sparsity:.1f}%)")
    print(f"    Sparsity (희소성): {sparsity:.1f}%")

# ============================================================================
# 시각화 - Medium grid 사용
# ============================================================================

# Medium grid (dx=0.1) 선택
idx = 1
x = x_grids[idx]
V = potentials[idx]
H, T, V_matrix = hamiltonians[idx]
N = len(x)

print(f"\n{'='*80}")
print(f"시각화에 사용된 그리드: dx = {grid_configs[idx]['dx']} (N = {N})")
print(f"{'='*80}")

# ============================================================================
# 고유값 문제 풀기
# ============================================================================

print("\n[고유값 문제 해결]")
eigenvalues, eigenvectors = np.linalg.eigh(H)

# 낮은 에너지 상태만 선택
n_states = 5
print(f"\n낮은 {n_states}개 에너지 준위:")
for i in range(n_states):
    E = eigenvalues[i]
    print(f"  n={i}: E = {E:10.4f} (상대 에너지: {E+V0:8.4f})")

# ============================================================================
# 시각화
# ============================================================================

fig = plt.figure(figsize=(18, 12))
gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)

# ============================================================================
# 1. 포텐셜 에너지 함수
# ============================================================================
ax1 = fig.add_subplot(gs[0, :])
ax1.plot(x, V, 'b-', linewidth=3, label='Finite Well Potential')
ax1.axhline(0, color='gray', linestyle='--', alpha=0.5)
ax1.axvline(-1, color='red', linestyle=':', alpha=0.5, label='Well boundary')
ax1.axvline(1, color='red', linestyle=':', alpha=0.5)
ax1.fill_between(x, V, 0, where=(V<0), alpha=0.2, color='blue')
ax1.set_xlabel('Position x', fontsize=13, fontweight='bold')
ax1.set_ylabel('Potential V(x)', fontsize=13, fontweight='bold')
ax1.set_title(f'Finite Well Potential (V₀ = {V0}, Well: [-1, 1])', 
             fontsize=14, fontweight='bold')
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)
ax1.set_ylim(-V0-2, 2)

# 텍스트 주석
ax1.text(0, -V0/2, f'-V₀ = {-V0}', fontsize=12, ha='center', 
        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))

# ============================================================================
# 2. 전체 해밀토니안 행렬 (히트맵)
# ============================================================================
ax2 = fig.add_subplot(gs[1, 0])
im2 = ax2.imshow(H, cmap='RdBu_r', aspect='auto', interpolation='nearest')
ax2.set_xlabel('Column Index j', fontsize=11, fontweight='bold')
ax2.set_ylabel('Row Index i', fontsize=11, fontweight='bold')
ax2.set_title(f'Hamiltonian Matrix H ({N}×{N})', fontsize=12, fontweight='bold')
plt.colorbar(im2, ax=ax2, label='Matrix Element Value')

# 중심부 확대 표시
center = N // 2
zoom_size = 20
rect = plt.Rectangle((center-zoom_size//2, center-zoom_size//2), 
                     zoom_size, zoom_size, 
                     fill=False, edgecolor='yellow', linewidth=2)
ax2.add_patch(rect)

# ============================================================================
# 3. 운동 에너지 행렬 T
# ============================================================================
ax3 = fig.add_subplot(gs[1, 1])
im3 = ax3.imshow(T, cmap='viridis', aspect='auto', interpolation='nearest')
ax3.set_xlabel('Column Index j', fontsize=11, fontweight='bold')
ax3.set_ylabel('Row Index i', fontsize=11, fontweight='bold')
ax3.set_title(f'Kinetic Energy Matrix T', fontsize=12, fontweight='bold')
plt.colorbar(im3, ax=ax3, label='T[i,j]')

# ============================================================================
# 4. 포텐셜 에너지 행렬 V
# ============================================================================
ax4 = fig.add_subplot(gs[1, 2])
im4 = ax4.imshow(V_matrix, cmap='coolwarm', aspect='auto', interpolation='nearest')
ax4.set_xlabel('Column Index j', fontsize=11, fontweight='bold')
ax4.set_ylabel('Row Index i', fontsize=11, fontweight='bold')
ax4.set_title(f'Potential Energy Matrix V', fontsize=12, fontweight='bold')
plt.colorbar(im4, ax=ax4, label='V[i,j]')

# ============================================================================
# 5. 행렬 구조 상세 (중심부 확대)
# ============================================================================
ax5 = fig.add_subplot(gs[2, 0])
center = N // 2
zoom_range = 20
zoom_H = H[center-zoom_range//2:center+zoom_range//2, 
           center-zoom_range//2:center+zoom_range//2]
im5 = ax5.imshow(zoom_H, cmap='RdBu_r', aspect='auto', interpolation='nearest')
ax5.set_xlabel('Column (zoomed)', fontsize=11, fontweight='bold')
ax5.set_ylabel('Row (zoomed)', fontsize=11, fontweight='bold')
ax5.set_title(f'Hamiltonian Zoom (center {zoom_range}×{zoom_range})', 
             fontsize=12, fontweight='bold')
plt.colorbar(im5, ax=ax5)

# 원소 값 표시 (가독성을 위해 작은 영역만)
for i in range(min(10, zoom_H.shape[0])):
    for j in range(min(10, zoom_H.shape[1])):
        text = ax5.text(j, i, f'{zoom_H[i, j]:.1f}',
                       ha="center", va="center", color="black", fontsize=6)

# ============================================================================
# 6. 특정 행의 원소 값 (Tridiagonal 구조 확인)
# ============================================================================
ax6 = fig.add_subplot(gs[2, 1])
row_idx = N // 2
row_values = H[row_idx, :]
ax6.stem(range(N), row_values, linefmt='b-', markerfmt='bo', basefmt='k-')
ax6.axvline(row_idx, color='red', linestyle='--', alpha=0.7, 
           label=f'Diagonal (i={row_idx})')
ax6.set_xlabel('Column Index j', fontsize=11, fontweight='bold')
ax6.set_ylabel('H[i,j] Value', fontsize=11, fontweight='bold')
ax6.set_title(f'Row {row_idx} of Hamiltonian', fontsize=12, fontweight='bold')
ax6.legend(fontsize=9)
ax6.grid(True, alpha=0.3, axis='y')
ax6.set_xlim(row_idx-20, row_idx+20)  # 주변만 표시

# ============================================================================
# 7. 에너지 고유값 스펙트럼
# ============================================================================
ax7 = fig.add_subplot(gs[2, 2])
n_show = 15
E_show = eigenvalues[:n_show]
ax7.plot(range(n_show), E_show, 'ro-', markersize=8, linewidth=2)
ax7.axhline(-V0, color='blue', linestyle='--', alpha=0.5, label='Well bottom')
ax7.axhline(0, color='gray', linestyle='--', alpha=0.5, label='V=0 level')
ax7.set_xlabel('Quantum State n', fontsize=11, fontweight='bold')
ax7.set_ylabel('Energy Eigenvalue', fontsize=11, fontweight='bold')
ax7.set_title(f'Energy Spectrum (first {n_show} states)', 
             fontsize=12, fontweight='bold')
ax7.legend(fontsize=9)
ax7.grid(True, alpha=0.3)

plt.suptitle(f'Hamiltonian Matrix Analysis: Finite Well (V₀={V0}, N={N})', 
            fontsize=16, fontweight='bold', y=0.995)

plt.savefig('./hamiltonian_matrix_analysis.png', 
           dpi=150, bbox_inches='tight')
print("\n✓ 그래프 저장: hamiltonian_matrix_analysis.png")
plt.close()

# ============================================================================
# 파동함수 시각화
# ============================================================================

fig2, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()

n_states_show = 6

for n in range(n_states_show):
    ax = axes[n]
    
    # 파동함수 (고유벡터)
    psi = eigenvectors[:, n]
    
    # 정규화 확인
    norm = np.trapezoid(psi**2, x)
    psi = psi / np.sqrt(norm)
    
    # 확률 밀도
    prob_density = psi**2
    
    # 포텐셜 (스케일 조정하여 함께 표시)
    V_scaled = V / V0 * np.max(prob_density) * 0.3
    
    # 플롯
    ax.fill_between(x, 0, prob_density, alpha=0.3, color='blue', label='|ψ|²')
    ax.plot(x, psi, 'b-', linewidth=2, label='ψ(x)', alpha=0.7)
    ax.plot(x, V_scaled, 'r--', linewidth=1.5, label='V(x) (scaled)', alpha=0.5)
    ax.axhline(0, color='gray', linestyle='-', alpha=0.3, linewidth=0.5)
    
    # 에너지 준위 표시
    E = eigenvalues[n]
    E_line = (E + V0) / V0 * np.max(prob_density) * 0.3
    ax.axhline(E_line, color='green', linestyle=':', alpha=0.5, linewidth=1)
    
    ax.set_xlabel('Position x', fontsize=11, fontweight='bold')
    ax.set_ylabel('Wave Function / Probability', fontsize=11, fontweight='bold')
    ax.set_title(f'State n={n}: E = {E:.4f}', fontsize=12, fontweight='bold')
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-5, 5)

plt.suptitle(f'Wave Functions of Finite Well (V₀={V0})', 
            fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('./finite_well_wavefunctions.png', 
           dpi=150, bbox_inches='tight')
print("✓ 그래프 저장: finite_well_wavefunctions.png")
plt.close()

# ============================================================================
# 행렬 구조 요약 텍스트 파일 생성
# ============================================================================

with open('./hamiltonian_matrix_structure.txt', 'w') as f:
    f.write("="*80 + "\n")
    f.write("Hamiltonian Matrix Structure Analysis\n")
    f.write("="*80 + "\n\n")
    
    f.write(f"Grid Configuration:\n")
    f.write(f"  Spatial range: [{x.min():.2f}, {x.max():.2f}]\n")
    f.write(f"  Grid spacing (dx): {x[1]-x[0]:.4f}\n")
    f.write(f"  Number of points (N): {N}\n")
    f.write(f"  Matrix size: {N} × {N}\n")
    f.write(f"  Total elements: {N*N:,}\n")
    f.write(f"  Memory: {N*N*8/1024:.2f} KB\n\n")
    
    f.write(f"Matrix Structure:\n")
    f.write(f"  Type: Tridiagonal (kinetic) + Diagonal (potential)\n")
    f.write(f"  Non-zero elements per row: 3 (off-diagonal) + 1 (potential)\n")
    f.write(f"  Sparsity: ~{(1 - 3/N)*100:.1f}%\n\n")
    
    f.write(f"Typical Matrix Elements:\n")
    f.write(f"  Kinetic energy (diagonal): {T[N//2, N//2]:.4f}\n")
    f.write(f"  Kinetic energy (off-diagonal): {T[N//2, N//2+1]:.4f}\n")
    f.write(f"  Potential (in well): {V_matrix[N//2, N//2]:.4f}\n")
    f.write(f"  Potential (outside): {V_matrix[0, 0]:.4f}\n\n")
    
    f.write(f"Energy Eigenvalues (first 10):\n")
    for i in range(min(10, len(eigenvalues))):
        f.write(f"  n={i}: E = {eigenvalues[i]:10.6f}\n")
    
    f.write("\n" + "="*80 + "\n")
    f.write("Matrix Expression:\n")
    f.write("="*80 + "\n")
    f.write("H = T + V\n\n")
    f.write("where:\n")
    f.write(f"  T[i,i]   = 2 x hbar^2/(2m x dx^2) = {T[N//2,N//2]:.4f}\n")
    f.write(f"  T[i,i±1] = -1 x hbar^2/(2m x dx^2) = {T[N//2,N//2+1]:.4f}\n")
    f.write(f"  V[i,i]   = V(x_i)\n\n")
    
    f.write("Example (5×5 submatrix at center):\n")
    center = N // 2
    sub = H[center-2:center+3, center-2:center+3]
    for i in range(5):
        f.write("  [")
        for j in range(5):
            f.write(f"{sub[i,j]:8.2f}")
        f.write(" ]\n")

print("✓ 텍스트 파일 저장: hamiltonian_matrix_structure.txt")

print("\n" + "="*80)
print("분석 완료!")
print("="*80)
print("\n생성된 파일:")
print("  1. hamiltonian_matrix_analysis.png - 행렬 구조 종합 분석")
print("  2. finite_well_wavefunctions.png - 에너지 고유상태")
print("  3. hamiltonian_matrix_structure.txt - 행렬 구조 상세 정보")