"""
07. Maxwell 2D Wave Equation
2D 전자기파 전파 (FDTD 방법)

물리 배경:
- 2D 파동 방정식: ∂²E/∂t² = c²(∂²E/∂x² + ∂²E/∂y²)
- 점 소스에서 원형 파면 전파
- 장애물에 의한 회절과 간섭

학습 목표:
1. 2D FDTD 알고리즘
2. 원형 파면 전파
3. 회절과 간섭 현상
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os

# 한글 폰트 설정
def set_korean_font():
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

# 출력 디렉토리 확인
output_dir = 'outputs'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print("="*70)
print("07. Maxwell 2D Wave Equation - Advanced FDTD")
print("="*70)

# 물리 상수
c = 3e8  # 광속

# 시뮬레이션 파라미터
Lx, Ly = 1.0, 1.0  # m
Nx, Ny = 150, 150  # 격자점
dx = Lx / Nx
dy = Ly / Ny

# CFL 조건 (2D)
CFL = 0.7  # 2D에서는 더 엄격
dt = CFL / (c * np.sqrt(1/dx**2 + 1/dy**2))

# 시간 스텝
T_total = 2 * Lx / c
Nt = int(T_total / dt)

print(f"\n공간 격자:")
print(f"  Lx × Ly = {Lx} × {Ly} m")
print(f"  Nx × Ny = {Nx} × {Ny}")
print(f"  dx = dy = {dx*1e3:.2f} mm")
print(f"\n시간 격자:")
print(f"  dt = {dt*1e9:.3f} ns")
print(f"  Nt = {Nt}")
print(f"  Total time = {T_total*1e9:.1f} ns")
print(f"\nCFL 수: {CFL:.2f}")

# 격자 생성
x = np.linspace(0, Lx, Nx)
y = np.linspace(0, Ly, Ny)
X, Y = np.meshgrid(x, y)

# 전기장 초기화
E = np.zeros((Ny, Nx))
E_old = np.zeros((Ny, Nx))
E_new = np.zeros((Ny, Nx))

# 점 소스 위치
source_x = int(0.2 * Nx)
source_y = int(0.5 * Ny)

print(f"\n점 소스 위치: ({source_x}, {source_y})")

# 장애물 추가 (선택적)
obstacle = np.zeros((Ny, Nx), dtype=bool)

# 슬릿 (회절 실험)
slit_x = int(0.6 * Nx)
slit_width = int(0.1 * Ny)
slit_center = int(0.5 * Ny)

for j in range(Ny):
    if abs(j - slit_center) > slit_width // 2:
        obstacle[j, slit_x] = True

print(f"장애물 (슬릿): x = {slit_x}, 폭 = {slit_width}")

# 시간 evolution
print("\n시간 진화 계산 중 (시간이 걸릴 수 있습니다)...")

snapshot_times = [int(0.15*Nt), int(0.35*Nt), int(0.55*Nt), int(0.75*Nt)]
snapshots = []
snapshot_labels = []

# 소스 파라미터 (시간에 따라 변하는 소스)
freq = 3e9  # 3 GHz
omega = 2 * np.pi * freq
wavelength = c / freq

print(f"소스 주파수: {freq/1e9:.1f} GHz")
print(f"파장: {wavelength*1e2:.2f} cm")

for n in range(Nt):
    # 소스 업데이트 (시간 의존)
    t = n * dt
    source_value = np.sin(omega * t) * np.exp(-((t - 3/freq)**2) / (2*(1/freq)**2))
    
    # FDTD 업데이트 (2D)
    for j in range(1, Ny-1):
        for i in range(1, Nx-1):
            if not obstacle[j, i]:
                laplacian = (E[j, i+1] - 2*E[j, i] + E[j, i-1]) / dx**2 + \
                           (E[j+1, i] - 2*E[j, i] + E[j-1, i]) / dy**2
                
                E_new[j, i] = 2*E[j, i] - E_old[j, i] + (c*dt)**2 * laplacian
    
    # 소스 추가
    E_new[source_y, source_x] += source_value
    
    # 경계 조건 (흡수 경계 - 간단한 버전)
    E_new[0, :] = 0
    E_new[-1, :] = 0
    E_new[:, 0] = 0
    E_new[:, -1] = 0
    
    # 장애물에서는 0
    E_new[obstacle] = 0
    
    # 업데이트
    E_old = E.copy()
    E = E_new.copy()
    
    # 스냅샷
    if n in snapshot_times:
        snapshots.append(E.copy())
        snapshot_labels.append(f't = {t*1e9:.1f} ns')
        print(f"  스냅샷 {len(snapshots)}/{len(snapshot_times)}: t = {t*1e9:.1f} ns")

print("\n시각화 중...")

# 시각화
fig, axes = plt.subplots(2, 2, figsize=(14, 14))
axes = axes.ravel()

for idx, (snap, label) in enumerate(zip(snapshots, snapshot_labels)):
    ax = axes[idx]
    
    # 장애물 표시
    snap_display = snap.copy()
    snap_display[obstacle] = np.nan  # NaN으로 표시하여 다른 색으로
    
    im = ax.imshow(snap_display, extent=[0, Lx*1e2, 0, Ly*1e2], 
                   origin='lower', cmap='RdBu_r', vmin=-0.5, vmax=0.5,
                   interpolation='bilinear')
    
    # 장애물 표시
    ax.contourf(X*1e2, Y*1e2, obstacle.astype(float), levels=[0.5, 1.5], 
                colors='black', alpha=0.5)
    
    # 소스 위치
    ax.plot(x[source_x]*1e2, y[source_y]*1e2, 'r*', markersize=15, 
            markeredgecolor='yellow', markeredgewidth=1)
    
    ax.set_xlabel('x (cm)', fontsize=11, fontweight='bold')
    ax.set_ylabel('y (cm)', fontsize=11, fontweight='bold')
    ax.set_title(label, fontsize=12, fontweight='bold')
    ax.set_aspect('equal')
    
    plt.colorbar(im, ax=ax, label='E field')

plt.suptitle('2D Wave Propagation with Diffraction (FDTD)', 
             fontsize=15, fontweight='bold')
plt.tight_layout()

output_path1 = f'{output_dir}/07_wave_2d_snapshots.png'
plt.savefig(output_path1, dpi=150, bbox_inches='tight')
print(f"\n[OK] 그래프 저장: {output_path1}")
plt.close()

# 추가: 장애물 없는 경우 (원형 파면)
print("\n추가 시뮬레이션: 자유 공간 전파")
print("-" * 70)

E2 = np.zeros((Ny, Nx))
E2_old = np.zeros((Ny, Nx))
E2_new = np.zeros((Ny, Nx))

# 중심 소스
source2_x = Nx // 2
source2_y = Ny // 2

snapshots2 = []
snapshot_times2 = [int(0.2*Nt), int(0.4*Nt), int(0.6*Nt), int(0.8*Nt)]

for n in range(Nt):
    t = n * dt
    source_value = np.sin(omega * t) * np.exp(-((t - 3/freq)**2) / (2*(1/freq)**2))
    
    for j in range(1, Ny-1):
        for i in range(1, Nx-1):
            laplacian = (E2[j, i+1] - 2*E2[j, i] + E2[j, i-1]) / dx**2 + \
                       (E2[j+1, i] - 2*E2[j, i] + E2[j-1, i]) / dy**2
            
            E2_new[j, i] = 2*E2[j, i] - E2_old[j, i] + (c*dt)**2 * laplacian
    
    E2_new[source2_y, source2_x] += source_value
    
    E2_new[0, :] = 0
    E2_new[-1, :] = 0
    E2_new[:, 0] = 0
    E2_new[:, -1] = 0
    
    E2_old = E2.copy()
    E2 = E2_new.copy()
    
    if n in snapshot_times2:
        snapshots2.append(E2.copy())

# 시각화 2
fig2, axes2 = plt.subplots(2, 2, figsize=(14, 14))
axes2 = axes2.ravel()

for idx, snap in enumerate(snapshots2):
    ax = axes2[idx]
    
    im = ax.imshow(snap, extent=[0, Lx*1e2, 0, Ly*1e2], 
                   origin='lower', cmap='RdBu_r', vmin=-0.5, vmax=0.5,
                   interpolation='bilinear')
    
    ax.plot(x[source2_x]*1e2, y[source2_y]*1e2, 'r*', markersize=15,
            markeredgecolor='yellow', markeredgewidth=1)
    
    t_frac = snapshot_times2[idx] / Nt
    ax.set_xlabel('x (cm)', fontsize=11, fontweight='bold')
    ax.set_ylabel('y (cm)', fontsize=11, fontweight='bold')
    ax.set_title(f't = {t_frac:.2f} × T_total', fontsize=12, fontweight='bold')
    ax.set_aspect('equal')
    
    plt.colorbar(im, ax=ax, label='E field')

plt.suptitle('2D Circular Wave (Free Space)', fontsize=15, fontweight='bold')
plt.tight_layout()

output_path2 = f'{output_dir}/07_wave_2d_circular.png'
plt.savefig(output_path2, dpi=150, bbox_inches='tight')
print(f"[OK] 그래프 저장: {output_path2}")
plt.close()

print("\n" + "="*70)
print("분석 완료!")
print("="*70)
print("\n핵심 개념:")
print("  1. 2D 파동 방정식: ∂²E/∂t² = c²∇²E")
print("  2. 점 소스 → 원형 파면 전파")
print("  3. 장애물에 의한 회절")
print("  4. 슬릿 통과 후 회절 패턴")
print("  5. 2D CFL 조건: 더 엄격함")
print("\n수치적 도전:")
print("  - 계산량: O(Nx × Ny × Nt)")
print("  - 메모리: 2D 배열")
print("  - 안정성: 2D CFL 조건")
print("  - 경계 조건: 흡수 경계 (PML 등)")
print("\n응용:")
print("  - 마이크로파 회로 설계")
print("  - 광학 소자 시뮬레이션")
print("  - 레이더 단면적 계산")
print("\n생성된 파일:")
print(f"  - {output_path1}")
print(f"  - {output_path2}")

