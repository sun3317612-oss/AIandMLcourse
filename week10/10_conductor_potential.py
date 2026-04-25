"""
10. Conductor Potential Distribution
도체 내부 전위 분포 계산 (라플라스 방정식)

물리 배경:
- 라플라스 방정식: ∇²V = 0 (전하가 없는 영역)
- 유한 차분법 + 반복법 (Gauss-Seidel)
- 경계 조건: 도체 표면에서 전위 고정

학습 목표:
1. 라플라스 방정식의 수치 해법
2. 경계 조건 처리
3. 반복법의 수렴
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
print("10. Conductor Potential Distribution - Laplace Equation")
print("="*70)

# 시뮬레이션 파라미터
Lx, Ly = 1.0, 1.0  # m
Nx, Ny = 100, 100  # 격자점
dx = Lx / (Nx - 1)
dy = Ly / (Ny - 1)

print(f"\n격자 설정:")
print(f"  영역: {Lx} × {Ly} m")
print(f"  격자점: {Nx} × {Ny}")
print(f"  간격: dx = {dx*1e3:.2f} mm, dy = {dy*1e3:.2f} mm")

# 격자 생성
x = np.linspace(0, Lx, Nx)
y = np.linspace(0, Ly, Ny)
X, Y = np.meshgrid(x, y)

# 전위 초기화
V = np.zeros((Ny, Nx))

# 경계 조건 설정
boundary = np.zeros((Ny, Nx), dtype=bool)

# 시나리오 1: 평행판 커패시터
print("\n시나리오 1: 평행판 커패시터")
print("-" * 70)

# 위쪽 판: V = +100 V
V[0, :] = 100.0
boundary[0, :] = True

# 아래쪽 판: V = -100 V
V[-1, :] = -100.0
boundary[-1, :] = True

# 좌우는 주기 경계 조건 또는 고립 (여기서는 Neumann)
# 실제로는 양쪽을 0으로 설정
boundary[:, 0] = True
boundary[:, -1] = True
V[:, 0] = 0.0
V[:, -1] = 0.0

def solve_laplace_gauss_seidel(V_initial, boundary, max_iter=10000, tol=1e-6):
    """
    가우스-자이델 반복법으로 라플라스 방정식 풀기
    
    ∇²V = 0
    
    유한 차분:
    (V[i+1,j] - 2V[i,j] + V[i-1,j])/dx² + (V[i,j+1] - 2V[i,j] + V[i,j-1])/dy² = 0
    
    V[i,j] = (V[i+1,j] + V[i-1,j] + V[i,j+1] + V[i,j-1]) / 4
    """
    V = V_initial.copy()
    Ny, Nx = V.shape
    
    residual_history = []
    
    print(f"  반복법 시작 (max_iter={max_iter}, tol={tol})")
    
    for iteration in range(max_iter):
        V_old = V.copy()
        
        # 가우스-자이델 업데이트
        for j in range(1, Ny-1):
            for i in range(1, Nx-1):
                if not boundary[j, i]:
                    V[j, i] = 0.25 * (V[j, i+1] + V[j, i-1] + V[j+1, i] + V[j-1, i])
        
        # 수렴 체크
        residual = np.max(np.abs(V - V_old))
        residual_history.append(residual)
        
        if (iteration + 1) % 500 == 0:
            print(f"    반복 {iteration+1}: residual = {residual:.2e}")
        
        if residual < tol:
            print(f"  수렴! 반복 횟수: {iteration+1}, residual = {residual:.2e}")
            break
    
    return V, residual_history

print("\n라플라스 방정식 풀이 중...")
V_solved, residual_history = solve_laplace_gauss_seidel(V, boundary, max_iter=5000, tol=1e-6)

# 전기장 계산 (E = -∇V)
Ey, Ex = np.gradient(V_solved, dy, dx)
Ex = -Ex
Ey = -Ey
E_mag = np.sqrt(Ex**2 + Ey**2)

print(f"\n전위 범위: [{np.min(V_solved):.1f}, {np.max(V_solved):.1f}] V")
print(f"최대 전기장: {np.max(E_mag):.2e} V/m")

# 시각화
fig = plt.figure(figsize=(16, 12))

# 1. 전위 분포 (컬러맵)
ax1 = fig.add_subplot(2, 3, 1)
im1 = ax1.contourf(X*1e2, Y*1e2, V_solved, levels=30, cmap='coolwarm')
cbar1 = plt.colorbar(im1, ax=ax1)
cbar1.set_label('Potential (V)', fontsize=11, fontweight='bold')

# 경계 표시
boundary_y, boundary_x = np.where(boundary)
ax1.plot(X[boundary_y, boundary_x]*1e2, Y[boundary_y, boundary_x]*1e2, 
         'k.', markersize=1, alpha=0.5)

ax1.set_xlabel('x (cm)', fontsize=11, fontweight='bold')
ax1.set_ylabel('y (cm)', fontsize=11, fontweight='bold')
ax1.set_title('Potential Distribution', fontsize=12, fontweight='bold')
ax1.set_aspect('equal')

# 2. 등전위선
ax2 = fig.add_subplot(2, 3, 2)
contour = ax2.contour(X*1e2, Y*1e2, V_solved, levels=20, colors='black', linewidths=1.5)
ax2.clabel(contour, inline=True, fontsize=8, fmt='%.0f V')

ax2.set_xlabel('x (cm)', fontsize=11, fontweight='bold')
ax2.set_ylabel('y (cm)', fontsize=11, fontweight='bold')
ax2.set_title('Equipotential Lines', fontsize=12, fontweight='bold')
ax2.set_aspect('equal')
ax2.grid(True, alpha=0.3)

# 3. 전기장 벡터
ax3 = fig.add_subplot(2, 3, 3)

# 배경
im3 = ax3.contourf(X*1e2, Y*1e2, np.log10(E_mag + 1), levels=20, cmap='plasma', alpha=0.6)
cbar3 = plt.colorbar(im3, ax=ax3)
cbar3.set_label('log10(E) [V/m]', fontsize=11, fontweight='bold')

# 벡터 (성글게)
skip = 5
E_norm = np.sqrt(Ex[::skip, ::skip]**2 + Ey[::skip, ::skip]**2)
Ex_norm = Ex[::skip, ::skip] / (E_norm + 1e-10)
Ey_norm = Ey[::skip, ::skip] / (E_norm + 1e-10)

ax3.quiver(X[::skip, ::skip]*1e2, Y[::skip, ::skip]*1e2, 
           Ex_norm, Ey_norm,
           angles='xy', scale_units='xy', scale=15,
           color='white', alpha=0.8, width=0.003)

ax3.set_xlabel('x (cm)', fontsize=11, fontweight='bold')
ax3.set_ylabel('y (cm)', fontsize=11, fontweight='bold')
ax3.set_title('Electric Field (E = -∇V)', fontsize=12, fontweight='bold')
ax3.set_aspect('equal')

# 4. 수렴 history
ax4 = fig.add_subplot(2, 3, 4)
ax4.semilogy(residual_history, 'b-', linewidth=2)
ax4.set_xlabel('Iteration', fontsize=11, fontweight='bold')
ax4.set_ylabel('Residual', fontsize=11, fontweight='bold')
ax4.set_title('Convergence History', fontsize=12, fontweight='bold')
ax4.grid(True, alpha=0.3)

# 5. x=0.5에서 전위 프로파일
ax5 = fig.add_subplot(2, 3, 5)
x_mid = Nx // 2
V_profile = V_solved[:, x_mid]
y_profile = y * 1e2

ax5.plot(V_profile, y_profile, 'b-', linewidth=2.5, label='V(x=0.5m, y)')

# 이론적 해 (평행판)
V_theory = 100 - 200 * y
ax5.plot(V_theory, y_profile, 'r--', linewidth=2, label='Theory (linear)')

ax5.set_xlabel('Potential (V)', fontsize=11, fontweight='bold')
ax5.set_ylabel('y (cm)', fontsize=11, fontweight='bold')
ax5.set_title('Potential Profile at x = 0.5 m', fontsize=12, fontweight='bold')
ax5.legend(fontsize=10)
ax5.grid(True, alpha=0.3)

# 6. 전기장 y 성분
ax6 = fig.add_subplot(2, 3, 6)
Ey_profile = Ey[:, x_mid]

ax6.plot(Ey_profile, y_profile, 'g-', linewidth=2.5, label='Ey(x=0.5m, y)')

# 이론값 (평행판: E = V/d)
E_theory = 200 / Ly  # V/m
ax6.axvline(E_theory, color='r', linestyle='--', linewidth=2, label=f'Theory = {E_theory:.0f} V/m')

ax6.set_xlabel('Ey (V/m)', fontsize=11, fontweight='bold')
ax6.set_ylabel('y (cm)', fontsize=11, fontweight='bold')
ax6.set_title('Electric Field (y-component)', fontsize=12, fontweight='bold')
ax6.legend(fontsize=10)
ax6.grid(True, alpha=0.3)

plt.suptitle('Laplace Equation: Parallel Plate Capacitor', fontsize=15, fontweight='bold')
plt.tight_layout()

output_path1 = f'{output_dir}/10_conductor_potential.png'
plt.savefig(output_path1, dpi=150, bbox_inches='tight')
print(f"\n[OK] 그래프 저장: {output_path1}")
plt.close()

# 시나리오 2: 원형 도체
print("\n시나리오 2: 동심 원통 도체 (실린더 커패시터)")
print("-" * 70)

V2 = np.zeros((Ny, Nx))
boundary2 = np.zeros((Ny, Nx), dtype=bool)

# 중심
center_x, center_y = Nx // 2, Ny // 2

# 내부 원통 (V = 100 V)
r_inner = 0.2  # m
for j in range(Ny):
    for i in range(Nx):
        r = np.sqrt((x[i] - x[center_x])**2 + (y[j] - y[center_y])**2)
        if r <= r_inner:
            V2[j, i] = 100.0
            boundary2[j, i] = True

# 외부 원통 (V = 0 V)
r_outer = 0.45  # m
for j in range(Ny):
    for i in range(Nx):
        r = np.sqrt((x[i] - x[center_x])**2 + (y[j] - y[center_y])**2)
        if r >= r_outer:
            V2[j, i] = 0.0
            boundary2[j, i] = True

print(f"  내부 반지름: {r_inner*1e2:.1f} cm, V = 100 V")
print(f"  외부 반지름: {r_outer*1e2:.1f} cm, V = 0 V")

print("\n라플라스 방정식 풀이 중...")
V2_solved, residual_history2 = solve_laplace_gauss_seidel(V2, boundary2, max_iter=5000, tol=1e-6)

# 전기장
Ey2, Ex2 = np.gradient(V2_solved, dy, dx)
Ex2 = -Ex2
Ey2 = -Ey2
E_mag2 = np.sqrt(Ex2**2 + Ey2**2)

# 시각화 2
fig2 = plt.figure(figsize=(16, 10))

# 1. 전위
ax1 = fig2.add_subplot(2, 3, 1)
im1 = ax1.contourf(X*1e2, Y*1e2, V2_solved, levels=30, cmap='coolwarm')
plt.colorbar(im1, ax=ax1, label='Potential (V)')
ax1.set_xlabel('x (cm)', fontsize=11, fontweight='bold')
ax1.set_ylabel('y (cm)', fontsize=11, fontweight='bold')
ax1.set_title('Potential Distribution', fontsize=12, fontweight='bold')
ax1.set_aspect('equal')

# 2. 등전위선 + 전기력선
ax2 = fig2.add_subplot(2, 3, 2)

# 등전위선
contour2 = ax2.contour(X*1e2, Y*1e2, V2_solved, levels=15, colors='green', 
                       linewidths=1.5, linestyles='dashed')

# 전기력선 (streamplot)
stream = ax2.streamplot(X*1e2, Y*1e2, Ex2, Ey2, color='blue', 
                        linewidth=1.5, density=2.0, arrowsize=1.2)

# 도체 경계
circle_inner = plt.Circle((x[center_x]*1e2, y[center_y]*1e2), r_inner*1e2, 
                          fill=False, edgecolor='red', linewidth=2)
circle_outer = plt.Circle((x[center_x]*1e2, y[center_y]*1e2), r_outer*1e2, 
                          fill=False, edgecolor='black', linewidth=2)
ax2.add_patch(circle_inner)
ax2.add_patch(circle_outer)

ax2.set_xlabel('x (cm)', fontsize=11, fontweight='bold')
ax2.set_ylabel('y (cm)', fontsize=11, fontweight='bold')
ax2.set_title('Field Lines (blue) & Equipotentials (green)', fontsize=12, fontweight='bold')
ax2.set_aspect('equal')
ax2.set_xlim([0, Lx*1e2])
ax2.set_ylim([0, Ly*1e2])

# 3. 전기장 크기
ax3 = fig2.add_subplot(2, 3, 3)
im3 = ax3.contourf(X*1e2, Y*1e2, np.log10(E_mag2 + 1), levels=20, cmap='plasma')
plt.colorbar(im3, ax=ax3, label='log10(E) [V/m]')

ax3.add_patch(plt.Circle((x[center_x]*1e2, y[center_y]*1e2), r_inner*1e2, 
                         fill=False, edgecolor='white', linewidth=2))
ax3.add_patch(plt.Circle((x[center_x]*1e2, y[center_y]*1e2), r_outer*1e2, 
                         fill=False, edgecolor='white', linewidth=2))

ax3.set_xlabel('x (cm)', fontsize=11, fontweight='bold')
ax3.set_ylabel('y (cm)', fontsize=11, fontweight='bold')
ax3.set_title('Electric Field Magnitude', fontsize=12, fontweight='bold')
ax3.set_aspect('equal')

# 4. 반지름 방향 전위
ax4 = fig2.add_subplot(2, 3, 4)

# 반지름 배열
r_array = np.linspace(r_inner, r_outer, 100)
V_radial = []

for r_val in r_array:
    # y = center_y인 점에서 샘플링
    i_sample = int((r_val + x[center_x]) / dx)
    if 0 <= i_sample < Nx:
        V_radial.append(V2_solved[center_y, i_sample])
    else:
        V_radial.append(0)

ax4.plot(r_array*1e2, V_radial, 'b-', linewidth=2.5, label='Numerical')

# 이론해 (원통 커패시터)
# V(r) = V_inner * ln(r_outer/r) / ln(r_outer/r_inner)
V_theory_radial = 100 * np.log(r_outer/r_array) / np.log(r_outer/r_inner)
ax4.plot(r_array*1e2, V_theory_radial, 'r--', linewidth=2, label='Theory')

ax4.set_xlabel('Radius (cm)', fontsize=11, fontweight='bold')
ax4.set_ylabel('Potential (V)', fontsize=11, fontweight='bold')
ax4.set_title('Radial Potential Profile', fontsize=12, fontweight='bold')
ax4.legend(fontsize=10)
ax4.grid(True, alpha=0.3)

# 5. 수렴
ax5 = fig2.add_subplot(2, 3, 5)
ax5.semilogy(residual_history2, 'b-', linewidth=2)
ax5.set_xlabel('Iteration', fontsize=11, fontweight='bold')
ax5.set_ylabel('Residual', fontsize=11, fontweight='bold')
ax5.set_title('Convergence History', fontsize=12, fontweight='bold')
ax5.grid(True, alpha=0.3)

# 6. 전기장 (반지름)
ax6 = fig2.add_subplot(2, 3, 6)

E_radial_numerical = []
for r_val in r_array:
    i_sample = int((r_val + x[center_x]) / dx)
    if 0 <= i_sample < Nx:
        E_radial_numerical.append(Ex2[center_y, i_sample])
    else:
        E_radial_numerical.append(0)

ax6.plot(r_array*1e2, np.abs(E_radial_numerical), 'b-', linewidth=2.5, label='Numerical')

# 이론해: E = V_inner / (r * ln(r_outer/r_inner))
E_theory_radial = 100 / (r_array * np.log(r_outer/r_inner))
ax6.plot(r_array*1e2, E_theory_radial, 'r--', linewidth=2, label='Theory (1/r)')

ax6.set_xlabel('Radius (cm)', fontsize=11, fontweight='bold')
ax6.set_ylabel('Electric Field (V/m)', fontsize=11, fontweight='bold')
ax6.set_title('Radial Electric Field', fontsize=12, fontweight='bold')
ax6.legend(fontsize=10)
ax6.grid(True, alpha=0.3)

plt.suptitle('Laplace Equation: Cylindrical Capacitor', fontsize=15, fontweight='bold')
plt.tight_layout()

output_path2 = f'{output_dir}/10_cylindrical_capacitor.png'
plt.savefig(output_path2, dpi=150, bbox_inches='tight')
print(f"[OK] 그래프 저장: {output_path2}")
plt.close()

print("\n" + "="*70)
print("분석 완료!")
print("="*70)
print("\n핵심 개념:")
print("  1. 라플라스 방정식: ∇²V = 0")
print("  2. 유한 차분법: 2차 미분을 격자로 근사")
print("  3. 가우스-자이델 반복법: 점진적 수렴")
print("  4. 경계 조건이 해를 결정")
print("  5. 수치해가 이론해와 일치")
print("\n수치 방법:")
print("  - 5점 stencil (2D)")
print("  - 반복법 (Jacobi, Gauss-Seidel, SOR)")
print("  - 수렴 조건: ||V_new - V_old|| < tol")
print("\n응용:")
print("  - 커패시터 설계")
print("  - 정전기 차폐")
print("  - 고전압 장치")
print("  - 반도체 소자")
print("\n생성된 파일:")
print(f"  - {output_path1}")
print(f"  - {output_path2}")

