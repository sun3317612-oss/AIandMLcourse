"""
02. Electric Potential
점전하의 전위(Potential) 계산 및 시각화

물리 배경:
- 전위: V = k*Q/r
- 전기장과 전위의 관계: E = -∇V
- 등전위선: 전위가 같은 점들을 연결한 선

학습 목표:
1. 전위의 개념 이해
2. 등전위선 시각화
3. 전기장과 전위의 관계 확인
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
print("02. Electric Potential - Equipotential Lines")
print("="*70)

# 물리 상수
k_e = 8.99e9  # 쿨롱 상수

# 시뮬레이션 파라미터
Q = 1e-9  # 전하량
x_range = np.linspace(-0.5, 0.5, 200)  # 고해상도 격자
y_range = np.linspace(-0.5, 0.5, 200)
X, Y = np.meshgrid(x_range, y_range)

# 전하 위치
q_pos = np.array([0.0, 0.0])

print(f"\n전하량: Q = {Q*1e9:.1f} nC")
print(f"격자 크기: {len(x_range)} x {len(y_range)}")

def electric_potential(x, y, Q, q_pos):
    """
    점전하의 전위 계산
    
    V = k*Q/r
    
    전위는 스칼라장 (각 점에서 크기만 존재)
    """
    dx = x - q_pos[0]
    dy = y - q_pos[1]
    r = np.sqrt(dx**2 + dy**2)
    r = np.maximum(r, 1e-10)
    
    V = k_e * Q / r
    
    return V

def electric_field_from_potential(V, dx, dy):
    """
    전위로부터 전기장 계산
    
    E = -∇V = -(dV/dx, dV/dy)
    
    수치 미분 사용
    """
    # 중앙 차분법
    Ey, Ex = np.gradient(V, dy, dx)
    Ex = -Ex
    Ey = -Ey
    
    return Ex, Ey

print("\n전위 및 전기장 계산 중...")

# 전위 계산
V = electric_potential(X, Y, Q, q_pos)

print(f"최대 전위: {np.max(V):.2e} V")
print(f"최소 전위: {np.min(V):.2e} V")

# 전위로부터 전기장 계산
dx = x_range[1] - x_range[0]
dy = y_range[1] - y_range[0]
Ex, Ey = electric_field_from_potential(V, dx, dy)

# 시각화
fig = plt.figure(figsize=(16, 6))

# 1. 전위 (3D 표면)
ax1 = fig.add_subplot(131, projection='3d')

# 전위 제한 (시각화를 위해)
V_limited = np.clip(V, -100, 100)

surf = ax1.plot_surface(X, Y, V_limited, cmap='coolwarm', alpha=0.8, 
                        rstride=5, cstride=5, linewidth=0.5, edgecolor='gray')
ax1.set_xlabel('x (m)', fontsize=10, fontweight='bold')
ax1.set_ylabel('y (m)', fontsize=10, fontweight='bold')
ax1.set_zlabel('V (V)', fontsize=10, fontweight='bold')
ax1.set_title('Electric Potential (3D)', fontsize=12, fontweight='bold')
ax1.view_init(elev=25, azim=45)
fig.colorbar(surf, ax=ax1, shrink=0.5, aspect=5)

# 2. 등전위선
ax2 = fig.add_subplot(132)

# 등전위선 (로그 간격)
V_levels = np.linspace(-50, 50, 21)
contour = ax2.contour(X, Y, V, levels=V_levels, colors='black', linewidths=1.5, alpha=0.7)
ax2.clabel(contour, inline=True, fontsize=8, fmt='%.0f V')

# 배경 색상
contourf = ax2.contourf(X, Y, V, levels=50, cmap='coolwarm', alpha=0.5)
cbar2 = plt.colorbar(contourf, ax=ax2)
cbar2.set_label('V (V)', fontsize=11, fontweight='bold')

# 전하 표시
if Q > 0:
    ax2.plot(q_pos[0], q_pos[1], 'ro', markersize=12, 
             markeredgecolor='black', markeredgewidth=2, label='Positive Charge')
else:
    ax2.plot(q_pos[0], q_pos[1], 'bo', markersize=12, 
             markeredgecolor='black', markeredgewidth=2, label='Negative Charge')

ax2.set_xlabel('x (m)', fontsize=11, fontweight='bold')
ax2.set_ylabel('y (m)', fontsize=11, fontweight='bold')
ax2.set_title('Equipotential Lines', fontsize=12, fontweight='bold')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)
ax2.set_aspect('equal')

# 3. 전기장과 등전위선
ax3 = fig.add_subplot(133)

# 등전위선
contour3 = ax3.contour(X, Y, V, levels=V_levels, colors='blue', 
                       linewidths=1.5, alpha=0.6, linestyles='solid')
ax3.clabel(contour3, inline=True, fontsize=7, fmt='%.0f V')

# 전기장 벡터 (성글게)
skip = 8
X_sparse = X[::skip, ::skip]
Y_sparse = Y[::skip, ::skip]
Ex_sparse = Ex[::skip, ::skip]
Ey_sparse = Ey[::skip, ::skip]

# 정규화
E_norm = np.sqrt(Ex_sparse**2 + Ey_sparse**2)
Ex_normalized = Ex_sparse / (E_norm + 1e-10)
Ey_normalized = Ey_sparse / (E_norm + 1e-10)

ax3.quiver(X_sparse, Y_sparse, Ex_normalized, Ey_normalized, 
           angles='xy', scale_units='xy', scale=10,
           color='red', alpha=0.7, width=0.004, label='E field')

# 전하 표시
if Q > 0:
    ax3.plot(q_pos[0], q_pos[1], 'ro', markersize=12, 
             markeredgecolor='black', markeredgewidth=2)
else:
    ax3.plot(q_pos[0], q_pos[1], 'bo', markersize=12, 
             markeredgecolor='black', markeredgewidth=2)

ax3.set_xlabel('x (m)', fontsize=11, fontweight='bold')
ax3.set_ylabel('y (m)', fontsize=11, fontweight='bold')
ax3.set_title('E = -grad(V)', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3)
ax3.set_aspect('equal')

plt.suptitle(f'Electric Potential and Field (Q = {Q*1e9:.1f} nC)', 
             fontsize=14, fontweight='bold')
plt.tight_layout()

output_path = f'{output_dir}/02_potential_contours.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\n[OK] 그래프 저장: {output_path}")
plt.close()

# 추가: 두 전하의 전위
print("\n두 전하 시스템 계산 중...")

fig2, axes = plt.subplots(1, 2, figsize=(14, 6))

# Case 1: 같은 부호 (양전하 두 개)
Q1, Q2 = 1e-9, 1e-9
pos1 = np.array([-0.2, 0.0])
pos2 = np.array([0.2, 0.0])

V1 = electric_potential(X, Y, Q1, pos1)
V2 = electric_potential(X, Y, Q2, pos2)
V_total = V1 + V2

ax = axes[0]
contour = ax.contour(X, Y, V_total, levels=20, colors='black', linewidths=1.5, alpha=0.7)
ax.clabel(contour, inline=True, fontsize=7, fmt='%.0f V')
contourf = ax.contourf(X, Y, V_total, levels=50, cmap='coolwarm', alpha=0.5)

ax.plot(pos1[0], pos1[1], 'ro', markersize=12, markeredgecolor='black', markeredgewidth=2)
ax.plot(pos2[0], pos2[1], 'ro', markersize=12, markeredgecolor='black', markeredgewidth=2)

ax.set_xlabel('x (m)', fontsize=11, fontweight='bold')
ax.set_ylabel('y (m)', fontsize=11, fontweight='bold')
ax.set_title('Two Positive Charges', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.set_aspect('equal')
plt.colorbar(contourf, ax=ax, label='V (V)')

# Case 2: 다른 부호 (전기 쌍극자)
Q1, Q2 = 1e-9, -1e-9

V1 = electric_potential(X, Y, Q1, pos1)
V2 = electric_potential(X, Y, Q2, pos2)
V_total = V1 + V2

ax = axes[1]
contour = ax.contour(X, Y, V_total, levels=20, colors='black', linewidths=1.5, alpha=0.7)
ax.clabel(contour, inline=True, fontsize=7, fmt='%.0f V')
contourf = ax.contourf(X, Y, V_total, levels=50, cmap='coolwarm', alpha=0.5)

ax.plot(pos1[0], pos1[1], 'ro', markersize=12, markeredgecolor='black', markeredgewidth=2)
ax.plot(pos2[0], pos2[1], 'bo', markersize=12, markeredgecolor='black', markeredgewidth=2)

ax.set_xlabel('x (m)', fontsize=11, fontweight='bold')
ax.set_ylabel('y (m)', fontsize=11, fontweight='bold')
ax.set_title('Electric Dipole (+ and -)', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.set_aspect('equal')
plt.colorbar(contourf, ax=ax, label='V (V)')

plt.suptitle('Equipotential Lines for Two Charges', fontsize=14, fontweight='bold')
plt.tight_layout()

output_path2 = f'{output_dir}/02_two_charges_potential.png'
plt.savefig(output_path2, dpi=150, bbox_inches='tight')
print(f"[OK] 그래프 저장: {output_path2}")
plt.close()

print("\n" + "="*70)
print("분석 완료!")
print("="*70)
print("\n핵심 개념:")
print("  1. 전위는 스칼라장 (방향 없음, 크기만 존재)")
print("  2. 등전위선: 전위가 같은 점들 (전기장에 수직)")
print("  3. 전기장 = -∇V (전위의 기울기)")
print("  4. 전기장은 높은 전위에서 낮은 전위로")
print("  5. 전하는 전위가 낮은 곳으로 이동")
print("\n생성된 파일:")
print(f"  - {output_path}")
print(f"  - {output_path2}")

