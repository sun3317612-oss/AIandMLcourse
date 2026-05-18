"""
04. Magnetic Field Basics
무한 직선 전류의 자기장

물리 배경:
- 비오-사바르 법칙: dB = (μ₀/4π) * (I*dl × r̂)/r²
- 무한 직선 전류: B = μ₀*I/(2π*r)
- 오른손 법칙: 엄지=전류 방향, 나머지 손가락=자기장 방향

학습 목표:
1. 전류가 만드는 자기장 이해
2. 원형 자기장 패턴 관찰
3. 비오-사바르 법칙 적용
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
print("04. Magnetic Field Basics - Straight Wire")
print("="*70)

# 물리 상수
mu_0 = 4 * np.pi * 1e-7  # 진공 투자율 (T*m/A)

# 시뮬레이션 파라미터
I = 10.0  # 전류 (A)
x_range = np.linspace(-0.1, 0.1, 30)
y_range = np.linspace(-0.1, 0.1, 30)
X, Y = np.meshgrid(x_range, y_range)

# 전류 위치 (z축 방향, 원점 통과)
wire_pos = np.array([0.0, 0.0])

print(f"\n전류: I = {I} A")
print(f"격자 크기: {len(x_range)} x {len(y_range)}")
print(f"방향: z축 (종이 수직 방향)")

def magnetic_field_straight_wire(x, y, I, wire_pos):
    """
    무한 직선 전류의 자기장 계산 (z축 방향 전류)
    
    B = (μ₀*I)/(2π*r) * φ̂
    
    여기서:
    - r: 전선으로부터의 거리
    - φ̂: 원주 방향 단위 벡터 (오른손 법칙)
    """
    # 전선으로부터의 벡터
    dx = x - wire_pos[0]
    dy = y - wire_pos[1]
    
    # 거리
    r = np.sqrt(dx**2 + dy**2)
    r = np.maximum(r, 1e-10)
    
    # 자기장 크기
    B_magnitude = (mu_0 * I) / (2 * np.pi * r)
    
    # 자기장 방향 (원주 방향)
    # φ̂ = (-sin(φ), cos(φ)) = (-y/r, x/r) (z축이 종이에서 나오는 방향일 때)
    Bx = -dy / r * B_magnitude
    By = dx / r * B_magnitude
    
    return Bx, By, B_magnitude

print("\n자기장 계산 중...")

# 자기장 계산
Bx, By, B_mag = magnetic_field_straight_wire(X, Y, I, wire_pos)

print(f"최대 자기장 크기: {np.max(B_mag)*1e6:.2f} μT")
print(f"최소 자기장 크기: {np.min(B_mag)*1e6:.2f} μT")

# 시각화
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 1. 벡터장
ax1 = axes[0]

# 배경: 자기장 크기
B_mag_log = np.log10(B_mag * 1e6 + 1e-10)  # μT 단위로 변환
contourf = ax1.contourf(X*100, Y*100, B_mag_log, levels=20, cmap='viridis', alpha=0.6)
cbar1 = plt.colorbar(contourf, ax=ax1)
cbar1.set_label('log10(B) [μT]', fontsize=11, fontweight='bold')

# 벡터 화살표 (정규화)
B_norm = np.sqrt(Bx**2 + By**2)
Bx_normalized = Bx / (B_norm + 1e-20)
By_normalized = By / (B_norm + 1e-20)

ax1.quiver(X*100, Y*100, Bx_normalized, By_normalized, 
           angles='xy', scale_units='xy', scale=10,
           color='white', alpha=0.8, width=0.003)

# 전선 위치 (z축 방향, 종이에서 나옴 = ⊙, 들어감 = ⊗)
circle = plt.Circle((wire_pos[0]*100, wire_pos[1]*100), 0.5, 
                     color='red', fill=True, zorder=10)
ax1.add_patch(circle)
ax1.plot(wire_pos[0]*100, wire_pos[1]*100, 'k.', markersize=3)
ax1.text(wire_pos[0]*100 + 2, wire_pos[1]*100 + 2, 'I (out)', 
         fontsize=10, fontweight='bold', color='red')

ax1.set_xlabel('x (cm)', fontsize=12, fontweight='bold')
ax1.set_ylabel('y (cm)', fontsize=12, fontweight='bold')
ax1.set_title('Magnetic Field Vector Map', fontsize=13, fontweight='bold')
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.set_aspect('equal')

# 2. 원형 자기장 라인
ax2 = axes[1]

# 자기장 라인 (원형)
theta = np.linspace(0, 2*np.pi, 100)
radii = np.array([0.02, 0.04, 0.06, 0.08, 0.10])  # m

for r in radii:
    x_circle = wire_pos[0] + r * np.cos(theta)
    y_circle = wire_pos[1] + r * np.sin(theta)
    
    # 해당 반지름에서의 자기장 크기
    B_at_r = (mu_0 * I) / (2 * np.pi * r)
    
    ax2.plot(x_circle*100, y_circle*100, 'b-', linewidth=2, alpha=0.7,
             label=f'r={r*100:.0f}cm, B={B_at_r*1e6:.1f}μT' if r == radii[0] or r == radii[-1] else '')

# 화살표로 방향 표시
for r in radii[::2]:
    for angle in [0, np.pi/2, np.pi, 3*np.pi/2]:
        x_pos = wire_pos[0] + r * np.cos(angle)
        y_pos = wire_pos[1] + r * np.sin(angle)
        
        # 접선 방향
        dx_tangent = -np.sin(angle) * r * 0.3
        dy_tangent = np.cos(angle) * r * 0.3
        
        ax2.arrow(x_pos*100, y_pos*100, dx_tangent*100, dy_tangent*100,
                  head_width=1.5, head_length=1.0, fc='blue', ec='blue', alpha=0.7)

# 전선
circle2 = plt.Circle((wire_pos[0]*100, wire_pos[1]*100), 0.5, 
                      color='red', fill=True, zorder=10)
ax2.add_patch(circle2)
ax2.plot(wire_pos[0]*100, wire_pos[1]*100, 'k.', markersize=3)

ax2.set_xlabel('x (cm)', fontsize=12, fontweight='bold')
ax2.set_ylabel('y (cm)', fontsize=12, fontweight='bold')
ax2.set_title('Circular Magnetic Field Lines', fontsize=13, fontweight='bold')
ax2.legend(fontsize=9, loc='upper right')
ax2.grid(True, alpha=0.3, linestyle='--')
ax2.set_aspect('equal')
ax2.set_xlim([-11, 11])
ax2.set_ylim([-11, 11])

plt.suptitle(f'Straight Wire: Magnetic Field (I = {I} A)', 
             fontsize=15, fontweight='bold')
plt.tight_layout()

output_path = f'{output_dir}/04_magnetic_field.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\n[OK] 그래프 저장: {output_path}")
plt.close()

# 추가: B vs r 그래프 및 두 전류
print("\n추가 분석 중...")

fig2, axes2 = plt.subplots(1, 2, figsize=(14, 6))

# 왼쪽: B vs r
ax_left = axes2[0]

r_plot = np.linspace(0.001, 0.15, 200)
B_plot = (mu_0 * I) / (2 * np.pi * r_plot)

ax_left.plot(r_plot*100, B_plot*1e6, 'b-', linewidth=2.5, label=f'I = {I} A')
ax_left.plot(r_plot*100, B_plot*1e6*2, 'r--', linewidth=2, label=f'I = {I*2} A')
ax_left.plot(r_plot*100, B_plot*1e6*0.5, 'g:', linewidth=2, label=f'I = {I*0.5} A')

ax_left.set_xlabel('Distance from wire (cm)', fontsize=11, fontweight='bold')
ax_left.set_ylabel('Magnetic Field (μT)', fontsize=11, fontweight='bold')
ax_left.set_title('B = μ₀I/(2πr)', fontsize=12, fontweight='bold')
ax_left.legend(fontsize=10)
ax_left.grid(True, alpha=0.3)
ax_left.set_xlim([0, 15])

# 오른쪽: 두 평행 전류
ax_right = axes2[1]

# 같은 방향 전류
I1, I2 = 10.0, 10.0
pos1 = np.array([-0.03, 0.0])
pos2 = np.array([0.03, 0.0])

Bx1, By1, _ = magnetic_field_straight_wire(X, Y, I1, pos1)
Bx2, By2, _ = magnetic_field_straight_wire(X, Y, I2, pos2)

Bx_total = Bx1 + Bx2
By_total = By1 + By2
B_total = np.sqrt(Bx_total**2 + By_total**2)

# 배경
B_total_log = np.log10(B_total * 1e6 + 1e-10)
contourf2 = ax_right.contourf(X*100, Y*100, B_total_log, levels=20, cmap='viridis', alpha=0.6)
cbar2 = plt.colorbar(contourf2, ax=ax_right)
cbar2.set_label('log10(B) [μT]', fontsize=11, fontweight='bold')

# 벡터
B_total_norm = np.sqrt(Bx_total**2 + By_total**2)
Bx_total_normalized = Bx_total / (B_total_norm + 1e-20)
By_total_normalized = By_total / (B_total_norm + 1e-20)

skip = 2
ax_right.quiver(X[::skip,::skip]*100, Y[::skip,::skip]*100, 
                Bx_total_normalized[::skip,::skip], By_total_normalized[::skip,::skip],
                angles='xy', scale_units='xy', scale=10,
                color='white', alpha=0.8, width=0.004)

# 전선들
for pos in [pos1, pos2]:
    circle = plt.Circle((pos[0]*100, pos[1]*100), 0.5, 
                         color='red', fill=True, zorder=10)
    ax_right.add_patch(circle)
    ax_right.plot(pos[0]*100, pos[1]*100, 'k.', markersize=3)

ax_right.set_xlabel('x (cm)', fontsize=11, fontweight='bold')
ax_right.set_ylabel('y (cm)', fontsize=11, fontweight='bold')
ax_right.set_title('Two Parallel Currents (Same Direction)', fontsize=12, fontweight='bold')
ax_right.grid(True, alpha=0.3)
ax_right.set_aspect('equal')

plt.suptitle('Magnetic Field Analysis', fontsize=14, fontweight='bold')
plt.tight_layout()

output_path2 = f'{output_dir}/04_magnetic_analysis.png'
plt.savefig(output_path2, dpi=150, bbox_inches='tight')
print(f"[OK] 그래프 저장: {output_path2}")
plt.close()

print("\n" + "="*70)
print("분석 완료!")
print("="*70)
print("\n핵심 개념:")
print("  1. 전류가 자기장을 만듦 (비오-사바르 법칙)")
print("  2. 직선 전류: 원형 자기장 (B ∝ 1/r)")
print("  3. 오른손 법칙으로 방향 결정")
print("  4. 자기장 크기는 거리에 반비례")
print("  5. 두 전류가 같은 방향: 서로 끌어당김")
print("\n응용:")
print("  - 전자석")
print("  - 전동기")
print("  - 변압기")
print("\n생성된 파일:")
print(f"  - {output_path}")
print(f"  - {output_path2}")

