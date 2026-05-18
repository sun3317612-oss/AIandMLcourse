"""
03. Electric Field Lines
전기력선 시각화

물리 배경:
- 전기력선: 전기장의 방향을 나타내는 곡선
- 성질:
  1. 양전하에서 시작, 음전하에서 끝
  2. 서로 교차하지 않음
  3. 밀도 ∝ 전기장 세기

학습 목표:
1. 전기력선의 의미 이해
2. Streamplot을 이용한 시각화
3. 다양한 전하 배치의 전기력선 관찰
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
print("03. Electric Field Lines - Streamplot Visualization")
print("="*70)

# 물리 상수
k_e = 8.99e9

# 시뮬레이션 파라미터
x_range = np.linspace(-1.0, 1.0, 100)
y_range = np.linspace(-1.0, 1.0, 100)
X, Y = np.meshgrid(x_range, y_range)

def electric_field_multiple(x, y, charges, positions):
    """
    여러 점전하의 전기장 계산 (중첩 원리)
    
    charges: 전하량 리스트
    positions: 전하 위치 리스트 [(x1,y1), (x2,y2), ...]
    """
    Ex_total = np.zeros_like(x)
    Ey_total = np.zeros_like(y)
    
    for Q, pos in zip(charges, positions):
        dx = x - pos[0]
        dy = y - pos[1]
        r = np.sqrt(dx**2 + dy**2)
        r = np.maximum(r, 1e-10)
        
        Ex_total += k_e * Q * dx / r**3
        Ey_total += k_e * Q * dy / r**3
    
    return Ex_total, Ey_total

# 시나리오들
scenarios = [
    {
        'name': 'Dipole',
        'charges': [1e-9, -1e-9],
        'positions': [(-0.3, 0.0), (0.3, 0.0)],
        'title': 'Electric Dipole'
    },
    {
        'name': 'Two Positive',
        'charges': [1e-9, 1e-9],
        'positions': [(-0.3, 0.0), (0.3, 0.0)],
        'title': 'Two Positive Charges'
    },
    {
        'name': 'Linear Triple',
        'charges': [1e-9, -2e-9, 1e-9],
        'positions': [(-0.4, 0.0), (0.0, 0.0), (0.4, 0.0)],
        'title': 'Linear Quadrupole'
    },
    {
        'name': 'Triangle',
        'charges': [1e-9, 1e-9, -2e-9],
        'positions': [(-0.3, -0.3), (0.3, -0.3), (0.0, 0.4)],
        'title': 'Triangular Configuration'
    }
]

print(f"\n총 {len(scenarios)}개의 시나리오 계산 중...")

# 메인 시각화
fig, axes = plt.subplots(2, 2, figsize=(14, 14))
axes = axes.ravel()

for idx, scenario in enumerate(scenarios):
    print(f"\n{idx+1}. {scenario['name']} 계산 중...")
    
    charges = scenario['charges']
    positions = scenario['positions']
    
    # 전기장 계산
    Ex, Ey = electric_field_multiple(X, Y, charges, positions)
    
    # 전기장 크기
    E_mag = np.sqrt(Ex**2 + Ey**2)
    
    ax = axes[idx]
    
    # 배경: 전기장 크기 (로그 스케일)
    E_mag_log = np.log10(E_mag + 1e-10)
    contourf = ax.contourf(X, Y, E_mag_log, levels=20, cmap='YlOrRd', alpha=0.5)
    
    # 전기력선 (Streamplot)
    # 전기장이 너무 강한 곳은 제외 (시각화를 위해)
    Ex_plot = np.copy(Ex)
    Ey_plot = np.copy(Ey)
    
    # 정규화 (streamplot은 자동으로 하지만 명시적으로)
    speed = np.sqrt(Ex_plot**2 + Ey_plot**2)
    
    # Streamplot
    stream = ax.streamplot(X, Y, Ex_plot, Ey_plot, 
                           color='blue', linewidth=1.5, 
                           density=2.0, arrowsize=1.5, arrowstyle='->')
    
    # 전하 표시
    for Q, pos in zip(charges, positions):
        if Q > 0:
            ax.plot(pos[0], pos[1], 'ro', markersize=15, 
                   markeredgecolor='black', markeredgewidth=2, 
                   label=f'+{abs(Q)*1e9:.1f}nC' if idx == 0 else '')
        else:
            ax.plot(pos[0], pos[1], 'bo', markersize=15, 
                   markeredgecolor='black', markeredgewidth=2,
                   label=f'-{abs(Q)*1e9:.1f}nC' if idx == 0 else '')
    
    ax.set_xlabel('x (m)', fontsize=11, fontweight='bold')
    ax.set_ylabel('y (m)', fontsize=11, fontweight='bold')
    ax.set_title(scenario['title'], fontsize=12, fontweight='bold')
    ax.set_xlim([x_range[0], x_range[-1]])
    ax.set_ylim([y_range[0], y_range[-1]])
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_aspect('equal')
    
    if idx == 0:
        ax.legend(fontsize=9, loc='upper right')

plt.suptitle('Electric Field Lines for Various Charge Configurations', 
             fontsize=15, fontweight='bold')
plt.tight_layout()

output_path = f'{output_dir}/03_field_lines.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\n[OK] 그래프 저장: {output_path}")
plt.close()

# 추가: 상세한 쌍극자 분석
print("\n쌍극자 상세 분석 중...")

fig2, axes2 = plt.subplots(1, 2, figsize=(14, 6))

# 쌍극자
charges_dipole = [1e-9, -1e-9]
positions_dipole = [(-0.3, 0.0), (0.3, 0.0)]

Ex_dipole, Ey_dipole = electric_field_multiple(X, Y, charges_dipole, positions_dipole)
E_mag_dipole = np.sqrt(Ex_dipole**2 + Ey_dipole**2)

# 왼쪽: 전기력선 + 등전위선
ax1 = axes2[0]

# 전위 계산
V = np.zeros_like(X)
for Q, pos in zip(charges_dipole, positions_dipole):
    dx = X - pos[0]
    dy = Y - pos[1]
    r = np.sqrt(dx**2 + dy**2)
    r = np.maximum(r, 1e-10)
    V += k_e * Q / r

# 등전위선
V_limited = np.clip(V, -50, 50)
contour = ax1.contour(X, Y, V_limited, levels=15, colors='green', 
                      linewidths=1.5, alpha=0.7, linestyles='dashed')

# 전기력선
stream = ax1.streamplot(X, Y, Ex_dipole, Ey_dipole, 
                        color='blue', linewidth=1.5, 
                        density=2.5, arrowsize=1.5, arrowstyle='->')

# 전하
ax1.plot(positions_dipole[0][0], positions_dipole[0][1], 'ro', 
         markersize=15, markeredgecolor='black', markeredgewidth=2, label='Positive')
ax1.plot(positions_dipole[1][0], positions_dipole[1][1], 'bo', 
         markersize=15, markeredgecolor='black', markeredgewidth=2, label='Negative')

ax1.set_xlabel('x (m)', fontsize=11, fontweight='bold')
ax1.set_ylabel('y (m)', fontsize=11, fontweight='bold')
ax1.set_title('Field Lines (blue) and Equipotentials (green)', fontsize=12, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_aspect('equal')

# 오른쪽: 전기장 크기
ax2 = axes2[1]

E_mag_log = np.log10(E_mag_dipole + 1e-10)
im = ax2.contourf(X, Y, E_mag_log, levels=20, cmap='plasma')
cbar = plt.colorbar(im, ax=ax2)
cbar.set_label('log10(E) [N/C]', fontsize=11, fontweight='bold')

# 전하
ax2.plot(positions_dipole[0][0], positions_dipole[0][1], 'ro', 
         markersize=15, markeredgecolor='white', markeredgewidth=2)
ax2.plot(positions_dipole[1][0], positions_dipole[1][1], 'bo', 
         markersize=15, markeredgecolor='white', markeredgewidth=2)

ax2.set_xlabel('x (m)', fontsize=11, fontweight='bold')
ax2.set_ylabel('y (m)', fontsize=11, fontweight='bold')
ax2.set_title('Electric Field Magnitude', fontsize=12, fontweight='bold')
ax2.set_aspect('equal')

plt.suptitle('Detailed Dipole Analysis', fontsize=14, fontweight='bold')
plt.tight_layout()

output_path2 = f'{output_dir}/03_dipole_detailed.png'
plt.savefig(output_path2, dpi=150, bbox_inches='tight')
print(f"[OK] 그래프 저장: {output_path2}")
plt.close()

print("\n" + "="*70)
print("분석 완료!")
print("="*70)
print("\n핵심 개념:")
print("  1. 전기력선: 양전하에서 시작 → 음전하에서 끝")
print("  2. 전기력선은 전기장의 방향을 나타냄")
print("  3. 전기력선 밀도 = 전기장 세기")
print("  4. 전기력선과 등전위선은 서로 수직")
print("  5. 전기력선은 절대 교차하지 않음")
print("\n물리적 의미:")
print("  - 쌍극자: 가장 간단한 비대칭 전하 분포")
print("  - 사극자: 쌍극자보다 복잡한 구조")
print("  - 실제 분자의 전하 분포 모델링")
print("\n생성된 파일:")
print(f"  - {output_path}")
print(f"  - {output_path2}")

