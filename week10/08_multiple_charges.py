"""
08. Multiple Point Charges - Advanced Configuration
다중 점전하의 전기장 계산

물리 배경:
- 중첩 원리: 전체 전기장 = 개별 전기장의 벡터 합
- 복잡한 전하 배치 분석
- 사극자, 육극자 등

학습 목표:
1. 중첩 원리 적용
2. 복잡한 전하 배치 분석
3. 전기력선과 등전위선 종합
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
print("08. Multiple Point Charges - Complex Configurations")
print("="*70)

# 물리 상수
k_e = 8.99e9

# 시뮬레이션 파라미터
x_range = np.linspace(-1.5, 1.5, 150)
y_range = np.linspace(-1.5, 1.5, 150)
X, Y = np.meshgrid(x_range, y_range)

def electric_field_multiple(x, y, charges, positions):
    """여러 점전하의 전기장 (중첩 원리)"""
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

def electric_potential_multiple(x, y, charges, positions):
    """여러 점전하의 전위"""
    V_total = np.zeros_like(x)
    
    for Q, pos in zip(charges, positions):
        dx = x - pos[0]
        dy = y - pos[1]
        r = np.sqrt(dx**2 + dy**2)
        r = np.maximum(r, 1e-10)
        
        V_total += k_e * Q / r
    
    return V_total

# 복잡한 전하 배치들
configurations = [
    {
        'name': 'Quadrupole (Linear)',
        'charges': [1e-9, -2e-9, 1e-9],
        'positions': [(-0.6, 0.0), (0.0, 0.0), (0.6, 0.0)],
        'description': 'Linear quadrupole configuration'
    },
    {
        'name': 'Square Configuration',
        'charges': [1e-9, -1e-9, 1e-9, -1e-9],
        'positions': [(-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5)],
        'description': 'Alternating charges at square corners'
    },
    {
        'name': 'Hexagonal Configuration',
        'charges': [1e-9, 1e-9, 1e-9, 1e-9, 1e-9, 1e-9, -6e-9],
        'positions': [
            (0.6*np.cos(i*np.pi/3), 0.6*np.sin(i*np.pi/3)) for i in range(6)
        ] + [(0.0, 0.0)],
        'description': 'Six positive charges around central negative'
    },
    {
        'name': 'Molecular Model',
        'charges': [2e-9, -1e-9, -1e-9],
        'positions': [(0.0, 0.0), (-0.7, 0.0), (0.35, 0.606)],
        'description': 'Simple molecular charge distribution'
    }
]

print(f"\n총 {len(configurations)}개의 구성 분석 중...")

# 메인 시각화
fig, axes = plt.subplots(2, 2, figsize=(16, 16))
axes = axes.ravel()

for idx, config in enumerate(configurations):
    print(f"\n{idx+1}. {config['name']}")
    print(f"   전하 개수: {len(config['charges'])}")
    
    charges = config['charges']
    positions = config['positions']
    
    # 전기장과 전위 계산
    Ex, Ey = electric_field_multiple(X, Y, charges, positions)
    V = electric_potential_multiple(X, Y, charges, positions)
    
    E_mag = np.sqrt(Ex**2 + Ey**2)
    
    ax = axes[idx]
    
    # 배경: 전위
    V_limited = np.clip(V, -100, 100)
    contourf = ax.contourf(X, Y, V_limited, levels=30, cmap='coolwarm', alpha=0.6)
    
    # 등전위선
    V_levels = np.linspace(-50, 50, 21)
    contour = ax.contour(X, Y, V_limited, levels=V_levels, colors='gray', 
                         linewidths=1, alpha=0.5, linestyles='dashed')
    
    # 전기력선
    stream = ax.streamplot(X, Y, Ex, Ey, color='blue', linewidth=1.5, 
                           density=2.0, arrowsize=1.2, arrowstyle='->')
    
    # 전하 표시
    for Q, pos in zip(charges, positions):
        if Q > 0:
            ax.plot(pos[0], pos[1], 'ro', markersize=12, 
                   markeredgecolor='black', markeredgewidth=2)
            ax.text(pos[0], pos[1]-0.15, f'+{abs(Q)*1e9:.1f}', 
                   ha='center', fontsize=8, fontweight='bold')
        else:
            ax.plot(pos[0], pos[1], 'bo', markersize=12, 
                   markeredgecolor='black', markeredgewidth=2)
            ax.text(pos[0], pos[1]-0.15, f'-{abs(Q)*1e9:.1f}', 
                   ha='center', fontsize=8, fontweight='bold')
    
    ax.set_xlabel('x (m)', fontsize=11, fontweight='bold')
    ax.set_ylabel('y (m)', fontsize=11, fontweight='bold')
    ax.set_title(f'{config["name"]}\n{config["description"]}', 
                fontsize=11, fontweight='bold')
    ax.set_xlim([x_range[0], x_range[-1]])
    ax.set_ylim([y_range[0], y_range[-1]])
    ax.grid(True, alpha=0.2, linestyle='--')
    ax.set_aspect('equal')

plt.suptitle('Multiple Point Charges: Complex Configurations', 
             fontsize=16, fontweight='bold')
plt.tight_layout()

output_path1 = f'{output_dir}/08_multiple_charges.png'
plt.savefig(output_path1, dpi=150, bbox_inches='tight')
print(f"\n[OK] 그래프 저장: {output_path1}")
plt.close()

# 추가 분석: 사극자 상세
print("\n사극자 상세 분석...")

fig2, axes2 = plt.subplots(2, 2, figsize=(14, 12))

# 선형 사극자
charges_q = [1e-9, -2e-9, 1e-9]
positions_q = [(-0.6, 0.0), (0.0, 0.0), (0.6, 0.0)]

Ex_q, Ey_q = electric_field_multiple(X, Y, charges_q, positions_q)
V_q = electric_potential_multiple(X, Y, charges_q, positions_q)
E_mag_q = np.sqrt(Ex_q**2 + Ey_q**2)

# 1. 전기력선 + 등전위선
ax1 = axes2[0, 0]
V_q_limited = np.clip(V_q, -50, 50)
contour1 = ax1.contour(X, Y, V_q_limited, levels=15, colors='green', 
                       linewidths=1.5, alpha=0.7, linestyles='dashed')
stream1 = ax1.streamplot(X, Y, Ex_q, Ey_q, color='blue', linewidth=1.5, 
                         density=2.5, arrowsize=1.5)

for Q, pos in zip(charges_q, positions_q):
    color = 'r' if Q > 0 else 'b'
    ax1.plot(pos[0], pos[1], f'{color}o', markersize=15, 
            markeredgecolor='black', markeredgewidth=2)

ax1.set_xlabel('x (m)', fontsize=11, fontweight='bold')
ax1.set_ylabel('y (m)', fontsize=11, fontweight='bold')
ax1.set_title('Field Lines (blue) & Equipotentials (green)', 
             fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.set_aspect('equal')

# 2. 전기장 크기
ax2 = axes2[0, 1]
E_mag_q_log = np.log10(E_mag_q + 1)
im2 = ax2.contourf(X, Y, E_mag_q_log, levels=20, cmap='plasma')
plt.colorbar(im2, ax=ax2, label='log10(E)')

for Q, pos in zip(charges_q, positions_q):
    if Q > 0:
        ax2.plot(pos[0], pos[1], 'ro', markersize=12, 
                markeredgecolor='white', markeredgewidth=2)
    else:
        ax2.plot(pos[0], pos[1], 'co', markersize=12, 
                markeredgecolor='white', markeredgewidth=2)

ax2.set_xlabel('x (m)', fontsize=11, fontweight='bold')
ax2.set_ylabel('y (m)', fontsize=11, fontweight='bold')
ax2.set_title('Electric Field Magnitude', fontsize=12, fontweight='bold')
ax2.set_aspect('equal')

# 3. x축 따라 전위
ax3 = axes2[1, 0]
y_slice = len(y_range) // 2
x_slice = x_range
V_slice = V_q[y_slice, :]

ax3.plot(x_slice, V_slice, 'b-', linewidth=2.5, label='V(x, y=0)')
ax3.axhline(0, color='k', linestyle='--', alpha=0.3)

for Q, pos in zip(charges_q, positions_q):
    if pos[1] == 0.0:
        ax3.axvline(pos[0], color='r', linestyle=':', alpha=0.5)

ax3.set_xlabel('x (m)', fontsize=11, fontweight='bold')
ax3.set_ylabel('Potential (V)', fontsize=11, fontweight='bold')
ax3.set_title('Potential along x-axis', fontsize=12, fontweight='bold')
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.3)

# 4. 전기장 x축
ax4 = axes2[1, 1]
Ex_slice = Ex_q[y_slice, :]
Ey_slice = Ey_q[y_slice, :]
E_mag_slice = np.sqrt(Ex_slice**2 + Ey_slice**2)

ax4.plot(x_slice, Ex_slice/1e6, 'r-', linewidth=2, label='Ex')
ax4.plot(x_slice, Ey_slice/1e6, 'g-', linewidth=2, label='Ey')
ax4.plot(x_slice, E_mag_slice/1e6, 'b-', linewidth=2.5, label='|E|')
ax4.axhline(0, color='k', linestyle='--', alpha=0.3)

ax4.set_xlabel('x (m)', fontsize=11, fontweight='bold')
ax4.set_ylabel('E field (10^6 N/C)', fontsize=11, fontweight='bold')
ax4.set_title('E field components along x-axis', fontsize=12, fontweight='bold')
ax4.legend(fontsize=10)
ax4.grid(True, alpha=0.3)

plt.suptitle('Quadrupole Detailed Analysis', fontsize=14, fontweight='bold')
plt.tight_layout()

output_path2 = f'{output_dir}/08_quadrupole_analysis.png'
plt.savefig(output_path2, dpi=150, bbox_inches='tight')
print(f"[OK] 그래프 저장: {output_path2}")
plt.close()

print("\n" + "="*70)
print("분석 완료!")
print("="*70)
print("\n핵심 개념:")
print("  1. 중첩 원리: E_total = ΣE_i")
print("  2. 사극자: 쌍극자보다 더 빠르게 감소 (∝ 1/r³)")
print("  3. 대칭성이 전기장 패턴 결정")
print("  4. 전기력선 밀도 = 전기장 세기")
print("  5. 복잡한 분자 구조 모델링")
print("\n물리적 의미:")
print("  - 쌍극자: 1/r² 감소")
print("  - 사극자: 1/r³ 감소")
print("  - 멀리 갈수록 영향 급감")
print("  - 분자간 상호작용 이해")
print("\n응용:")
print("  - 분자 동역학")
print("  - 정전기 트랩")
print("  - 이온 렌즈")
print("\n생성된 파일:")
print(f"  - {output_path1}")
print(f"  - {output_path2}")

