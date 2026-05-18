"""
01. Electric Field Basics
단일 점전하의 전기장 시각화

물리 배경:
- 쿨롱의 법칙: F = k*q1*q2/r^2
- 전기장 정의: E = F/q = k*Q/r^2
- 벡터장 표현: 방향과 크기

학습 목표:
1. 점전하가 만드는 전기장 이해
2. 벡터장 시각화 방법
3. 전기장의 방향과 크기 관계
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
print("01. Electric Field Basics - Single Point Charge")
print("="*70)

# 물리 상수
k_e = 8.99e9  # 쿨롱 상수 (N*m^2/C^2)

# 시뮬레이션 파라미터
Q = 1e-9  # 전하량 (C) - 1 나노쿨롱
x_range = np.linspace(-0.5, 0.5, 25)  # 격자점 (m)
y_range = np.linspace(-0.5, 0.5, 25)
X, Y = np.meshgrid(x_range, y_range)

# 전하 위치 (원점)
q_pos = np.array([0.0, 0.0])

print(f"\n전하량: Q = {Q*1e9:.1f} nC")
print(f"격자 크기: {len(x_range)} x {len(y_range)}")
print(f"계산 범위: [{x_range[0]}, {x_range[-1]}] m")

def electric_field(x, y, Q, q_pos):
    """
    점전하가 만드는 전기장 계산
    
    E = k*Q/r^2 * r_hat
    
    여기서:
    - k: 쿨롱 상수
    - Q: 전하량
    - r: 전하로부터의 거리
    - r_hat: 단위 벡터 (방향)
    """
    # 전하로부터의 벡터
    dx = x - q_pos[0]
    dy = y - q_pos[1]
    
    # 거리 (0으로 나누는 것 방지)
    r = np.sqrt(dx**2 + dy**2)
    r = np.maximum(r, 1e-10)  # 최소 거리 설정
    
    # 전기장 크기
    E_magnitude = k_e * abs(Q) / r**2
    
    # 전기장 방향 (양전하면 밖으로, 음전하면 안으로)
    Ex = k_e * Q * dx / r**3
    Ey = k_e * Q * dy / r**3
    
    return Ex, Ey, E_magnitude

print("\n전기장 계산 중...")

# 전기장 계산
Ex, Ey, E_mag = electric_field(X, Y, Q, q_pos)

print(f"최대 전기장 크기: {np.max(E_mag):.2e} N/C")
print(f"최소 전기장 크기: {np.min(E_mag):.2e} N/C")

# 시각화
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 1. 벡터장 (Quiver plot)
ax1 = axes[0]

# 전기장 크기로 색상 표시 (로그 스케일)
E_mag_log = np.log10(E_mag + 1e-10)
color_map = ax1.contourf(X, Y, E_mag_log, levels=20, cmap='hot', alpha=0.6)
cbar1 = plt.colorbar(color_map, ax=ax1)
cbar1.set_label('log10(E) [N/C]', fontsize=11, fontweight='bold')

# 벡터 화살표 (크기 포함, 성글게 표시)
skip = 3  # 3칸마다 하나씩만 표시
X_sparse = X[::skip, ::skip]
Y_sparse = Y[::skip, ::skip]
Ex_sparse = Ex[::skip, ::skip]
Ey_sparse = Ey[::skip, ::skip]
E_mag_sparse = E_mag[::skip, ::skip]

# 로그 스케일로 크기 조정하되, 더 큰 동적 범위 사용
E_mag_log_sparse = np.log10(E_mag_sparse + 1e-10)
E_mag_log_max = np.max(E_mag_log_sparse)
E_mag_log_min = np.min(E_mag_log_sparse)

# 0.3 ~ 2.5 범위로 정규화 (약 2배 증가)
scale_factor = (E_mag_log_sparse - E_mag_log_min) / (E_mag_log_max - E_mag_log_min)
scale_factor = 0.3 + 2.2 * scale_factor  # 최소 0.3, 최대 2.5

# 방향 유지하면서 크기 적용
Ex_scaled = Ex_sparse / (E_mag_sparse + 1e-10) * scale_factor
Ey_scaled = Ey_sparse / (E_mag_sparse + 1e-10) * scale_factor

ax1.quiver(X_sparse, Y_sparse, Ex_scaled, Ey_scaled, 
           angles='xy', scale_units='xy', scale=6, 
           color='white', alpha=0.9, width=0.006, headwidth=4, headlength=5)

# 전하 위치 표시
if Q > 0:
    ax1.plot(q_pos[0], q_pos[1], 'ro', markersize=15, 
             markeredgecolor='black', markeredgewidth=2, label='Positive Charge')
else:
    ax1.plot(q_pos[0], q_pos[1], 'bo', markersize=15, 
             markeredgecolor='black', markeredgewidth=2, label='Negative Charge')

ax1.set_xlabel('x (m)', fontsize=12, fontweight='bold')
ax1.set_ylabel('y (m)', fontsize=12, fontweight='bold')
ax1.set_title('Electric Field Vector Map', fontsize=13, fontweight='bold')
ax1.legend(fontsize=10, loc='upper right')
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.set_aspect('equal')

# 2. 전기장 크기 (히트맵)
ax2 = axes[1]

# 로그 스케일로 표시
im = ax2.imshow(E_mag_log, extent=[x_range[0], x_range[-1], y_range[0], y_range[-1]], 
                origin='lower', cmap='plasma', aspect='auto')
cbar2 = plt.colorbar(im, ax=ax2)
cbar2.set_label('log10(E) [N/C]', fontsize=11, fontweight='bold')

# 전하 위치 표시
if Q > 0:
    ax2.plot(q_pos[0], q_pos[1], 'ro', markersize=15, 
             markeredgecolor='white', markeredgewidth=2)
else:
    ax2.plot(q_pos[0], q_pos[1], 'bo', markersize=15, 
             markeredgecolor='white', markeredgewidth=2)

ax2.set_xlabel('x (m)', fontsize=12, fontweight='bold')
ax2.set_ylabel('y (m)', fontsize=12, fontweight='bold')
ax2.set_title('Electric Field Magnitude', fontsize=13, fontweight='bold')
ax2.grid(False)

plt.suptitle(f'Single Point Charge (Q = {Q*1e9:.1f} nC)', 
             fontsize=15, fontweight='bold')
plt.tight_layout()

# 저장
output_path = f'{output_dir}/01_single_charge_field.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\n[OK] 그래프 저장: {output_path}")
plt.close()

# 추가 시각화: 양전하와 음전하 비교
fig2, axes2 = plt.subplots(1, 2, figsize=(14, 6))

for idx, (Q_test, title) in enumerate([(1e-9, 'Positive Charge'), (-1e-9, 'Negative Charge')]):
    ax = axes2[idx]
    
    Ex_test, Ey_test, E_mag_test = electric_field(X, Y, Q_test, q_pos)
    
    # 배경 색상
    E_mag_log_test = np.log10(E_mag_test + 1e-10)
    color_map = ax.contourf(X, Y, E_mag_log_test, levels=20, cmap='hot', alpha=0.6)
    
    # 벡터 (크기 포함, 성글게)
    skip_test = 3
    X_sparse_test = X[::skip_test, ::skip_test]
    Y_sparse_test = Y[::skip_test, ::skip_test]
    Ex_sparse_test = Ex_test[::skip_test, ::skip_test]
    Ey_sparse_test = Ey_test[::skip_test, ::skip_test]
    E_mag_sparse_test = E_mag_test[::skip_test, ::skip_test]
    
    # 로그 스케일로 크기 조정, 더 큰 동적 범위
    E_mag_log_sparse_test = np.log10(E_mag_sparse_test + 1e-10)
    E_mag_log_max_test = np.max(E_mag_log_sparse_test)
    E_mag_log_min_test = np.min(E_mag_log_sparse_test)
    
    scale_factor_test = (E_mag_log_sparse_test - E_mag_log_min_test) / (E_mag_log_max_test - E_mag_log_min_test)
    scale_factor_test = 0.3 + 2.2 * scale_factor_test  # 0.3 ~ 2.5 범위 (약 2배 증가)
    
    Ex_scaled_test = Ex_sparse_test / (E_mag_sparse_test + 1e-10) * scale_factor_test
    Ey_scaled_test = Ey_sparse_test / (E_mag_sparse_test + 1e-10) * scale_factor_test
    
    ax.quiver(X_sparse_test, Y_sparse_test, Ex_scaled_test, Ey_scaled_test, 
              angles='xy', scale_units='xy', scale=6,
              color='white', alpha=0.9, width=0.006, headwidth=4, headlength=5)
    
    # 전하 표시
    if Q_test > 0:
        ax.plot(q_pos[0], q_pos[1], 'ro', markersize=15, 
                markeredgecolor='black', markeredgewidth=2)
    else:
        ax.plot(q_pos[0], q_pos[1], 'bo', markersize=15, 
                markeredgecolor='black', markeredgewidth=2)
    
    ax.set_xlabel('x (m)', fontsize=12, fontweight='bold')
    ax.set_ylabel('y (m)', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_aspect('equal')

plt.suptitle('Comparison: Positive vs Negative Charge', fontsize=15, fontweight='bold')
plt.tight_layout()

output_path2 = f'{output_dir}/01_charge_comparison.png'
plt.savefig(output_path2, dpi=150, bbox_inches='tight')
print(f"[OK] 그래프 저장: {output_path2}")
plt.close()

print("\n" + "="*70)
print("분석 완료!")
print("="*70)
print("\n핵심 개념:")
print("  1. 전기장은 벡터장 (각 점에서 방향과 크기 존재)")
print("  2. 전기장 크기는 거리의 제곱에 반비례 (E ∝ 1/r^2)")
print("  3. 양전하: 전기장이 밖으로 (발산)")
print("  4. 음전하: 전기장이 안으로 (수렴)")
print("  5. 가까울수록 전기장이 강함")
print("\n생성된 파일:")
print(f"  - {output_path}")
print(f"  - {output_path2}")

