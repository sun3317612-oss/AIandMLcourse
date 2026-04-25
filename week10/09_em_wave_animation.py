"""
09. Electromagnetic Wave Animation
전자기파 전파 애니메이션 (E, B 필드)

물리 배경:
- 맥스웰 방정식의 해: 평면파
- E ⊥ B ⊥ k (서로 수직)
- E × B의 방향 = 전파 방향 (포인팅 벡터)

학습 목표:
1. 전자기파의 구조 이해
2. E, B 필드의 상호 관계
3. 파동의 전파 시각화
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
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
print("09. Electromagnetic Wave Animation")
print("="*70)

# 물리 상수
c = 3e8  # 광속 (m/s)
epsilon_0 = 8.85e-12  # 진공 유전율
mu_0 = 4*np.pi*1e-7  # 진공 투자율

# 파동 파라미터
freq = 1e9  # 1 GHz
wavelength = c / freq  # 파장
k = 2 * np.pi / wavelength  # 파수
omega = 2 * np.pi * freq  # 각진동수
period = 1 / freq  # 주기

print(f"\n전자기파 파라미터:")
print(f"  주파수: f = {freq/1e9:.1f} GHz")
print(f"  파장: λ = {wavelength*1e2:.1f} cm")
print(f"  주기: T = {period*1e9:.2f} ns")
print(f"  파수: k = {k:.2f} rad/m")
print(f"  각진동수: ω = {omega:.2e} rad/s")

# 공간 격자
z_range = np.linspace(0, 2*wavelength, 200)  # z 방향 (전파 방향)

# E 필드 진폭 (V/m)
E0 = 1.0
# B 필드 진폭 (T)
B0 = E0 / c

print(f"\n진폭:")
print(f"  E0 = {E0} V/m")
print(f"  B0 = {B0*1e9:.2f} nT")
print(f"  E0/B0 = {E0/B0:.2e} m/s (= c)")

# 애니메이션 프레임 생성
n_frames = 60  # 60 프레임
time_span = period  # 1 주기

print(f"\n애니메이션:")
print(f"  프레임 수: {n_frames}")
print(f"  시간 범위: {time_span*1e9:.2f} ns (1 period)")

# ============================================================================
# 애니메이션 설정
# ============================================================================
fig = plt.figure(figsize=(14, 10))
ax1 = fig.add_subplot(2, 2, (1, 3), projection='3d')
ax2 = fig.add_subplot(2, 2, 2)
ax3 = fig.add_subplot(2, 2, 4)

def update(frame):
    t = frame * time_span / n_frames
    
    # 전기장 (y 방향)
    Ey = E0 * np.sin(k * z_range - omega * t)
    # 자기장 (x 방향)
    Bx = B0 * np.sin(k * z_range - omega * t)
    
    # 1. 3D 표현
    ax1.clear()
    
    # E 필드 (y 방향)
    x_e = np.zeros_like(z_range)
    y_e = Ey
    ax1.plot(x_e, y_e, z_range*1e2, 'b-', linewidth=3, label='E field (y)')
    
    # B 필드 (x 방향)
    x_b = Bx * (E0/B0) * 0.1  # 스케일 조정
    y_b = np.zeros_like(z_range)
    ax1.plot(x_b, y_b, z_range*1e2, 'r-', linewidth=3, label='B field (x)')
    
    # 축과 화살표
    ax1.quiver(0, 0, 0, 0, E0*1.5, 0, color='blue', arrow_length_ratio=0.1, linewidth=2)
    ax1.quiver(0, 0, 0, E0*0.15, 0, 0, color='red', arrow_length_ratio=0.1, linewidth=2)
    ax1.quiver(0, 0, 0, 0, 0, wavelength*1e2*0.3, color='green', arrow_length_ratio=0.1, linewidth=2)
    
    ax1.text(0, E0*1.7, 0, 'E', fontsize=14, fontweight='bold', color='blue')
    ax1.text(E0*0.2, 0, 0, 'B', fontsize=14, fontweight='bold', color='red')
    ax1.text(0, 0, wavelength*1e2*0.35, 'k', fontsize=12, fontweight='bold', color='green')
    
    ax1.set_xlabel('x', fontsize=11, fontweight='bold')
    ax1.set_ylabel('y (E field)', fontsize=11, fontweight='bold')
    ax1.set_zlabel('z (cm)', fontsize=11, fontweight='bold')
    ax1.set_title(f'EM Wave: E ⊥ B ⊥ k\nt = {t*1e9:.2f} ns', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10, loc='upper right')
    ax1.set_xlim([-E0*0.2, E0*0.2])
    ax1.set_ylim([-E0*1.5, E0*1.5])
    ax1.set_zlim([0, 2*wavelength*1e2])
    ax1.view_init(elev=20, azim=45)
    
    # 2. E 필드 (2D)
    ax2.clear()
    ax2.plot(z_range*1e2, Ey, 'b-', linewidth=3, label='Ey')
    ax2.axhline(0, color='k', linestyle='--', alpha=0.3)
    ax2.fill_between(z_range*1e2, 0, Ey, alpha=0.3, color='blue')
    
    ax2.set_xlabel('z (cm)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('E field (V/m)', fontsize=11, fontweight='bold')
    ax2.set_title(f'Electric Field', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([-E0*1.5, E0*1.5])
    
    # 3. B 필드 (2D)
    ax3.clear()
    ax3.plot(z_range*1e2, Bx*1e9, 'r-', linewidth=3, label='Bx')
    ax3.axhline(0, color='k', linestyle='--', alpha=0.3)
    ax3.fill_between(z_range*1e2, 0, Bx*1e9, alpha=0.3, color='red')
    
    ax3.set_xlabel('z (cm)', fontsize=11, fontweight='bold')
    ax3.set_ylabel('B field (nT)', fontsize=11, fontweight='bold')
    ax3.set_title(f'Magnetic Field', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim([-B0*1e9*1.5, B0*1e9*1.5])
    
    plt.suptitle(f'Electromagnetic Wave Propagation', fontsize=15, fontweight='bold')

print("\n애니메이션 생성 중...")
ani = animation.FuncAnimation(fig, update, frames=n_frames, interval=50)

# GIF 저장
output_gif = f'{output_dir}/09_em_wave.gif'
try:
    ani.save(output_gif, writer='pillow', fps=20)
    print(f"[OK] 애니메이션 저장: {output_gif}")
except Exception as e:
    print(f"[Warning] GIF 저장 실패 (ffmpeg/pillow 확인 필요): {e}")

# ============================================================================
# 요약 그래프 생성
# ============================================================================
print("\n요약 그래프 생성 중...")

fig_summary = plt.figure(figsize=(16, 10))

# 여러 시간대의 스냅샷
time_snapshots = [0, period/4, period/2, 3*period/4]
snapshot_labels = ['t = 0', 't = T/4', 't = T/2', 't = 3T/4']

for idx, (t_snap, label) in enumerate(zip(time_snapshots, snapshot_labels)):
    Ey_snap = E0 * np.sin(k * z_range - omega * t_snap)
    Bx_snap = B0 * np.sin(k * z_range - omega * t_snap)
    
    # E 필드
    ax_e = fig_summary.add_subplot(4, 2, 2*idx+1)
    ax_e.plot(z_range*1e2, Ey_snap, 'b-', linewidth=2.5)
    ax_e.axhline(0, color='k', linestyle='--', alpha=0.3)
    ax_e.fill_between(z_range*1e2, 0, Ey_snap, alpha=0.3, color='blue')
    ax_e.set_ylabel('Ey (V/m)', fontsize=10, fontweight='bold')
    ax_e.set_title(f'E field: {label}', fontsize=11, fontweight='bold')
    ax_e.grid(True, alpha=0.3)
    ax_e.set_ylim([-E0*1.2, E0*1.2])
    
    if idx == 3:
        ax_e.set_xlabel('z (cm)', fontsize=10, fontweight='bold')
    
    # B 필드
    ax_b = fig_summary.add_subplot(4, 2, 2*idx+2)
    ax_b.plot(z_range*1e2, Bx_snap*1e9, 'r-', linewidth=2.5)
    ax_b.axhline(0, color='k', linestyle='--', alpha=0.3)
    ax_b.fill_between(z_range*1e2, 0, Bx_snap*1e9, alpha=0.3, color='red')
    ax_b.set_ylabel('Bx (nT)', fontsize=10, fontweight='bold')
    ax_b.set_title(f'B field: {label}', fontsize=11, fontweight='bold')
    ax_b.grid(True, alpha=0.3)
    ax_b.set_ylim([-B0*1e9*1.2, B0*1e9*1.2])
    
    if idx == 3:
        ax_b.set_xlabel('z (cm)', fontsize=10, fontweight='bold')

plt.suptitle(f'EM Wave Evolution Over One Period (f = {freq/1e9:.1f} GHz)', 
             fontsize=15, fontweight='bold')
plt.tight_layout()

output_summary = f'{output_dir}/09_wave_summary.png'
plt.savefig(output_summary, dpi=150, bbox_inches='tight')
print(f"[OK] 요약 그래프 저장: {output_summary}")
# plt.close() # 요약 그래프는 닫지 않고 보여줄 수도 있지만, 애니메이션 창과 겹칠 수 있음.

print("\n" + "="*70)
print("분석 완료!")
print("="*70)
print(f"  - {output_summary}")
print(f"  - {output_gif}")

# 애니메이션 화면 표시
print("\n애니메이션 창을 닫으면 프로그램이 종료됩니다.")
plt.show()
