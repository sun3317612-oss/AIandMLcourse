"""
05. Lorentz Force - Charged Particle Motion
전자기장 내 하전 입자의 운동

물리 배경:
- 로렌츠 힘: F = q(E + v×B)
- 전기력: F_E = qE (가속)
- 자기력: F_B = qv×B (방향 변경, 일은 안 함)
- 사이클로트론 운동: 자기장에 수직인 원운동

학습 목표:
1. 로렌츠 힘의 이해
2. 사이클로트론 운동 관찰
3. E×B 표류(drift) 현상
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
print("05. Lorentz Force - Charged Particle in EM Fields")
print("="*70)

# 물리 상수
q = 1.6e-19  # 전하량 (C) - 전자
m = 9.11e-31  # 질량 (kg) - 전자

def rk4_step(f, y, t, dt):
    """Runge-Kutta 4차 방법"""
    k1 = f(y, t)
    k2 = f(y + 0.5*dt*k1, t + 0.5*dt)
    k3 = f(y + 0.5*dt*k2, t + 0.5*dt)
    k4 = f(y + dt*k3, t + dt)
    return y + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)

def lorentz_force(y, t, q, m, E_func, B_func):
    """
    로렌츠 힘 운동 방정식
    
    y = [x, y, z, vx, vy, vz]
    
    F = q(E + v×B)
    a = F/m
    """
    x, y_pos, z = y[0], y[1], y[2]
    vx, vy, vz = y[3], y[4], y[5]
    
    # 전기장과 자기장
    E = E_func(x, y_pos, z, t)
    B = B_func(x, y_pos, z, t)
    
    # 로렌츠 힘
    # F = q*E + q*(v × B)
    # v × B = (vy*Bz - vz*By, vz*Bx - vx*Bz, vx*By - vy*Bx)
    
    v_cross_B = np.array([
        vy * B[2] - vz * B[1],
        vz * B[0] - vx * B[2],
        vx * B[1] - vy * B[0]
    ])
    
    F = q * E + q * v_cross_B
    a = F / m
    
    return np.array([vx, vy, vz, a[0], a[1], a[2]])

def simulate_particle(y0, t_max, dt, q, m, E_func, B_func):
    """입자 운동 시뮬레이션"""
    n_steps = int(t_max / dt)
    trajectory = np.zeros((n_steps, 6))
    t_array = np.zeros(n_steps)
    
    y = y0.copy()
    t = 0
    
    for i in range(n_steps):
        trajectory[i] = y
        t_array[i] = t
        y = rk4_step(lambda y_arg, t_arg: lorentz_force(y_arg, t_arg, q, m, E_func, B_func), 
                     y, t, dt)
        t += dt
    
    return t_array, trajectory

# 시나리오 1: 순수 자기장 (사이클로트론)
print("\n시나리오 1: 사이클로트론 운동")
print("-" * 70)

B0 = 1e-3  # 1 mT (지구 자기장 정도)

def E_zero(x, y, z, t):
    return np.array([0.0, 0.0, 0.0])

def B_uniform_z(x, y, z, t):
    return np.array([0.0, 0.0, B0])

# 초기 조건
v0 = 1e6  # 1,000 km/s
y0_cyclotron = np.array([0.0, 0.0, 0.0, v0, 0.0, 0.0])

# 사이클로트론 주파수와 반지름
omega_c = abs(q) * B0 / m
T_c = 2 * np.pi / omega_c
r_c = m * v0 / (abs(q) * B0)

print(f"자기장: B = {B0*1e3:.2f} mT")
print(f"초기 속도: v = {v0/1e6:.1f} × 10^6 m/s")
print(f"사이클로트론 주파수: f = {omega_c/(2*np.pi)*1e-6:.2f} MHz")
print(f"사이클로트론 주기: T = {T_c*1e9:.2f} ns")
print(f"사이클로트론 반지름: r = {r_c*1e3:.4f} mm")

t_max_1 = 3 * T_c
dt_1 = T_c / 200

t1, traj1 = simulate_particle(y0_cyclotron, t_max_1, dt_1, q, m, E_zero, B_uniform_z)

# 시나리오 2: E×B 표류
print("\n시나리오 2: E×B Drift")
print("-" * 70)

E0 = 1e3  # 1 kV/m

def E_uniform_x(x, y, z, t):
    return np.array([E0, 0.0, 0.0])

# 표류 속도
v_drift = E0 / B0

print(f"전기장: E = {E0/1e3:.1f} kV/m (x 방향)")
print(f"자기장: B = {B0*1e3:.2f} mT (z 방향)")
print(f"표류 속도: v_drift = E/B = {v_drift/1e3:.1f} km/s (y 방향)")

y0_drift = np.array([0.0, 0.0, 0.0, v0, 0.0, 0.0])
t_max_2 = 5 * T_c
dt_2 = T_c / 200

t2, traj2 = simulate_particle(y0_drift, t_max_2, dt_2, q, m, E_uniform_x, B_uniform_z)

# 시나리오 3: 나선 운동 (pitch angle)
print("\n시나리오 3: 나선 운동")
print("-" * 70)

v_perp = 0.8 * v0  # 수직 성분
v_para = 0.6 * v0  # 평행 성분

y0_helix = np.array([0.0, 0.0, 0.0, v_perp, 0.0, v_para])

pitch = v_para / v_perp * r_c

print(f"수직 속도: v⊥ = {v_perp/1e6:.1f} × 10^6 m/s")
print(f"평행 속도: v∥ = {v_para/1e6:.1f} × 10^6 m/s")
print(f"나선 피치: {pitch*1e3:.4f} mm")

t_max_3 = 3 * T_c
dt_3 = T_c / 200

t3, traj3 = simulate_particle(y0_helix, t_max_3, dt_3, q, m, E_zero, B_uniform_z)

# 시각화
fig = plt.figure(figsize=(16, 10))

# 1. 사이클로트론 (x-y 평면)
ax1 = fig.add_subplot(2, 3, 1)
ax1.plot(traj1[:, 0]*1e3, traj1[:, 1]*1e3, 'b-', linewidth=2)
ax1.plot(traj1[0, 0]*1e3, traj1[0, 1]*1e3, 'go', markersize=10, label='Start')
ax1.plot(traj1[-1, 0]*1e3, traj1[-1, 1]*1e3, 'ro', markersize=10, label='End')
ax1.set_xlabel('x (mm)', fontsize=11, fontweight='bold')
ax1.set_ylabel('y (mm)', fontsize=11, fontweight='bold')
ax1.set_title('Cyclotron Motion (x-y plane)', fontsize=12, fontweight='bold')
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)
ax1.set_aspect('equal')

# 2. 사이클로트론 속도
ax2 = fig.add_subplot(2, 3, 4)
v_mag_1 = np.sqrt(traj1[:, 3]**2 + traj1[:, 4]**2 + traj1[:, 5]**2)
ax2.plot(t1*1e9, v_mag_1/1e6, 'b-', linewidth=2)
ax2.axhline(v0/1e6, color='r', linestyle='--', label='Initial speed')
ax2.set_xlabel('Time (ns)', fontsize=11, fontweight='bold')
ax2.set_ylabel('Speed (10^6 m/s)', fontsize=11, fontweight='bold')
ax2.set_title('Speed Conservation (B does no work)', fontsize=12, fontweight='bold')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

# 3. E×B 표류
ax3 = fig.add_subplot(2, 3, 2)
ax3.plot(traj2[:, 0]*1e3, traj2[:, 1]*1e3, 'r-', linewidth=2)
ax3.plot(traj2[0, 0]*1e3, traj2[0, 1]*1e3, 'go', markersize=10)
ax3.plot(traj2[-1, 0]*1e3, traj2[-1, 1]*1e3, 'ro', markersize=10)
ax3.arrow(0, 0, 0, 0.5, head_width=0.05, head_length=0.1, fc='blue', ec='blue', alpha=0.5)
ax3.text(0.05, 0.3, 'Drift', fontsize=10, color='blue', fontweight='bold')
ax3.set_xlabel('x (mm)', fontsize=11, fontweight='bold')
ax3.set_ylabel('y (mm)', fontsize=11, fontweight='bold')
ax3.set_title('E×B Drift Motion', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3)
ax3.set_aspect('equal')

# 4. 표류 분석
ax4 = fig.add_subplot(2, 3, 5)
ax4.plot(t2*1e9, traj2[:, 1]*1e3, 'r-', linewidth=2, label='y position')
# 이론적 표류
y_drift_theory = v_drift * t2
ax4.plot(t2*1e9, y_drift_theory*1e3, 'k--', linewidth=2, label='Theory (v_drift*t)')
ax4.set_xlabel('Time (ns)', fontsize=11, fontweight='bold')
ax4.set_ylabel('y position (mm)', fontsize=11, fontweight='bold')
ax4.set_title('Drift Velocity Verification', fontsize=12, fontweight='bold')
ax4.legend(fontsize=9)
ax4.grid(True, alpha=0.3)

# 5. 나선 운동 (3D)
ax5 = fig.add_subplot(2, 3, 3, projection='3d')
ax5.plot(traj3[:, 0]*1e3, traj3[:, 1]*1e3, traj3[:, 2]*1e3, 'g-', linewidth=2)
ax5.plot([traj3[0, 0]*1e3], [traj3[0, 1]*1e3], [traj3[0, 2]*1e3], 
         'go', markersize=10, label='Start')
ax5.set_xlabel('x (mm)', fontsize=10, fontweight='bold')
ax5.set_ylabel('y (mm)', fontsize=10, fontweight='bold')
ax5.set_zlabel('z (mm)', fontsize=10, fontweight='bold')
ax5.set_title('Helical Motion', fontsize=12, fontweight='bold')
ax5.legend(fontsize=9)

# 6. 나선 z vs t
ax6 = fig.add_subplot(2, 3, 6)
ax6.plot(t3*1e9, traj3[:, 2]*1e3, 'g-', linewidth=2, label='z position')
# 이론적 직선
z_theory = v_para * t3
ax6.plot(t3*1e9, z_theory*1e3, 'k--', linewidth=2, label='Theory (v_para*t)')
ax6.set_xlabel('Time (ns)', fontsize=11, fontweight='bold')
ax6.set_ylabel('z position (mm)', fontsize=11, fontweight='bold')
ax6.set_title('Parallel Motion', fontsize=12, fontweight='bold')
ax6.legend(fontsize=9)
ax6.grid(True, alpha=0.3)

plt.suptitle('Lorentz Force: Charged Particle Dynamics', fontsize=15, fontweight='bold')
plt.tight_layout()

output_path = f'{output_dir}/05_particle_trajectory.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\n[OK] 그래프 저장: {output_path}")
plt.close()

print("\n" + "="*70)
print("분석 완료!")
print("="*70)
print("\n핵심 개념:")
print("  1. 로렌츠 힘: F = q(E + v×B)")
print("  2. 전기력은 가속, 자기력은 방향만 바꿈")
print("  3. 사이클로트론: 원운동 (속도 일정)")
print("  4. E×B 표류: 전기장과 자기장이 교차하면 표류")
print("  5. 나선 운동: 자기장에 평행 + 수직 성분")
print("\n응용:")
print("  - 사이클로트론 가속기")
print("  - 질량 분석기")
print("  - 자기권의 하전 입자 (Van Allen Belt)")
print("  - 핵융합 플라즈마 제어")
print(f"\n생성된 파일: {output_path}")

