"""
HW 02. Foucault Pendulum — 지구 자전의 가시화
Foucault Pendulum Simulation with Coriolis Force

레옹 푸코(1851)가 파리 팡테옹에서 진자를 이용해 지구 자전을 직접 증명했습니다.
진자 운동 평면이 천천히 회전하는 것은 코리올리 힘(관성력) 때문입니다.

회전 좌표계 운동 방정식 (수평면, 소각도 근사):
  ẍ = -ω₀²x + 2Ωsin(λ)·ẏ
  ÿ = -ω₀²y - 2Ωsin(λ)·ẋ

where:
  ω₀  = √(g/L)       진자 고유 진동수 (rad/s)
  Ω   = 7.292×10⁻⁵   지구 자전 각속도 (rad/s)
  λ   = 위도 (latitude)
  2Ωsin(λ) = Ω_eff   유효 코리올리 매개변수
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os

output_dir = 'outputs'
os.makedirs(output_dir, exist_ok=True)

print("=" * 70)
print("Foucault Pendulum — Coriolis Force and Earth's Rotation")
print("=" * 70)

# ============================================================================
# 물리 상수 및 파라미터
# ============================================================================

g   = 9.81              # 중력 가속도 (m/s²)
L   = 67.0              # 진자 길이 (m) — 팡테옹 원본과 동일
Omega_E = 7.292e-5      # 지구 자전 각속도 (rad/s)

omega0 = np.sqrt(g / L)  # 진자 고유 진동수 (rad/s)
T0     = 2 * np.pi / omega0  # 진자 주기 (s)

print(f"\n진자 파라미터:")
print(f"  길이 L   = {L} m  (파리 팡테옹 원본)")
print(f"  고유 진동수 ω₀ = {omega0:.4f} rad/s")
print(f"  진자 주기 T₀  = {T0:.2f} s ({T0/60:.2f} 분)")
print(f"  지구 자전 Ω   = {Omega_E:.4e} rad/s")

# ============================================================================
# 운동 방정식 (상태 벡터: [x, vx, y, vy])
# ============================================================================

def foucault_derivs(state, t, latitude_deg):
    """
    푸코 진자 운동 방정식 (회전 좌표계)

    state = [x, vx, y, vy]

    코리올리 가속도: a_Cor = -2(Ω × v)
    수직 방향 Ω_eff = Ω sin(λ)
    """
    x, vx, y, vy = state
    lam  = np.radians(latitude_deg)
    Oc   = Omega_E * np.sin(lam)  # 유효 코리올리 매개변수

    ax = -omega0**2 * x + 2 * Oc * vy
    ay = -omega0**2 * y - 2 * Oc * vx

    return np.array([vx, ax, vy, ay])


def rk4_step(f, y, t, dt, **kwargs):
    """4차 Runge-Kutta"""
    k1 = f(y, t, **kwargs)
    k2 = f(y + 0.5*dt*k1, t + 0.5*dt, **kwargs)
    k3 = f(y + 0.5*dt*k2, t + 0.5*dt, **kwargs)
    k4 = f(y + dt*k3,     t + dt,     **kwargs)
    return y + (dt/6.0) * (k1 + 2*k2 + 2*k3 + k4)


def simulate_foucault(latitude_deg, x0, t_max, dt):
    """
    푸코 진자 시뮬레이션

    Parameters
    ----------
    latitude_deg : float  위도 (도)
    x0           : float  초기 진폭 x(0) (m)
    t_max        : float  시뮬레이션 시간 (s)
    dt           : float  시간 간격 (s)
    """
    n = int(t_max / dt)
    states = np.zeros((n, 4))
    t_arr  = np.zeros(n)

    state = np.array([x0, 0.0, 0.0, 0.0])
    t = 0.0
    for i in range(n):
        t_arr[i]  = t
        states[i] = state
        state = rk4_step(foucault_derivs, state, t, dt, latitude_deg=latitude_deg)
        t += dt

    return t_arr, states[:, 0], states[:, 2]  # t, x, y

# ============================================================================
# 시뮬레이션 — 여러 위도
# ============================================================================

x0_amp = 1.0    # 초기 진폭 1 m
dt     = 0.5    # 시간 간격 (s)

latitudes = {
    '북극 (90°N)':     90.0,
    '부산 (35.1°N)':   35.1,
    '파리 (48.9°N)':   48.9,
    '적도 (0°)':        0.1,   # 0이면 코리올리 = 0이라 수치 문제 방지
}

# 각 위도의 세차 주기 (이론값): T_precession = 24h / sin(λ)
print("\n위도별 세차 주기 (이론값):")
for label, lat in latitudes.items():
    lam = np.radians(lat)
    if np.sin(lam) > 1e-6:
        T_prec_h = 24.0 / np.sin(lam)
        print(f"  {label}: {T_prec_h:.1f} 시간")
    else:
        print(f"  {label}: ∞ (세차 없음)")

# 세차 주기가 가장 짧은 위도(북극)의 1/4 세차 주기 시뮬레이션
t_max_sim = 6 * 3600  # 6시간

print(f"\n시뮬레이션 시간: {t_max_sim/3600:.0f} 시간")
print("시뮬레이션 실행 중...")

results = {}
for label, lat in latitudes.items():
    t, xr, yr = simulate_foucault(lat, x0_amp, t_max_sim, dt)
    results[label] = (t, xr, yr)
    print(f"  [{label}] 완료: {len(t)} 스텝")

# ============================================================================
# 시각화
# ============================================================================

# 색상 팔레트
colors = ['royalblue', 'tomato', 'forestgreen', 'darkorange']
labels = list(latitudes.keys())

# --- 그림 1: 진자 흔적 (2D 평면) ---
fig1, axes1 = plt.subplots(2, 2, figsize=(14, 12))
axes1 = axes1.flatten()

for idx, (label, (t, xr, yr)) in enumerate(results.items()):
    ax = axes1[idx]
    # 시간에 따른 색상 그라데이션
    n = len(t)
    for j in range(0, n-1, max(1, n//1000)):
        frac = j / n
        c = plt.cm.viridis(frac)
        ax.plot(xr[j:j+2], yr[j:j+2], color=c, linewidth=0.6, alpha=0.8)
    ax.plot(xr[0], yr[0], 'go', markersize=8, label='Start', zorder=5)
    ax.set_xlabel('x (m)', fontsize=11, fontweight='bold')
    ax.set_ylabel('y (m)', fontsize=11, fontweight='bold')
    ax.set_title(f'{label}', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

plt.suptitle(f'Foucault Pendulum Trajectory (t = {t_max_sim/3600:.0f} hr)\n'
             f'L = {L} m, Amplitude = {x0_amp} m',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{output_dir}/02_foucault_trajectory.png', dpi=150, bbox_inches='tight')
print(f"\n[OK] 저장: {output_dir}/02_foucault_trajectory.png")
plt.close()

# --- 그림 2: 세차 각도 vs 시간 + x(t) ---
fig2 = plt.figure(figsize=(16, 10))
gs2 = GridSpec(2, 2, figure=fig2, hspace=0.35, wspace=0.3)

# 세차 각도 분석 (부산 위도 기준)
label_busan = '부산 (35.1°N)'
t_b, xb, yb = results[label_busan]

# 진자 끝이 x축과 이루는 각도 → 세차 각도
prec_angle = np.degrees(np.arctan2(yb, xb))

ax21 = fig2.add_subplot(gs2[0, :])
ax21.plot(t_b / 3600, prec_angle, 'tomato', linewidth=1.0, alpha=0.8)
ax21.set_xlabel('Time (hr)', fontweight='bold')
ax21.set_ylabel('Precession Angle (degrees)', fontweight='bold')
ax21.set_title(f'Precession Angle vs Time — {label_busan}', fontsize=13, fontweight='bold')
ax21.grid(True, alpha=0.3)

# x(t) 비교 (초반 3주기)
n_show = int(3 * T0 / dt)
ax22 = fig2.add_subplot(gs2[1, 0])
for idx, (label, (t, xr, yr)) in enumerate(results.items()):
    ax22.plot(t[:n_show], xr[:n_show], color=colors[idx],
              linewidth=1.2, label=label, alpha=0.8)
ax22.set_xlabel('Time (s)', fontweight='bold')
ax22.set_ylabel('x(t) (m)', fontweight='bold')
ax22.set_title('x(t): First 3 Periods', fontsize=12, fontweight='bold')
ax22.legend(fontsize=8)
ax22.grid(True, alpha=0.3)

# 요약 텍스트
ax23 = fig2.add_subplot(gs2[1, 1])
ax23.axis('off')

lat_b = latitudes[label_busan]
lam_b = np.radians(lat_b)
T_prec_b = 24.0 / np.sin(lam_b)
deg_per_hr_b = 360.0 / T_prec_b

summary = f"""
FOUCAULT PENDULUM SUMMARY
{'='*40}

Pendulum:
  Length L = {L} m
  Period T₀ = {T0:.1f} s ({T0/60:.2f} min)
  Amplitude = {x0_amp} m

Coriolis Parameter:
  Ω_E = {Omega_E:.3e} rad/s

Precession Period (theory):
  90°N (북극): 24.0 hr
  {lat_b:.1f}°N (부산): {T_prec_b:.1f} hr
  48.9°N (파리): {24/np.sin(np.radians(48.9)):.1f} hr
  0° (적도): ∞ hr (세차 없음)

Busan (35.1°N) Result:
  Precession rate ≈ {deg_per_hr_b:.2f} deg/hr
  → 1도 세차하는 데 {60/deg_per_hr_b:.1f} 분 소요

Physical Interpretation:
  진자는 관성 좌표계에서는
  고정된 면에서 진동.
  지구가 그 아래에서 자전 →
  관찰자에게 진자 면이 돌아
  보임 (코리올리 힘).
"""
ax23.text(0.02, 0.5, summary, fontsize=9.5, family='monospace',
          verticalalignment='center', transform=ax23.transAxes)

plt.suptitle('Foucault Pendulum Analysis', fontsize=15, fontweight='bold')
plt.savefig(f'{output_dir}/02_foucault_analysis.png', dpi=150, bbox_inches='tight')
print(f"[OK] 저장: {output_dir}/02_foucault_analysis.png")
plt.close()

print("\n" + "=" * 70)
print("완료!")
print("=" * 70)
print("\n생성된 파일:")
print(f"  1. {output_dir}/02_foucault_trajectory.png — 4개 위도의 진자 흔적")
print(f"  2. {output_dir}/02_foucault_analysis.png   — 세차 각도 분석")
print("\n물리적 결론:")
for label, lat in latitudes.items():
    lam = np.radians(lat)
    if np.sin(lam) > 1e-3:
        T_p = 24.0 / np.sin(lam)
        print(f"  · {label}: 세차 주기 ≈ {T_p:.1f} hr")
    else:
        print(f"  · {label}: 코리올리 효과 거의 없음 (세차 없음)")
