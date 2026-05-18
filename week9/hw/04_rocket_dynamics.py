"""
HW 04. Rocket Dynamics — 치올코프스키 방정식과 다단 로켓
Variable-Mass Rocket with Tsiolkovsky Equation

콘스탄틴 치올코프스키(1903)가 유도한 로켓 방정식은
우주 공학의 근본 방정식입니다.

가변 질량 뉴턴 법칙 (Rocket Equation):
  m(t) · dv/dt = v_e · |dm/dt| - m(t) · g - ½ρv²C_D·A

적분하면 치올코프스키 방정식 (무중력, 항력 무시):
  Δv = v_e · ln(m₀ / m_f)

  Δv: 최종 속도 변화량 (delta-v)
  v_e: 배기 속도 (exhaust velocity)
  m₀: 초기 질량 (연료 포함)
  m_f: 최종 질량 (연료 소진)

시뮬레이션에서는 중력 + 항력을 포함한 완전한 방정식을 RK4로 풉니다.
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
print("Rocket Dynamics — Tsiolkovsky Equation and Multi-Stage Rockets")
print("=" * 70)

# ============================================================================
# 물리 상수
# ============================================================================

g0    = 9.81        # 지표면 중력 가속도 (m/s²)
R_E   = 6.371e6     # 지구 반지름 (m)
rho0  = 1.225       # 해수면 공기 밀도 (kg/m³)
H_atm = 8500.0      # 대기 규모 고도 (m)

# ============================================================================
# 로켓 파라미터 (단단 로켓 기준값)
# ============================================================================

m_dry  = 5000.0     # 로켓 빈 질량 (kg) — 구조체 + 페이로드
m_fuel = 45000.0    # 연료 질량 (kg)
m0     = m_dry + m_fuel  # 초기 전체 질량 (kg)

v_e      = 3500.0   # 배기 속도 (m/s) — 케로신/LOX 기준
thrust_F = 6.0e5    # 추력 (N) = 600 kN
mdot     = thrust_F / v_e  # 질량 유량 (kg/s)
t_burn   = m_fuel / mdot   # 연소 시간 (s)

C_D    = 0.3        # 항력 계수
A_ref  = 3.14       # 기준 단면적 (m²)

print(f"\n로켓 파라미터:")
print(f"  초기 질량 m₀   = {m0/1000:.1f} t  (연료 {m_fuel/1000:.1f} t + 구조 {m_dry/1000:.1f} t)")
print(f"  배기 속도 v_e  = {v_e} m/s")
print(f"  추력 F         = {thrust_F/1000:.0f} kN")
print(f"  질량 유량 ṁ    = {mdot:.2f} kg/s")
print(f"  연소 시간 t_b  = {t_burn:.1f} s")

# 치올코프스키 이상적 Δv
delta_v_ideal = v_e * np.log(m0 / m_dry)
print(f"\n치올코프스키 Δv (이상, 무중력·무항력):")
print(f"  Δv = v_e · ln(m₀/m_f) = {v_e} · ln({m0/m_dry:.2f}) = {delta_v_ideal:.1f} m/s")

# ============================================================================
# 운동 방정식
# ============================================================================

def gravity(h):
    """고도에 따른 중력 가속도"""
    return g0 * (R_E / (R_E + h))**2


def air_density(h):
    """지수 대기 모델"""
    if h < 0:
        return rho0
    return rho0 * np.exp(-h / H_atm)


def rocket_derivs(state, t, is_burning):
    """
    로켓 운동 방정식 (1D 수직 발사)
    state = [h, v, m]
      h: 고도 (m)
      v: 속도 (m/s), 위 방향 양수
      m: 현재 질량 (kg)
    """
    h, v, m = state

    g   = gravity(h)
    rho = air_density(max(h, 0))

    # 항력 (항상 속도 반대 방향)
    F_drag = 0.5 * rho * v * abs(v) * C_D * A_ref

    if is_burning and m > m_dry:
        # 연소 중: 추력 - 중력 - 항력
        dv = (thrust_F - m * g - F_drag) / m
        dm = -mdot
    else:
        # 연소 종료: 중력 + 항력만
        dv = (-m * g - F_drag) / m
        dm = 0.0

    return np.array([v, dv, dm])


def rk4_step(f, y, t, dt, **kwargs):
    k1 = f(y, t, **kwargs)
    k2 = f(y + 0.5*dt*k1, t + 0.5*dt, **kwargs)
    k3 = f(y + 0.5*dt*k2, t + 0.5*dt, **kwargs)
    k4 = f(y + dt*k3,     t + dt,     **kwargs)
    return y + (dt/6.0) * (k1 + 2*k2 + 2*k3 + k4)


def simulate_rocket(m0, m_dry, v_e, thrust, mdot, t_max, dt=0.5):
    """단단 로켓 시뮬레이션"""
    n = int(t_max / dt)
    result = np.zeros((n, 4))  # [t, h, v, m]
    state = np.array([0.0, 0.0, m0])
    t = 0.0

    for i in range(n):
        result[i] = [t, state[0], state[1], state[2]]

        burning = (state[2] > m_dry) and (t < m_fuel / mdot + 1)
        state_new = rk4_step(rocket_derivs, state, t, dt, is_burning=burning)

        # 질량 하한 및 지면 충돌 처리
        state_new[2] = max(state_new[2], m_dry)
        if state_new[0] < 0 and state_new[1] < 0:
            state_new[0] = 0.0
            state_new[1] = 0.0

        state = state_new
        t += dt

    return result

# ============================================================================
# 시뮬레이션 1: 단단 로켓
# ============================================================================

print("\n[1] 단단 로켓 시뮬레이션...")
t_max_1 = 600.0  # 10분
dt      = 0.5

traj1 = simulate_rocket(m0, m_dry, v_e, thrust_F, mdot, t_max_1, dt)
t1, h1, v1, m1 = traj1[:, 0], traj1[:, 1], traj1[:, 2], traj1[:, 3]

h_max1 = h1.max()
v_max1 = v1.max()
print(f"    최대 고도: {h_max1/1000:.1f} km")
print(f"    최대 속도: {v_max1:.1f} m/s  (마하 {v_max1/343:.2f})")
print(f"    실제 Δv 손실: 이론값 {delta_v_ideal:.0f} m/s → 실제 {v_max1:.0f} m/s")

# ============================================================================
# 시뮬레이션 2: 배기 속도 비교 (연료 종류)
# ============================================================================

print("\n[2] 배기 속도 비교 (연료 종류)...")

fuels = {
    '고체연료 (v_e=2500)':    2500,
    '케로신/LOX (v_e=3500)':  3500,
    '수소/LOX (v_e=4500)':    4500,
}

fuel_trajs = {}
for label, ve in fuels.items():
    mdot_f = thrust_F / ve
    traj = simulate_rocket(m0, m_dry, ve, thrust_F, mdot_f, t_max_1, dt)
    fuel_trajs[label] = traj
    h_max = traj[:, 1].max()
    dv_ideal = ve * np.log(m0 / m_dry)
    print(f"    {label}: Δv_ideal={dv_ideal:.0f} m/s, h_max={h_max/1000:.1f} km")

# ============================================================================
# 시뮬레이션 3: 다단 로켓 — Staging의 이점
# ============================================================================

print("\n[3] 다단 로켓 vs 단단 로켓 Δv 비교 (치올코프스키 해석)...")

# ----------------------------------------------------------------
# 공정한 비교: 총 발사 질량 50t, 최종 페이로드 5t, 연료 40t
# 구조체: 1단 3t + 2단 2t = 5t (총 동일)
# ----------------------------------------------------------------
m_struct2, m_fuel2, m_pay2 = 2000, 10000, 5000
m_struct1, m_fuel1         = 3000, 30000
# 총 발사 질량 = 3+30+2+10+5 = 50t (동일)

# Case A: 단단 (staging 없음) — 1단 구조체도 끝까지 메고 감
# m0 = 50t, m_f = 3(struct1) + 2(struct2) + 5(payload) = 10t
dv_no_stage = v_e * np.log((m_struct1 + m_fuel1 + m_struct2 + m_fuel2 + m_pay2) /
                             (m_struct1 + m_struct2 + m_pay2))

# Case B: 2단 (staging 있음) — 1단 연소 후 struct1(3t) 분리
m_payload1 = m_struct2 + m_fuel2 + m_pay2  # 2단 전체 = 17t
dv_stage1 = v_e * np.log((m_struct1 + m_fuel1 + m_payload1) / (m_struct1 + m_payload1))
dv_stage2 = v_e * np.log((m_struct2 + m_fuel2 + m_pay2) / (m_struct2 + m_pay2))
dv_two_stage = dv_stage1 + dv_stage2

print(f"\n    [동일 조건: m_launch=50t, payload=5t, 연료=40t, 구조=5t]")
print(f"    Case A (단단, no staging): Δv = {dv_no_stage:.0f} m/s")
print(f"      → 1단 구조체(3t)를 끝까지 탑재, m0/m_f = 50/10 = 5")
print(f"    Case B (2단, staging):     Δv = {dv_stage1:.0f} + {dv_stage2:.0f} = {dv_two_stage:.0f} m/s")
print(f"      → 1단 연소 후 3t 분리, m0/m_f per stage 개선")
print(f"    staging 이점: +{dv_two_stage - dv_no_stage:.0f} m/s ({(dv_two_stage/dv_no_stage-1)*100:.1f}% 향상)")
print(f"\n    참고: 저궤도(LEO) Δv ≈ 9,400 m/s, 달까지 ≈ 12,000 m/s")

# ============================================================================
# 시각화
# ============================================================================

colors_fuel = ['coral', 'steelblue', 'seagreen']

# --- 그림 1: 단단 로켓 전체 분석 ---
fig1 = plt.figure(figsize=(16, 12))
gs1 = GridSpec(2, 3, figure=fig1, hspace=0.35, wspace=0.35)

# 고도 vs 시간
ax11 = fig1.add_subplot(gs1[0, :2])
ax11.plot(t1, h1/1000, 'royalblue', linewidth=2)
ax11.axvline(t_burn, color='r', linestyle='--', linewidth=1.5, label=f'Burnout t={t_burn:.0f}s')
ax11.axhline(h_max1/1000, color='g', linestyle=':', linewidth=1.5, label=f'h_max={h_max1/1000:.1f} km')
ax11.set_xlabel('Time (s)', fontweight='bold')
ax11.set_ylabel('Altitude (km)', fontweight='bold')
ax11.set_title('Altitude vs Time', fontsize=12, fontweight='bold')
ax11.legend(fontsize=10)
ax11.grid(True, alpha=0.3)

# 속도 vs 시간
ax12 = fig1.add_subplot(gs1[0, 2])
ax12.plot(t1, v1, 'tomato', linewidth=2)
ax12.axvline(t_burn, color='r', linestyle='--', linewidth=1.5, label=f'Burnout')
ax12.axhline(v_max1, color='g', linestyle=':', linewidth=1.5, label=f'v_max={v_max1:.0f} m/s')
ax12.set_xlabel('Time (s)', fontweight='bold')
ax12.set_ylabel('Velocity (m/s)', fontweight='bold')
ax12.set_title('Velocity vs Time', fontsize=12, fontweight='bold')
ax12.legend(fontsize=10)
ax12.grid(True, alpha=0.3)

# 질량 vs 시간
ax13 = fig1.add_subplot(gs1[1, 0])
ax13.plot(t1, m1/1000, 'darkorange', linewidth=2)
ax13.axhline(m_dry/1000, color='k', linestyle='--', linewidth=1.5, label=f'm_dry={m_dry/1000:.0f} t')
ax13.set_xlabel('Time (s)', fontweight='bold')
ax13.set_ylabel('Mass (t)', fontweight='bold')
ax13.set_title('Mass vs Time', fontsize=12, fontweight='bold')
ax13.legend(fontsize=10)
ax13.grid(True, alpha=0.3)

# 추력-중력비 (T/W ratio)
g_arr  = np.array([gravity(max(hh, 0)) for hh in h1])
rho_arr = np.array([air_density(max(hh, 0)) for hh in h1])
drag_arr = 0.5 * rho_arr * v1 * np.abs(v1) * C_D * A_ref
burn_mask = m1 > m_dry
TW = np.where(burn_mask, thrust_F / (m1 * g_arr), 0)

ax14 = fig1.add_subplot(gs1[1, 1])
ax14.plot(t1[burn_mask], TW[burn_mask], 'purple', linewidth=2)
ax14.axhline(1.0, color='r', linestyle='--', linewidth=1.5, label='T/W = 1 (liftoff threshold)')
ax14.set_xlabel('Time (s)', fontweight='bold')
ax14.set_ylabel('Thrust-to-Weight Ratio', fontweight='bold')
ax14.set_title('T/W Ratio During Burn', fontsize=12, fontweight='bold')
ax14.legend(fontsize=10)
ax14.grid(True, alpha=0.3)

# 요약
ax15 = fig1.add_subplot(gs1[1, 2])
ax15.axis('off')
summary1 = f"""
SINGLE-STAGE ROCKET
{'='*36}

Parameters:
  m₀ = {m0/1000:.0f} t  (total)
  m_fuel = {m_fuel/1000:.0f} t
  m_dry  = {m_dry/1000:.0f} t
  v_e = {v_e} m/s
  F   = {thrust_F/1000:.0f} kN
  ṁ   = {mdot:.1f} kg/s
  t_burn = {t_burn:.1f} s

Tsiolkovsky (ideal):
  Δv = v_e ln(m₀/m_f)
     = {v_e} × ln({m0/m_dry:.2f})
     = {delta_v_ideal:.0f} m/s

Simulation (with gravity+drag):
  v_max  = {v_max1:.0f} m/s
  h_max  = {h_max1/1000:.1f} km
  Loss   = {delta_v_ideal-v_max1:.0f} m/s
           (gravity + drag loss)
"""
ax15.text(0.03, 0.5, summary1, fontsize=9.5, family='monospace',
          verticalalignment='center', transform=ax15.transAxes)

plt.suptitle('Single-Stage Rocket: Full Trajectory Analysis', fontsize=14, fontweight='bold')
plt.savefig(f'{output_dir}/04_rocket_single_stage.png', dpi=150, bbox_inches='tight')
print(f"\n[OK] 저장: {output_dir}/04_rocket_single_stage.png")
plt.close()

# --- 그림 2: 배기 속도 비교 + 다단 Δv ---
fig2 = plt.figure(figsize=(16, 10))
gs2 = GridSpec(2, 2, figure=fig2, hspace=0.35, wspace=0.3)

# 배기 속도별 고도 비교
ax21 = fig2.add_subplot(gs2[0, :])
for (label, ve), color in zip(fuels.items(), colors_fuel):
    traj = fuel_trajs[label]
    ax21.plot(traj[:, 0], traj[:, 1]/1000, color=color, linewidth=2, label=label)
ax21.set_xlabel('Time (s)', fontweight='bold')
ax21.set_ylabel('Altitude (km)', fontweight='bold')
ax21.set_title('Altitude vs Time: Effect of Exhaust Velocity', fontsize=13, fontweight='bold')
ax21.legend(fontsize=10)
ax21.grid(True, alpha=0.3)

# 치올코프스키 곡선 (Δv vs 질량비)
ax22 = fig2.add_subplot(gs2[1, 0])
R_vals = np.linspace(1.05, 20, 200)
for ve_label, ve_val, color in zip(['2500', '3500', '4500'],
                                    [2500, 3500, 4500], colors_fuel):
    dv_vals = ve_val * np.log(R_vals)
    ax22.plot(R_vals, dv_vals/1000, color=color, linewidth=2,
              label=f'v_e = {ve_label} m/s')
ax22.axhline(9.4, color='k', linestyle='--', linewidth=1.5, label='LEO Δv ≈ 9.4 km/s')
ax22.axhline(11.2, color='gray', linestyle=':', linewidth=1.5, label='Moon Δv ≈ 11.2 km/s')
ax22.axvline(m0/m_dry, color='purple', linestyle='--', linewidth=1.5,
             label=f'This rocket (R={m0/m_dry:.1f})')
ax22.set_xlabel('Mass Ratio m₀/m_f', fontweight='bold')
ax22.set_ylabel('Δv (km/s)', fontweight='bold')
ax22.set_title('Tsiolkovsky Rocket Equation', fontsize=12, fontweight='bold')
ax22.legend(fontsize=9)
ax22.grid(True, alpha=0.3)
ax22.set_xlim(1, 20)

# 단단(no staging) vs 2단(staging) Δv 막대 그래프
ax23 = fig2.add_subplot(gs2[1, 1])
stages_labels = ['No Staging\n(m0/mf=5)', '2-Stage\nStage1', '2-Stage\nStage2', '2-Stage\nTotal']
stages_vals   = [dv_no_stage/1000, dv_stage1/1000, dv_stage2/1000, dv_two_stage/1000]
bar_colors    = ['steelblue', 'tomato', 'tomato', 'forestgreen']
bars = ax23.bar(stages_labels, stages_vals, color=bar_colors, alpha=0.8, edgecolor='black')
ax23.axhline(9.4, color='k', linestyle='--', linewidth=1.5, label='LEO requirement')
for bar, val in zip(bars, stages_vals):
    ax23.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
              f'{val:.2f} km/s', ha='center', va='bottom', fontsize=10, fontweight='bold')
ax23.set_ylabel('Δv (km/s)', fontweight='bold')
ax23.set_title('Single-Stage vs 2-Stage Δv', fontsize=12, fontweight='bold')
ax23.legend(fontsize=10)
ax23.grid(True, alpha=0.3, axis='y')

plt.suptitle('Rocket Performance Analysis', fontsize=14, fontweight='bold')
plt.savefig(f'{output_dir}/04_rocket_performance.png', dpi=150, bbox_inches='tight')
print(f"[OK] 저장: {output_dir}/04_rocket_performance.png")
plt.close()

print("\n" + "=" * 70)
print("완료!")
print("=" * 70)
print("\n생성된 파일:")
print(f"  1. {output_dir}/04_rocket_single_stage.png  — 단단 로켓 전체 분석")
print(f"  2. {output_dir}/04_rocket_performance.png   — 배기속도 비교 + 다단 Δv")
print(f"\n물리적 결론:")
print(f"  · 치올코프스키 Δv = {delta_v_ideal:.0f} m/s → 실제 {v_max1:.0f} m/s (중력+항력 손실)")
print(f"  · 배기 속도 1000 m/s 증가 → 최대 고도 {(fuel_trajs[list(fuels.keys())[2]][:,1].max() - fuel_trajs[list(fuels.keys())[0]][:,1].max())/1000:.0f} km 이상 향상")
print(f"  · staging 없음 (빈 1단 탑재): Δv = {dv_no_stage:.0f} m/s")
print(f"  · staging 있음 (1단 분리):   Δv = {dv_two_stage:.0f} m/s")
print(f"    → 다단 분리로 +{dv_two_stage-dv_no_stage:.0f} m/s 이득: LEO 도달의 핵심 기술")
