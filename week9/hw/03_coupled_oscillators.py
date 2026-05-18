"""
HW 03. Coupled Oscillators — 노멀 모드와 에너지 전달
Coupled Harmonic Oscillators: Normal Modes, Resonance, and Energy Transfer

두 진동자를 스프링으로 연결하면 독립적인 "정규 모드(normal mode)"가 생깁니다.
초기 조건이 정규 모드와 다르면 에너지가 두 진동자 사이를 주기적으로 오가는
"맥놀이(beat)" 현상이 나타납니다.

계 구성:
  질량 m₁ ── k₁ ── [고정벽] ── k₂ ── m₂ ── k₃ ── [고정벽]
  (k₂: 두 질량을 연결하는 결합 스프링)

운동 방정식:
  m₁ẍ₁ = -k₁x₁ - k₂(x₁ - x₂)
  m₂ẍ₂ = -k₃x₂ + k₂(x₁ - x₂)

정규 모드 진동수 (m₁=m₂=m, k₁=k₃=k):
  ω₋ = √(k/m)          (in-phase mode)
  ω₊ = √((k + 2k₂)/m)  (out-of-phase mode)
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
print("Coupled Oscillators — Normal Modes and Energy Transfer")
print("=" * 70)

# ============================================================================
# 물리 파라미터
# ============================================================================

m  = 1.0   # 질량 (kg)
k  = 4.0   # 벽-질량 스프링 상수 (N/m)
k2 = 0.5   # 결합 스프링 상수 (N/m) — 약한 결합

# 정규 모드 진동수 (해석해)
omega_minus = np.sqrt(k / m)
omega_plus  = np.sqrt((k + 2*k2) / m)

# 맥놀이 진동수
omega_beat   = (omega_plus - omega_minus) / 2
omega_avg    = (omega_plus + omega_minus) / 2
T_beat = 2 * np.pi / omega_beat   # 에너지 전달 주기

print(f"\n파라미터:")
print(f"  m  = {m} kg,  k  = {k} N/m,  k₂ = {k2} N/m")
print(f"\n정규 모드 진동수 (이론):")
print(f"  ω₋ (in-phase)     = {omega_minus:.4f} rad/s  → T = {2*np.pi/omega_minus:.3f} s")
print(f"  ω₊ (out-of-phase) = {omega_plus:.4f} rad/s  → T = {2*np.pi/omega_plus:.3f} s")
print(f"\n맥놀이:")
print(f"  에너지 전달 주기 T_beat = {T_beat:.2f} s")

# ============================================================================
# 운동 방정식
# ============================================================================

def coupled_osc_derivs(state, t):
    """
    결합 진동자 운동 방정식
    state = [x1, v1, x2, v2]
    """
    x1, v1, x2, v2 = state
    ax1 = -(k + k2)/m * x1 + k2/m * x2
    ax2 =  k2/m * x1 - (k + k2)/m * x2
    return np.array([v1, ax1, v2, ax2])


def rk4_step(f, y, t, dt):
    k1 = f(y, t)
    k2_ = f(y + 0.5*dt*k1, t + 0.5*dt)
    k3 = f(y + 0.5*dt*k2_, t + 0.5*dt)
    k4 = f(y + dt*k3,      t + dt)
    return y + (dt/6.0) * (k1 + 2*k2_ + 2*k3 + k4)


def simulate(state0, t_max, dt):
    n = int(t_max / dt)
    states = np.zeros((n, 4))
    t_arr  = np.zeros(n)
    state  = np.array(state0, dtype=float)
    t = 0.0
    for i in range(n):
        t_arr[i]  = t
        states[i] = state
        state = rk4_step(coupled_osc_derivs, state, t, dt)
        t += dt
    return t_arr, states

# ============================================================================
# 시뮬레이션 1: 정규 모드 (In-phase)
# 두 질량이 같은 위상으로 진동 — 결합 스프링은 늘어나지 않음
# ============================================================================

print("\n[1] In-phase Normal Mode 시뮬레이션...")

dt    = 0.01
t_max = 5 * 2 * np.pi / omega_minus  # 5주기

state0_ip = [1.0, 0.0, 1.0, 0.0]   # x1=x2=1, v1=v2=0
t_ip, s_ip = simulate(state0_ip, t_max, dt)

print(f"    ω₋ 측정값 확인:")
zero_cross = np.where(np.diff(np.sign(s_ip[:, 0])))[0]
if len(zero_cross) >= 2:
    T_measured = 2 * (t_ip[zero_cross[1]] - t_ip[zero_cross[0]])
    print(f"    측정 주기 = {T_measured:.4f} s  (이론: {2*np.pi/omega_minus:.4f} s)")

# ============================================================================
# 시뮬레이션 2: 정규 모드 (Out-of-phase)
# 두 질량이 반대 위상으로 진동 — 결합 스프링이 최대로 신축
# ============================================================================

print("\n[2] Out-of-phase Normal Mode 시뮬레이션...")

state0_oop = [1.0, 0.0, -1.0, 0.0]  # x1=-x2=1
t_oop, s_oop = simulate(state0_oop, t_max, dt)

# ============================================================================
# 시뮬레이션 3: 맥놀이 — 한쪽만 초기 변위
# 에너지가 m₁ → m₂ → m₁ 로 완전히 전달됨
# ============================================================================

print("\n[3] Beat (에너지 전달) 시뮬레이션...")

t_max_beat = 2.5 * T_beat
state0_beat = [1.0, 0.0, 0.0, 0.0]   # x1=1, x2=0
t_bt, s_bt  = simulate(state0_beat, t_max_beat, dt)

# 에너지 계산
E_kin = 0.5 * m * (s_bt[:, 1]**2 + s_bt[:, 3]**2)
E_pot = (0.5 * k * s_bt[:, 0]**2 +
         0.5 * k * s_bt[:, 2]**2 +
         0.5 * k2 * (s_bt[:, 0] - s_bt[:, 2])**2)
E_tot = E_kin + E_pot
E_1   = 0.5 * m * s_bt[:, 1]**2 + 0.5 * (k + k2) * s_bt[:, 0]**2
E_2   = 0.5 * m * s_bt[:, 3]**2 + 0.5 * (k + k2) * s_bt[:, 2]**2

# 이론적 에너지 전달 포락선
A = 1.0  # 초기 진폭
E0 = 0.5 * k * A**2
E1_theory = E0 * np.cos(omega_beat * t_bt)**2
E2_theory = E0 * np.sin(omega_beat * t_bt)**2

E_init = E_tot[0]
E_drift = abs(E_tot - E_init).max() / E_init * 100
print(f"    총 에너지 보존: 최대 오차 {E_drift:.4f}%")
print(f"    에너지 전달 주기 (이론): {T_beat:.2f} s")

# ============================================================================
# 시뮬레이션 4: 결합 강도 스캔 — k₂ 변화에 따른 모드 분리
# ============================================================================

print("\n[4] 결합 강도 스캔...")

k2_values = [0.1, 0.5, 1.0, 2.0]
freq_data = {}
for k2_val in k2_values:
    om_m = np.sqrt(k / m)
    om_p = np.sqrt((k + 2*k2_val) / m)
    freq_data[k2_val] = (om_m, om_p, om_p - om_m)
    print(f"    k₂={k2_val:.1f}: ω₋={om_m:.3f}, ω₊={om_p:.3f}, Δω={om_p-om_m:.3f}")

# ============================================================================
# 시각화
# ============================================================================

# --- 그림 1: 정규 모드 비교 ---
fig1 = plt.figure(figsize=(16, 10))
gs1 = GridSpec(2, 2, figure=fig1, hspace=0.35, wspace=0.3)

n_show_ip = int(3 * 2*np.pi/omega_minus / dt)

ax11 = fig1.add_subplot(gs1[0, 0])
ax11.plot(t_ip[:n_show_ip], s_ip[:n_show_ip, 0], 'b-', linewidth=2, label='Mass 1 (x₁)')
ax11.plot(t_ip[:n_show_ip], s_ip[:n_show_ip, 2], 'r--', linewidth=2, label='Mass 2 (x₂)')
ax11.set_xlabel('Time (s)', fontweight='bold')
ax11.set_ylabel('Displacement (m)', fontweight='bold')
ax11.set_title(f'In-Phase Mode  ω₋ = {omega_minus:.3f} rad/s', fontsize=12, fontweight='bold')
ax11.legend(fontsize=10)
ax11.grid(True, alpha=0.3)

n_show_oop = int(3 * 2*np.pi/omega_plus / dt)

ax12 = fig1.add_subplot(gs1[0, 1])
ax12.plot(t_oop[:n_show_oop], s_oop[:n_show_oop, 0], 'b-', linewidth=2, label='Mass 1 (x₁)')
ax12.plot(t_oop[:n_show_oop], s_oop[:n_show_oop, 2], 'r--', linewidth=2, label='Mass 2 (x₂)')
ax12.set_xlabel('Time (s)', fontweight='bold')
ax12.set_ylabel('Displacement (m)', fontweight='bold')
ax12.set_title(f'Out-of-Phase Mode  ω₊ = {omega_plus:.3f} rad/s', fontsize=12, fontweight='bold')
ax12.legend(fontsize=10)
ax12.grid(True, alpha=0.3)

# 위상 공간
ax13 = fig1.add_subplot(gs1[1, 0])
ax13.plot(s_ip[:, 0], s_ip[:, 2], 'b-', linewidth=1, alpha=0.7, label='In-phase')
ax13.plot(s_oop[:, 0], s_oop[:, 2], 'r-', linewidth=1, alpha=0.7, label='Out-of-phase')
ax13.set_xlabel('x₁ (m)', fontweight='bold')
ax13.set_ylabel('x₂ (m)', fontweight='bold')
ax13.set_title('Configuration Space (x₁ vs x₂)', fontsize=12, fontweight='bold')
ax13.legend(fontsize=10)
ax13.grid(True, alpha=0.3)
ax13.set_aspect('equal')

# 결합 강도 — 모드 진동수
ax14 = fig1.add_subplot(gs1[1, 1])
k2_scan = np.linspace(0, 3, 100)
om_m_scan = np.sqrt(k / m) * np.ones_like(k2_scan)
om_p_scan = np.sqrt((k + 2*k2_scan) / m)
ax14.plot(k2_scan, om_m_scan, 'b-', linewidth=2, label='ω₋ (in-phase)')
ax14.plot(k2_scan, om_p_scan, 'r-', linewidth=2, label='ω₊ (out-of-phase)')
ax14.fill_between(k2_scan, om_m_scan, om_p_scan, alpha=0.15, color='green', label='Band gap')
ax14.axvline(k2, color='k', linestyle='--', linewidth=1.5, label=f'k₂={k2}')
ax14.set_xlabel('Coupling Constant k₂ (N/m)', fontweight='bold')
ax14.set_ylabel('Normal Mode Frequency (rad/s)', fontweight='bold')
ax14.set_title('Mode Splitting vs Coupling Strength', fontsize=12, fontweight='bold')
ax14.legend(fontsize=10)
ax14.grid(True, alpha=0.3)

plt.suptitle('Coupled Oscillators — Normal Modes', fontsize=15, fontweight='bold')
plt.savefig(f'{output_dir}/03_normal_modes.png', dpi=150, bbox_inches='tight')
print(f"\n[OK] 저장: {output_dir}/03_normal_modes.png")
plt.close()

# --- 그림 2: 맥놀이와 에너지 전달 ---
fig2 = plt.figure(figsize=(16, 12))
gs2 = GridSpec(3, 2, figure=fig2, hspace=0.4, wspace=0.3)

ax21 = fig2.add_subplot(gs2[0, :])
ax21.plot(t_bt, s_bt[:, 0], 'b-', linewidth=1.2, label='Mass 1 (x₁)', alpha=0.9)
ax21.plot(t_bt, s_bt[:, 2], 'r-', linewidth=1.2, label='Mass 2 (x₂)', alpha=0.9)
# 포락선 (이론)
A_env = A * np.abs(np.cos(omega_beat * t_bt))
ax21.plot(t_bt,  A_env, 'b:', linewidth=2, alpha=0.6, label='Envelope (theory)')
ax21.plot(t_bt, -A_env, 'b:', linewidth=2, alpha=0.6)
ax21.set_xlabel('Time (s)', fontweight='bold')
ax21.set_ylabel('Displacement (m)', fontweight='bold')
ax21.set_title('Beat Phenomenon — Energy Transfer Between Masses', fontsize=13, fontweight='bold')
ax21.legend(fontsize=10)
ax21.grid(True, alpha=0.3)

ax22 = fig2.add_subplot(gs2[1, :])
ax22.plot(t_bt, E1_theory, 'b-', linewidth=2, label='E₁ (theory)', alpha=0.7)
ax22.plot(t_bt, E2_theory, 'r-', linewidth=2, label='E₂ (theory)', alpha=0.7)
ax22.plot(t_bt, E_1, 'b--', linewidth=1.0, label='E₁ (RK4)')
ax22.plot(t_bt, E_2, 'r--', linewidth=1.0, label='E₂ (RK4)')
ax22.axhline(E_tot.mean(), color='k', linestyle=':', linewidth=1.5, label='E_total (conserved)')
ax22.set_xlabel('Time (s)', fontweight='bold')
ax22.set_ylabel('Energy (J)', fontweight='bold')
ax22.set_title('Energy Transfer: E₁ ↔ E₂ (Beat Period)', fontsize=13, fontweight='bold')
ax22.legend(fontsize=10)
ax22.grid(True, alpha=0.3)

ax23 = fig2.add_subplot(gs2[2, 0])
ax23.plot(s_bt[:, 0], s_bt[:, 1], 'b-', linewidth=0.5, alpha=0.7, label='Mass 1')
ax23.plot(s_bt[:, 2], s_bt[:, 3], 'r-', linewidth=0.5, alpha=0.7, label='Mass 2')
ax23.set_xlabel('x (m)', fontweight='bold')
ax23.set_ylabel('v (m/s)', fontweight='bold')
ax23.set_title('Phase Space (x-v)', fontsize=12, fontweight='bold')
ax23.legend(fontsize=10)
ax23.grid(True, alpha=0.3)

ax24 = fig2.add_subplot(gs2[2, 1])
ax24.axis('off')
summary = f"""
COUPLED OSCILLATORS SUMMARY
{'='*42}

System Parameters:
  m = {m} kg,  k = {k} N/m,  k₂ = {k2} N/m

Normal Mode Frequencies:
  ω₋ = {omega_minus:.4f} rad/s  (in-phase)
  ω₊ = {omega_plus:.4f} rad/s  (out-of-phase)
  Δω = {omega_plus - omega_minus:.4f} rad/s

Beat (Energy Transfer):
  ω_beat = Δω/2 = {omega_beat:.4f} rad/s
  T_beat = {T_beat:.2f} s
  → m₁ → m₂ 완전 에너지 전달 주기

Energy Conservation:
  Total energy drift < {E_drift:.4f}%

Physical Analogy:
  결합 진동자 ↔ 분자 진동 모드
  k₂ 클수록 모드 분리 커짐
  → 분자 스펙트로스코피의 기초
"""
ax24.text(0.03, 0.5, summary, fontsize=9.5, family='monospace',
          verticalalignment='center', transform=ax24.transAxes)

plt.suptitle('Beat Phenomenon — Energy Transfer in Coupled Oscillators',
             fontsize=14, fontweight='bold')
plt.savefig(f'{output_dir}/03_beat_energy_transfer.png', dpi=150, bbox_inches='tight')
print(f"[OK] 저장: {output_dir}/03_beat_energy_transfer.png")
plt.close()

print("\n" + "=" * 70)
print("완료!")
print("=" * 70)
print("\n생성된 파일:")
print(f"  1. {output_dir}/03_normal_modes.png        — 정규 모드 + 결합 강도 스캔")
print(f"  2. {output_dir}/03_beat_energy_transfer.png — 맥놀이 + 에너지 전달")
print(f"\n물리적 결론:")
print(f"  · In-phase 모드(ω₋): 두 질량이 같이 움직여 결합 스프링 미신축")
print(f"  · Out-of-phase 모드(ω₊): 반대로 움직여 결합 스프링 최대 신축")
print(f"  · 비정규 초기 조건 → 에너지가 주기 T_beat = {T_beat:.2f} s로 완전 전달")
print(f"  · k₂ 증가 → 모드 분리(Δω) 증가 → T_beat 감소 (더 빠른 에너지 교환)")
