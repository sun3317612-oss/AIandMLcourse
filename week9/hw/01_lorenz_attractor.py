"""
HW 01. Lorenz Attractor — 혼돈 역학의 고전
Lorenz System: Chaos and Butterfly Effect

에드워드 로렌츠(1963)가 기상 모델을 단순화하여 발견한 3차원 혼돈 시스템.
결정론적 방정식에서 예측 불가능한 혼돈 거동이 나타남을 보여줍니다.

운동 방정식 (Lorenz Equations):
  dx/dt = σ(y - x)
  dy/dt = x(ρ - z) - y
  dz/dt = xy - βz

표준 파라미터 (혼돈 영역):
  σ = 10  (Prandtl 수)
  ρ = 28  (Rayleigh 수 비)
  β = 8/3 (기하학적 인자)
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.gridspec import GridSpec
import os

output_dir = 'outputs'
os.makedirs(output_dir, exist_ok=True)

print("=" * 70)
print("Lorenz Attractor -- Chaos and Butterfly Effect")
print("=" * 70)

# ============================================================================
# 파라미터 설정
# ============================================================================

SIGMA = 10.0   # Prandtl 수
RHO   = 28.0   # Rayleigh 수 비
BETA  = 8.0/3  # 기하학적 인자

print(f"\n로렌츠 파라미터:")
print(f"  σ = {SIGMA}  (Prandtl number)")
print(f"  ρ = {RHO}   (Rayleigh number ratio)")
print(f"  β = {BETA:.4f}  (geometric factor)")

# ============================================================================
# 수치 적분 — RK4
# ============================================================================

def lorenz_derivs(state, t):
    """
    로렌츠 방정식 우변
    state = [x, y, z]
    """
    x, y, z = state
    dx = SIGMA * (y - x)
    dy = x * (RHO - z) - y
    dz = x * y - BETA * z
    return np.array([dx, dy, dz])


def rk4_step(f, y, t, dt):
    """4차 Runge-Kutta 적분"""
    k1 = f(y, t)
    k2 = f(y + 0.5*dt*k1, t + 0.5*dt)
    k3 = f(y + 0.5*dt*k2, t + 0.5*dt)
    k4 = f(y + dt*k3,     t + dt)
    return y + (dt/6.0) * (k1 + 2*k2 + 2*k3 + k4)


def simulate_lorenz(x0, y0, z0, t_max, dt):
    """로렌츠 시스템 시뮬레이션"""
    n_steps = int(t_max / dt)
    states = np.zeros((n_steps, 3))
    t_arr  = np.zeros(n_steps)

    state = np.array([x0, y0, z0], dtype=float)
    t = 0.0
    for i in range(n_steps):
        t_arr[i]  = t
        states[i] = state
        state = rk4_step(lorenz_derivs, state, t, dt)
        t += dt

    return t_arr, states

# ============================================================================
# 시뮬레이션 1: 단일 궤적 (어트랙터 구조)
# ============================================================================

print("\n[1] 단일 궤적 시뮬레이션 (t = 50 s)...")

t_max = 50.0
dt    = 0.01

t, traj = simulate_lorenz(0.1, 0.0, 0.0, t_max, dt)
x, y, z = traj[:, 0], traj[:, 1], traj[:, 2]

print(f"    완료: {len(t)} 스텝, x∈[{x.min():.2f}, {x.max():.2f}], z∈[{z.min():.2f}, {z.max():.2f}]")

# ============================================================================
# 시뮬레이션 2: 나비 효과 — 극미세 초기 조건 차이
# ============================================================================

print("\n[2] 나비 효과 시뮬레이션...")

epsilon = 1e-8  # 초기 조건 차이 (1 억분의 1)

t_A, traj_A = simulate_lorenz(0.1,         0.0, 0.0, 40.0, dt)
t_B, traj_B = simulate_lorenz(0.1+epsilon, 0.0, 0.0, 40.0, dt)

xA, yA, zA = traj_A[:, 0], traj_A[:, 1], traj_A[:, 2]
xB, yB, zB = traj_B[:, 0], traj_B[:, 1], traj_B[:, 2]

# 두 궤적 사이의 거리
dist = np.sqrt((xA - xB)**2 + (yA - yB)**2 + (zA - zB)**2)

# 리아푸노프 지수 추정 (선형 성장 구간에서의 기울기)
valid = (dist > 0) & (dist < 1.0)
if valid.sum() > 100:
    log_dist = np.log(dist[valid])
    t_valid  = t_A[valid]
    # 선형 피팅
    coeffs = np.polyfit(t_valid, log_dist, 1)
    lyapunov = coeffs[0]
else:
    lyapunov = np.nan

print(f"    초기 차이: Δx₀ = {epsilon:.1e}")
print(f"    최대 거리: {dist.max():.4f}")
if not np.isnan(lyapunov):
    print(f"    리아푸노프 지수 추정: λ ≈ {lyapunov:.3f} s⁻¹")
    print(f"    예측 가능 시간: ~{1/lyapunov:.2f} s (이후 완전한 혼돈)")

# ============================================================================
# 시뮬레이션 3: 평형점 분석
# ============================================================================

print("\n[3] 고정점(Fixed Points) 분석...")

# 로렌츠 방정식의 고정점: dx=dy=dz=0 → 3개 해
# C0 = (0, 0, 0)
# C± = (±√(β(ρ-1)), ±√(β(ρ-1)), ρ-1)

c_pm = np.sqrt(BETA * (RHO - 1))
FP = {
    'O':  np.array([0.0,  0.0,  0.0]),
    'C+': np.array([ c_pm,  c_pm, RHO-1]),
    'C-': np.array([-c_pm, -c_pm, RHO-1]),
}

for name, fp in FP.items():
    print(f"    {name}: ({fp[0]:.4f}, {fp[1]:.4f}, {fp[2]:.4f})")

# ============================================================================
# 시각화
# ============================================================================

# --- 그림 1: 3D 어트랙터 ---
fig1 = plt.figure(figsize=(18, 7))

ax1 = fig1.add_subplot(1, 3, 1, projection='3d')
sc = ax1.scatter(x[::5], y[::5], z[::5],
                 c=t[::5], cmap='plasma', s=0.3, alpha=0.6)
for name, fp in FP.items():
    ax1.scatter(*fp, s=60, zorder=5,
                color='red' if name != 'O' else 'black',
                marker='*', label=f'FP {name}')
ax1.set_xlabel('x', fontweight='bold')
ax1.set_ylabel('y', fontweight='bold')
ax1.set_zlabel('z', fontweight='bold')
ax1.set_title('Lorenz Attractor (3D)', fontsize=12, fontweight='bold')
ax1.legend(fontsize=8)
plt.colorbar(sc, ax=ax1, label='time (s)', shrink=0.5)

# --- 그림 1 (중앙): x-z 투영 ---
ax2 = fig1.add_subplot(1, 3, 2)
ax2.plot(x, z, linewidth=0.2, alpha=0.8, color='steelblue')
for name, fp in FP.items():
    ax2.plot(fp[0], fp[2], 'r*' if name != 'O' else 'k*', markersize=10, label=f'FP {name}')
ax2.set_xlabel('x', fontweight='bold')
ax2.set_ylabel('z', fontweight='bold')
ax2.set_title('x-z Projection', fontsize=12, fontweight='bold')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

# --- 그림 1 (오른쪽): x(t) 시계열 ---
ax3 = fig1.add_subplot(1, 3, 3)
ax3.plot(t[:3000], x[:3000], linewidth=0.8, color='steelblue')
ax3.set_xlabel('Time (s)', fontweight='bold')
ax3.set_ylabel('x(t)', fontweight='bold')
ax3.set_title('x(t) Time Series', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3)

plt.suptitle('Lorenz Attractor — Chaotic Dynamics', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{output_dir}/01_lorenz_attractor.png', dpi=150, bbox_inches='tight')
print(f"\n[OK] 저장: {output_dir}/01_lorenz_attractor.png")
plt.close()

# --- 그림 2: 나비 효과 분석 ---
fig2 = plt.figure(figsize=(16, 10))
gs2 = GridSpec(2, 3, figure=fig2, hspace=0.35, wspace=0.35)

# 두 궤적 x(t) 비교
ax21 = fig2.add_subplot(gs2[0, :2])
ax21.plot(t_A[:4000], xA[:4000], 'b-', linewidth=1.2, alpha=0.8, label='Trajectory A')
ax21.plot(t_B[:4000], xB[:4000], 'r--', linewidth=1.2, alpha=0.8, label=f'Trajectory B (Δx₀={epsilon:.0e})')
ax21.set_xlabel('Time (s)', fontweight='bold')
ax21.set_ylabel('x(t)', fontweight='bold')
ax21.set_title('Butterfly Effect: x(t) Comparison', fontsize=13, fontweight='bold')
ax21.legend(fontsize=10)
ax21.grid(True, alpha=0.3)

# 발산 거리 (log scale)
ax22 = fig2.add_subplot(gs2[0, 2])
valid_idx = dist > 0
ax22.semilogy(t_A[valid_idx], dist[valid_idx], 'purple', linewidth=1.5)
ax22.axhline(epsilon, color='g', linestyle='--', label=f'Initial Δ={epsilon:.0e}')
ax22.set_xlabel('Time (s)', fontweight='bold')
ax22.set_ylabel('Distance |ΔX|', fontweight='bold')
ax22.set_title('Exponential Divergence', fontsize=12, fontweight='bold')
ax22.legend(fontsize=9)
ax22.grid(True, alpha=0.3, which='both')

# 두 궤적 3D (짧은 시간)
ax23 = fig2.add_subplot(gs2[1, 0], projection='3d')
n_short = 2000
ax23.plot(xA[:n_short], yA[:n_short], zA[:n_short], 'b-', linewidth=0.8, alpha=0.7, label='A')
ax23.plot(xB[:n_short], yB[:n_short], zB[:n_short], 'r-', linewidth=0.8, alpha=0.7, label='B')
ax23.set_xlabel('x'); ax23.set_ylabel('y'); ax23.set_zlabel('z')
ax23.set_title('3D (t < 20s)', fontsize=11, fontweight='bold')
ax23.legend(fontsize=8)

# x-z 투영 비교
ax24 = fig2.add_subplot(gs2[1, 1])
ax24.plot(xA, zA, 'b-', linewidth=0.3, alpha=0.6, label='A')
ax24.plot(xB, zB, 'r-', linewidth=0.3, alpha=0.6, label='B')
ax24.set_xlabel('x', fontweight='bold'); ax24.set_ylabel('z', fontweight='bold')
ax24.set_title('x-z Projection (both)', fontsize=11, fontweight='bold')
ax24.legend(fontsize=9); ax24.grid(True, alpha=0.3)

# 요약 텍스트
ax25 = fig2.add_subplot(gs2[1, 2])
ax25.axis('off')
lyap_str = f"{lyapunov:.3f} s⁻¹" if not np.isnan(lyapunov) else "N/A"
pred_str = f"{1/lyapunov:.2f} s"  if not np.isnan(lyapunov) else "N/A"
summary = f"""
LORENZ SYSTEM SUMMARY
{'='*38}

Parameters:
  σ = {SIGMA}, ρ = {RHO}, β = {BETA:.4f}

Fixed Points:
  O  = (0, 0, 0)
  C+ = (+{c_pm:.3f}, +{c_pm:.3f}, {RHO-1:.1f})
  C- = (-{c_pm:.3f}, -f{c_pm:.3f}, {RHO-1:.1f})

Butterfly Effect:
  Initial Δx₀ = {epsilon:.1e}
  Lyapunov λ  ≈ {lyap_str}
  Predictability ≈ {pred_str}

Conclusion:
  결정론적 방정식이지만
  초기 조건에 극도로 민감.

  → "나비 효과"의 수학적 기원
  → 장기 기상 예보의 한계
"""
ax25.text(0.05, 0.5, summary, fontsize=9, family='monospace',
          verticalalignment='center', transform=ax25.transAxes)

plt.suptitle('Butterfly Effect Analysis', fontsize=15, fontweight='bold')
plt.savefig(f'{output_dir}/01_butterfly_effect.png', dpi=150, bbox_inches='tight')
print(f"[OK] 저장: {output_dir}/01_butterfly_effect.png")
plt.close()

print("\n" + "=" * 70)
print("완료!")
print("=" * 70)
print("\n생성된 파일:")
print(f"  1. {output_dir}/01_lorenz_attractor.png  — 3D 어트랙터 + 고정점 + 시계열")
print(f"  2. {output_dir}/01_butterfly_effect.png  — 나비 효과 분석")
print("\n물리적 해석:")
print(f"  · 로렌츠 어트랙터는 두 날개(C+, C-) 주변을 불규칙하게 이동")
print(f"  · 리아푸노프 지수 λ ≈ {lyap_str}: 오차가 e배 커지는 데 {pred_str} 소요")
print(f"  · 이것이 기상 예보의 근본적 한계 (예측 가능 기간 ~2주)")
