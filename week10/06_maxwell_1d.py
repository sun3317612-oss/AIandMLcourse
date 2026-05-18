"""
06. Maxwell 1D Wave Equation
1D 전자기파 전파 (FDTD 방법)

물리 배경:
- 파동 방정식: ∂²E/∂t² = c²∂²E/∂x²
- FDTD (Finite Difference Time Domain) 방법
- CFL 안정성 조건: c*dt/dx ≤ 1

학습 목표:
1. 파동 방정식의 수치 해법
2. FDTD 알고리즘 이해
3. 파동의 전파와 반사 관찰
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
print("06. Maxwell 1D Wave Equation - FDTD Method")
print("="*70)

# 물리 상수
c = 3e8  # 광속 (m/s)

# 시뮬레이션 파라미터
Lx = 1.0  # 공간 크기 (m)
Nx = 500  # 공간 격자점
dx = Lx / Nx

# CFL 조건: c*dt/dx <= 1 (안정성)
CFL = 0.9
dt = CFL * dx / c

# 시간 스텝
T_total = 2 * Lx / c  # 파동이 왕복하는 시간
Nt = int(T_total / dt)

print(f"\n공간 격자:")
print(f"  Lx = {Lx} m")
print(f"  Nx = {Nx}")
print(f"  dx = {dx*1e3:.3f} mm")
print(f"\n시간 격자:")
print(f"  dt = {dt*1e9:.3f} ns")
print(f"  Nt = {Nt}")
print(f"  Total time = {T_total*1e9:.1f} ns")
print(f"\nCFL 수: {CFL:.2f} (< 1 = stable)")

# 격자 생성
x = np.linspace(0, Lx, Nx)

# 전기장 초기화
E = np.zeros(Nx)
E_old = np.zeros(Nx)
E_new = np.zeros(Nx)

# 초기 조건: 가우시안 펄스
x0 = 0.2 * Lx  # 펄스 중심
sigma = 0.05 * Lx  # 펄스 폭
E = np.exp(-((x - x0)**2) / (2*sigma**2))
E_old = E.copy()

print(f"\n초기 펄스:")
print(f"  위치: x0 = {x0*1e2:.1f} cm")
print(f"  폭: σ = {sigma*1e2:.1f} cm")

# 시간 evolution (FDTD)
print("\n시간 진화 계산 중...")

# 스냅샷 저장할 시간들
snapshot_times = [0, int(0.25*Nt), int(0.5*Nt), int(0.75*Nt), Nt-1]
snapshots = []
snapshot_labels = []

for n in range(Nt):
    # FDTD 업데이트
    # ∂²E/∂t² = c²∂²E/∂x²
    # E_new = 2*E - E_old + (c*dt/dx)^2 * (E[i+1] - 2*E[i] + E[i-1])
    
    for i in range(1, Nx-1):
        E_new[i] = 2*E[i] - E_old[i] + (c*dt/dx)**2 * (E[i+1] - 2*E[i] + E[i-1])
    
    # 경계 조건 (반사)
    E_new[0] = 0  # 완전 반사 (고정 끝)
    E_new[-1] = 0
    
    # 업데이트
    E_old = E.copy()
    E = E_new.copy()
    
    # 스냅샷 저장
    if n in snapshot_times:
        snapshots.append(E.copy())
        t_current = n * dt
        snapshot_labels.append(f't = {t_current*1e9:.1f} ns')
        print(f"  스냅샷 {len(snapshots)}: t = {t_current*1e9:.1f} ns")

print("\n시각화 중...")

# 시각화 1: 스냅샷들
fig1, axes1 = plt.subplots(len(snapshots), 1, figsize=(12, 10))

for idx, (snap, label) in enumerate(zip(snapshots, snapshot_labels)):
    ax = axes1[idx]
    ax.plot(x*1e2, snap, 'b-', linewidth=2)
    ax.axhline(0, color='k', linestyle='--', alpha=0.3)
    ax.set_ylabel('E field', fontsize=11, fontweight='bold')
    ax.set_title(label, fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, Lx*1e2])
    ax.set_ylim([-1.5, 1.5])
    
    if idx == len(snapshots) - 1:
        ax.set_xlabel('x (cm)', fontsize=12, fontweight='bold')

plt.suptitle('1D Wave Propagation (FDTD Method)', fontsize=15, fontweight='bold')
plt.tight_layout()

output_path1 = f'{output_dir}/06_wave_1d_snapshots.png'
plt.savefig(output_path1, dpi=150, bbox_inches='tight')
print(f"\n[OK] 그래프 저장: {output_path1}")
plt.close()

# 추가 분석: 다양한 초기 조건
print("\n추가 시뮬레이션: 다양한 초기 조건")
print("-" * 70)

def run_fdtd_1d(E_initial, Nt_run, dx, dt, c):
    """1D FDTD 시뮬레이션"""
    Nx = len(E_initial)
    E = E_initial.copy()
    E_old = E.copy()
    E_new = np.zeros(Nx)
    
    history = [E.copy()]
    
    for n in range(Nt_run):
        for i in range(1, Nx-1):
            E_new[i] = 2*E[i] - E_old[i] + (c*dt/dx)**2 * (E[i+1] - 2*E[i] + E[i-1])
        
        E_new[0] = 0
        E_new[-1] = 0
        
        E_old = E.copy()
        E = E_new.copy()
        
        if n % (Nt_run // 10) == 0 or n == Nt_run - 1:
            history.append(E.copy())
    
    return history

# 시나리오들
scenarios = [
    {
        'name': 'Single Pulse',
        'E_init': np.exp(-((x - 0.2*Lx)**2) / (2*(0.05*Lx)**2))
    },
    {
        'name': 'Two Pulses',
        'E_init': np.exp(-((x - 0.3*Lx)**2) / (2*(0.04*Lx)**2)) + \
                  np.exp(-((x - 0.7*Lx)**2) / (2*(0.04*Lx)**2))
    },
    {
        'name': 'Standing Wave',
        'E_init': np.sin(4*np.pi*x/Lx)
    }
]

fig2, axes2 = plt.subplots(3, 4, figsize=(16, 10))

for row, scenario in enumerate(scenarios):
    print(f"  시뮬레이션: {scenario['name']}")
    
    history = run_fdtd_1d(scenario['E_init'], Nt, dx, dt, c)
    
    # 4개의 시간대 표시
    time_indices = [0, len(history)//3, 2*len(history)//3, -1]
    
    for col, tidx in enumerate(time_indices):
        ax = axes2[row, col]
        ax.plot(x*1e2, history[tidx], 'b-', linewidth=2)
        ax.axhline(0, color='k', linestyle='--', alpha=0.3)
        ax.set_ylim([-2, 2])
        ax.grid(True, alpha=0.3)
        
        if row == 0:
            t_frac = tidx / (len(history) - 1)
            ax.set_title(f't = {t_frac:.2f} × T', fontsize=11, fontweight='bold')
        
        if row == 2:
            ax.set_xlabel('x (cm)', fontsize=10, fontweight='bold')
        
        if col == 0:
            ax.set_ylabel(f'{scenario["name"]}\nE field', 
                         fontsize=10, fontweight='bold')

plt.suptitle('1D Wave Equation: Different Initial Conditions', 
             fontsize=14, fontweight='bold')
plt.tight_layout()

output_path2 = f'{output_dir}/06_wave_1d_scenarios.png'
plt.savefig(output_path2, dpi=150, bbox_inches='tight')
print(f"\n[OK] 그래프 저장: {output_path2}")
plt.close()

print("\n" + "="*70)
print("분석 완료!")
print("="*70)
print("\n핵심 개념:")
print("  1. 파동 방정식: ∂²E/∂t² = c²∂²E/∂x²")
print("  2. FDTD: 시간과 공간을 격자로 나눔")
print("  3. CFL 조건: 안정성을 위한 dt/dx 제한")
print("  4. 경계에서 반사 (고정 끝)")
print("  5. 파동의 중첩 원리")
print("\n수치 방법:")
print("  - 중앙 차분법 (Central Difference)")
print("  - 2차 정확도 (시간, 공간)")
print("  - 명시적 방법 (Explicit scheme)")
print("\n응용:")
print("  - 전자기파 전파")
print("  - 레이더 시뮬레이션")
print("  - 안테나 설계")
print("\n생성된 파일:")
print(f"  - {output_path1}")
print(f"  - {output_path2}")

