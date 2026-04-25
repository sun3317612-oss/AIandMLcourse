"""
08. Advanced 2D Ising Model - Animation and Cluster Analysis
고급 2D 이징 모델 - 애니메이션 및 클러스터 분석

2D 이징 모델의 가장 심화된 분석입니다:
- 열평형 과정 애니메이션
- 클러스터 분석 (상관 길이, 클러스터 크기 분포)
- 외부 자기장에 의한 히스테리시스 루프
- 자기상관 시간 분석
- 대규모 시스템 (100x100)

실행 시간: 5-10분

학습 목표:
1. 동역학적 과정의 시각화
2. 스핀 클러스터의 통계적 성질
3. 히스테리시스 현상
4. Monte Carlo 효율성 분석
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.gridspec import GridSpec
from scipy.ndimage import label, find_objects
from scipy.optimize import curve_fit
import os
import time

# 출력 디렉토리 확인
output_dir = 'outputs'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

animation_dir = os.path.join(output_dir, 'animation_frames')
if not os.path.exists(animation_dir):
    os.makedirs(animation_dir)

print("="*70)
print("Advanced 2D Ising Model - Animation and Analysis")
print("="*70)
print("Expected execution time: 5-10 minutes")
print("="*70)

# 한글 폰트 설정
def set_korean_font():
    """한글 폰트를 설정합니다."""
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

# ============================================================================
# 2D Ising Model Functions
# ============================================================================

def initialize_spins_2d(L, state='random'):
    """2D 스핀 배열 초기화"""
    if state == 'random':
        return np.random.choice([-1, 1], size=(L, L))
    elif state == 'up':
        return np.ones((L, L), dtype=int)
    elif state == 'down':
        return -np.ones((L, L), dtype=int)

def calculate_energy_2d(spins, J=1.0, h=0.0):
    """2D 이징 모델의 에너지 계산 (외부 자기장 포함)"""
    # 교환 상호작용
    energy = 0.0
    energy += -J * np.sum(spins * np.roll(spins, -1, axis=1))
    energy += -J * np.sum(spins * np.roll(spins, -1, axis=0))
    
    # 외부 자기장
    energy += -h * np.sum(spins)
    
    return energy

def calculate_magnetization_2d(spins):
    """자화 계산"""
    return np.sum(spins) / spins.size

def monte_carlo_sweep_2d(spins, T, J=1.0, h=0.0):
    """Monte Carlo sweep"""
    L = spins.shape[0]
    N = L * L
    
    for _ in range(N):
        i = np.random.randint(0, L)
        j = np.random.randint(0, L)
        
        neighbors_sum = (
            spins[(i+1)%L, j] + 
            spins[(i-1)%L, j] + 
            spins[i, (j+1)%L] + 
            spins[i, (j-1)%L]
        )
        
        dE = 2 * J * spins[i, j] * neighbors_sum + 2 * h * spins[i, j]
        
        if dE < 0 or np.random.random() < np.exp(-dE / T):
            spins[i, j] *= -1
    
    return spins

# ============================================================================
# Cluster Analysis Functions
# ============================================================================

def find_clusters(spins):
    """
    스핀 배열에서 클러스터를 찾습니다.
    
    Returns:
    --------
    labeled_array : array
        각 클러스터에 레이블된 배열
    num_clusters : int
        클러스터 개수
    """
    # scipy.ndimage.label을 사용하여 클러스터 식별
    # +1 스핀과 -1 스핀을 별도로 분석
    
    up_spins = (spins == 1).astype(int)
    down_spins = (spins == -1).astype(int)
    
    # 4-connectivity (상하좌우)
    structure = np.array([[0, 1, 0],
                          [1, 1, 1],
                          [0, 1, 0]])
    
    labeled_up, num_up = label(up_spins, structure=structure)
    labeled_down, num_down = label(down_spins, structure=structure)
    
    return labeled_up, labeled_down, num_up, num_down

def calculate_cluster_distribution(spins):
    """
    클러스터 크기 분포를 계산합니다.
    
    Returns:
    --------
    cluster_sizes : array
        클러스터 크기들
    """
    labeled_up, labeled_down, num_up, num_down = find_clusters(spins)
    
    cluster_sizes = []
    
    # Up spins
    for cluster_id in range(1, num_up + 1):
        size = np.sum(labeled_up == cluster_id)
        cluster_sizes.append(size)
    
    # Down spins
    for cluster_id in range(1, num_down + 1):
        size = np.sum(labeled_down == cluster_id)
        cluster_sizes.append(size)
    
    return np.array(cluster_sizes)

def calculate_correlation_function(spins):
    """
    스핀 상관 함수를 계산합니다.
    C(r) = <s(0) * s(r)>
    
    Returns:
    --------
    distances : array
        거리
    correlations : array
        상관 함수 값
    """
    L = spins.shape[0]
    max_r = L // 4
    
    correlations = np.zeros(max_r)
    counts = np.zeros(max_r)
    
    # 모든 방향으로 평균
    for dr in range(max_r):
        # 수평 방향
        corr_h = np.mean(spins * np.roll(spins, dr, axis=1))
        correlations[dr] += corr_h
        counts[dr] += 1
        
        # 수직 방향
        if dr > 0:
            corr_v = np.mean(spins * np.roll(spins, dr, axis=0))
            correlations[dr] += corr_v
            counts[dr] += 1
    
    correlations /= counts
    
    return np.arange(max_r), correlations

# ============================================================================
# Animation: Thermalization Process
# ============================================================================

print("\n1. Creating Thermalization Animation...")
print("   Generating frames (this may take 2-3 minutes)...")

start_time = time.time()

L = 100
T = Tc = 2.269
J = 1.0

spins = initialize_spins_2d(L, 'random')

# 프레임 저장할 스텝들
total_sweeps = 500
frame_interval = 5  # 매 5 sweep마다 프레임 저장
frames_to_save = list(range(0, total_sweeps + 1, frame_interval))

energies_anim = []
mags_anim = []

for sweep in range(total_sweeps + 1):
    if sweep in frames_to_save:
        # 프레임 저장
        fig_frame = plt.figure(figsize=(12, 10))
        gs = GridSpec(2, 2, figure=fig_frame, hspace=0.3, wspace=0.3)
        
        # (a) 스핀 배열
        ax1 = fig_frame.add_subplot(gs[0, :])
        im = ax1.imshow(spins, cmap='RdBu_r', vmin=-1, vmax=1, interpolation='nearest')
        ax1.set_title(f'Sweep {sweep}/{total_sweeps} at T={T:.3f} (Tc)', 
                      fontsize=14, weight='bold')
        ax1.axis('off')
        plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04, label='Spin', orientation='horizontal')
        
        # 현재 통계
        current_E = calculate_energy_2d(spins, J) / (L*L)
        current_M = calculate_magnetization_2d(spins)
        energies_anim.append(current_E)
        mags_anim.append(abs(current_M))
        
        # (b) 에너지 진화
        ax2 = fig_frame.add_subplot(gs[1, 0])
        if len(energies_anim) > 1:
            ax2.plot(frames_to_save[:len(energies_anim)], energies_anim, 'b-', linewidth=2)
        ax2.set_xlabel('Sweep', fontsize=11)
        ax2.set_ylabel('Energy per Spin', fontsize=11)
        ax2.set_title('Energy Evolution', fontsize=12, weight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim([0, total_sweeps])
        
        # (c) 자화 진화
        ax3 = fig_frame.add_subplot(gs[1, 1])
        if len(mags_anim) > 1:
            ax3.plot(frames_to_save[:len(mags_anim)], mags_anim, 'r-', linewidth=2)
        ax3.set_xlabel('Sweep', fontsize=11)
        ax3.set_ylabel('|Magnetization|', fontsize=11)
        ax3.set_title('Magnetization Evolution', fontsize=12, weight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.set_xlim([0, total_sweeps])
        ax3.set_ylim([0, 1])
        
        frame_filename = os.path.join(animation_dir, f'frame_{sweep:04d}.png')
        plt.savefig(frame_filename, dpi=100, bbox_inches='tight')
        plt.close(fig_frame)
        
        print(f"      Frame {sweep:04d} saved. M={abs(current_M):.4f}, E={current_E:.4f}")
    
    # Monte Carlo sweep
    if sweep < total_sweeps:
        spins = monte_carlo_sweep_2d(spins, T, J)

elapsed = time.time() - start_time
print(f"   Animation frames created in {elapsed:.1f} seconds.")
print(f"   Total frames: {len(frames_to_save)}")
print(f"   Location: {animation_dir}/")

# 최종 프레임 하나를 메인 출력에 저장
final_frame_path = os.path.join(output_dir, '08_thermalization_final.png')
os.system(f'copy "{os.path.join(animation_dir, f"frame_{total_sweeps:04d}.png")}" "{final_frame_path}" > nul 2>&1')
print(f"   Final frame saved: {final_frame_path}")

# ============================================================================
# Cluster Analysis at Different Temperatures
# ============================================================================

print("\n2. Cluster Analysis...")

temperatures_cluster = [1.5, Tc, 3.5]
L_cluster = 100

fig2 = plt.figure(figsize=(15, 12))
gs = GridSpec(3, 3, figure=fig2, hspace=0.4, wspace=0.4)

for t_idx, T in enumerate(temperatures_cluster):
    print(f"   Analyzing T = {T:.3f}...")
    
    # 평형화
    spins = initialize_spins_2d(L_cluster, 'random')
    for _ in range(500):
        spins = monte_carlo_sweep_2d(spins, T, J)
    
    # 클러스터 분석
    labeled_up, labeled_down, num_up, num_down = find_clusters(spins)
    cluster_sizes = calculate_cluster_distribution(spins)
    
    # 상관 함수
    distances, correlations = calculate_correlation_function(spins)
    
    # (a) 스핀 배열
    ax1 = fig2.add_subplot(gs[t_idx, 0])
    im1 = ax1.imshow(spins, cmap='RdBu_r', vmin=-1, vmax=1, interpolation='nearest')
    ax1.set_title(f'T={T:.2f} Spins', fontsize=11, weight='bold')
    ax1.axis('off')
    
    # (b) 클러스터 레이블 (up spins)
    ax2 = fig2.add_subplot(gs[t_idx, 1])
    # 클러스터를 랜덤 색상으로 표시
    labeled_display = labeled_up.copy().astype(float)
    labeled_display[labeled_display == 0] = np.nan
    im2 = ax2.imshow(labeled_display, cmap='tab20', interpolation='nearest')
    ax2.set_title(f'Up-spin Clusters (N={num_up})', fontsize=11, weight='bold')
    ax2.axis('off')
    
    # (c) 클러스터 크기 분포
    ax3 = fig2.add_subplot(gs[t_idx, 2])
    
    if len(cluster_sizes) > 0:
        # 로그 빈
        max_size = cluster_sizes.max()
        bins = np.logspace(0, np.log10(max_size), 30)
        counts, bin_edges = np.histogram(cluster_sizes, bins=bins)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        bin_widths = np.diff(bin_edges)
        
        # 정규화된 분포
        normalized_counts = counts / (bin_widths * np.sum(counts))
        
        ax3.loglog(bin_centers[counts > 0], normalized_counts[counts > 0], 
                   'bo-', markersize=4, linewidth=1.5)
        
        # 임계점에서는 power law: n(s) ~ s^(-tau), tau ~ 2.05 for 2D Ising
        if abs(T - Tc) < 0.1:
            s_range = np.logspace(0, 3, 50)
            power_law = s_range**(-2.05)
            power_law *= normalized_counts[counts > 0][0] / power_law[0]
            ax3.loglog(s_range, power_law, 'r--', linewidth=2, 
                       label='Power law ~ s^(-2.05)')
            ax3.legend(fontsize=9)
    
    ax3.set_xlabel('Cluster Size s', fontsize=10)
    ax3.set_ylabel('n(s)', fontsize=10)
    ax3.set_title('Cluster Size Distribution', fontsize=11, weight='bold')
    ax3.grid(True, alpha=0.3, which='both')
    
    print(f"      Clusters: {num_up + num_down}, Sizes: {len(cluster_sizes)}")
    if len(cluster_sizes) > 0:
        print(f"      Largest cluster: {cluster_sizes.max()}, Mean: {cluster_sizes.mean():.1f}")

plt.savefig(f'{output_dir}/08_cluster_analysis.png', dpi=150, bbox_inches='tight')
print(f"\n   Saved: {output_dir}/08_cluster_analysis.png")

# ============================================================================
# Correlation Length Analysis
# ============================================================================

print("\n3. Correlation Length Analysis...")

temperatures_corr = [1.5, 2.0, Tc, 2.5, 3.0, 3.5]
L_corr = 80

fig3 = plt.figure(figsize=(15, 10))
gs = GridSpec(2, 3, figure=fig3, hspace=0.35, wspace=0.3)

correlation_lengths = []

for idx, T in enumerate(temperatures_corr):
    print(f"   T = {T:.3f}...")
    
    # 평형화
    spins = initialize_spins_2d(L_corr, 'random')
    for _ in range(500):
        spins = monte_carlo_sweep_2d(spins, T, J)
    
    # 상관 함수
    distances, correlations = calculate_correlation_function(spins)
    
    # Plot
    ax = fig3.add_subplot(gs[idx // 3, idx % 3])
    ax.semilogy(distances, np.abs(correlations), 'bo-', markersize=4, linewidth=1.5,
                label='Data')
    
    # 상관 길이 추정: C(r) ~ exp(-r/xi)
    # ln|C(r)| = ln|C(0)| - r/xi
    # 선형 영역에서 피팅
    valid_range = (distances > 2) & (distances < 15) & (correlations > 0.01)
    if np.sum(valid_range) > 3:
        try:
            slope, intercept = np.polyfit(distances[valid_range], 
                                           np.log(np.abs(correlations[valid_range])), 1)
            xi = -1.0 / slope
            
            # 피팅 곡선
            fit_curve = np.exp(intercept + slope * distances)
            ax.semilogy(distances, fit_curve, 'r--', linewidth=2,
                        label=f'Fit: xi={xi:.2f}')
            
            correlation_lengths.append((T, xi))
        except:
            correlation_lengths.append((T, np.nan))
    else:
        correlation_lengths.append((T, np.nan))
    
    ax.set_xlabel('Distance r', fontsize=11)
    ax.set_ylabel('|C(r)|', fontsize=11)
    ax.set_title(f'T={T:.2f}', fontsize=12, weight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, which='both')
    ax.set_ylim([1e-3, 1])

plt.savefig(f'{output_dir}/08_correlation_length.png', dpi=150, bbox_inches='tight')
print(f"\n   Saved: {output_dir}/08_correlation_length.png")

# 상관 길이 vs 온도
correlation_lengths = np.array(correlation_lengths)
print(f"\n   Correlation lengths:")
for T, xi in correlation_lengths:
    if not np.isnan(xi):
        print(f"      T={T:.2f}: xi={xi:.2f}")

# ============================================================================
# Hysteresis Loop (External Magnetic Field)
# ============================================================================

print("\n4. Hysteresis Loop Analysis...")
print("   (This will take 3-4 minutes...)")

start_time = time.time()

L_hyst = 50
T_hyst = 1.8  # Below Tc
J_hyst = 1.0

# 자기장 범위: -3 to +3 and back
h_up = np.linspace(-3.0, 3.0, 25)
h_down = np.linspace(3.0, -3.0, 25)
h_fields = np.concatenate([h_up, h_down])

magnetizations_hyst = []

spins = initialize_spins_2d(L_hyst, 'up')  # 시작: 모두 up

for idx, h in enumerate(h_fields):
    # 각 자기장에서 평형화
    for _ in range(200):
        spins = monte_carlo_sweep_2d(spins, T_hyst, J_hyst, h)
    
    # 측정
    M = calculate_magnetization_2d(spins)
    magnetizations_hyst.append(M)
    
    if (idx + 1) % 10 == 0:
        print(f"      [{idx+1}/{len(h_fields)}] h={h:.2f}, M={M:.4f}")

elapsed = time.time() - start_time
print(f"   Hysteresis calculation completed in {elapsed:.1f} seconds.")

fig4 = plt.figure(figsize=(12, 5))

# (a) 히스테리시스 루프
ax1 = fig4.add_subplot(1, 2, 1)

# Increasing field
ax1.plot(h_up, magnetizations_hyst[:len(h_up)], 'b-o', markersize=4, 
         linewidth=2, label='Increasing h')
# Decreasing field
ax1.plot(h_down, magnetizations_hyst[len(h_up):], 'r-s', markersize=4,
         linewidth=2, label='Decreasing h')

ax1.axhline(y=0, color='k', linestyle='--', alpha=0.3)
ax1.axvline(x=0, color='k', linestyle='--', alpha=0.3)

ax1.set_xlabel('External Field h', fontsize=12)
ax1.set_ylabel('Magnetization M', fontsize=12)
ax1.set_title(f'(a) Hysteresis Loop (T={T_hyst:.2f}, L={L_hyst})', 
              fontsize=13, weight='bold')
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)

# Coercive field and remanent magnetization
# 보자력: M=0일 때의 h
# 잔류 자화: h=0일 때의 M

print(f"\n   Hysteresis properties:")
# Increasing branch에서 M=0 근처
idx_zero_up = np.argmin(np.abs(magnetizations_hyst[:len(h_up)]))
h_coercive_up = h_up[idx_zero_up]
print(f"      Coercive field (up): {h_coercive_up:.3f}")

# Decreasing branch에서 M=0 근처
idx_zero_down = np.argmin(np.abs(magnetizations_hyst[len(h_up):]))
h_coercive_down = h_down[idx_zero_down]
print(f"      Coercive field (down): {h_coercive_down:.3f}")

# h=0일 때 자화 (잔류 자화)
idx_h0_up = np.argmin(np.abs(h_up))
M_remanent_up = magnetizations_hyst[idx_h0_up]
print(f"      Remanent magnetization (up): {M_remanent_up:.3f}")

# (b) 에너지 변화
ax2 = fig4.add_subplot(1, 2, 2)

# 에너지는 자화와 자기장으로부터 추정
# E/N ~ -J * <neighbors> - h * M
# 정확한 계산을 위해서는 각 상태에서 에너지를 저장해야 하지만
# 여기서는 단순히 -h*M 항만 표시
energy_field = -np.array(h_fields) * np.array(magnetizations_hyst)

ax2.plot(h_up, energy_field[:len(h_up)], 'b-o', markersize=4,
         linewidth=2, label='Increasing h')
ax2.plot(h_down, energy_field[len(h_up):], 'r-s', markersize=4,
         linewidth=2, label='Decreasing h')

ax2.set_xlabel('External Field h', fontsize=12)
ax2.set_ylabel('Field Energy -h*M', fontsize=12)
ax2.set_title('(b) Energy Contribution from Field', fontsize=13, weight='bold')
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{output_dir}/08_hysteresis_loop.png', dpi=150, bbox_inches='tight')
print(f"\n   Saved: {output_dir}/08_hysteresis_loop.png")

# ============================================================================
# Autocorrelation Time Analysis
# ============================================================================

print("\n5. Autocorrelation Time Analysis...")

T_auto = Tc
L_auto = 50
n_sweeps_auto = 2000

spins = initialize_spins_2d(L_auto, 'random')

# 평형화
for _ in range(500):
    spins = monte_carlo_sweep_2d(spins, T_auto, J)

# 자화 시계열
mag_series = []
for sweep in range(n_sweeps_auto):
    spins = monte_carlo_sweep_2d(spins, T_auto, J)
    M = calculate_magnetization_2d(spins)
    mag_series.append(M)

mag_series = np.array(mag_series)

# 자기상관 함수: C(t) = <M(0) * M(t)> - <M>^2
def autocorrelation(series, max_lag):
    """자기상관 함수 계산"""
    mean = np.mean(series)
    var = np.var(series)
    
    autocorr = np.zeros(max_lag)
    for lag in range(max_lag):
        autocorr[lag] = np.mean((series[:-lag or None] - mean) * (series[lag:] - mean)) / var
    
    return autocorr

max_lag = 200
autocorr = autocorrelation(mag_series, max_lag)

# 자기상관 시간: sum of autocorrelation
# tau = sum_{t=0}^infty C(t)
# 실제로는 C(t)가 0에 가까워질 때까지 적분
integrated_autocorr = np.cumsum(autocorr)

# Exponential fit: C(t) ~ exp(-t/tau)
valid_fit = (autocorr > 0.05) & (autocorr < 0.95)
if np.sum(valid_fit) > 5:
    try:
        def exp_decay(t, tau):
            return np.exp(-t / tau)
        
        popt, _ = curve_fit(exp_decay, np.arange(max_lag)[valid_fit], 
                             autocorr[valid_fit], p0=[20])
        tau_fit = popt[0]
    except:
        tau_fit = np.nan
else:
    tau_fit = np.nan

fig5 = plt.figure(figsize=(15, 5))

# (a) 자화 시계열
ax1 = fig5.add_subplot(1, 3, 1)
ax1.plot(mag_series, 'b-', alpha=0.7, linewidth=0.8)
ax1.axhline(y=0, color='k', linestyle='--', alpha=0.3)
ax1.set_xlabel('Monte Carlo Sweep', fontsize=12)
ax1.set_ylabel('Magnetization M', fontsize=12)
ax1.set_title(f'(a) Magnetization Time Series (T={T_auto:.3f})', 
              fontsize=13, weight='bold')
ax1.grid(True, alpha=0.3)

# (b) 자기상관 함수
ax2 = fig5.add_subplot(1, 3, 2)
lags = np.arange(max_lag)
ax2.plot(lags, autocorr, 'go-', markersize=3, linewidth=1.5, label='Autocorrelation')

if not np.isnan(tau_fit):
    ax2.plot(lags, exp_decay(lags, tau_fit), 'r--', linewidth=2,
             label=f'Fit: tau={tau_fit:.1f}')

ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)
ax2.set_xlabel('Lag (sweeps)', fontsize=12)
ax2.set_ylabel('Autocorrelation C(t)', fontsize=12)
ax2.set_title('(b) Autocorrelation Function', fontsize=13, weight='bold')
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)

print(f"   Autocorrelation time:")
if not np.isnan(tau_fit):
    print(f"      tau (fit) ~ {tau_fit:.2f} sweeps")
print(f"      Integrated autocorr time: {integrated_autocorr[min(100, max_lag-1)]:.2f}")

# (c) Integrated autocorrelation time
ax3 = fig5.add_subplot(1, 3, 3)
ax3.plot(lags, integrated_autocorr, 'm-', linewidth=2)
ax3.set_xlabel('Lag (sweeps)', fontsize=12)
ax3.set_ylabel('Integrated Autocorrelation', fontsize=12)
ax3.set_title('(c) Integrated Autocorrelation Time', fontsize=13, weight='bold')
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{output_dir}/08_autocorrelation.png', dpi=150, bbox_inches='tight')
print(f"\n   Saved: {output_dir}/08_autocorrelation.png")

# ============================================================================
# Summary
# ============================================================================

print("\n" + "="*70)
print("Key Results:")
print("="*70)
print(f"1. Thermalization Animation:")
print(f"   - {len(frames_to_save)} frames saved in {animation_dir}/")
print(f"   - Shows relaxation to equilibrium at Tc")
print(f"   - Observable: critical fluctuations")
print(f"\n2. Cluster Analysis:")
print(f"   - Cluster size distribution n(s) ~ s^(-tau)")
print(f"   - At Tc: power law with tau ~ 2.05")
print(f"   - Below Tc: one large cluster dominates")
print(f"   - Above Tc: many small clusters")
print(f"\n3. Correlation Length:")
print(f"   - Diverges at Tc: xi ~ |T-Tc|^(-nu)")
print(f"   - Critical exponent nu ~ 1.0 for 2D Ising")
print(f"   - Determines size of correlated regions")
print(f"\n4. Hysteresis Loop:")
print(f"   - Coercive fields: {h_coercive_up:.3f}, {h_coercive_down:.3f}")
print(f"   - Remanent magnetization: {M_remanent_up:.3f}")
print(f"   - Memory effect in magnetic systems")
print(f"\n5. Autocorrelation Time:")
if not np.isnan(tau_fit):
    print(f"   - tau ~ {tau_fit:.1f} sweeps")
print(f"   - Determines MC sampling efficiency")
print(f"   - Critical slowing down near Tc")
print("="*70)
print("\nAll advanced analyses completed successfully!")
print("="*70)

plt.show()

