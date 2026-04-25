"""
02. Monte Carlo Integration: Pi Estimation
Monte Carlo 적분: 원주율 추정

Monte Carlo 방법을 이용한 적분 계산의 대표적인 예제입니다:
- 랜덤 샘플링을 통한 면적 계산
- 원주율 π 추정
- 수렴 속도 분석
- 오차 분석

학습 목표:
1. Monte Carlo 적분의 원리 이해
2. 샘플 수에 따른 수렴 특성
3. 통계적 오차 분석
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.gridspec import GridSpec
import os

# 출력 디렉토리 확인
output_dir = 'outputs'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print("="*70)
print("Monte Carlo Integration: Pi Estimation")
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
# Monte Carlo Pi Estimation
# ============================================================================

def estimate_pi_monte_carlo(n_samples):
    """
    Monte Carlo 방법으로 π를 추정합니다.
    
    원리: 단위 정사각형 내의 1/4 원의 면적 = π/4
    랜덤 점들이 원 안에 들어갈 확률 = π/4
    따라서 π ≈ 4 * (원 안의 점 개수) / (전체 점 개수)
    
    Parameters:
    -----------
    n_samples : int
        샘플 점의 개수
    
    Returns:
    --------
    pi_estimate : float
        추정된 π 값
    x, y : arrays
        샘플 점들의 좌표
    inside : array (bool)
        각 점이 원 안에 있는지 여부
    """
    # [0, 1] x [0, 1] 영역에서 랜덤 샘플링
    x = np.random.uniform(0, 1, n_samples)
    y = np.random.uniform(0, 1, n_samples)
    
    # 원의 방정식: x^2 + y^2 <= 1
    distance_squared = x**2 + y**2
    inside = distance_squared <= 1.0
    
    # π 추정
    n_inside = np.sum(inside)
    pi_estimate = 4.0 * n_inside / n_samples
    
    return pi_estimate, x, y, inside

def estimate_pi_convergence(max_samples, n_runs=10):
    """
    샘플 수에 따른 π 추정값의 수렴을 분석합니다.
    
    Parameters:
    -----------
    max_samples : int
        최대 샘플 수
    n_runs : int
        각 샘플 수에서의 반복 횟수
    
    Returns:
    --------
    sample_counts : array
        샘플 수 리스트
    pi_estimates : array (n_sample_counts, n_runs)
        각 조건에서의 π 추정값들
    """
    # 샘플 수를 로그 스케일로 설정
    sample_counts = np.logspace(2, np.log10(max_samples), 20, dtype=int)
    pi_estimates = np.zeros((len(sample_counts), n_runs))
    
    for i, n_samples in enumerate(sample_counts):
        for j in range(n_runs):
            pi_est, _, _, _ = estimate_pi_monte_carlo(n_samples)
            pi_estimates[i, j] = pi_est
    
    return sample_counts, pi_estimates

# ============================================================================
# Simulation 1: Visualization of Monte Carlo Sampling
# ============================================================================

print("\n1. Visualizing Monte Carlo Sampling...")

fig1 = plt.figure(figsize=(15, 10))
gs = GridSpec(2, 2, figure=fig1, hspace=0.3, wspace=0.3)

# 다양한 샘플 수로 시각화
sample_sizes = [100, 1000, 10000, 100000]

for idx, n_samples in enumerate(sample_sizes):
    ax = fig1.add_subplot(gs[idx // 2, idx % 2])
    
    pi_est, x, y, inside = estimate_pi_monte_carlo(n_samples)
    
    # 원 안과 밖의 점들을 다른 색으로 표시
    ax.scatter(x[inside], y[inside], c='red', s=1, alpha=0.5, label='Inside circle')
    ax.scatter(x[~inside], y[~inside], c='blue', s=1, alpha=0.5, label='Outside circle')
    
    # 1/4 원 그리기
    theta = np.linspace(0, np.pi/2, 100)
    circle_x = np.cos(theta)
    circle_y = np.sin(theta)
    ax.plot(circle_x, circle_y, 'k-', linewidth=2, label='Quarter circle')
    
    # 정사각형
    ax.plot([0, 1, 1, 0, 0], [0, 0, 1, 1, 0], 'k--', linewidth=1.5, alpha=0.5)
    
    error = abs(pi_est - np.pi)
    error_pct = 100 * error / np.pi
    
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_aspect('equal')
    ax.set_xlabel('X', fontsize=11)
    ax.set_ylabel('Y', fontsize=11)
    ax.set_title(f'N = {n_samples:,}\npi estimate = {pi_est:.6f} (error: {error_pct:.3f}%)', 
                 fontsize=12, weight='bold')
    if idx == 0:
        ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    n_inside = np.sum(inside)
    print(f"   N={n_samples:>6}: pi = {pi_est:.6f}, error = {error:.6f} ({error_pct:.3f}%)")

plt.savefig(f'{output_dir}/02_pi_sampling_visualization.png', dpi=150, bbox_inches='tight')
print(f"\n   Saved: {output_dir}/02_pi_sampling_visualization.png")

# ============================================================================
# Simulation 2: Convergence Analysis
# ============================================================================

print("\n2. Analyzing Convergence...")

max_samples = 1000000
n_runs = 50

sample_counts, pi_estimates = estimate_pi_convergence(max_samples, n_runs)

fig2 = plt.figure(figsize=(15, 10))
gs = GridSpec(2, 2, figure=fig2, hspace=0.3, wspace=0.3)

# (a) Pi estimates vs sample size
ax1 = fig2.add_subplot(gs[0, :])

# 모든 실행 결과를 연한 선으로
for j in range(n_runs):
    ax1.plot(sample_counts, pi_estimates[:, j], 'b-', alpha=0.1, linewidth=0.5)

# 평균값을 굵은 선으로
pi_mean = np.mean(pi_estimates, axis=1)
ax1.plot(sample_counts, pi_mean, 'r-', linewidth=2.5, label='Mean estimate')

# 실제 π 값
ax1.axhline(y=np.pi, color='g', linestyle='--', linewidth=2, label=f'True pi = {np.pi:.6f}')

ax1.set_xscale('log')
ax1.set_xlabel('Number of Samples', fontsize=12)
ax1.set_ylabel('Pi Estimate', fontsize=12)
ax1.set_title('(a) Convergence of Pi Estimation', fontsize=13, weight='bold')
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3, which='both')
ax1.set_ylim([2.9, 3.4])

# (b) Error vs sample size
ax2 = fig2.add_subplot(gs[1, 0])

errors = np.abs(pi_estimates - np.pi)
mean_error = np.mean(errors, axis=1)
std_error = np.std(errors, axis=1)

ax2.loglog(sample_counts, mean_error, 'bo-', linewidth=2, markersize=6, label='Mean error')
ax2.fill_between(sample_counts, mean_error - std_error, mean_error + std_error, 
                  alpha=0.3, color='blue', label='Standard deviation')

# 이론적 수렴 속도: error ~ 1/sqrt(N)
theoretical_error = 0.5 / np.sqrt(sample_counts)
ax2.loglog(sample_counts, theoretical_error, 'r--', linewidth=2, 
           label='Theory: 1/sqrt(N)')

ax2.set_xlabel('Number of Samples', fontsize=12)
ax2.set_ylabel('Absolute Error', fontsize=12)
ax2.set_title('(b) Error vs Sample Size (log-log)', fontsize=13, weight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3, which='both')

print(f"   Error scaling check:")
for i in [0, len(sample_counts)//2, -1]:
    n = sample_counts[i]
    err = mean_error[i]
    theo = theoretical_error[i]
    print(f"   N={n:>7}: error={err:.6f}, theory~{theo:.6f}")

# (c) Histogram of final estimates
ax3 = fig2.add_subplot(gs[1, 1])

final_estimates = pi_estimates[-1, :]
mean_final = np.mean(final_estimates)
std_final = np.std(final_estimates)

counts, bins, patches = ax3.hist(final_estimates, bins=20, density=True, 
                                   alpha=0.7, color='skyblue', edgecolor='black')

# 이론적 가우시안 분포
x_gauss = np.linspace(final_estimates.min(), final_estimates.max(), 100)
gaussian = (1/(std_final * np.sqrt(2*np.pi))) * np.exp(-0.5*((x_gauss - mean_final)/std_final)**2)
ax3.plot(x_gauss, gaussian, 'r-', linewidth=2, label=f'Gaussian fit')

ax3.axvline(x=np.pi, color='g', linestyle='--', linewidth=2, label=f'True pi')
ax3.axvline(x=mean_final, color='orange', linestyle='--', linewidth=2, 
            label=f'Mean = {mean_final:.4f}')

ax3.set_xlabel('Pi Estimate', fontsize=12)
ax3.set_ylabel('Probability Density', fontsize=12)
ax3.set_title(f'(c) Distribution of Estimates (N={max_samples:,}, {n_runs} runs)', 
              fontsize=13, weight='bold')
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.3)

print(f"\n   Final statistics (N={max_samples:,}):")
print(f"   Mean estimate: {mean_final:.6f}")
print(f"   Std deviation: {std_final:.6f}")
print(f"   Error: {abs(mean_final - np.pi):.6f}")
print(f"   Theoretical std: {np.pi * np.sqrt((4 - np.pi)/(4*max_samples)):.6f}")

plt.savefig(f'{output_dir}/02_pi_convergence.png', dpi=150, bbox_inches='tight')
print(f"\n   Saved: {output_dir}/02_pi_convergence.png")

# ============================================================================
# Simulation 3: Other Integration Examples
# ============================================================================

print("\n3. General Monte Carlo Integration...")

def monte_carlo_integrate(func, x_min, x_max, y_min, y_max, n_samples):
    """
    일반적인 Monte Carlo 적분을 수행합니다.
    
    Parameters:
    -----------
    func : function
        적분할 함수 (x, y 입력)
    x_min, x_max : float
        x 적분 범위
    y_min, y_max : float
        y 적분 범위
    n_samples : int
        샘플 수
    
    Returns:
    --------
    integral : float
        적분 추정값
    """
    # 직사각형 영역에서 랜덤 샘플링
    x = np.random.uniform(x_min, x_max, n_samples)
    y = np.random.uniform(y_min, y_max, n_samples)
    
    # 함수가 0보다 큰 영역의 비율
    area = (x_max - x_min) * (y_max - y_min)
    
    # 점들이 함수 아래에 있는지 확인
    inside = (y >= 0) & (y <= func(x))
    
    # 적분 추정
    integral = area * np.sum(inside) / n_samples
    
    return integral, x, y, inside

# 예제: 반원의 면적
fig3 = plt.figure(figsize=(15, 5))

# (a) Semicircle: int sqrt(1-x^2) dx from -1 to 1
ax1 = fig3.add_subplot(1, 3, 1)
def semicircle(x):
    return np.sqrt(np.maximum(1 - x**2, 0))

n_samples_int = 10000
integral_est, x_samp, y_samp, inside = monte_carlo_integrate(
    semicircle, -1, 1, 0, 1, n_samples_int
)

ax1.scatter(x_samp[inside], y_samp[inside], c='red', s=1, alpha=0.3, label='Inside')
ax1.scatter(x_samp[~inside], y_samp[~inside], c='blue', s=1, alpha=0.3, label='Outside')

x_curve = np.linspace(-1, 1, 200)
ax1.plot(x_curve, semicircle(x_curve), 'k-', linewidth=2, label='y = sqrt(1-x^2)')

ax1.set_xlabel('X', fontsize=12)
ax1.set_ylabel('Y', fontsize=12)
ax1.set_title(f'(a) Semicircle Area\nEstimate = {integral_est:.4f} (True = {np.pi/2:.4f})', 
              fontsize=12, weight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

print(f"   Semicircle area: {integral_est:.6f} (true: {np.pi/2:.6f}, error: {abs(integral_est - np.pi/2):.6f})")

# (b) Parabola: int x^2 dx from 0 to 1
ax2 = fig3.add_subplot(1, 3, 2)
def parabola(x):
    return x**2

integral_est2, x_samp2, y_samp2, inside2 = monte_carlo_integrate(
    parabola, 0, 1, 0, 1, n_samples_int
)

ax2.scatter(x_samp2[inside2], y_samp2[inside2], c='red', s=1, alpha=0.3, label='Inside')
ax2.scatter(x_samp2[~inside2], y_samp2[~inside2], c='blue', s=1, alpha=0.3, label='Outside')

x_curve2 = np.linspace(0, 1, 200)
ax2.plot(x_curve2, parabola(x_curve2), 'k-', linewidth=2, label='y = x^2')

ax2.set_xlabel('X', fontsize=12)
ax2.set_ylabel('Y', fontsize=12)
ax2.set_title(f'(b) Parabola Area\nEstimate = {integral_est2:.4f} (True = {1/3:.4f})', 
              fontsize=12, weight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

print(f"   Parabola area: {integral_est2:.6f} (true: {1/3:.6f}, error: {abs(integral_est2 - 1/3):.6f})")

# (c) Exponential: int exp(-x) dx from 0 to 2
ax3 = fig3.add_subplot(1, 3, 3)
def exponential(x):
    return np.exp(-x)

y_max_exp = 1.0
integral_est3, x_samp3, y_samp3, inside3 = monte_carlo_integrate(
    exponential, 0, 2, 0, y_max_exp, n_samples_int
)

ax3.scatter(x_samp3[inside3], y_samp3[inside3], c='red', s=1, alpha=0.3, label='Inside')
ax3.scatter(x_samp3[~inside3], y_samp3[~inside3], c='blue', s=1, alpha=0.3, label='Outside')

x_curve3 = np.linspace(0, 2, 200)
ax3.plot(x_curve3, exponential(x_curve3), 'k-', linewidth=2, label='y = exp(-x)')

true_val = 1 - np.exp(-2)
ax3.set_xlabel('X', fontsize=12)
ax3.set_ylabel('Y', fontsize=12)
ax3.set_title(f'(c) Exponential Area\nEstimate = {integral_est3:.4f} (True = {true_val:.4f})', 
              fontsize=12, weight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)

print(f"   Exponential area: {integral_est3:.6f} (true: {true_val:.6f}, error: {abs(integral_est3 - true_val):.6f})")

plt.tight_layout()
plt.savefig(f'{output_dir}/02_pi_general_integration.png', dpi=150, bbox_inches='tight')
print(f"\n   Saved: {output_dir}/02_pi_general_integration.png")

# ============================================================================
# Summary
# ============================================================================

print("\n" + "="*70)
print("Key Results:")
print("="*70)
print(f"1. Monte Carlo Integration:")
print(f"   - Random sampling estimates integrals without analytical methods")
print(f"   - Works well for high-dimensional integrals")
print(f"   - Error scales as 1/sqrt(N) (not dependent on dimension!)")
print(f"\n2. Pi Estimation:")
print(f"   - Classical example of Monte Carlo method")
print(f"   - Converges to true value with more samples")
print(f"   - Statistical fluctuations decrease with sqrt(N)")
print(f"\n3. General Principle:")
print(f"   - Area/Volume = (bounding box area) * (fraction of points inside)")
print(f"   - Applicable to any integrable function")
print(f"   - Foundation for Monte Carlo in statistical physics")
print("="*70)

plt.show()

