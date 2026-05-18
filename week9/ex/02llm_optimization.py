"""
02. LLM Code Optimization Example
LLM 코드 최적화 예제

실제 LLM API를 사용하지 않고, 코드 최적화의 개념과 중요성을 보여주는 교육용 프로그램입니다.

최적화 기법:
1. 벡터화 (Vectorization)
2. 알고리즘 복잡도 개선
3. 메모리 효율화
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import time
import os

# 출력 디렉토리 확인
output_dir = 'outputs'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print("="*70)
print("LLM Code Optimization Example")
print("="*70)
print("다양한 최적화 기법의 성능 비교")
print()

# ============================================================================
# 예제 1: 행렬 곱셈 - 벡터화의 중요성
# ============================================================================

print("="*70)
print("Example 1: Matrix Multiplication - Vectorization")
print("="*70)

def matrix_multiply_naive(A, B):
    """
    순진한 구현 (Naive Implementation)
    - 3중 for 루프 사용
    - Python 리스트 사용
    - 매우 느림!
    """
    n = len(A)
    m = len(B[0])
    k = len(B)
    
    C = [[0 for _ in range(m)] for _ in range(n)]
    
    for i in range(n):
        for j in range(m):
            for p in range(k):
                C[i][j] += A[i][p] * B[p][j]
    
    return C

def matrix_multiply_optimized(A, B):
    """
    최적화된 구현 (Optimized Implementation)
    - NumPy 벡터화 사용
    - C로 구현된 BLAS 라이브러리 활용
    - 매우 빠름!
    """
    return np.dot(A, B)

# 벤치마크
sizes = [10, 50, 100, 200]
times_naive = []
times_optimized = []

print("\n벤치마크 실행 중...")
for n in sizes:
    print(f"  행렬 크기: {n}x{n}")
    
    # NumPy 배열 생성
    A_np = np.random.rand(n, n)
    B_np = np.random.rand(n, n)
    
    # Python 리스트 변환 (naive용)
    A_list = A_np.tolist()
    B_list = B_np.tolist()
    
    # Naive 방법
    start = time.time()
    C_naive = matrix_multiply_naive(A_list, B_list)
    time_naive = time.time() - start
    times_naive.append(time_naive)
    print(f"    Naive: {time_naive:.4f}s")
    
    # Optimized 방법
    start = time.time()
    C_optimized = matrix_multiply_optimized(A_np, B_np)
    time_optimized = time.time() - start
    times_optimized.append(time_optimized)
    print(f"    Optimized: {time_optimized:.6f}s")
    print(f"    가속비: {time_naive/time_optimized:.1f}x")

# ============================================================================
# 예제 2: 거리 계산 - 알고리즘 개선
# ============================================================================

print("\n" + "="*70)
print("Example 2: Distance Calculation - Algorithm Improvement")
print("="*70)

def pairwise_distances_naive(points):
    """
    순진한 구현: O(n^2) 시간, O(n^2) 공간
    - 모든 쌍의 거리를 계산
    - 중복 계산 존재
    """
    n = len(points)
    distances = []
    
    for i in range(n):
        for j in range(n):
            if i != j:
                dx = points[i][0] - points[j][0]
                dy = points[i][1] - points[j][1]
                dist = (dx**2 + dy**2)**0.5
                distances.append(dist)
    
    return distances

def pairwise_distances_optimized(points):
    """
    최적화된 구현: 벡터화 + 브로드캐스팅
    - NumPy의 브로드캐스팅 활용
    - 대칭성 활용 (선택적)
    """
    points = np.array(points)
    # 브로드캐스팅을 사용한 거리 계산
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    distances = np.sqrt(np.sum(diff**2, axis=2))
    return distances

# 벤치마크
n_points_list = [10, 50, 100, 500]
times_dist_naive = []
times_dist_optimized = []

print("\n벤치마크 실행 중...")
for n_points in n_points_list:
    print(f"  점의 개수: {n_points}")
    
    # 무작위 점 생성
    points = np.random.rand(n_points, 2).tolist()
    
    # Naive 방법
    start = time.time()
    dist_naive = pairwise_distances_naive(points)
    time_naive = time.time() - start
    times_dist_naive.append(time_naive)
    print(f"    Naive: {time_naive:.4f}s")
    
    # Optimized 방법
    start = time.time()
    dist_optimized = pairwise_distances_optimized(points)
    time_optimized = time.time() - start
    times_dist_optimized.append(time_optimized)
    print(f"    Optimized: {time_optimized:.6f}s")
    print(f"    가속비: {time_naive/time_optimized:.1f}x")

# ============================================================================
# 예제 3: 필터링 - 조건부 연산
# ============================================================================

print("\n" + "="*70)
print("Example 3: Filtering - Conditional Operations")
print("="*70)

def filter_and_transform_naive(data, threshold):
    """
    순진한 구현
    - for 루프로 조건 확인
    - 리스트 append 사용
    """
    result = []
    for x in data:
        if x > threshold:
            result.append(x ** 2)
    return result

def filter_and_transform_optimized(data, threshold):
    """
    최적화된 구현
    - NumPy boolean indexing
    - 벡터화된 연산
    """
    data = np.array(data)
    mask = data > threshold
    return data[mask] ** 2

# 벤치마크
data_sizes = [1000, 10000, 100000, 1000000]
times_filter_naive = []
times_filter_optimized = []

print("\n벤치마크 실행 중...")
for size in data_sizes:
    print(f"  데이터 크기: {size:,}")
    
    # 무작위 데이터 생성
    data = np.random.rand(size).tolist()
    threshold = 0.5
    
    # Naive 방법
    start = time.time()
    result_naive = filter_and_transform_naive(data, threshold)
    time_naive = time.time() - start
    times_filter_naive.append(time_naive)
    print(f"    Naive: {time_naive:.4f}s")
    
    # Optimized 방법
    start = time.time()
    result_optimized = filter_and_transform_optimized(data, threshold)
    time_optimized = time.time() - start
    times_filter_optimized.append(time_optimized)
    print(f"    Optimized: {time_optimized:.6f}s")
    print(f"    가속비: {time_naive/time_optimized:.1f}x")

# ============================================================================
# 시각화
# ============================================================================

# 그림 1: 성능 비교
fig1 = plt.figure(figsize=(16, 12))
gs1 = GridSpec(3, 2, figure=fig1, hspace=0.35, wspace=0.3)

# 1-1: 행렬 곱셈 시간
ax11 = fig1.add_subplot(gs1[0, 0])
x_pos = np.arange(len(sizes))
width = 0.35
ax11.bar(x_pos - width/2, times_naive, width, label='Naive', alpha=0.8, color='red')
ax11.bar(x_pos + width/2, times_optimized, width, label='Optimized', alpha=0.8, color='blue')
ax11.set_xlabel('Matrix Size', fontsize=11, fontweight='bold')
ax11.set_ylabel('Time (seconds)', fontsize=11, fontweight='bold')
ax11.set_title('Matrix Multiplication Performance', fontsize=12, fontweight='bold')
ax11.set_xticks(x_pos)
ax11.set_xticklabels([f'{n}x{n}' for n in sizes])
ax11.legend(fontsize=10)
ax11.grid(True, alpha=0.3, axis='y')
ax11.set_yscale('log')

# 1-2: 행렬 곱셈 가속비
ax12 = fig1.add_subplot(gs1[0, 1])
speedups_matrix = [t_n/t_o for t_n, t_o in zip(times_naive, times_optimized)]
ax12.plot(sizes, speedups_matrix, 'go-', linewidth=2, markersize=10)
ax12.set_xlabel('Matrix Size', fontsize=11, fontweight='bold')
ax12.set_ylabel('Speedup Factor', fontsize=11, fontweight='bold')
ax12.set_title('Matrix Multiplication Speedup', fontsize=12, fontweight='bold')
ax12.grid(True, alpha=0.3)
for i, (s, sp) in enumerate(zip(sizes, speedups_matrix)):
    ax12.text(s, sp, f'{sp:.0f}x', ha='center', va='bottom', fontsize=10, fontweight='bold')

# 2-1: 거리 계산 시간
ax21 = fig1.add_subplot(gs1[1, 0])
x_pos = np.arange(len(n_points_list))
ax21.bar(x_pos - width/2, times_dist_naive, width, label='Naive', alpha=0.8, color='red')
ax21.bar(x_pos + width/2, times_dist_optimized, width, label='Optimized', alpha=0.8, color='blue')
ax21.set_xlabel('Number of Points', fontsize=11, fontweight='bold')
ax21.set_ylabel('Time (seconds)', fontsize=11, fontweight='bold')
ax21.set_title('Distance Calculation Performance', fontsize=12, fontweight='bold')
ax21.set_xticks(x_pos)
ax21.set_xticklabels(n_points_list)
ax21.legend(fontsize=10)
ax21.grid(True, alpha=0.3, axis='y')
ax21.set_yscale('log')

# 2-2: 거리 계산 가속비
ax22 = fig1.add_subplot(gs1[1, 1])
speedups_dist = [t_n/t_o for t_n, t_o in zip(times_dist_naive, times_dist_optimized)]
ax22.plot(n_points_list, speedups_dist, 'mo-', linewidth=2, markersize=10)
ax22.set_xlabel('Number of Points', fontsize=11, fontweight='bold')
ax22.set_ylabel('Speedup Factor', fontsize=11, fontweight='bold')
ax22.set_title('Distance Calculation Speedup', fontsize=12, fontweight='bold')
ax22.grid(True, alpha=0.3)
for i, (n, sp) in enumerate(zip(n_points_list, speedups_dist)):
    ax22.text(n, sp, f'{sp:.0f}x', ha='center', va='bottom', fontsize=10, fontweight='bold')

# 3-1: 필터링 시간
ax31 = fig1.add_subplot(gs1[2, 0])
x_pos = np.arange(len(data_sizes))
ax31.bar(x_pos - width/2, times_filter_naive, width, label='Naive', alpha=0.8, color='red')
ax31.bar(x_pos + width/2, times_filter_optimized, width, label='Optimized', alpha=0.8, color='blue')
ax31.set_xlabel('Data Size', fontsize=11, fontweight='bold')
ax31.set_ylabel('Time (seconds)', fontsize=11, fontweight='bold')
ax31.set_title('Filtering Performance', fontsize=12, fontweight='bold')
ax31.set_xticks(x_pos)
ax31.set_xticklabels([f'{s:,}' for s in data_sizes], rotation=45)
ax31.legend(fontsize=10)
ax31.grid(True, alpha=0.3, axis='y')
ax31.set_yscale('log')

# 3-2: 필터링 가속비
ax32 = fig1.add_subplot(gs1[2, 1])
speedups_filter = [t_n/t_o for t_n, t_o in zip(times_filter_naive, times_filter_optimized)]
ax32.plot(data_sizes, speedups_filter, 'co-', linewidth=2, markersize=10)
ax32.set_xlabel('Data Size', fontsize=11, fontweight='bold')
ax32.set_ylabel('Speedup Factor', fontsize=11, fontweight='bold')
ax32.set_title('Filtering Speedup', fontsize=12, fontweight='bold')
ax32.grid(True, alpha=0.3)
ax32.set_xscale('log')
for i, (s, sp) in enumerate(zip(data_sizes, speedups_filter)):
    ax32.text(s, sp, f'{sp:.0f}x', ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.suptitle('Code Optimization: Performance Comparison', fontsize=16, fontweight='bold')
plt.savefig(f'{output_dir}/02_optimization_comparison.png', dpi=150, bbox_inches='tight')
print(f"\n[OK] 그래프 저장: {output_dir}/02_optimization_comparison.png")
plt.close()

# 그림 2: 복잡도 분석
fig2, axes2 = plt.subplots(1, 3, figsize=(16, 5))

# 행렬 곱셈: O(n^3) vs O(n^2.376) (실제 BLAS)
ax1 = axes2[0]
n_range = np.linspace(10, 200, 100)
# Naive: O(n^3)
complexity_naive = (n_range / 10) ** 3 * times_naive[0]
# Optimized: 실제 측정값 기반
actual_times = np.interp(n_range, sizes, times_optimized)

ax1.plot(n_range, complexity_naive, 'r--', linewidth=2, label='Naive O(n^3)', alpha=0.7)
ax1.plot(sizes, times_naive, 'ro', markersize=10, label='Naive (measured)')
ax1.plot(n_range, actual_times, 'b-', linewidth=2, label='Optimized', alpha=0.7)
ax1.plot(sizes, times_optimized, 'bs', markersize=10, label='Optimized (measured)')
ax1.set_xlabel('Matrix Size n', fontsize=11, fontweight='bold')
ax1.set_ylabel('Time (seconds)', fontsize=11, fontweight='bold')
ax1.set_title('Complexity: Matrix Multiplication', fontsize=12, fontweight='bold')
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)
ax1.set_yscale('log')

# 거리 계산: O(n^2)
ax2 = axes2[1]
n_range_dist = np.linspace(10, 500, 100)
complexity_dist = (n_range_dist / 10) ** 2 * times_dist_naive[0]
actual_times_dist = np.interp(n_range_dist, n_points_list, times_dist_optimized)

ax2.plot(n_range_dist, complexity_dist, 'r--', linewidth=2, label='Naive O(n^2)', alpha=0.7)
ax2.plot(n_points_list, times_dist_naive, 'ro', markersize=10, label='Naive (measured)')
ax2.plot(n_range_dist, actual_times_dist, 'b-', linewidth=2, label='Optimized O(n^2)', alpha=0.7)
ax2.plot(n_points_list, times_dist_optimized, 'bs', markersize=10, label='Optimized (measured)')
ax2.set_xlabel('Number of Points n', fontsize=11, fontweight='bold')
ax2.set_ylabel('Time (seconds)', fontsize=11, fontweight='bold')
ax2.set_title('Complexity: Distance Calculation', fontsize=12, fontweight='bold')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)
ax2.set_yscale('log')

# 필터링: O(n)
ax3 = axes2[2]
n_range_filter = np.linspace(1000, 1000000, 100)
complexity_filter = n_range_filter / 1000 * times_filter_naive[0]
actual_times_filter = np.interp(n_range_filter, data_sizes, times_filter_optimized)

ax3.plot(n_range_filter, complexity_filter, 'r--', linewidth=2, label='Naive O(n)', alpha=0.7)
ax3.plot(data_sizes, times_filter_naive, 'ro', markersize=10, label='Naive (measured)')
ax3.plot(n_range_filter, actual_times_filter, 'b-', linewidth=2, label='Optimized O(n)', alpha=0.7)
ax3.plot(data_sizes, times_filter_optimized, 'bs', markersize=10, label='Optimized (measured)')
ax3.set_xlabel('Data Size n', fontsize=11, fontweight='bold')
ax3.set_ylabel('Time (seconds)', fontsize=11, fontweight='bold')
ax3.set_title('Complexity: Filtering', fontsize=12, fontweight='bold')
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3)
ax3.set_xscale('log')
ax3.set_yscale('log')

plt.suptitle('Algorithmic Complexity Analysis', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{output_dir}/02_complexity_analysis.png', dpi=150, bbox_inches='tight')
print(f"[OK] 그래프 저장: {output_dir}/02_complexity_analysis.png")
plt.close()

# ============================================================================
# 최적화 보고서 생성
# ============================================================================

report = f"""
CODE OPTIMIZATION REPORT
{'='*70}

1. MATRIX MULTIPLICATION
{'='*70}
Task: Multiply two nxn matrices

Naive Implementation:
  - Algorithm: Triple nested loops
  - Complexity: O(n^3)
  - Language: Pure Python with lists

Optimized Implementation:
  - Algorithm: NumPy dot product (BLAS)
  - Complexity: O(n^2.376) (Strassen-like)
  - Language: C/Fortran backend

Results:
  Size        Naive (s)    Optimized (s)    Speedup
  {'='*60}
"""

for i, n in enumerate(sizes):
    report += f"  {n:4d}x{n:<4d}    {times_naive[i]:8.4f}     {times_optimized[i]:11.6f}    {times_naive[i]/times_optimized[i]:6.1f}x\n"

avg_speedup_matrix = np.mean([t_n/t_o for t_n, t_o in zip(times_naive, times_optimized)])
report += f"\nAverage Speedup: {avg_speedup_matrix:.1f}x\n"

report += f"""

2. PAIRWISE DISTANCE CALCULATION
{'='*70}
Task: Calculate distances between all pairs of n points

Naive Implementation:
  - Algorithm: Nested loops with manual distance
  - Complexity: O(n^2)
  - Memory: O(n^2) list

Optimized Implementation:
  - Algorithm: NumPy broadcasting
  - Complexity: O(n^2) (same but vectorized)
  - Memory: O(n^2) numpy array

Results:
  Points      Naive (s)    Optimized (s)    Speedup
  {'='*60}
"""

for i, n in enumerate(n_points_list):
    report += f"  {n:4d}        {times_dist_naive[i]:8.4f}     {times_dist_optimized[i]:11.6f}    {times_dist_naive[i]/times_dist_optimized[i]:6.1f}x\n"

avg_speedup_dist = np.mean([t_n/t_o for t_n, t_o in zip(times_dist_naive, times_dist_optimized)])
report += f"\nAverage Speedup: {avg_speedup_dist:.1f}x\n"

report += f"""

3. FILTER AND TRANSFORM
{'='*70}
Task: Filter elements > threshold and square them

Naive Implementation:
  - Algorithm: For loop with if condition
  - Complexity: O(n)
  - Memory: List append (dynamic)

Optimized Implementation:
  - Algorithm: Boolean indexing + vectorized operation
  - Complexity: O(n)
  - Memory: Pre-allocated numpy array

Results:
  Data Size   Naive (s)    Optimized (s)    Speedup
  {'='*60}
"""

for i, size in enumerate(data_sizes):
    report += f"  {size:8,}    {times_filter_naive[i]:8.4f}     {times_filter_optimized[i]:11.6f}    {times_filter_naive[i]/times_filter_optimized[i]:6.1f}x\n"

avg_speedup_filter = np.mean([t_n/t_o for t_n, t_o in zip(times_filter_naive, times_filter_optimized)])
report += f"\nAverage Speedup: {avg_speedup_filter:.1f}x\n"

report += f"""

OPTIMIZATION TECHNIQUES SUMMARY
{'='*70}

1. Vectorization
   - Replace loops with NumPy operations
   - Utilize SIMD instructions
   - Typical speedup: 10-100x

2. Better Algorithms
   - Lower complexity (e.g., O(n^3) to O(n^2.376))
   - Use specialized libraries (BLAS, LAPACK)
   - Typical speedup: 2-10x

3. Memory Efficiency
   - Pre-allocate arrays
   - Avoid dynamic lists
   - Cache-friendly access patterns
   - Typical speedup: 2-5x

KEY TAKEAWAYS
{'='*70}

1. NumPy is ESSENTIAL for numerical Python
   - Always use NumPy arrays instead of lists
   - Vectorize operations whenever possible

2. Avoid explicit loops for array operations
   - Use broadcasting, indexing, and built-in functions
   - Let compiled code do the work

3. Profile before optimizing
   - Measure actual bottlenecks
   - Don't optimize prematurely

4. Use the right libraries
   - NumPy for arrays
   - SciPy for scientific computing
   - Numba/Cython for critical sections

CONCLUSION
{'='*70}

Average speedups achieved:
  - Matrix Multiplication: {avg_speedup_matrix:.1f}x
  - Distance Calculation: {avg_speedup_dist:.1f}x
  - Filtering: {avg_speedup_filter:.1f}x
  
Overall average: {(avg_speedup_matrix + avg_speedup_dist + avg_speedup_filter)/3:.1f}x

These optimizations are crucial for:
  - Real-time applications
  - Large-scale data processing
  - Scientific simulations
  - Machine learning training

Remember: Fast code = More experiments = Better science!
"""

# 보고서 저장
with open(f'{output_dir}/optimization_report.txt', 'w', encoding='utf-8') as f:
    f.write(report)

print(f"[OK] 보고서 저장: {output_dir}/optimization_report.txt")

print("\n" + "="*70)
print("분석 완료!")
print("="*70)
print("\n생성된 파일:")
print(f"  1. {output_dir}/02_optimization_comparison.png - 성능 비교")
print(f"  2. {output_dir}/02_complexity_analysis.png - 복잡도 분석")
print(f"  3. {output_dir}/optimization_report.txt - 최적화 보고서")
print("\n주요 결과:")
print(f"  - 행렬 곱셈: 평균 {avg_speedup_matrix:.0f}x 빠름")
print(f"  - 거리 계산: 평균 {avg_speedup_dist:.0f}x 빠름")
print(f"  - 필터링: 평균 {avg_speedup_filter:.0f}x 빠름")
print(f"  - 전체 평균: {(avg_speedup_matrix + avg_speedup_dist + avg_speedup_filter)/3:.0f}x 가속!")
print("\n결론: NumPy 벡터화는 필수입니다!")

