"""
Week 14 모든 예제 자동 실행 스크립트
All PINN examples runner
"""

import subprocess
import time
import os

# 실행할 스크립트 목록 (순서대로)
scripts = [
    "01_basic_pinn.py",
    "02_heat_equation_1d.py",
    "03_wave_equation_1d.py",
    "04_heat_equation_2d.py",
    "05_burgers_equation.py",
    "06_wave_equation_2d.py",
    "07_complex_boundary.py"
]

print("="*70)
print("Week 14: PINN 예제 전체 실행")
print("="*70)
print(f"\n총 {len(scripts)}개의 스크립트를 순차적으로 실행합니다.\n")

results = []

for i, script in enumerate(scripts, 1):
    print(f"\n[{i}/{len(scripts)}] {script} 실행 중...")
    print("-" * 70)
    
    start_time = time.time()
    
    try:
        # uv run으로 스크립트 실행
        result = subprocess.run(
            ["uv", "run", script],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            capture_output=False,  # 출력을 콘솔에 표시
            text=True
        )
        
        elapsed_time = time.time() - start_time
        
        if result.returncode == 0:
            status = "✅ 성공"
            results.append((script, True, elapsed_time))
            print(f"\n✅ {script} 완료 (실행 시간: {elapsed_time:.1f}초)")
        else:
            status = "❌ 실패"
            results.append((script, False, elapsed_time))
            print(f"\n❌ {script} 실패 (오류 코드: {result.returncode})")
            
    except Exception as e:
        elapsed_time = time.time() - start_time
        results.append((script, False, elapsed_time))
        print(f"\n❌ {script} 실행 중 예외 발생: {e}")
    
    print("-" * 70)

# 최종 결과 요약
print("\n" + "="*70)
print("실행 결과 요약")
print("="*70)

total_time = sum(r[2] for r in results)
success_count = sum(1 for r in results if r[1])

for script, success, elapsed in results:
    status = "✅" if success else "❌"
    print(f"{status} {script:30s} ({elapsed:.1f}초)")

print("\n" + "="*70)
print(f"성공: {success_count}/{len(scripts)}")
print(f"총 실행 시간: {total_time:.1f}초 ({total_time/60:.1f}분)")
print("="*70)

if success_count == len(scripts):
    print("\n🎉 모든 예제가 성공적으로 실행되었습니다!")
else:
    print(f"\n⚠️  {len(scripts) - success_count}개 예제에서 오류가 발생했습니다.")
