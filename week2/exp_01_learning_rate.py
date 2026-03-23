import numpy as np
import matplotlib.pyplot as plt
import os

# 스스로 해보기 1: learning_rate를 바꿔보기
# 원본: 0.1 → 실험: 1.0 (너무 크면 어떻게 되는가?)
output_dir = 'outputs'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def loss_function(x):
    return x**2

def gradient(x):
    return 2 * x

# learning_rate 비교: 0.1 (원본) vs 1.0 (실험)
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
x_range = np.linspace(-5, 5, 100)

for ax, lr, title in zip(axes,
                          [0.1, 1.0],
                          ['learning_rate = 0.1 (원본)', 'learning_rate = 1.0 (너무 큼)']):
    current_x = -4.0
    steps = []
    for _ in range(20):
        steps.append((current_x, loss_function(current_x)))
        current_x = current_x - lr * gradient(current_x)

    steps = np.array(steps)
    ax.plot(x_range, loss_function(x_range), 'k-', label='Loss Function')
    ax.scatter(steps[:, 0], steps[:, 1], color='red', s=80, zorder=5)
    ax.plot(steps[:, 0], steps[:, 1], 'r--', label='Path')
    ax.text(steps[0, 0], steps[0, 1] + 1, 'Start', ha='center', color='red', fontweight='bold')
    ax.set_title(title)
    ax.set_xlabel('x')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True)
    ax.set_xlim(-5, 5)
    ax.set_ylim(-1, 20)
    print(f"[lr={lr}] 최종 x: {steps[-1, 0]:.4f}")

plt.suptitle('[실험 1] Learning Rate 비교: 0.1 vs 1.0', fontsize=14, fontweight='bold')
plt.tight_layout()
save_path = os.path.join(output_dir, 'exp_01_learning_rate.png')
plt.savefig(save_path)
print(f"\n저장: {save_path}")
print("관찰: lr=1.0이면 x=-4 → +4 → -4 → ... 수렴하지 않고 진동!")
