import matplotlib.pyplot as plt
import numpy as np
import re


def smooth(data, weight=0.85):
    """
    学术界常用的指数移动平均 (EMA) 平滑算法
    weight 越高，曲线越平滑 (范围 0-1)
    """
    scalar = data[0]
    smoothed = []
    for value in data:
        scalar = scalar * weight + value * (1 - weight)
        smoothed.append(scalar)
    return smoothed


def draw_academic_figure():
    print("🕵️‍♂️ 正在提取分数...")
    rewards = []
    with open("train_log.txt", "r", encoding="utf-8") as f:
        for line in f:
            match = re.search(r"累计获得奖励:\s*(-?\d+\.\d+)", line)
            if match:
                rewards.append(float(match.group(1)))

    if not rewards:
        return

    # 生成平滑数据
    smoothed_rewards = smooth(rewards, weight=0.90)

    # 绘制学术级高颜值图表
    plt.figure(figsize=(10, 5), dpi=300)

    # 1. 画出原始的震荡背景 (半透明)
    plt.plot(rewards, color='#FFB07C', alpha=0.4, linewidth=1, label='Raw Reward')

    # 2. 画出平滑后的趋势线 (加粗)
    plt.plot(smoothed_rewards, color='#E84A27', linewidth=2, label='Smoothed Trend (EMA)')

    plt.title('Training Reward Convergence', fontsize=16, fontweight='bold', pad=15)
    plt.xlabel('Training Episodes', fontsize=12, fontweight='bold')
    plt.ylabel('Cumulative Reward', fontsize=12, fontweight='bold')

    # 优化网格和图例
    plt.grid(True, linestyle='--', alpha=0.5, color='gray')
    plt.legend(loc='lower right', frameon=True, fontsize=11)

    # 隐藏上方和右方的边框，更符合高水平论文规范
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    save_path = "../data/reward_convergence_academic.png"
    plt.savefig(save_path, bbox_inches='tight', transparent=False)
    print(f"🎉 学术版收敛图已生成！保存在: {save_path}")


if __name__ == "__main__":
    draw_academic_figure()