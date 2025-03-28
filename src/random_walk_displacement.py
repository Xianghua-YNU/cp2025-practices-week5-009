import numpy as np
import matplotlib.pyplot as plt

def random_walk_displacement(num_steps, num_simulations):
    """
    模拟随机行走并返回每次模拟的最终位移

    参数:
    num_steps (int): 随机行走的步数
    num_simulations (int): 模拟的次数

    返回:
    numpy.ndarray: 形状为(2, num_simulations)的数组，表示每次模拟的最终位移
    """
    # TODO: 检查输入参数的有效性
    if num_steps <= 0 or num_simulations <= 0:
        raise ValueError("num_steps and num_simulations must be positive integers.")
    # TODO: 实现随机行走算法
    # 提示：
    # 1. 使用 np.random.choice 生成随机步长 ([-1, 1])
    steps_x = np.random.choice([-1, 1], size=(num_simulations, num_steps))
    steps_y = np.random.choice([-1, 1], size=(num_simulations, num_steps))
    # 2. 生成形状为 (2, num_simulations, num_steps) 的数组
    # 3. 对步数维度求和得到最终位移
    final_displacements_x = steps_x.sum(axis=1)
    final_displacements_y = steps_y.sum(axis=1)
    #生成形状为 (2, num_simulations, num_steps) 的数组
    return np.array([final_displacements_x, final_displacements_y])
    

def plot_displacement_distribution(final_displacements, bins=30):
    """
    绘制位移分布直方图

    参数:
    final_displacements (list): 包含每次模拟最终位移的列表
    bins (int): 直方图的组数
    """
    # TODO: 实现位移分布的直方图绘制
    # 1. 计算每次模拟的最终位移
    # 2. 使用plt.hist绘制直方图
    plt.hist(final_displacements[0], bins=bins, density=True, alpha=0.7, color='blue', label='X Displacement')
    plt.hist(final_displacements[1], bins=bins, density=True, alpha=0.7, color='orange', label='Y Displacement')
    plt.title("Displacement Distribution")
    plt.xlabel("Displacement")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True)
    plt.show()
    # 3. 添加标题和标签

def plot_displacement_square_distribution(final_displacements, bins=30):
    """
    绘制位移平方分布直方图

    参数:
    final_displacements (list): 包含每次模拟最终位移的列表
    bins (int): 直方图的组数
    """
    # TODO: 实现位移平方分布的直方图绘制
    # 1. 计算位移平方
    square_displacements_x = final_displacements[0] ** 2
    square_displacements_y = final_displacements[1] ** 2
    # 2. 使用plt.hist绘制直方图
    plt.hist(square_displacements_x, bins=bins, density=True, alpha=0.7, color='blue', label='X Squared Displacement')
    plt.hist(square_displacements_y, bins=bins, density=True, alpha=0.7, color='orange', label='Y Squared Displacement')
    plt.title("Square of Displacement Distribution")
    plt.xlabel("Displacement Squared")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True)
    plt.show()
    # 3. 添加标题和标签

if __name__ == "__main__":
    # 可调整的参数
    num_steps = 1000  # 随机行走的步数
    num_simulations = 1000  # 模拟的次数
    bins = 30  # 直方图的组数

    # TODO: 完成主程序逻辑
    # 1. 调用random_walk_displacement获取模拟结果
    final_displacements = random_walk_displacement(num_steps, num_simulations)
    # 2. 绘制位移分布直方图
    plot_displacement_distribution(final_displacements, bins)
    # 3. 绘制位移平方分布直方图
    plot_displacement_square_distribution(final_displacements, bins)
