import numpy as np
import matplotlib.pyplot as plt
from scipy.special import factorial

def plot_poisson_pmf(lambda_param=8, max_l=20):
    """绘制泊松分布的概率质量函数
    
    参数:
        lambda_param (float): 泊松分布参数λ
        max_l (int): 最大的l值
    """
    # TODO: 实现泊松分布概率质量函数的计算和绘制
    # 提示：
    # 1. 使用np.arange生成l值序列
    # 2. 使用给定公式计算PMF
    # 3. 使用plt绘制图形并设置标签
    l_values = np.arange(0, max_l+1)  # 生成0到max_l的整数序列
    pmf = (lambda_param ** l_values )* np.exp(-lambda_param) / factorial(l_values)  # 计算PMF
    
    plt.figure(figsize=(10, 6))
    plt.stem(l_values, pmf, linefmt='b-', markerfmt='bo', basefmt=' ')  # 绘制离散分布
    plt.title(f'Poisson Probability Mass Function (λ={lambda_param})')
    plt.xlabel('Number of Successes (l)')
    plt.ylabel('Probability p(l)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xticks(l_values)

def simulate_coin_flips(n_experiments=10000, n_flips=100, p_head=0.08):
    """模拟多组抛硬币实验
    
    参数:
        n_experiments (int): 实验组数N
        n_flips (int): 每组抛硬币次数
        p_head (float): 正面朝上的概率
        
    返回:
        ndarray: 每组实验中正面朝上的次数
    """
    # TODO: 实现多组抛硬币实验
    # 提示：
    # 1. 使用np.random.choice模拟硬币抛掷
    # 2. 统计每组实验中正面的次数
    return np.random.binomial(n=n_flips, p=p_head, size=n_experiments)

def compare_simulation_theory(n_experiments=10000, lambda_param=8):
    """比较实验结果与理论分布
    
    参数:
        n_experiments (int): 实验组数
        lambda_param (float): 泊松分布参数λ
    """
    # TODO: 实现实验结果与理论分布的对比
    # 提示：
    # 1. 调用simulate_coin_flips获取实验结果
    # 2. 计算理论分布
    # 3. 绘制直方图和理论曲线
    # 4. 计算并打印统计信息
    # 进行硬币抛掷模拟
    simulated_data = simulate_coin_flips(n_experiments)
    
    # 动态确定可视化范围
    data_max = simulated_data.max()
    l_values = np.arange(0, data_max + 1)
    
    # 计算理论泊松分布
    pmf = (lambda_param ** l_values) * np.exp(-lambda_param) / factorial(l_values)
    theory_counts = pmf * n_experiments
    
    # 设置直方图分箱
    bins = np.arange(-0.5, data_max + 1.5, 1)
    
    # 绘制图形
    plt.figure(figsize=(12, 6))
    plt.hist(simulated_data, bins=bins, 
             alpha=0.7, color='skyblue', 
             edgecolor='black', 
             density=False,
             label='Simulation Results')
    
    # 绘制理论曲线（转换为频数）
    plt.plot(l_values, theory_counts, 'r--o', 
             linewidth=2, markersize=6,
             label='Theoretical Poisson')
    
    # 添加统计信息
    sim_mean = np.mean(simulated_data)
    sim_var = np.var(simulated_data)
    binomial_var = 100 * 0.08 * (1 - 0.08)  # 真实二项分布的方差
    
    plt.title('Simulation vs Theoretical Distribution Comparison')
    plt.xlabel('Number of Heads')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # 打印统计信息
    print("\nStatistical Comparison:")
    print(f"Simulation Mean: {sim_mean:.4f} | Poisson Theory: {lambda_param}")
    print(f"Simulation Variance: {sim_var:.4f} | Poisson Theory: {lambda_param}")
    print(f"Binomial Theoretical Variance: {binomial_var:.4f}")


if __name__ == "__main__":
    # 设置随机种子
    np.random.seed(42)
    
    # 1. 绘制理论分布
    plot_poisson_pmf()
    
    # 2&3. 进行实验模拟并比较结果
    compare_simulation_theory()
    
    plt.show()
