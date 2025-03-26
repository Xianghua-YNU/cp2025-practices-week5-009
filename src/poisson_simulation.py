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
    l_values = np.arange(max_l)
    pmf = (lambda_param ** l_values) * np.exp(-lambda_param) / factorial(l_values)
    
    plt.figure()
    plt.plot(l_values, pmf, 'bo', ms=8, label='Poisson PMF')
    plt.vlines(l_values, 0, pmf, colors='b', lw=5, alpha=0.5)
    plt.title(f'Poisson Probability Mass Function (λ={lambda_param})')
    plt.xlabel('l')
    plt.ylabel('Probability')
    plt.legend()
    return pmf

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
    n_flips = 100
    p_head = lambda_param / n_flips
    results = simulate_coin_flips(n_experiments, n_flips, p_head)
    
    max_observed = results.max()
    std_dev = np.sqrt(lambda_param)
    max_l_theory = max(max_observed, int(lambda_param + 4 * std_dev)) + 2
    l_values = np.arange(0, max_l_theory + 1)
    
    pmf = (lambda_param ** l_values) * np.exp(-lambda_param) / factorial(l_values)
    theory_counts = pmf * n_experiments
    
    bins = np.arange(-0.5, max_l_theory + 1.5)
    plt.hist(results, bins=bins, alpha=0.5, label='Simulation Results')
    plt.plot(l_values, theory_counts, 'r-', label='Theoretical Distribution')
    
    plt.xlabel('Number of Heads')
    plt.ylabel('Frequency')
    plt.title(f'Simulation vs Theory (λ={lambda_param})')
    plt.legend()
    
    sim_mean = np.mean(results)
    sim_var = np.var(results)
    print(f"Simulation Mean: {sim_mean:.2f}, Theory Mean: {lambda_param:.2f}")
    print(f"Simulation Variance: {sim_var:.2f}, Theory Variance: {lambda_param:.2f}")
if __name__ == "__main__":
    # 设置随机种子
    np.random.seed(42)
    
    # 1. 绘制理论分布
    plot_poisson_pmf()
    
    # 2&3. 进行实验模拟并比较结果
    compare_simulation_theory()
    
    plt.show()
