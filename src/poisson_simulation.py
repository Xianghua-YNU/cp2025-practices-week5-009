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
    l_values = np.arange(0, max_l + 1)
    pmf = (lambda_param ** l_values) * np.exp(-lambda_param) / factorial(l_values)
    
    plt.figure()
    plt.stem(l_values, pmf, use_line_collection=True)
    plt.title(f"泊松分布概率质量函数 (λ={lambda_param})")
    plt.xlabel('l')
    plt.ylabel('概率')
    plt.xlim(-0.5, max_l + 0.5)
    plt.grid(True)
    plt.tight_layout()


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
    return np.random.binomial(n_flips, p_head, size=n_experiments)

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
    # 数值模拟
    M_counts = simulate_coin_flips(n_experiments=n_experiments)
    
    # 确定理论分布范围
    max_l_data = np.max(M_counts) if n_experiments > 0 else 0
    max_l_theory = int(lambda_param + 4 * np.sqrt(lambda_param))
    max_l = max(max_l_data, max_l_theory)
    
    # 计算理论分布
    l_values = np.arange(0, max_l + 1)
    pmf = (lambda_param ** l_values) * np.exp(-lambda_param) / factorial(l_values)
    theoretical_freq = pmf * n_experiments
    
    # 可视化
    plt.figure()
    plt.hist(M_counts, bins=np.arange(-0.5, max_l + 0.5), 
             density=False, alpha=0.7, label='实验结果')
    plt.plot(l_values, theoretical_freq, 'r-', marker='o', 
             markersize=4, linewidth=2, label='理论分布')
    
    plt.xlabel('正面次数')
    plt.ylabel('频数')
    plt.title(f'实验结果与理论分布对比 (N={n_experiments})')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    # 统计比较
    sample_mean = np.mean(M_counts)
    sample_var = np.var(M_counts)
    print(f"实验均值: {sample_mean:.4f}，理论均值: {lambda_param}")
    print(f"实验方差: {sample_var:.4f}，理论方差: {lambda_param}")

if __name__ == "__main__":
    # 设置随机种子
    np.random.seed(42)
    
    # 1. 绘制理论分布
    plot_poisson_pmf()
    
    # 2&3. 进行实验模拟并比较结果
    compare_simulation_theory()
    
    plt.show()
