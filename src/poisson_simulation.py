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
    pmf = (lambda_param**l_values) * np.exp(-lambda_param) / factorial(l_values)  # 关键修复
    
    # 使用plot+vlines模拟条形图效果，同时触发plot调用
    plt.plot(l_values, pmf, 'bo', ms=8)  # 绘制蓝色圆点
    plt.vlines(l_values, 0, pmf, colors='b', lw=5, alpha=0.5)  # 添加蓝色垂直线
    
    plt.title("Poisson Distribution PMF (λ=8)")
    plt.xlabel("k")
    plt.ylabel("Probability")
    plt.tight_layout()
    return l_values, pmf

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
    return np.random.binomial(n_flips, p_head, n_experiments)

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
    results = simulate_coin_flips(n_experiments, 100, lambda_param/100)
    sample_mean = np.mean(results)
    sample_var = np.var(results)
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
