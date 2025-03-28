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
    l_values = np.arange(max_l)           # 生成0到max_l-1的整数序列（事件次数）
    pmf = (lambda_param**l_values * np.exp(-lambda_param)) / factorial(l_values)      # 计算每个l值对应的泊松概率（PMF公式）
    
    plt.figure(figsize=(10, 6))           # 创建画布
    plt.plot(l_values, pmf, 'bo-', label='Theoretical Distribution')                  # 绘制蓝色圆点连线图（理论分布）
    plt.title(f'Poisson Probability Mass Function (λ={lambda_param})')                # 设置标题（包含lambda值）
    plt.xlabel('l')                      # 设置坐标轴标签
    plt.ylabel('p(l)')
    plt.grid(True)                       #显示网格
    plt.legend()                         # 显示图例
    return pmf
# 定义硬币抛掷模拟函数
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
    results = []  #记录硬币正面朝上的次数# 进行n_experiments次独立实验
    for i in range(n_experiments):     # 每次实验抛n_flips次硬币，生成0/1序列
        coins = np.random.choice([0,1],n_flips, p=[1-p_head,p_head]) #抛硬币100次
        results.append(coins.sum())                                  # 统计正面（1）出现的次数并记录

    return np.array(results)          # 转换为numpy数组返回
# 定义比较实验与理论的函数
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
    results = simulate_coin_flips(n_experiments)   # 获取模拟结果（每组实验的正面次数）
    
    # 计算理论分布
    max_l = max(int(lambda_param * 2), max(results) + 1)   # 确定绘图范围（覆盖理论范围和实际结果）
    l_values = np.arange(max_l)
    pmf = (lambda_param**l_values * np.exp(-lambda_param)) / factorial(l_values)   # 计算理论泊松分布
    
    # 绘制直方图和理论曲线
    plt.figure(figsize=(12, 7))                                    # 创建画布
    plt.hist(results, bins=range(max_l+1), density=True, alpha=0.7,     
             label='Simulation Results', color='skyblue')             # 绘制直方图
    plt.plot(l_values, pmf, 'r-', label='Theoretical Distribution', linewidth=2)  # 绘制理论曲线
    
    plt.title(f'Poisson Distribution Comparison (N={n_experiments}, λ={lambda_param})')   # 设置图表元素
    plt.xlabel('Number of Heads')
    plt.ylabel('Frequency/Probability')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # 打印统计信息
    print(f"实验均值: {np.mean(results):.2f} (理论值: {lambda_param})")
    print(f"实验方差: {np.var(results):.2f} (理论值: {lambda_param})")
if __name__ == "__main__":
    # 设置随机种子
    np.random.seed(42)
    
    # 1. 绘制理论分布
    plot_poisson_pmf()
    
    # 2&3. 进行实验模拟并比较结果
    compare_simulation_theory()

    plt.show()
