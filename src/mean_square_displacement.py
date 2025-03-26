import numpy as np
import matplotlib.pyplot as plt

def random_walk_finals(num_steps=1000, num_walks=1000):
    """生成多个二维随机游走的终点位置
    
    通过模拟多次随机游走，每次在x和y方向上随机选择±1移动，
    计算所有随机游走的终点坐标。

    参数:
        num_steps (int, optional): 每次随机游走的步数. 默认值为1000
        num_walks (int, optional): 随机游走的次数. 默认值为1000
        
    返回:
        tuple: 包含两个numpy数组的元组 (x_finals, y_finals)
            - x_finals: 所有随机游走终点的x坐标数组
            - y_finals: 所有随机游走终点的y坐标数组
    """
    # TODO: 实现随机游走算法
    # 提示：
    # 1. 使用np.zeros初始化数组
    # 2. 使用np.random.choice生成随机步长
    # 3. 使用np.sum计算总位移
    x_finals = np.zeros(num_walks)
    y_finals = np.zeros(num_walks)
    for walk in range(num_walks):
        x, y = 0, 0
        
        # 内层循环：遍历每个步长
        for step in range(num_steps):
            # 生成x方向步长
            dx = np.random.choice([-1, 1])
            # 生成y方向步长
            dy = np.random.choice([-1, 1])
            
            # 更新位置
            x += dx
            y += dy
        
        # 记录最终位置
        x_finals[walk] = x
        y_finals[walk] = y
    
    return (x_finals, y_finals)

def calculate_mean_square_displacement():
    """计算不同步数下的均方位移
    
    对于预设的步数序列[1000, 2000, 3000, 4000]，分别进行多次随机游走模拟，
    计算每种步数下的均方位移。每次模拟默认进行1000次随机游走取平均。
    
    返回:
        tuple: 包含两个numpy数组的元组 (steps, msd)
            - steps: 步数数组 [1000, 2000, 3000, 4000]
            - msd: 对应的均方位移数组
    """
    # TODO: 实现均方位移计算
    # 提示：
    # 1. 使用random_walk_finals获取终点坐标
    # 2. 计算位移平方和
    # 3. 使用np.mean计算平均值
    steps = np.array([1000, 2000, 3000, 4000])
    msd = []
    
    for n in steps:
        x, y = random_walk_finals(num_steps=n)
        r_sq = x**2 + y**2
        msd.append(np.mean(r_sq))
    
    return steps, np.array(msd)


def analyze_step_dependence():
    """分析均方位移与步数的关系，并进行最小二乘拟合
    
    返回:
        tuple: (steps, msd, k)
            - steps: 步数数组
            - msd: 对应的均方位移数组
            - k: 拟合得到的比例系数
    """
    # TODO: 实现数据分析
    # 提示：
    # 1. 调用calculate_mean_square_displacement获取数据
    # 2. 使用最小二乘法拟合 msd = k * steps
    # 3. k = Σ(N·msd)/Σ(N²)
steps, msd = calculate_mean_square_displacement()
    k = np.sum(steps * msd) / np.sum(steps**2)
    return steps, msd, k

if __name__ == "__main__":
    # TODO: 完成主程序
    # 提示：
    # 1. 获取数据和拟合结果
    # 2. 绘制实验数据点和理论曲线
    # 3. 设置图形属性
    # 4. 打印数据分析结果
    steps, msd, k = analyze_step_dependence()
    
    plt.figure(figsize=(10,6))
    plt.scatter(steps, msd, s=100, c='red', edgecolor='black', label='模拟数据')
    plt.plot(steps, k*steps, 'b--', label=f'拟合曲线: k={k:.2f}')
    plt.plot(steps, 2*steps, 'g-', label='理论值: k=2')
    
    plt.title("双重循环实现的随机游走分析", fontsize=14)
    plt.xlabel("步数N", fontsize=12)
    plt.ylabel("均方位移", fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    print("双重循环实现结果：")
    print(f"拟合系数k = {k:.4f}")
    print(f"理论值k = 2.0")
    print(f"相对误差: {(k-2)/2*100:.2f}%")
    
    plt.show()
