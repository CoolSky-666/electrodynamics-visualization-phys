import numpy as np
import matplotlib.pyplot as plt

# 设置全局默认字体
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Times New Roman + SimSun'
plt.rcParams['font.size'] = 16

# 定义电场分量函数
def E_x(x, y, Q, a, b, d, epsilon_0):
    r = np.sqrt(x**2 + y**2)
    theta = np.arcsin(y/r)
    # 计算分母
    denominator1 = (r**2 + a**2 - 2*r*a*np.cos(theta))**(3/2)
    denominator2 = (r**2 + b**2 - 2*r*b*np.cos(theta))**(3/2)
    # 计算径向分量 Er
    Er = (1 / (4 * np.pi * epsilon_0)) * (-Q * (r - a * np.cos(theta)) / denominator1 +
                                        (a * Q) / d * (r - b * np.cos(theta)) / denominator2)
    # 计算角向分量 Etheta
    Etheta = (1 / (4 * np.pi * epsilon_0)) * (-Q * (r * a * np.sin(theta)) / denominator1 +
                                            (a * Q) / d * (r * b * np.sin(theta)) / denominator2)
    return np.cos(theta)*Er-np.sin(theta)*Etheta

def E_y(x, y, Q, a, b, d, epsilon_0):
    r = np.sqrt(x**2 + y**2)
    theta = np.arcsin(y/r)
    # 计算分母
    denominator1 = (r**2 + a**2 - 2*r*a*np.cos(theta))**(3/2)
    denominator2 = (r**2 + b**2 - 2*r*b*np.cos(theta))**(3/2)
    # 计算径向分量 Er
    Er = (1 / (4 * np.pi * epsilon_0)) * (-Q * (r - a * np.cos(theta)) / denominator1 +
                                        (a * Q) / d * (r - b * np.cos(theta)) / denominator2)
    # 计算角向分量 Etheta
    Etheta = (1 / (4 * np.pi * epsilon_0)) * (-Q * (r * a * np.sin(theta)) / denominator1 +
                                            (a * Q) / d * (r * b * np.sin(theta)) / denominator2)
    return np.sin(theta)*Er+np.cos(theta)*Etheta

# 设置参数
Q = 1  # 电荷量
a = 2  # 位置参数a
d = 4  # 距离参数d
b = a**2 / d  # 位置参数b
epsilon_0 = 1  # 真空介电常数

# 创建网格
x = np.linspace(-5, 5, 500)
y = np.linspace(-5, 5, 500)
X, Y = np.meshgrid(x, y)

# 计算电场
Ex = np.vectorize(E_x)(X, Y, Q, a, b, d, epsilon_0)
Ey = np.vectorize(E_y)(X, Y, Q, a, b, d, epsilon_0)

# 绘制流线图
plt.figure(figsize=(8, 8))
plt.streamplot(X, Y, Ex, Ey, color='black', linewidth=1, density=0.5, broken_streamlines=False,zorder=0)
# 绘制导体球
circle = plt.Circle((0, 0), a, color='black', fill=False)
plt.gca().add_patch(circle)

# 绘制点电荷
plt.scatter(d, 0, color='red',s=100,zorder=5)
plt.text(d, -0.4, 'Q', color='red', ha='center', va='center',zorder=5)  # 调整 y 位置
plt.scatter(b, 0, color='blue',s=100,zorder=5)
plt.text(b, -0.4, "Q'", color='blue', ha='center', va='center',zorder=5)  # 调整 y 位置

plt.title('点电荷与接地导体球')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(False)
plt.show()