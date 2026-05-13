import numpy as np
import matplotlib.pyplot as plt

# 定义电场分量函数
def E_x(x, y, Q, a, b, d, epsilon_0):
    term1 = -Q * (x - a) / ((x - a)**2 + y**2)**(3/2)
    term2 = (a * Q / d) * (x - b) / ((x - b)**2 + y**2)**(3/2)
    return 1 / (4 * np.pi * epsilon_0) * (term1 + term2)

def E_y(x, y, Q, a, b, d, epsilon_0):
    term1 = Q * y / ((x - a)**2 + y**2)**(3/2)
    term2 = -(a * Q / d) * y / ((x - b)**2 + y**2)**(3/2)
    return 1 / (4 * np.pi * epsilon_0) * (term1 + term2)

# 设置参数
Q = 1  # 电荷量
a = 2  # 位置参数a
d = 4  # 距离参数d
b = a**2/d  # 位置参数b
epsilon_0 = 1  # 真空介电常数

# 创建网格
x = np.linspace(-5, 5, 1000)
y = np.linspace(-5, 5, 1000)
X, Y = np.meshgrid(x, y)

# 计算电场
Ex = np.vectorize(E_x)(X, Y, Q, a, b, d, epsilon_0)
Ey = np.vectorize(E_y)(X, Y, Q, a, b, d, epsilon_0)

# 绘制流线图
plt.figure(figsize=(8, 8))
plt.streamplot(X, Y, Ex, Ey, color='black',linewidth=1,density=0.5,broken_streamlines=True)
# 绘制导体球
circle = plt.Circle((0, 0), a, color='black', fill=False)
plt.gca().add_patch(circle)

# 绘制点电荷
plt.scatter(d, 0, color='red')
plt.text(d, 0, f'Q', color='red', ha='center')
plt.scatter(b, 0, color='blue')
plt.text(b, 0, f'Q\'', color='blue', ha='center')

plt.title('Electric Field Streamlines')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(False)
plt.show()
