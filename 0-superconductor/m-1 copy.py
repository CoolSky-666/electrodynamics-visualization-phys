import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

# 设置全局默认字体
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Times New Roman + SimSun'
plt.rcParams['font.size'] = 16

# 定义参数
Q = 1.0  # 磁单极子电荷
a = 4.0  # 超导体半径
d = 8.0  # 磁单极子到超导体中心的距离

# 创建网格
x = np.linspace(-10, 10, 1000)
y = np.linspace(-10, 10, 1000)
X, Y = np.meshgrid(x, y)

# 计算每个点的电场
# Q_image = -Q * a / d  # 镜像电荷
# R = np.sqrt((X - d)**2 + Y**2)  # 点电荷到各点的距离
# R_prime = np.sqrt((X - a**2/d)**2 + Y**2)  # 镜像电荷到各点的距离
# r = np.sqrt(X**2 + Y**2)  # 球心到各点的距离
term1 = Q / (4 * np.pi * np.sqrt(X**2 + Y**2 + d**2 - 2 * d * X))
term2 = (Q * a) / (4 * np.pi * d * np.sqrt(X**2+Y**2 + (a**2 / d)**2 - 2 * X * (a**2 / d)))
term3 = (Q / (4 * np.pi * a)) * np.log((np.sqrt(X**2 + Y**2) + X) / 
                                        ((X - a**2 / d) + np.sqrt(X**2 + Y**2 + (a**2 / d)**2 - 2 * X * (a**2 / d))))

phi = term1 + term2 - term3
# 计算电势
phi[np.isnan(phi)] = 0
# 计算电场（电势的梯度）


# 处理无效值
phi[np.isnan(phi)] = 0

Ex = np.gradient(-phi, axis=1)
Ey = np.gradient(-phi, axis=0)

# 绘制电场线
plt.figure(figsize=(8, 8))
plt.streamplot(X, Y, Ex, Ey, linewidth=1,density=0.35,color='black',broken_streamlines=False,zorder=0)

# 处理无效值
# U[np.isnan(U)] = 0
# V[np.isnan(V)] = 0

# 绘制超导体
circle = plt.Circle((0, 0), a, color='black',linewidth=2, fill=False)
plt.gca().add_patch(circle)

# 绘制磁单极子及其镜像
plt.scatter(d, 0, color='red', label='Q',s=100,zorder=5)
plt.text(d, -1, f'Q', color='red', ha='center',zorder=6)
plt.scatter(a**2/d, 0, color='red', label='Qa/d',s=100,zorder=5)
plt.text(a**2/d, -1, f'Qa/d', color='red', ha='center',zorder=6)
plt.plot([0,a**2/d],[0,0],linestyle='-',color = 'blue',lw=5,zorder=4)
plt.text(0.05, -1, f'-Q/a', color='blue', ha='center',zorder=6)


plt.xlabel('X')
plt.ylabel('Y')
plt.title('超导球体与磁单极子')
# plt.legend()
plt.grid(False)
plt.axis('equal')
plt.show()