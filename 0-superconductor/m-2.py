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
d = 7.0  # 磁单极子到超导体中心的距离
m = 1.0
b = a**2/d

# 创建网格
x = np.linspace(-10, 10, 1000)
y = np.linspace(-10, 10, 1000)
X, Y = np.meshgrid(x, y)

# 计算每个点的电场
# Q_image = -Q * a / d  # 镜像电荷
# R = np.sqrt((X - d)**2 + Y**2)  # 点电荷到各点的距离
# R_prime = np.sqrt((X - a**2/d)**2 + Y**2)  # 镜像电荷到各点的距离
# r = np.sqrt(X**2 + Y**2)  # 球心到各点的距离
term2 = (m / (4 * np.pi)) * ((X - d) / (np.power((X - d)**2 + Y**2, 1.5)))
term3 = (m / (4 * np.pi)) * ((X - a**2 / d)*a**3 / (d**3 * np.power((X - a**2 / d)**2 + Y**2, 1.5)))
phi = term2 - term3
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

# 添加磁偶极子矢量箭头
plt.quiver(d, 0, 2, 0, angles='xy', scale_units='xy',color='darkorange', scale=1,zorder=3,pivot ='middle')
plt.quiver(b, 0, -1.5, 0,angles='xy', scale_units='xy', zorder=3,pivot = 'middle',scale=1, color='darkorange')

# plt.scatter(b, 0, color='red',s=50,zorder=5)
plt.text(d, -0.8, f'm', color='darkorange', ha='center',zorder=6)
plt.text(b, -0.8, f'm\'', color='darkorange', ha='center',zorder=6)
# plt.text(b, 0.5, r'$-ma/d^2$', color='red', ha='center',zorder=6)

plt.xlabel('X')
plt.ylabel('Y')
# plt.title('超导球体与磁偶极子')
# plt.legend()
plt.grid(False)
plt.axis('equal')
plt.show()