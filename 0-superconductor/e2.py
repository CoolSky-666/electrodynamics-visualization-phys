import numpy as np
import matplotlib.pyplot as plt

# 设置全局默认字体
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Times New Roman + SimSun'
plt.rcParams['font.size'] = 16

# 定义参数
Q = 2.0  # 点电荷的电荷量
d = 8.0  # 点电荷到球体表面的距离
a = 4.0  # 球体的半径
Q0 = 0
# 创建网格
x = np.linspace(-10, 10, 100)
y = np.linspace(-10, 10, 100)
X, Y = np.meshgrid(x, y)

# 计算每个点的电场
Q_image = -Q * a / d  # 镜像电荷
R = np.sqrt((X - d)**2 + Y**2)  # 点电荷到各点的距离
R_prime = np.sqrt((X - a**2/d)**2 + Y**2)  # 镜像电荷到各点的距离
r = np.sqrt(X**2 + Y**2)  # 球心到各点的距离

# 计算电势
phi = (1/(4*np.pi*8.85e-12)) * (Q/R + Q_image/R_prime - (Q0-Q_image)/r)

# 计算电场（电势的梯度）
Ex = np.gradient(-phi, axis=1)
Ey = np.gradient(-phi, axis=0)

# 绘制电场线
plt.figure(figsize=(8, 8))
plt.streamplot(X, Y, Ex, Ey, linewidth=1,density=0.35,color='black', cmap='viridis',broken_streamlines=False,zorder=0)

# 绘制球体
circle = plt.Circle((0, 0), a, fill=False, edgecolor='black')
plt.gca().add_patch(circle)

# 标记点电荷和镜像电荷
plt.scatter([d], [0], color='red', label='点电荷 Q')
plt.scatter([a**2/d], [0], color='blue', label='镜像电荷 Q_image')

# 绘制导体球
circle = plt.Circle((0, 0), a, color='black',linewidth=2, fill=False,zorder=1)
plt.gca().add_patch(circle)

# 绘制点电荷
plt.scatter(d, 0, color='red',s=100,zorder=5)
plt.text(d, -1, 'Q', color='red', ha='center', va='center',zorder=5)  # 调整 y 位置
plt.scatter(a**2/d, 0, color='blue',s=100,zorder=5)
plt.text(a**2/d, -1, "-Qa/d", color='blue', ha='center', va='center',zorder=5)
plt.scatter(0, 0, color='red',s=100,zorder=5)
plt.text(0, -1, "Qa/d", color='red', ha='center', va='center',zorder=5) 


# 设置坐标轴范围和标签
# plt.legend()
plt.xlim(-10, 10)
plt.ylim(-10, 10)
plt.xlabel('X')
plt.ylabel('Y')
# plt.title('带电导体球与点电荷的电场线')
plt.grid(False)
plt.show()