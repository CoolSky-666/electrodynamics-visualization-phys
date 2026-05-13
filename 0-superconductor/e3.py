import numpy as np
import matplotlib.pyplot as plt

# 设置全局默认字体
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Times New Roman + SimSun'
plt.rcParams['font.size'] = 16

# 定义参数
p = 1  # 点电荷的电荷量
d = 7.0  # 点电荷到球体表面的距离
a = 5.0  # 球体的半径
b = a**2/d

# 创建网格
x = np.linspace(-10, 10, 100)
y = np.linspace(-10, 10, 100)
X, Y = np.meshgrid(x, y)

# 计算每个点的电场
# Q_image = -Q * a / d  # 镜像电荷
# R = np.sqrt((X - d)**2 + Y**2)  # 点电荷到各点的距离
# R_prime = np.sqrt((X - a**2/d)**2 + Y**2)  # 镜像电荷到各点的距离
# r = np.sqrt(X**2 + Y**2)  # 球心到各点的距离

# 计算电势
phi = (1/(4*np.pi*8.85e-12)) * (p*(X-b)/((X-b)**2+Y**2)**1.5-(a**3/d**3)*p*(X-d)/((X-d)**2+Y**2)**1.5)

# 计算电场（电势的梯度）
Ex = np.gradient(-phi, axis=1)
Ey = np.gradient(-phi, axis=0)

# 绘制电场线
plt.figure(figsize=(8, 8))
plt.streamplot(X, Y, Ex, Ey, linewidth=1,density=0.35,color='black', cmap='viridis',broken_streamlines=False,zorder=0)

# 绘制导体球
circle = plt.Circle((0, 0), a, color='black',linewidth=2, fill=False,zorder=1)
plt.gca().add_patch(circle)

# 添加电偶极子矢量箭头
dipole_length = 1  # 电偶极子矢量箭头的长度
quiver_x = d
quiver_y = 0
quiver_dx = dipole_length * np.cos(0)  # 45度方向
quiver_dy = dipole_length * np.sin(0)  # 45度方向
plt.quiver(d, 0, 1.5, 0, angles='xy', scale_units='xy',color='darkorange', scale=1,zorder=3,pivot = 'middle')
plt.quiver(b, 0, -1, 0,angles='xy', scale_units='xy', zorder=3,pivot = 'middle',scale=1, color='darkorange')
# 添加电偶极子矢量箭头
dipole_length2 = 1  # 电偶极子矢量箭头的长度


plt.text(b, -0.5, "p\'", color='darkorange', ha='center', va='center',zorder=5) 
plt.text(d, -0.5, "p", color='darkorange', ha='center', va='center',zorder=5) 


# 设置坐标轴范围和标签
# plt.legend()
plt.xlim(-10, 10)
plt.ylim(-10, 10)
plt.xlabel('X')
plt.ylabel('Y')
# plt.title('带电导体球与点电荷的电场线')
plt.grid(False)
plt.show()