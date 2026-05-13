import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

# 定义参数
Q = 1.0  # 点电荷电荷量
a = 2.0  # 导体球半径
d = 4.0  # 点电荷到导体球中心的距离
b = a**2 / d  # 根据b的定义
epsilon_0 = 1

# 创建更高分辨率的均匀笛卡尔网格
x = np.linspace(-5, 5, 1000)  # 增加网格点数
y = np.linspace(-5, 5, 1000)  # 增加网格点数
X, Y = np.meshgrid(x, y)

# 计算极坐标
R = np.sqrt(X**2 + Y**2)
Theta = np.arctan2(Y, X)

# 处理 R=0 的情况，避免除零错误
R[R == 0] = 1e-10  # 避免除零错误

# 计算电势
phi = (1 / (4 * np.pi *epsilon_0)) * (
    (Q / np.sqrt(R**2 + a**2 - 2 * R * a * np.cos(Theta))) -
    (a * Q / d / np.sqrt(R**2 + b**2 - 2 * R * b * np.cos(Theta)))
)

# 处理无效值
phi[np.isnan(phi)] = 0

# 绘制电场线
plt.figure(figsize=(8, 8))
# plt.contourf(X, Y, phi, levels=50, cmap='seismic', norm=Normalize(vmin=-0.5, vmax=0.5))
# plt.colorbar(label='Electric Scalar Potential')

# 使用更高分辨率的网格绘制流线图
# 计算电场分量
U = np.gradient(phi)[1]  # dx 方向
V = np.gradient(phi)[0]  # dy 方向

# 处理无效值
U[np.isnan(U)] = 0
V[np.isnan(V)] = 0

# # 生成对称的起始点
# num_lines = 8  # 电场线的数量
# angles = np.linspace(0, 2 * np.pi, num_lines, endpoint=False)  # 均匀分布的角度
# start_points = np.array([[-5 * np.cos(angle), -5 * np.sin(angle)] for angle in angles])  # 从边界上均匀选择起始点

# print(start_points)
# 绘制流线
# plt.streamplot(X, Y, U, V, color='black', linewidth=1, start_points=start_points)

plt.streamplot(X, Y, U, V, color='black', linewidth=1,density=0.5,broken_streamlines=False)

# 绘制导体球
circle = plt.Circle((0, 0), a, color='black', fill=False)
plt.gca().add_patch(circle)

# 绘制点电荷
plt.scatter(d, 0, color='red')
plt.text(d, 0, f'Q', color='red', ha='center')
plt.scatter(b, 0, color='blue')
plt.text(b, 0, f'Q\'', color='blue', ha='center')

plt.xlabel('x')
plt.ylabel('y')
plt.title('Electric Field Lines for a Point Charge in the Presence of a Conducting Sphere')
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.show()