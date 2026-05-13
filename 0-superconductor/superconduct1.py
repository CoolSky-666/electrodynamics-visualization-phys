import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

# 定义参数
Q = 1.0  # 磁单极子电荷
a = 2.0  # 超导体半径
d = 3.0  # 磁单极子到超导体中心的距离

# 创建均匀的笛卡尔网格
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)

# 计算极坐标
R = np.sqrt(X**2 + Y**2)
Theta = np.arctan2(Y, X)

# 处理 R=0 的情况，避免除零错误
R[R == 0] = 1e-10  # 避免除零错误

# 计算公式 (7) 中的各个部分
term1 = Q * a / (4 * np.pi * d * np.sqrt(R**2 + (a**2/d)**2 - 2 * R * (a**2/d) * np.cos(Theta)))
term2 = - Q / (4 * np.pi * a) * np.log(
    (R * (1 + np.cos(Theta))) / 
    ((R * np.cos(Theta) - a**2/d) + np.sqrt(R**2 + (a**2/d)**2 - 2 * R * (a**2/d) * np.cos(Theta)))
)
# 原磁场
term3 =  Q/ (4 * np.pi *np.sqrt(R**2 + d**2 - 2 * R * d * np.cos(Theta)))
# 计算总磁标势
phi = term1 + term2 + term3

# 处理无效值
phi[np.isnan(phi)] = 0

# 绘制磁感线
plt.figure(figsize=(8, 8))
plt.contourf(X, Y, phi, levels=50, cmap='seismic', norm=Normalize(vmin=-0.5, vmax=0.5))
plt.colorbar(label='Magnetic Scalar Potential')

# 使用均匀的笛卡尔网格绘制流线图
# # 计算磁场分量
U = np.gradient(phi)[1]  # dx 方向
V = np.gradient(phi)[0]  # dy 方向

# 处理无效值
U[np.isnan(U)] = 0
V[np.isnan(V)] = 0

plt.streamplot(X, Y, U, V, color='black', linewidth=1, density=1)

# 绘制超导体
circle = plt.Circle((0, 0), a, color='blue', fill=False)
plt.gca().add_patch(circle)

# # 绘制磁单极子及其镜像
plt.scatter(d, 0, color='black', label='Source Monopole Q')
plt.scatter(a**2/d, 0, color='black', label='Image Monopole Qa/d')
plt.text(0, 0, f'-Q/a', color='black', ha='center')

plt.xlabel('x')
plt.ylabel('y')
plt.title('Magnetic Induction Lines for a Magnetic Monopole in the Presence of a Superconducting Sphere')
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.show()