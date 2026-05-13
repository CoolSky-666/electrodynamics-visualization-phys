import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

# 设置全局默认字体
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Times New Roman + SimSun'
plt.rcParams['font.size'] = 16

# 定义参数
Q = 1.0  # 磁单极子电荷
a = 2.0  # 超导体半径
d = 3.0  # 磁单极子到超导体中心的距离

# 创建更高分辨率的均匀笛卡尔网格
x = np.linspace(-5, 5, 400)  # 增加网格点数
y = np.linspace(-5, 5, 400)  # 增加网格点数
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
# plt.contourf(X, Y, phi, levels=50, cmap='seismic', norm=Normalize(vmin=-0.5, vmax=0.5))
# plt.colorbar(label='Magnetic Scalar Potential')

# 使用更高分辨率的网格绘制流线图
# 计算磁场分量
U = np.gradient(phi)[1]  # dx 方向
V = np.gradient(phi)[0]  # dy 方向

# 处理无效值
U[np.isnan(U)] = 0
V[np.isnan(V)] = 0

# 绘制流线
plt.streamplot(X, Y, U, V, color='black',linewidth=1,density=0.5,broken_streamlines=True)

# 绘制超导体
circle = plt.Circle((0, 0), a, color='blue', fill=False)
plt.gca().add_patch(circle)

# 绘制磁单极子及其镜像
plt.scatter(d, 0, color='red', label='Q',s=100,zorder=5)
plt.text(d, -0.5, f'Q', color='red', ha='center',zorder=6)
plt.scatter(a**2/d, 0, color='red', label='Qa/d',s=100,zorder=5)
plt.text(a**2/d, -0.5, f'Qa/d', color='red', ha='center',zorder=6)
plt.plot([0,a**2/d],[0,0],linestyle='-',color = 'blue',lw=2,zorder=4)
plt.text(a**2/2/d, -0.5, f'-Q/a', color='blue', ha='center',zorder=6)




plt.xlabel('x')
plt.ylabel('y')
plt.title('超导球体与磁单极子')
# plt.legend()
plt.grid(True)
plt.axis('equal')
plt.show()