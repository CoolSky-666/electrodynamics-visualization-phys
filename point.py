import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 设置点电荷的位置和电荷量
q = 1.0  # 电荷量，单位可以是库仑
position = np.array([0, 0, 0])  # 点电荷的位置，坐标原点
k = 9e9  # 库仑常数

# 创建三维网格
x = np.linspace(-1, 1, 10)
y = np.linspace(-1, 1, 10)
z = np.linspace(-1, 1, 10)
X, Y, Z = np.meshgrid(x, y, z)

# 计算每个点到点电荷的距离
r = np.sqrt((X - position[0])**2 + (Y - position[1])**2 + (Z - position[2])**2)

# 计算电场强度
Ex = k * q * (X - position[0]) / (r**3)
Ey = k * q * (Y - position[1]) / (r**3)
Ez = k * q * (Z - position[2]) / (r**3)

# 创建图形
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 绘制电场线
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        for k in range(X.shape[2]):
            if r[i, j, k] > 0.1:  # 避免除以零
                ax.quiver(X[i, j, k], Y[i, j, k], Z[i, j, k], Ex[i, j, k], Ey[i, j, k], Ez[i, j, k], length=0.1, normalize=True, color='b')

# 设置坐标轴标签
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# 设置标题
ax.set_title('Electric Field Lines from a Point Charge')

# 显示图形
plt.show()