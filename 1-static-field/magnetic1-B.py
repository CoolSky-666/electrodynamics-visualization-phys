import numpy as np
import matplotlib.pyplot as plt

# 设置全局默认字体
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Times New Roman + SimSun'
plt.rcParams['font.size'] = 16

# 定义参数
B0 = 1.0   # 外加均匀磁场大小
mu1 = 1.0  # 圆柱外磁导率
mu2 = 5.0  # 圆柱内磁导率
R = 2.5    # 圆柱半径

alpha = (mu2 - mu1) / (mu1 + mu2)
B_inside = 2.0 * mu2 * B0 / (mu1 + mu2)
B_added_inside = B_inside - B0

# 创建网格
x = np.linspace(-7, 7, 600)
y = np.linspace(-7, 7, 600)
X, Y = np.meshgrid(x, y)
r2 = X**2 + Y**2
r = np.sqrt(r2)

# 分区：圆柱内外分别使用解析解
inside = r <= R
outside = ~inside

Bx0 = B0 * np.ones_like(X)
By0 = np.zeros_like(Y)

Bx = np.zeros_like(X)
By = np.zeros_like(Y)

# 圆柱内部：磁场均匀
Bx[inside] = B_inside
By[inside] = 0.0

# 圆柱外部：均匀场 + 二维偶极扰动
safe_r2 = np.where(outside, r2, 1.0)
safe_r4 = safe_r2**2

Bx[outside] = B0 * (
    1.0 + alpha * R**2 * (X[outside] ** 2 - Y[outside] ** 2) / safe_r4[outside]
)
By[outside] = B0 * (
    2.0 * alpha * R**2 * X[outside] * Y[outside] / safe_r4[outside]
)

# 将总场写成：原均匀场 + 感应附加场
Bx_added = Bx - Bx0
By_added = By - By0

# 只保留圆柱内的附加场，用于说明内部磁场叠加
Bx_inner_added = np.ma.array(Bx_added, mask=outside)
By_inner_added = np.ma.array(By_added, mask=outside)

# 只保留圆柱外的附加场，用于说明外部由等效磁荷产生的扰动
Bx_outer_added = np.ma.array(Bx_added, mask=inside)
By_outer_added = np.ma.array(By_added, mask=inside)


def decorate_axis(ax, title):
    ax.add_patch(plt.Circle((0, 0), R, color='black', linewidth=2, fill=False, zorder=2))
    ax.set_title(title, fontsize=15)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_xlim(-7, 7)
    ax.set_ylim(-7, 7)
    ax.set_aspect('equal')
    ax.grid(False)


fig, axes = plt.subplots(2, 2, figsize=(12, 12), constrained_layout=True)

# 1. 原外加均匀磁场
axes[0, 0].streamplot(
    X,
    Y,
    Bx0,
    By0,
    color='black',
    linewidth=1,
    density=0.55,
    broken_streamlines=False,
    zorder=0,
)
axes[0, 0].quiver(
    -5.8,
    5.8,
    1.8,
    0,
    angles='xy',
    scale_units='xy',
    scale=1,
    color='darkorange',
    zorder=3,
)
axes[0, 0].text(-4.9, 6.2, r'$\\vec{B}_0$', color='darkorange', ha='center')
decorate_axis(axes[0, 0], '原外加均匀磁场')

# 2. 圆柱内附加场：说明内部总场 = 原场 + 附加场
sample_x = np.linspace(-R * 0.75, R * 0.75, 6)
sample_y = np.linspace(-R * 0.75, R * 0.75, 6)
SX, SY = np.meshgrid(sample_x, sample_y)
inside_samples = SX**2 + SY**2 <= (0.82 * R) ** 2
axes[0, 1].quiver(
    SX[inside_samples],
    SY[inside_samples],
    B_added_inside * np.ones(np.count_nonzero(inside_samples)),
    np.zeros(np.count_nonzero(inside_samples)),
    angles='xy',
    scale_units='xy',
    scale=4.2,
    color='darkorange',
    zorder=3,
)
axes[0, 1].text(
    0,
    5.8,
    r'$\\Delta \\vec{B}_{\\mathrm{in}} = \\vec{B}_{\\mathrm{in}} - \\vec{B}_0$',
    color='darkorange',
    ha='center',
    fontsize=14,
)
axes[0, 1].text(
    0,
    -5.9,
    rf'$|\\Delta \\vec{{B}}_{{\\mathrm{{in}}}}| = {B_added_inside:.2f}$',
    color='darkorange',
    ha='center',
    fontsize=14,
)
decorate_axis(axes[0, 1], '圆柱内附加场（内部叠加项）')

# 3. 圆柱外等效磁荷产生的附加场
axes[1, 0].streamplot(
    X,
    Y,
    Bx_outer_added,
    By_outer_added,
    color='black',
    linewidth=1,
    density=0.75,
    broken_streamlines=False,
    zorder=0,
)
axes[1, 0].scatter(R, 0, color='red', s=80, zorder=4)
axes[1, 0].scatter(-R, 0, color='blue', s=80, zorder=4)
axes[1, 0].text(R, -0.55, r'$+q_m$', color='red', ha='center', fontsize=14)
axes[1, 0].text(-R, -0.55, r'$-q_m$', color='blue', ha='center', fontsize=14)
axes[1, 0].text(
    0,
    5.8,
    '外部附加场可理解为等效磁荷产生的扰动',
    color='darkorange',
    ha='center',
    fontsize=14,
)
decorate_axis(axes[1, 0], '圆柱外等效磁荷场')

# 4. 总场：原场 + 附加场
axes[1, 1].streamplot(
    X,
    Y,
    Bx,
    By,
    color='black',
    linewidth=1,
    density=0.55,
    broken_streamlines=False,
    zorder=0,
)
axes[1, 1].quiver(
    -5.8,
    5.8,
    1.8,
    0,
    angles='xy',
    scale_units='xy',
    scale=1,
    color='darkorange',
    zorder=3,
)
axes[1, 1].text(-4.7, 6.2, r'$\\vec{B} = \\vec{B}_0 + \\Delta \\vec{B}$', color='darkorange', ha='center', fontsize=14)
decorate_axis(axes[1, 1], '总磁场')

plt.suptitle('横向均匀磁场中的磁介质圆柱：原场与附加场的叠加', fontsize=18)
plt.show()
