import argparse
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['SimSun', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['font.size'] = 16
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['axes.unicode_minus'] = False

def triangle_solid_angle(X, Y, z, v1, v2, v3):
    r1x = v1[0] - X
    r1y = v1[1] - Y
    r1z = v1[2] - z
    r2x = v2[0] - X
    r2y = v2[1] - Y
    r2z = v2[2] - z
    r3x = v3[0] - X
    r3y = v3[1] - Y
    r3z = v3[2] - z

    c23x = r2y * r3z - r2z * r3y
    c23y = r2z * r3x - r2x * r3z
    c23z = r2x * r3y - r2y * r3x

    numer = r1x * c23x + r1y * c23y + r1z * c23z

    r1 = np.sqrt(r1x * r1x + r1y * r1y + r1z * r1z)
    r2 = np.sqrt(r2x * r2x + r2y * r2y + r2z * r2z)
    r3 = np.sqrt(r3x * r3x + r3y * r3y + r3z * r3z)

    d12 = r1x * r2x + r1y * r2y + r1z * r2z
    d23 = r2x * r3x + r2y * r3y + r2z * r3z
    d31 = r3x * r1x + r3y * r1y + r3z * r1z

    denom = r1 * r2 * r3 + d12 * r3 + d23 * r1 + d31 * r2
    return 2.0 * np.arctan2(numer, denom)


def polygon_solid_angle(X, Y, z, vertices):
    v0 = vertices[0]
    omega = np.zeros_like(X, dtype=float)
    for i in range(1, len(vertices) - 1):
        omega += triangle_solid_angle(X, Y, z, v0, vertices[i], vertices[i + 1])
    return omega


def compute_phi_on_grid(X, Y, z, vertices, current=1.0):
    omega = polygon_solid_angle(X, Y, z, vertices)
    return (current / (4.0 * np.pi)) * omega


def compute_B_on_plane(X, Y, z0, vertices, current=1.0, mu0=1.0):
    dx = float(X[0, 1] - X[0, 0])
    dy = float(Y[1, 0] - Y[0, 0])
    dz = 0.02 * max(np.max(np.abs(X)), np.max(np.abs(Y)), 1.0)

    phi0 = compute_phi_on_grid(X, Y, z0, vertices, current=current)
    dphidy, dphidx = np.gradient(phi0, dy, dx)
    Bx = -mu0 * dphidx
    By = -mu0 * dphidy

    phi_p = compute_phi_on_grid(X, Y, z0 + dz, vertices, current=current)
    phi_m = compute_phi_on_grid(X, Y, z0 - dz, vertices, current=current)
    dphidz = (phi_p - phi_m) / (2.0 * dz)
    Bz = -mu0 * dphidz

    return phi0, Bx, By, Bz


def make_square(side_length):
    a = side_length / 2.0
    return np.array(
        [
            [-a, -a, 0.0],
            [a, -a, 0.0],
            [a, a, 0.0],
            [-a, a, 0.0],
        ],
        dtype=float,
    )


def make_circle(radius, n=240):
    t = np.linspace(0, 2.0 * np.pi, n, endpoint=False)
    x = radius * np.cos(t)
    y = radius * np.sin(t)
    z = np.zeros_like(x)
    return np.stack([x, y, z], axis=1)


def make_arbitrary(radius, n=420):
    t = np.linspace(0, 2.0 * np.pi, n, endpoint=False)
    r = radius * (1.0 + 0.25 * np.cos(t) + 0.18 * np.cos(2.0 * t) - 0.12 * np.sin(3.0 * t))
    x = r * np.cos(t)
    y = 0.95 * r * np.sin(t)
    z = np.zeros_like(x)
    return np.stack([x, y, z], axis=1)


def analytic_Bz_center_circle(mu0, current, radius, z):
    return mu0 * current * radius * radius / (2.0 * (radius * radius + z * z) ** 1.5)


def analytic_Bz_center_square(mu0, current, half_side, z):
    a = half_side
    return 2.0 * mu0 * current * a * a / (np.pi * (a * a + z * z) * np.sqrt(2.0 * a * a + z * z))


def plot_case(ax_phi, ax_Bz, ax_stream, X, Y, z0, vertices, title, current=1.0, mu0=1.0, annotate=None):
    phi, Bx, By, Bz = compute_B_on_plane(X, Y, z0, vertices, current=current, mu0=mu0)

    extent = [float(X.min()), float(X.max()), float(Y.min()), float(Y.max())]
    phi_levels = 21
    Bz_levels = 21

    ax_phi.contour(X, Y, phi, levels=phi_levels, colors='black', linewidths=0.8)
    ax_phi.plot(vertices[:, 0], vertices[:, 1], color='darkorange', linewidth=2)
    ax_phi.set_title(f'{title}\n$\\varphi_m$ 等势线', fontsize=13)
    ax_phi.set_xlabel('x')
    ax_phi.set_ylabel('y')
    ax_phi.set_aspect('equal')
    ax_phi.set_xlim(extent[0], extent[1])
    ax_phi.set_ylim(extent[2], extent[3])
    ax_phi.grid(False)

    ax_Bz.contourf(X, Y, Bz, levels=Bz_levels, cmap='seismic')
    ax_Bz.plot(vertices[:, 0], vertices[:, 1], color='black', linewidth=1.5)
    ax_Bz.set_title(f'{title}\n$B_z(x,y,z_0)$', fontsize=13)
    ax_Bz.set_xlabel('x')
    ax_Bz.set_ylabel('y')
    ax_Bz.set_aspect('equal')
    ax_Bz.set_xlim(extent[0], extent[1])
    ax_Bz.set_ylim(extent[2], extent[3])
    ax_Bz.grid(False)

    stride = 2
    ax_stream.streamplot(
        X[::stride, ::stride],
        Y[::stride, ::stride],
        Bx[::stride, ::stride],
        By[::stride, ::stride],
        color='black',
        linewidth=1,
        density=1.0,
        broken_streamlines=False,
    )
    ax_stream.plot(vertices[:, 0], vertices[:, 1], color='darkorange', linewidth=2)
    ax_stream.set_title(f'{title}\n$\\vec{{B}}$ 在平面内投影', fontsize=13)
    ax_stream.set_xlabel('x')
    ax_stream.set_ylabel('y')
    ax_stream.set_aspect('equal')
    ax_stream.set_xlim(extent[0], extent[1])
    ax_stream.set_ylim(extent[2], extent[3])
    ax_stream.grid(False)

    if annotate is not None:
        Bz_center = float(Bz[Bz.shape[0] // 2, Bz.shape[1] // 2])
        ax_Bz.text(
            0.0,
            extent[3] * 0.88,
            annotate(Bz_center),
            ha='center',
            va='center',
            fontsize=12,
            color='black',
            bbox={'boxstyle': 'round', 'facecolor': 'white', 'alpha': 0.8, 'edgecolor': 'none'},
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mu0', type=float, default=1.0)
    parser.add_argument('--I', type=float, default=1.0)
    parser.add_argument('--z0', type=float, default=0.6)
    parser.add_argument('--side', type=float, default=2.0)
    parser.add_argument('--radius', type=float, default=1.0)
    parser.add_argument('--grid', type=int, default=240)
    parser.add_argument('--lim', type=float, default=2.5)
    parser.add_argument('--save', action='store_true')
    parser.add_argument('--no-show', action='store_true')
    args = parser.parse_args()

    x = np.linspace(-args.lim, args.lim, args.grid)
    y = np.linspace(-args.lim, args.lim, args.grid)
    X, Y = np.meshgrid(x, y)

    square = make_square(args.side)
    circle = make_circle(args.radius)
    arbitrary = make_arbitrary(args.radius)

    fig, axes = plt.subplots(3, 3, figsize=(14, 14), constrained_layout=True)

    def ann_circle(Bz_center):
        Bz_ana = analytic_Bz_center_circle(args.mu0, args.I, args.radius, args.z0)
        return rf'中心：数值 {Bz_center:.4f}，解析 {Bz_ana:.4f}'

    def ann_square(Bz_center):
        a = args.side / 2.0
        Bz_ana = analytic_Bz_center_square(args.mu0, args.I, a, args.z0)
        return rf'中心：数值 {Bz_center:.4f}，解析 {Bz_ana:.4f}'

    plot_case(axes[0, 0], axes[0, 1], axes[0, 2], X, Y, args.z0, square, '方形线圈', current=args.I, mu0=args.mu0, annotate=ann_square)
    plot_case(axes[1, 0], axes[1, 1], axes[1, 2], X, Y, args.z0, circle, '圆形线圈', current=args.I, mu0=args.mu0, annotate=ann_circle)
    plot_case(axes[2, 0], axes[2, 1], axes[2, 2], X, Y, args.z0, arbitrary, '任意形状线圈', current=args.I, mu0=args.mu0, annotate=None)

    fig.suptitle(r'立体角磁标势法：$\varphi_m = \dfrac{I}{4\pi}\Omega$，$\vec{B}=-\mu_0\nabla \varphi_m$（在 $z=z_0$ 平面）', fontsize=16)

    if args.save:
        fig.savefig('solid_angle_coils.png', dpi=200)

    if not args.no_show:
        plt.show()


if __name__ == '__main__':
    main()
