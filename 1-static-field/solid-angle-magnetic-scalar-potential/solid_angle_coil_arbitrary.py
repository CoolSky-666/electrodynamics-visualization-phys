import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def configure_matplotlib(font_size: int = 16) -> None:
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["SimSun", "Microsoft YaHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["font.size"] = font_size
    plt.rcParams["mathtext.fontset"] = "stix"
    plt.rcParams["axes.unicode_minus"] = False


def triangle_solid_angle(X: np.ndarray, Y: np.ndarray, z: float, v1: np.ndarray, v2: np.ndarray, v3: np.ndarray) -> np.ndarray:
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


def polygon_solid_angle(X: np.ndarray, Y: np.ndarray, z: float, vertices: np.ndarray) -> np.ndarray:
    v0 = vertices[0]
    omega = np.zeros_like(X, dtype=float)
    for i in range(1, len(vertices) - 1):
        omega += triangle_solid_angle(X, Y, z, v0, vertices[i], vertices[i + 1])
    return omega


def magnetic_scalar_potential(X: np.ndarray, Y: np.ndarray, z: float, vertices: np.ndarray, current: float) -> np.ndarray:
    omega = polygon_solid_angle(X, Y, z, vertices)
    return (current / (4.0 * np.pi)) * omega


def fields_on_plane(
    X: np.ndarray,
    Y: np.ndarray,
    z0: float,
    vertices: np.ndarray,
    current: float,
    mu0: float,
    dz: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    dx = float(X[0, 1] - X[0, 0])
    dy = float(Y[1, 0] - Y[0, 0])

    phi0 = magnetic_scalar_potential(X, Y, z0, vertices, current=current)
    dphidy, dphidx = np.gradient(phi0, dy, dx)
    Bx = -mu0 * dphidx
    By = -mu0 * dphidy

    phi_p = magnetic_scalar_potential(X, Y, z0 + dz, vertices, current=current)
    phi_m = magnetic_scalar_potential(X, Y, z0 - dz, vertices, current=current)
    dphidz = (phi_p - phi_m) / (2.0 * dz)
    Bz = -mu0 * dphidz

    return phi0, Bx, By, Bz


def make_demo_shape(radius: float, n: int) -> np.ndarray:
    t = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    r = radius * (1.0 + 0.25 * np.cos(t) + 0.18 * np.cos(2.0 * t) - 0.12 * np.sin(3.0 * t))
    x = r * np.cos(t)
    y = 0.95 * r * np.sin(t)
    z = np.zeros_like(x)
    return np.stack([x, y, z], axis=1)


def load_vertices(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        data = np.load(path)
    else:
        data = np.loadtxt(path, dtype=float, delimiter=None)

    data = np.asarray(data, dtype=float)
    if data.ndim != 2 or data.shape[1] not in (2, 3):
        raise ValueError("vertices must be a 2D array with 2 or 3 columns (x,y[,z])")

    if data.shape[1] == 2:
        z = np.zeros((data.shape[0], 1), dtype=float)
        data = np.concatenate([data, z], axis=1)
    return data


def plot_figure(
    X: np.ndarray,
    Y: np.ndarray,
    z0: float,
    vertices: np.ndarray,
    current: float,
    mu0: float,
    dz: float,
    title: str,
) -> tuple[plt.Figure, np.ndarray]:
    phi, Bx, By, Bz = fields_on_plane(X, Y, z0, vertices, current=current, mu0=mu0, dz=dz)
    extent = [float(X.min()), float(X.max()), float(Y.min()), float(Y.max())]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), constrained_layout=True)

    axes[0].contour(X, Y, phi, levels=21, colors="black", linewidths=0.8, zorder=0)
    axes[0].plot(vertices[:, 0], vertices[:, 1], color="black", linewidth=2.0, zorder=2)
    axes[0].set_title(rf"{title}" + "\n" + r"$\varphi_m(x,y,z_0)$ Contours", fontsize=13)
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    axes[0].set_aspect("equal")
    axes[0].set_xlim(extent[0], extent[1])
    axes[0].set_ylim(extent[2], extent[3])
    axes[0].grid(False)

    axes[1].contourf(X, Y, Bz, levels=21, cmap="seismic", zorder=0)
    axes[1].plot(vertices[:, 0], vertices[:, 1], color="black", linewidth=2.0, zorder=2)
    axes[1].set_title(rf"{title}" + "\n" + r"$B_z(x,y,z_0)$", fontsize=13)
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    axes[1].set_aspect("equal")
    axes[1].set_xlim(extent[0], extent[1])
    axes[1].set_ylim(extent[2], extent[3])
    axes[1].grid(False)

    stride = 2
    axes[2].streamplot(
        X[::stride, ::stride],
        Y[::stride, ::stride],
        Bx[::stride, ::stride],
        By[::stride, ::stride],
        color="black",
        linewidth=1.0,
        density=1.0,
        broken_streamlines=False,
        zorder=0,
    )
    axes[2].plot(vertices[:, 0], vertices[:, 1], color="black", linewidth=2.0, zorder=2)
    axes[2].set_title(rf"{title}" + "\n" + r"$\vec{B}_\parallel(x,y,z_0)$ Streamlines", fontsize=13)
    axes[2].set_xlabel("x")
    axes[2].set_ylabel("y")
    axes[2].set_aspect("equal")
    axes[2].set_xlim(extent[0], extent[1])
    axes[2].set_ylim(extent[2], extent[3])
    axes[2].grid(False)

    fig.suptitle(
        r"Solid-angle magnetic scalar potential: "
        r"$\varphi_m=\dfrac{I}{4\pi}\Omega$, "
        r"$\vec{B}=-\mu_0\nabla\varphi_m$ "
        r"(plane $z=z_0$)",
        fontsize=15,
    )
    return fig, axes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mu0", type=float, default=1.0)
    parser.add_argument("--I", type=float, default=1.0)
    parser.add_argument("--z0", type=float, default=0.6)
    parser.add_argument("--grid", type=int, default=240)
    parser.add_argument("--lim", type=float, default=2.5)
    parser.add_argument("--dz", type=float, default=None)
    parser.add_argument("--vertices", type=str, default=None)
    parser.add_argument("--radius", type=float, default=1.0)
    parser.add_argument("--coil-n", type=int, default=420)
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    configure_matplotlib()

    x = np.linspace(-args.lim, args.lim, args.grid)
    y = np.linspace(-args.lim, args.lim, args.grid)
    X, Y = np.meshgrid(x, y)

    if args.vertices is None:
        vertices = make_demo_shape(args.radius, n=args.coil_n)
        title = "Arbitrary Coil (Demo Shape)"
    else:
        vertices = load_vertices(Path(args.vertices))
        title = "Arbitrary Coil (Loaded Vertices)"

    dz = float(args.dz) if args.dz is not None else 0.02 * max(float(np.max(np.abs(X))), float(np.max(np.abs(Y))), 1.0)

    fig, _ = plot_figure(X, Y, args.z0, vertices, current=args.I, mu0=args.mu0, dz=dz, title=title)

    out_path = Path(args.out) if args.out is not None else Path("solid_angle_coil_arbitrary.png")
    fig.savefig(out_path, dpi=200)

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
