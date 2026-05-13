import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()
parser.add_argument("--mu0", type=float, default=1.0)
parser.add_argument("--I", type=float, default=1.0)
parser.add_argument("--lim", type=float, default=2.0)
parser.add_argument("--n", type=int, default=25)
parser.add_argument("--zmin", type=float, default=0.2)
parser.add_argument("--zmax", type=float, default=2.0)
parser.add_argument("--vertices", type=str, default=None)
parser.add_argument("--radius", type=float, default=1.0)
parser.add_argument("--coil-n", type=int, default=420)
parser.add_argument("--out", type=str, default=None)
parser.add_argument("--no-show", action="store_true")
args = parser.parse_args()

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["SimSun", "Microsoft YaHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["font.size"] = 16
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["axes.unicode_minus"] = False

if args.vertices is None:
    t = np.linspace(0.0, 2.0 * np.pi, args.coil_n, endpoint=False)
    r = args.radius * (1.0 + 0.25 * np.cos(t) + 0.18 * np.cos(2.0 * t) - 0.12 * np.sin(3.0 * t))
    coil = np.stack([r * np.cos(t), 0.95 * r * np.sin(t), np.zeros_like(t)], axis=1)
    title = "Arbitrary Coil (Demo Shape)"
else:
    path = Path(args.vertices)
    if path.suffix.lower() == ".npy":
        coil = np.load(path)
    else:
        coil = np.loadtxt(path, dtype=float, delimiter=None)

    coil = np.asarray(coil, dtype=float)
    if coil.ndim != 2 or coil.shape[1] not in (2, 3):
        raise ValueError("vertices must be a 2D array with 2 or 3 columns (x,y[,z])")
    if coil.shape[1] == 2:
        coil = np.concatenate([coil, np.zeros((coil.shape[0], 1), dtype=float)], axis=1)
    title = "Arbitrary Coil (Loaded Vertices)"

coil_for_plot = np.vstack([coil, coil[0]])

x = np.linspace(-args.lim, args.lim, max(int(args.n), 3))
y = np.linspace(-args.lim, args.lim, max(int(args.n), 3))
z = np.linspace(args.zmin, args.zmax, max(int(args.n), 3))
dx = float(x[1] - x[0])
dy = float(y[1] - y[0])
dz = float(z[1] - z[0])
X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

omega = np.zeros_like(X, dtype=float)
v0 = coil[0]
for i in range(1, coil.shape[0] - 1):
    v1 = v0
    v2 = coil[i]
    v3 = coil[i + 1]

    r1x = v1[0] - X
    r1y = v1[1] - Y
    r1z = v1[2] - Z
    r2x = v2[0] - X
    r2y = v2[1] - Y
    r2z = v2[2] - Z
    r3x = v3[0] - X
    r3y = v3[1] - Y
    r3z = v3[2] - Z

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
    omega += 2.0 * np.arctan2(numer, denom)

phi = (args.I / (4.0 * np.pi)) * omega
dphidx, dphidy, dphidz = np.gradient(phi, dx, dy, dz, edge_order=2)
Bx = -args.mu0 * dphidx
By = -args.mu0 * dphidy
Bz = -args.mu0 * dphidz
Bmag = np.sqrt(Bx * Bx + By * By + Bz * Bz)

fig = plt.figure(figsize=(9, 7), constrained_layout=True)
ax = fig.add_subplot(111, projection="3d")

Xs = X.ravel()
Ys = Y.ravel()
Zs = Z.ravel()
Cs = Bmag.ravel()

sc = ax.scatter(Xs, Ys, Zs, c=Cs, s=10, cmap="viridis", alpha=0.85, linewidths=0.0)
ax.plot(coil_for_plot[:, 0], coil_for_plot[:, 1], coil_for_plot[:, 2], color="black", linewidth=2.0, zorder=3)
ax.set_title(title + "\n" + r"$|\vec{B}(x,y,z)|$ Distribution", fontsize=13)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
ax.set_box_aspect((1.0, 1.0, 1.0))

cbar = fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.02)
cbar.set_label(r"$|\vec{B}|$")

out_path = Path(args.out) if args.out is not None else Path("solid_angle_coil_arbitrary_B_3d.png")
fig.savefig(out_path, dpi=200)

if not args.no_show:
    plt.show()
