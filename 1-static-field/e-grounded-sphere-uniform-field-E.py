import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

R = 1.0
E0 = 1.0
lim = 3.0
grid = 401
density = 1.4
out_path = Path("e-grounded-sphere-uniform-field-E.png")
no_show = False

for arg in sys.argv[1:]:
    if arg == "--no-show":
        no_show = True
        continue
    if not arg.startswith("--") or "=" not in arg:
        continue
    key, value = arg[2:].split("=", 1)
    if key == "R":
        R = float(value)
    elif key == "E0":
        E0 = float(value)
    elif key == "lim":
        lim = float(value)
    elif key == "grid":
        grid = int(float(value))
    elif key == "density":
        density = float(value)
    elif key == "out":
        out_path = Path(value)

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["SimSun", "Microsoft YaHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["font.size"] = 16
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["axes.unicode_minus"] = False

x = np.linspace(-lim, lim, grid)
z = np.linspace(-lim, lim, grid)
X, Z = np.meshgrid(x, z)

r = np.sqrt(X * X + Z * Z)
eps = 1e-12
r_safe = np.maximum(r, eps)

cos_theta = Z / r_safe
sin_theta = X / r_safe

outside = r >= R

Er = np.zeros_like(r, dtype=float)
Eth = np.zeros_like(r, dtype=float)
rr = r_safe[outside]
ct = cos_theta[outside]
st = sin_theta[outside]
Er[outside] = E0 * ct * (1.0 + 2.0 * (R**3) / (rr**3))
Eth[outside] = -E0 * st * (1.0 - (R**3) / (rr**3))

Ex = Er * sin_theta + Eth * cos_theta
Ez = Er * cos_theta - Eth * sin_theta
Emag = np.sqrt(Ex * Ex + Ez * Ez)

fig, ax = plt.subplots(1, 1, figsize=(7.6, 7.0), constrained_layout=True)

vmax = float(np.nanpercentile(Emag[outside], 98.0)) if np.any(outside) else float(np.nanmax(Emag))
im = ax.contourf(X, Z, np.clip(Emag, 0.0, vmax), levels=40, cmap="viridis", zorder=0)

ax.streamplot(
    X,
    Z,
    Ex,
    Ez,
    color="black",
    linewidth=1.0,
    density=float(density),
    broken_streamlines=False,
    zorder=1,
)

t = np.linspace(0.0, 2.0 * np.pi, 500)
ax.plot(R * np.cos(t), R * np.sin(t), color="black", linewidth=2.0, zorder=2)

ax.set_title(r"Electric Field of a Grounded Conducting Sphere in Uniform $\vec{E}_0$", fontsize=13)
ax.set_xlabel("x")
ax.set_ylabel("z")
ax.set_aspect("equal")
ax.set_xlim(-lim, lim)
ax.set_ylim(-lim, lim)

cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
cbar.set_label(r"$|\vec{E}|$")

fig.savefig(out_path, dpi=200)

if not no_show:
    plt.show()
