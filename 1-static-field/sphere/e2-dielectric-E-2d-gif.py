import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from pathlib import Path

'''
Electric Field Lines for a Dielectric Sphere in a Uniform Field
Dielectric -> Conductor limit as epsilon_2 increases
'''

# Parameters
E0 = 1.0
a = 2.0  # Sphere radius
epsilon_1 = 1.0
lim = 5.0
grid_n = 1000
density = 0.6
fps = 8

# Grid
x = np.linspace(-lim, lim, grid_n)
y = np.linspace(-lim, lim, grid_n)
X, Y = np.meshgrid(x, y)
r = np.sqrt(X**2 + Y**2)
theta = np.arctan2(Y, X)
cos_theta = np.cos(theta)
sin_theta = np.sin(theta)

r_safe = np.where(r == 0, 1e-10, r)
outside = r >= a
inside = ~outside
inv_r3 = 1.0 / (r_safe**3)

epsilon_ratio_values = np.logspace(0.0, 4.0, 20)
epsilon_2_values = epsilon_1 * epsilon_ratio_values

fig, ax = plt.subplots(1, 1, figsize=(8, 8))

def update(frame_index):
    epsilon_2 = float(epsilon_2_values[frame_index])
    epsilon_ratio_label = f"{epsilon_2 / epsilon_1:.1e}"
    k_out = (epsilon_2 - epsilon_1) / (2.0 * epsilon_1 + epsilon_2)
    k_in = 3.0 * epsilon_1 / (2.0 * epsilon_1 + epsilon_2)

    Er = np.empty_like(r, dtype=float)
    Eth = np.empty_like(r, dtype=float)

    Er[outside] = E0 * cos_theta[outside] + 2.0 * k_out * E0 * (a**3) * cos_theta[outside] * inv_r3[outside]
    Eth[outside] = -E0 * sin_theta[outside] + k_out * E0 * (a**3) * sin_theta[outside] * inv_r3[outside]

    Er[inside] = k_in * E0 * cos_theta[inside]
    Eth[inside] = -k_in * E0 * sin_theta[inside]

    Ex = Er * cos_theta - Eth * sin_theta
    Ey = Er * sin_theta + Eth * cos_theta

    ax.cla()
    ax.streamplot(
        X,
        Y,
        Ex,
        Ey,
        color="black",
        linewidth=1,
        density=float(density),
        broken_streamlines=False,
        zorder=0,
    )

    circle = plt.Circle((0, 0), a, linewidth=2, edgecolor="black", fill=False, zorder=2)
    ax.add_patch(circle)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(rf"Dielectric Sphere in Uniform $\vec{{E}}_0$  ($\epsilon_2/\epsilon_1$ = {epsilon_ratio_label})")
    ax.grid(False)
    ax.set_aspect("equal")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)

    return []


anim = FuncAnimation(fig, update, frames=len(epsilon_2_values), interval=1000.0 / fps, blit=False)

anim.save("images/e2-dielectric-E-2d-gif.gif", writer=PillowWriter(fps=fps), dpi=50)

plt.show()
