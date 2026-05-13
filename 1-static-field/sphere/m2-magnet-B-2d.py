import numpy as np
import matplotlib.pyplot as plt
import os

'''
Magnetic Field Lines for a Uniformly Magnetized Sphere (Permanent Magnet)
'''

# Grid
x = np.linspace(-5, 5, 500)
y = np.linspace(-5, 5, 500)
X, Y = np.meshgrid(x, y)
r = np.sqrt(X**2 + Y**2)
theta = np.arctan2(Y, X)

# Parameters
M0 = 1.0
a = 2.0  # Sphere radius
mu0 = 1.0

# Magnetic scalar potential phi_m
cos_theta = np.cos(theta)

r_safe = np.where(r == 0, 1e-10, r)
outside = r >= a

phi_m = np.empty_like(r, dtype=float)
phi_m[outside] = (M0 * a**3 / 3) * cos_theta[outside] / (r_safe[outside] ** 2)
phi_m[~outside] = (M0 / 3) * r[~outside] * cos_theta[~outside]

# Compute magnetic field components (H = -grad(phi_m), B = mu0 H outside, B = mu0(H + M) inside)
dx = x[1] - x[0]
dy = y[1] - y[0]
dphi_dy, dphi_dx = np.gradient(phi_m, dy, dx)
Hx = -dphi_dx
Hy = -dphi_dy

Bx = Hx.copy()
By = Hy.copy()
Bx *= mu0
By *= mu0
Bx[~outside] += mu0 * M0

fig, axes = plt.subplots(1, 2, figsize=(14, 7))

axes[0].streamplot(X, Y, Hx, Hy, color='black', linewidth=1, density=0.6, broken_streamlines=False, zorder=0)
circle_H = plt.Circle((0, 0), a, linewidth=2, edgecolor='black', fill=False, zorder=2)
axes[0].add_patch(circle_H)
axes[0].annotate('', xy=(0.8 * a, 0), xytext=(0, 0), arrowprops=dict(arrowstyle='->', color='darkorange', linewidth=2), zorder=3)
axes[0].text(0.85 * a, 0.1 * a, r'$\vec{M}$', color='darkorange', fontsize=12, zorder=3)
axes[0].set_xlabel('x')
axes[0].set_ylabel('y')
axes[0].set_title(r'$\vec{H}$ of a Uniformly Magnetized Sphere')
axes[0].grid(False)
axes[0].axis('equal')

axes[1].streamplot(X, Y, Bx, By, color='black', linewidth=1, density=0.6, broken_streamlines=False, zorder=0)
circle_B = plt.Circle((0, 0), a, linewidth=2, edgecolor='black', fill=False, zorder=2)
axes[1].add_patch(circle_B)
axes[1].annotate('', xy=(0.8 * a, 0), xytext=(0, 0), arrowprops=dict(arrowstyle='->', color='darkorange', linewidth=2), zorder=3)
axes[1].text(0.85 * a, 0.1 * a, r'$\vec{M}$', color='darkorange', fontsize=12, zorder=3)
axes[1].set_xlabel('x')
axes[1].set_ylabel('y')
axes[1].set_title(r'$\vec{B}$ of a Uniformly Magnetized Sphere')
axes[1].grid(False)
axes[1].axis('equal')

plt.savefig('images/m2-magnet-B-2d.png', dpi=200, bbox_inches='tight')
plt.show()
