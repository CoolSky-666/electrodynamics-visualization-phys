import numpy as np
import matplotlib.pyplot as plt
import os

'''
Magnetic Field Lines for a Permeable Sphere in a Uniform Field
exactly the same as E field lines
'''

# Grid
x = np.linspace(-5, 5, 1000)
y = np.linspace(-5, 5, 1000)
X, Y = np.meshgrid(x, y)
r = np.sqrt(X**2 + Y**2)
theta = np.arctan2(Y, X)

# Parameters
B0 = 1.0
a = 2.0  # Sphere radius
mu_1 = 1.0
mu_2 = 4.0

H0 = B0 / mu_1

# Magnetic scalar potential phi_m
cos_theta = np.cos(theta)
k_out = (mu_2 - mu_1) / (2 * mu_1 + mu_2)
k_in = 3 * mu_1 / (2 * mu_1 + mu_2)

r_safe = np.where(r == 0, 1e-10, r)
outside = r >= a

phi_m = np.empty_like(r, dtype=float)
phi_m[outside] = -H0 * r[outside] * cos_theta[outside] + k_out * H0 * a**3 * cos_theta[outside] / (r_safe[outside] ** 2)
phi_m[~outside] = -k_in * H0 * r[~outside] * cos_theta[~outside]

# Compute magnetic field components (H = -grad(phi_m), B = mu H)
dx = x[1] - x[0]
dy = y[1] - y[0]
dphi_dy, dphi_dx = np.gradient(phi_m, dy, dx)
Hx = -dphi_dx
Hy = -dphi_dy

Bx = Hx.copy()
By = Hy.copy()
Bx[outside] *= mu_1
By[outside] *= mu_1
Bx[~outside] *= mu_2
By[~outside] *= mu_2

plt.figure(figsize=(8, 8))

plt.streamplot(X, Y, Bx, By, color='black', linewidth=1, density=0.6, broken_streamlines=False, zorder=0)

circle = plt.Circle((0, 0), a, linewidth=2, edgecolor='black', fill=False, zorder=2)
plt.gca().add_patch(circle)

plt.xlabel('x')
plt.ylabel('y')
plt.title(r'$\vec{B}$ around a Permeable Sphere in Uniform $\vec{B}_0$')
plt.grid(False)
plt.axis('equal')
images_dir = os.path.join(os.path.dirname(__file__), 'images')
os.makedirs(images_dir, exist_ok=True)
plt.savefig(os.path.join(images_dir, 'm1-dielectric-B-2d.png'), dpi=200, bbox_inches='tight')
plt.show()
