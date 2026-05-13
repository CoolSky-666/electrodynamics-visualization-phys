import numpy as np
import matplotlib.pyplot as plt

'''
Electric Field Lines for a Dielectric Sphere in a Uniform Field
'''

# Grid
x = np.linspace(-5, 5, 1000)
y = np.linspace(-5, 5, 1000) 
X, Y = np.meshgrid(x, y)
r = np.sqrt(X**2 + Y**2)
theta = np.arctan2(Y, X)

# Parameters
E0 = 1.0
a = 2.0  # Sphere radius
epsilon_1 = 1.0
epsilon_2 = 4.0

# Potential \varphi
cos_theta = np.cos(theta)
k_out = (epsilon_2 - epsilon_1) / (2 * epsilon_1 + epsilon_2)
k_in = 3 * epsilon_1 / (2 * epsilon_1 + epsilon_2)

r_safe = np.where(r == 0, 1e-10, r)
outside = r >= a

phi = np.empty_like(r, dtype=float)
phi[outside] = -E0 * r[outside] * cos_theta[outside] + k_out * E0 * a**3 * cos_theta[outside] / (r_safe[outside] ** 2)
phi[~outside] = -k_in * E0 * r[~outside] * cos_theta[~outside]

# Compute electric field components (E = -grad(phi))
dx = x[1] - x[0]
dy = y[1] - y[0]
dphi_dy, dphi_dx = np.gradient(phi, dy, dx)
Ex = -dphi_dx
Ey = -dphi_dy

# Plot field lines
plt.figure(figsize=(8, 8))

# Streamplot
plt.streamplot(X, Y, Ex, Ey, color='black', linewidth=1, density=0.6, broken_streamlines=False, zorder=0)

# Draw the dielectric sphere
circle = plt.Circle((0, 0), a, linewidth=2, edgecolor='black', fill=False, zorder=2)
plt.gca().add_patch(circle)

plt.xlabel('x')
plt.ylabel('y')
plt.title(r'$\vec{E}$ around a Dielectric Sphere in $\vec{E}_0$')
plt.grid(False)
plt.axis('equal')
plt.show()
