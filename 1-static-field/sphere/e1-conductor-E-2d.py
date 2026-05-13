import numpy as np
import matplotlib.pyplot as plt

'''
Electric Field Lines for a Conducting Sphere in a Uniform Field
'''

# Grid
x = np.linspace(-5, 5, 1000)
y = np.linspace(-5, 5, 1000) 
X, Y = np.meshgrid(x, y)
R = np.sqrt(X**2 + Y**2)
Theta = np.arctan2(Y, X)

# Avoid division by zero
R[R == 0] = 1e-10

# Parameters
E0 = 1.0
a = 2.0  # Sphere radius

# Potential \varphi
phi = -E0 * R * np.cos(Theta) + E0 * a**3 * np.cos(Theta)/R**2

# Compute electric field components (E = -grad(phi))
Ex = -np.gradient(phi, axis=1)  # x-direction
Ey = -np.gradient(phi, axis=0)  # y-direction

# Mask the field inside the sphere to hollow out the streamlines
mask = R < a
Ex[mask] = np.nan
Ey[mask] = np.nan

# Plot field lines
plt.figure(figsize=(8, 8))

# Streamplot
plt.streamplot(X, Y, Ex, Ey, color='black', linewidth=1, density=0.6, broken_streamlines=False, zorder=0)

# Draw the conducting sphere
circle = plt.Circle((0, 0), a, linewidth=2, edgecolor='black', facecolor='white', fill=True, zorder=2)
plt.gca().add_patch(circle)

plt.xlabel('x')
plt.ylabel('y')
plt.title(r'$\vec{E}$ around a Conducting Sphere in $\vec{E_0}$')
plt.grid(False)
plt.axis('equal')
plt.savefig("images/conductor_E-2d.png")
plt.show()
