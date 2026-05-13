import numpy as np
import matplotlib.pyplot as plt

Q = 2.0  # Point charge magnitude
d = 8.0  # Distance from the point charge to the sphere surface
a = 4.0  # Sphere radius
b = a**2/d
# Create the grid
x = np.linspace(-10, 10, 100)
y = np.linspace(-10, 10, 100)
X, Y = np.meshgrid(x, y)

# Compute the potential contributions at each grid point
term1 = Q / np.sqrt(X**2 + Y**2 + a**2 - 2 * a * X)
term2 = (a * Q / d) / np.sqrt(X**2 + Y**2 + (a**2 / d)**2 - 2 * (a**2 / d) * X)

# Combine terms
phi = (term1 - term2)

# Electric field from the potential gradient
Ex = np.gradient(-phi, axis=1)
Ey = np.gradient(-phi, axis=0)

# Plot field lines
plt.figure(figsize=(8, 8))
plt.streamplot(X, Y, Ex, Ey, linewidth=1,density=0.4,color='black', cmap='viridis',broken_streamlines=False,zorder=0)

# Draw the conducting sphere
circle = plt.Circle((0, 0), a,linewidth=2, fill=True, edgecolor='black', facecolor='white')
plt.gca().add_patch(circle)

# Draw charges
plt.scatter(d, 0, color='red',s=100,zorder=5)
plt.text(d, -1, 'Q', color='red', ha='center', va='center',zorder=5)  # Adjust y position
plt.scatter(b, 0, color='blue',s=100,zorder=5)
plt.text(b, -1, "Q\'", color='blue', ha='center', va='center',zorder=5)  # Adjust y position

plt.title('Point Charge Near a Grounded Conducting Sphere')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(False)
plt.show()
