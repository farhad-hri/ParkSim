import numpy as np
import matplotlib.pyplot as plt

# Generate a base ellipse (scaled circle)
theta = np.linspace(0, 2 * np.pi, 100)
a, b = 3, 1.5  # Semi-major and semi-minor axes
ellipse = np.array([a * np.cos(theta), b * np.sin(theta)])

# Apply a shear transformation matrix to skew the ellipse in the first quadrant
shear_matrix = np.array([[1, -0.5],  # Shear in x-direction
                         [-0.3, 1]])  # Shear in y-direction

skewed_ellipse = shear_matrix @ ellipse

# Plot the skewed ellipse
plt.figure(figsize=(6, 6))
plt.plot(skewed_ellipse[0], skewed_ellipse[1], 'r', label="Skewed Ellipse")
plt.axhline(0, color='gray', linestyle='--')
plt.axvline(0, color='gray', linestyle='--')
plt.xlim(-4, 4)
plt.ylim(-4, 4)
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.title("Skewed Ellipse (Narrowing in First Quadrant)")
plt.legend()
plt.grid()
plt.axis('equal')
plt.show()