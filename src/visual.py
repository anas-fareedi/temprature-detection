import matplotlib.pyplot as plt
import numpy as np

# Values from 1-12 (representing months)
months = np.arange(1, 13)

sin_vals = np.sin(2 * np.pi * months / 12)
cos_vals = np.cos(2 * np.pi * months / 12)

# Plot points on a circle to visualize
fig, ax = plt.subplots()
ax.axis('equal')  # To make sure circle is not stretched

ax.scatter(sin_vals, cos_vals, color='teal')

for i, m in enumerate(months):
    ax.text(sin_vals[i], cos_vals[i], m, ha='center', va='center')

ax.set_title('Cyclical Representation of Months')
ax.set_xlabel('sin(2π * month / 12)')
ax.set_ylabel('cos(2π * month / 12)')
plt.show()
