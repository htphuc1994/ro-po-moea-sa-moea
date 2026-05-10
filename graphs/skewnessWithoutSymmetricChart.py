import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skewnorm

# Parameters for the skewed normal distributions
a_values = [-5, 5]  # Negative for left-skewed, 0 for normal, positive for right-skewed
labels = ['Negative Skew', 'Positive Skew']

# Set up the figure and axes for the subplots
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# Loop through each subplot to generate skewed normal distributions and mark mode, mean, and median
for ax, a, label in zip(axes, a_values, labels):
    rv = skewnorm(a)
    x = np.linspace(rv.ppf(0.01), rv.ppf(0.99), 1000)
    y = rv.pdf(x)

    # Calculating mean, median, and an approximation of mode
    mean = rv.mean()
    median = rv.median()
    mode = x[y.argmax()]  # The x value where the PDF has its maximum value

    # Plotting
    # ax.plot(x, y, 's-', label='Probability density')  # Scaling the line
    ax.plot(x, y, color='tab:pink', label=r'Probability $f(x)$')  # Scaling the line

    ax.axvline(mean, color='tab:orange', linestyle='dashed', label='Mean')
    ax.axvline(median, color='purple', linestyle='dotted', label='Median')
    ax.axvline(mode, color='tab:red', linestyle='dashdot', label='Mode')

    # ax.set_title(label)
    ax.set_xlabel(r'Random variable $x$', fontsize=22)
    ax.set_ylabel(r'Probability density $f(x)$', fontsize=22)
    ax.legend(frameon=False, fontsize=16)
    ax.grid(True, which='major', ls='--', alpha=0.3)
    ax.tick_params(axis='both', which='major', labelsize=20, length=6, width=1.2)

plt.tight_layout()
plt.show()