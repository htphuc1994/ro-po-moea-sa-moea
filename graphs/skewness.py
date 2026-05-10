import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skewnorm

# Parameters for the skewed normal distributions
a_values = [-5, 0, 5]  # Negative for left-skewed, 0 for normal, positive for right-skewed
labels = ['Negative Skew', 'Symmetric (Normal)', 'Positive Skew']

# Set up the figure and axes for the subplots
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

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
    # ax.plot(x, y, 's-', label='Probability density')
    ax.plot(x, y, label='Probability density')
    ax.axvline(mean, color='r', linestyle='--', label='Mean')
    ax.axvline(median, color='g', linestyle=':', label='Median')
    ax.axvline(mode, color='b', linestyle='-', label='Mode')

    ax.set_title(label)
    ax.set_xlabel('X')
    ax.set_ylabel('Probability density')
    ax.legend()

plt.tight_layout()
plt.show()