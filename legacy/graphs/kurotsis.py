import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from scipy.stats import kurtosis

x = np.linspace(-5, 5, 100)
fig, ax = plt.subplots(figsize=(7.5, 5))  # create axes after rcParams

distnames  = ['laplace', 'norm', 'uniform']
linestyles = ['-', '--', ':']

for i, name in enumerate(distnames):
    dist = getattr(stats, name)(loc=-2, scale=4) if name == 'uniform' else getattr(stats, name)
    y = dist.pdf(x)
    if name == 'uniform':
        label, color = 'Negative Kurtosis', 'tab:orange'
    elif name == 'laplace':
        label, color = 'Positive Kurtosis', 'tab:pink'
    else:
        label, color = 'Medium Kurtosis', 'tab:red'
    ax.plot(x, y, linestyle=linestyles[i], color=color, label=label)

# make labels and tick labels bigger
ax.set_xlabel(r'Random variable $x$', fontsize=22, labelpad=6)       # x-label size
ax.set_ylabel(r'Probability density $f(x)$', fontsize=22, labelpad=6) # y-label size
ax.tick_params(axis='both', which='major', labelsize=20, length=6, width=1.2)

ax.legend(frameon=False, fontsize=16)
ax.grid(True, which='major', ls='--', alpha=0.3)
plt.tight_layout()
plt.show()