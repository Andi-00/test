import numpy as np
# import tensorflow as tf
import seaborn as sns
from matplotlib import pyplot as plt

sns.set()

x = np.arange(1, 5, 0.01)
noi = np.random.normal(0, 0.5, len(x))
y = 2 * x + noi


fig, ax = plt.subplots(figsize = (16, 8))

ax.scatter(x, y, color = "royalblue")
ax.set_xlabel("time")
ax.set_ylabel("position")

plt.show()
