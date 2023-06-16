import numpy as np
import tensorflow as tf
import seaborn as sns

x = np.arange(1, 5, 0.01)
noi = np.random.normal(0, 0.3, len(x))
y = 2 * x + noi

sns.set_theme()

tips = sns.load_dataset("tips")

print(tips.datatype)