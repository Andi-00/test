import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten

plt.rcParams["axes.titlesize"] = 32
plt.rcParams["axes.labelsize"] = 30
plt.rcParams["axes.titleweight"] = "bold"
plt.rcParams["xtick.labelsize"] = 22
plt.rcParams["ytick.labelsize"] = 22
plt.rcParams["legend.fontsize"] = 22
plt.rcParams["axes.linewidth"] = 1.2
plt.rcParams["scatter.marker"] = "."
plt.rcParams["axes.grid"] = False

# Create the data
m = 500

n = 5000
f = np.random.uniform(0.5, 21, n)
a = np.random.uniform(0.5, 21, n)

x = np.linspace(0, 20, m)

result = np.c_[f, a]
data = np.array([np.array([np.array([x[i], a[j] * np.sin(f[j] * x[i])]) for i in range(len(x))]) for j in range(len(a))])

print(data.shape)
print(result.shape)

# model = Sequential()

# neurons = 128

# model.add(Dense(neurons, input_shape = (m, 2, ), activation = "relu"))
# model.add(Dense(neurons, activation = "relu"))
# model.add(Flatten())
# model.add(Dense(neurons, activation = "relu"))
# model.add(Dense(neurons, activation = "relu"))
# model.add(Dense(neurons, activation = "relu"))
# model.add(Dense(neurons, activation = "relu"))
# model.add(Dense(neurons, activation = "relu"))
# model.add(Dense(neurons, activation = "relu"))
# model.add(Dense(neurons, activation = "relu"))
# model.add(Dense(neurons, activation = "relu"))
# model.add(Dense(2))


# model.summary()
# print(model.output_shape)

# model.compile(optimizer = "adam", loss = 'mse')

model = keras.models.load_model("my_sin_model")

# history = model.fit(data, result, epochs = 10, verbose = 1)

# model.save("my_sin_model")

# mse = history.history['loss']
# epochs = range(1, len(mse) + 1)

# # Plot the MSE versus epoch
# plt.figure(figsize = (16, 8))
           
# plt.plot(epochs, mse, 'b-o', color = "crimson")
# plt.xlabel('Epoch')
# plt.ylabel('Mean Squared Error')
# plt.title('Mean Squared Error vs. Epoch')
# plt.grid(True)
# plt.savefig("mse_plot.png")

f0 = np.arange(1, 21)
a0 = np.arange(1, 21)

test = lambda f, a : np.array([np.array([np.array([x[i], a * np.sin(f * x[i])]) for i in range(len(x))])])

grid = np.array([np.array([test(f, a) for f in f0]) for a in a0[:: -1]])
grid = tf.reshape(grid, (-1, m, 2))

temp = model.predict(grid)
temp = tf.reshape(temp, (20, 20, 2))

f = temp[:, :, 0]
a = temp[:, :, 1]

t = np.arange(2, 21)[: : 2]



fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (20, 9), sharey = True)



sns.heatmap(f, ax = ax1, cmap = "mako", cbar_ax = None, yticklabels = t[::-1], xticklabels = t)
sns.heatmap(a, ax = ax2, cmap = "mako", yticklabels = t[::-1], xticklabels = t)

ax1.set_xticks(t - 0.5)
ax2.set_xticks(t - 0.5)
ax1.set_xticklabels(t)
ax2.set_xticklabels(t)

ax1.set_yticks(t - 1.5)
ax1.set_yticklabels(t[::-1])

ax1.set_title("Frequenz", y = 1.02)
ax2.set_title("Amplitude", y = 1.02)
ax1.set_ylabel("Amplitude")
ax1.set_xlabel("Frequenz")
ax2.set_xlabel("Frequenz")

plt.savefig("grid.png")

testf = np.array([np.array([np.array([f, a]) for f in f0]) for a in a0[:: -1]])[:, :, 0]
testa = np.array([np.array([np.array([f, a]) for f in f0]) for a in a0[:: -1]])[:, :, 1]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (20, 9), sharey = True)

sns.heatmap(abs(f - testf), ax = ax1, cmap = "mako", cbar_ax = None, yticklabels = t[::-1], xticklabels = t)
sns.heatmap(abs(a - testa), ax = ax2, cmap = "mako", yticklabels = t[::-1], xticklabels = t)

ax1.set_xticks(t - 0.5)
ax2.set_xticks(t - 0.5)
ax1.set_xticklabels(t)
ax2.set_xticklabels(t)

ax1.set_yticks(t - 1.5)
ax1.set_yticklabels(t[::-1])

ax1.set_title("Frequenz", y = 1.02)
ax2.set_title("Amplitude", y = 1.02)
ax1.set_ylabel("Amplitude")
ax1.set_xlabel("Frequenz")
ax2.set_xlabel("Frequenz")

plt.savefig("delta_grid.png")