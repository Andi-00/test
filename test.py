import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
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
n = 200
f = np.random.uniform(1, 10, n)
a = np.random.uniform(1, 5, n)

x = np.arange(0, 10, 0.1)

result = np.c_[f, a]
data = np.array([np.array([np.array([x[i], a[j] * np.sin(f[j] * x[i])]) for i in range(len(x))]) for j in range(len(a))])

print(data.shape)
print(result.shape)

model = Sequential()

model.add(Dense(32, input_shape = (100, 2, ), activation = "relu"))
model.add(Dense(64, activation = "relu"))
model.add(Flatten())
model.add(Dense(64, activation = "relu"))
model.add(Dense(64, activation = "relu"))
model.add(Dense(128, activation = "relu"))
model.add(Dense(128, activation = "relu"))
model.add(Dense(128, activation = "relu"))
model.add(Dense(64, activation = "relu"))
model.add(Dense(64, activation = "relu"))
model.add(Dense(32, activation = "relu"))
model.add(Dense(2))


model.summary()
print(model.output_shape)

model.compile(optimizer = "adam", loss = 'mean_absolute_percentage_error')
history = model.fit(data, result, epochs = 60, verbose = 1)

mse = history.history['loss']
epochs = range(1, len(mse) + 1)

# Plot the MSE versus epoch
plt.figure(figsize = (16, 8))
           
plt.plot(epochs, mse, 'b-o', color = "crimson")
plt.xlabel('Epoch')
plt.ylabel('Mean Squared Error')
plt.title('Mean Squared Error vs. Epoch')
plt.grid(True)
plt.show()

f0 = np.arange(1, 11)
a0 = np.arange(1, 11)

test = lambda f, a : np.array([np.array([np.array([x[i], a * np.sin(f * x[i])]) for i in range(len(x))])])

grid = np.array([np.array([test(f, a) for f in f0]) for a in a0[:: -1]])
grid = tf.reshape(grid, (-1, 100, 2))

temp = model.predict(grid)
temp = tf.reshape(temp, (10, 10, 2))

f = temp[:, :, 0]
a = temp[:, :, 1]

t = np.arange(1, 11)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (18, 8), sharey = True)

sns.heatmap(f, ax = ax1, cmap = "mako", cbar_ax = None, yticklabels = t[::-1], xticklabels = t)
sns.heatmap(a, ax = ax2, cmap = "mako", yticklabels = t[::-1], xticklabels = t)

ax1.set_title("Frequenz")
ax2.set_title("Amplitude")
ax1.set_ylabel("Amplitude")
ax1.set_xlabel("Frequenz")
ax2.set_xlabel("Frequenz")

plt.show()

testf = np.array([np.array([np.array([f, a]) for f in f0]) for a in a0[:: -1]])[:, :, 0]
testa = np.array([np.array([np.array([f, a]) for f in f0]) for a in a0[:: -1]])[:, :, 1]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (18, 8), sharey = True)

sns.heatmap(abs(f - testf), ax = ax1, cmap = "mako", cbar_ax = None, yticklabels = t[::-1], xticklabels = t)
sns.heatmap(abs(a - testa), ax = ax2, cmap = "mako", yticklabels = t[::-1], xticklabels = t)

ax1.set_title("Frequenz")
ax2.set_title("Amplitude")
ax1.set_ylabel("Amplitude")
ax1.set_xlabel("Frequenz")
ax2.set_xlabel("Frequenz")

plt.show()