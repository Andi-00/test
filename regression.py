import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense


plt.rcParams["axes.titlesize"] = 32
plt.rcParams["axes.labelsize"] = 30
plt.rcParams["axes.titleweight"] = "bold"
plt.rcParams["xtick.labelsize"] = 22
plt.rcParams["ytick.labelsize"] = 22
plt.rcParams["legend.fontsize"] = 22
plt.rcParams["axes.linewidth"] = 1.2
plt.rcParams["scatter.marker"] = "."
plt.rcParams["axes.grid"] = True



a = 1
b = 0.1

n = 1000

x_train = np.linspace(-a, a, n)
noi_train = np.random.normal(0, 0.2, len(x_train))
y_train = x_train ** 2 * np.exp(-b * x_train ** 2) + noi_train


model = Sequential()

model.add(Dense(32, input_shape = (1, ), activation = "relu"))
model.add(Dense(32, activation = "relu"))
model.add(Dense(64, activation = "relu"))
model.add(Dense(32, activation = "relu"))
model.add(Dense(16, activation = "relu"))
model.add(Dense(16, activation = "relu"))
model.add(Dense(8, activation = "relu"))
model.add(Dense(1))


model.summary()
model.compile(optimizer = "adam", loss = "mse")

history = model.fit(x_train, y_train, epochs = 100, verbose = 2)

mse = history.history['loss']
epochs = range(1, len(mse) + 1)

a = 10
b = 0.1
n = 1000

x_test = np.linspace(-a, a, n)
noi = np.random.normal(0, 0.2, len(x_test))
y_test = x_test ** 2 * np.exp(-b * x_test ** 2) + noi


ypre = model.predict(x_test)

# Plot the MSE versus epoch
fig, ax = plt.subplots(figsize = (16, 9))

ax.plot(epochs, mse, 'b-o', color = "crimson")
ax.set_xlabel('Epoch')
ax.set_ylabel('Mean Squared Error')
ax.set_title('Mean Squared Error vs. Epoch', y = 1.02)
ax.grid(True)
plt.show()

# Plot of the results

fig, ax = plt.subplots(figsize = (16, 9))
ax.scatter(x_test, y_test, color = "royalblue", label = "Test Data", s = 50)
ax.scatter(x_train, y_train, color = "crimson", label = "Train Data", s = 50)

ax.plot(x_test, ypre, color = "black", lw = 3, label = "Prediction")

mi = min(y_test)
ma = max(y_test)

d = ma - mi

ax.set_ylim(mi - d / 10, ma + d / 10)


ax.set_ylabel("y")
ax.set_xlabel("x")
ax.set_title("Comparison - Test Data vs. ML Prediction", y = 1.02)

ax.legend()

plt.show()