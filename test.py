import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten

# Create the data
n = 200
f = np.random.uniform(2, 10, n)
a = np.random.uniform(1, 5, n)

x = np.arange(0, 10, 0.1)

result = np.c_[f, a]
data = np.array([np.array([np.array([x[i], a[j] * np.sin(f[j] * x[i])]) for i in range(len(x))]) for j in range(len(a))])

print(data.shape)
print(result.shape)

model = Sequential()

model.add(Dense(32, input_shape = (100, 2, ), activation = "relu"))
model.add(Flatten())
model.add(Dense(64, activation = "relu"))
model.add(Dense(128, activation = "relu"))
model.add(Dense(256, activation = "relu"))
model.add(Dense(128, activation = "relu"))
model.add(Dense(128, activation = "relu"))
model.add(Dense(256, activation = "relu"))
model.add(Dense(128, activation = "relu"))
model.add(Dense(128, activation = "relu"))
model.add(Dense(64, activation = "relu"))
model.add(Dense(64, activation = "relu"))
model.add(Dense(32, activation = "relu"))


model.add(Dense(2))


model.summary()
print(model.output_shape)

model.compile(optimizer = "adam", loss = "mse")
history = model.fit(data, result, epochs = 60, verbose = 1)

mse = history.history['loss']
epochs = range(1, len(mse) + 1)

# Plot the MSE versus epoch
plt.plot(epochs, mse, 'b-o', color = "crimson")
plt.xlabel('Epoch')
plt.ylabel('Mean Squared Error')
plt.title('Mean Squared Error vs. Epoch')
plt.grid(True)
plt.show()

f = np.random.uniform(1, 10)
a = np.random.uniform(1, 10)

test = np.array([np.array([np.array([x[i], a * np.sin(f * x[i])]) for i in range(len(x))])])
temp = model.predict(test)
print(temp)
print(f, a)