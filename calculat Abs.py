import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense

plt.show(block = False)

# Generate the random values 
# Input shape (n, 2), Output shape (n, )
n = 1000

m = 200

x = np.random.uniform(-m, m, n)
y = np.random.uniform(-m, m, n)

values = np.abs(x - y)
values = tf.constant(values)

print(values)

l = np.c_[x, y]
l = tf.reshape(l, [n, -1])

print(l)

print(l.shape)
print(values.shape)


model = Sequential()
model.add(Dense(32, input_shape = (2, ), activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))

model.summary()



model.compile(optimizer='adam', loss='mse')
history = model.fit(l, values, epochs = 100, verbose=2)


mse = history.history['loss']
epochs = range(1, len(mse) + 1)

# Plot the MSE versus epoch
fig, ax = plt.subplots()

ax.plot(epochs, mse, 'b-o', color = "crimson")
ax.set_xlabel('Epoch')
ax.set_ylabel('Mean Squared Error')
ax.set_title('Mean Squared Error vs. Epoch')
ax.grid(True)
plt.show()

n = 100
grid = np.array([tf.reshape(np.array([np.array([i, j]) for i in range(n)]), [n, -1]) for j in range(n)])
grid = tf.reshape(grid, [len(grid) ** 2, -1])

predict = model.predict(grid)
predict = tf.reshape(predict, [n, n,])

comp = np.array([np.array([abs(i - j) for i in range(n)]) for j in range(n)])



ax = sns.heatmap(np.abs(predict - comp), cmap = "mako")
ax.invert_yaxis()
# plt.pcolormesh(predict)
plt.show()