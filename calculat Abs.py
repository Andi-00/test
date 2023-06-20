import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense

# Generate the random values 
# Input shape (n, 2), Output shape (n, )
n = 500

x = np.random.uniform(-10, 10, n)
y = np.random.uniform(-10, 10, n)

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
model.add(Dense(1))

model.summary()



model.compile(optimizer='adam', loss='mse')
history = model.fit(l, values, epochs=100, verbose=0)


mse = history.history['loss']
epochs = range(1, len(mse) + 1)

# Plot the MSE versus epoch
plt.plot(epochs, mse, 'b-o')
plt.xlabel('Epoch')
plt.ylabel('Mean Squared Error')
plt.title('Mean Squared Error vs. Epoch')
plt.grid(True)
plt.show()



x = 120
y = 40

l = np.c_[x, y]

a = np.abs(x - y)
b = model.predict(l)
print(a)
print(b)