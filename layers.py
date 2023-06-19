import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense

# Generate quadratic data with noise
np.random.seed(42)
X = np.random.uniform(low=-1, high=1, size=(100, 1))
y = 2 * X**2 + np.random.normal(scale=0.1, size=(100, 1))

# Create the model
model = Sequential()
model.add(Dense(10, input_shape=(1,), activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))

# Compile and train the model
model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=100, verbose=0)

# Generate test data
X_test = np.linspace(-1, 1, 100).reshape(-1, 1)

# Predict using the trained model
y_pred = model.predict(X_test)

# Plot the original data and the predicted values
plt.scatter(X, y, label='Original Data')
plt.plot(X_test, y_pred, 'r-', label='Predicted Values')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Regression on Quadratic Data')
plt.legend()
plt.show()