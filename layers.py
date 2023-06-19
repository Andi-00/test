import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
import seaborn as sns

# Generate quadratic data with noise
np.random.seed(42)
X = np.random.uniform(low=-10, high=10, size=(200, 1))
y = 2 * X ** 2 + np.random.normal(scale=0.1, size=(200, 1))

# Create the model
model = Sequential()
model.add(Dense(15, input_shape=(1,), activation='relu'))
model.add(Dense(25, activation='relu'))
model.add(Dense(35, activation='relu'))
model.add(Dense(45, activation='relu'))
model.add(Dense(55, activation='relu'))
model.add(Dense(65, activation='relu'))
model.add(Dense(55, activation='relu'))
model.add(Dense(45, activation='relu'))
model.add(Dense(35, activation='relu'))
model.add(Dense(25, activation='relu'))
model.add(Dense(15, activation='relu'))
model.add(Dense(1))

model.summary()

# Compile and train the model
model.compile(optimizer='adam', loss='mse')
history = model.fit(X, y, epochs=200, verbose=0)


# Extract the MSE values and epoch numbers from the training history
mse = history.history['loss']
epochs = range(1, len(mse) + 1)

# Plot the MSE versus epoch
plt.plot(epochs, mse, 'b-o')
plt.xlabel('Epoch')
plt.ylabel('Mean Squared Error')
plt.title('Mean Squared Error vs. Epoch')
plt.grid(True)
plt.show()



# Generate test data
X_test = np.linspace(-10, 10, 200).reshape(-1, 1)

# Predict using the trained model
y_pred = model.predict(X_test)

# Plot
sns.set(style="whitegrid")
plt.figure(figsize=(8, 6))
sns.scatterplot(x=X.flatten(), y=y.flatten(), label='Original Data', color = "royalblue")
sns.lineplot(x=X_test.flatten(), y=y_pred.flatten(), color='crimson', label='Predicted Values')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Regression on Quadratic Data')
plt.legend()
plt.show()