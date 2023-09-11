import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense

plt.rcParams['pgf.rcfonts'] = False
plt.rcParams['font.serif'] = []
plt.rcParams['font.family'] = 'serif'
plt.rcParams['text.usetex'] = True
plt.rcParams['axes.formatter.useoffset'] = False
plt.rcParams['lines.linewidth'] = 1.5
plt.rcParams['errorbar.capsize'] = 2

plt.rcParams['grid.linewidth'] = 0.5
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.title_fontsize'] = 12
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['savefig.pad_inches'] = 0.1

#plt.rcParams['savefig.transparent'] = True
plt.rcParams['figure.figsize'] = (8, 4)



a = 8
b = 0.2

n = 200


x_train = np.random.uniform(-a, a, n)
noi_train = np.random.normal(0, 0.1, len(x_train))
y_train = x_train ** 2 * np.exp(-b * x_train ** 2) + noi_train

c = int(0.8 * n)
d = int(0.9 * n)

x_valid = x_train[c : d]
x_test = x_train[d :]
x_train = x_train[: c]

y_valid = y_train[c : d]
y_test = y_train[d :]
y_train = y_train[: c]

neurons = 128


model = Sequential()

model.add(Dense(neurons, input_shape = (1, ), activation = "relu"))
model.add(Dense(neurons, activation = "relu"))
model.add(Dense(neurons, activation = "relu"))
model.add(Dense(neurons, activation = "relu"))
model.add(Dense(neurons, activation = "relu"))
model.add(Dense(neurons, activation = "relu"))
model.add(Dense(neurons, activation = "relu"))
model.add(Dense(1))


def schedular(epoch, lr):
    if epoch < 10: return lr
    else: return lr * tf.math.exp(-0.001)

lr_schedule = tf.keras.callbacks.LearningRateScheduler(schedular)

early_stopping = tf.keras.callbacks.EarlyStopping(monitor = "val_loss", patience = 200, restore_best_weights = True, verbose = 1)


model.compile(optimizer = "adam", loss = "mse")

history1 = model.fit(x_train, y_train, validation_data = (x_valid, y_valid), callbacks = [lr_schedule, early_stopping], epochs = 1000, verbose = 2)

model1 = Sequential()

model1.add(Dense(neurons, input_shape = (1, ), activation = "relu"))
model1.add(Dense(neurons, activation = "relu"))
model1.add(Dense(neurons, activation = "relu"))
model1.add(Dense(neurons, activation = "relu"))
model1.add(Dense(neurons, activation = "relu"))
model1.add(Dense(neurons, activation = "relu"))
model1.add(Dense(neurons, activation = "relu"))
model1.add(Dense(neurons, activation = "relu"))
model1.add(Dense(neurons, activation = "relu"))
model1.add(Dense(1))

model1.compile(optimizer = "adam", loss = "mse")
history2 = model1.fit(x_train, y_train, validation_data = (x_valid, y_valid), epochs = 1000, verbose = 2)

histories = [history1, history2]
names = ["A", "B"]
models = [model, model1]

for i in range(len(histories)):

    history = histories[i]
    c = names[i]
    model = models[i]

    mse = history.history['loss']
    v_loss = history.history["val_loss"]
    epochs = range(1, len(mse) + 1)



    print(model.evaluate(x_test, y_test))



    x = np.linspace(-a, a, 1000)
    ypre = model.predict(x)

    # Plot the MSE versus epoch
    fig, ax = plt.subplots()

    ax.plot(epochs, mse, color = "crimson", label = "Training loss")
    ax.plot(epochs, v_loss, color = "royalblue", label = "Validation loss")
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Mean Squared Error')
    ax.legend()
    ax.set_title(c, y = 1.02)


    ax.set_yscale("log")

    ax.grid()

    plt.savefig("./loss_{}.png".format(c))


    # Plot of the results

    fig, ax = plt.subplots()

    ax.scatter(x_train, y_train, color = "crimson", label = "Train Data")
    ax.scatter(x_valid, y_valid, color = "royalblue", label = "Validation Data")

    ax.plot(x, ypre, color = "black", label = "Prediction", zorder = 15)

    ax.plot(x, x ** 2 * np.exp(-b * x ** 2), color = "black", ls = "--", label = "True $f(x)$", zorder = 10)



    # ax.set_ylim(mi - d / 10, ma + d / 10)


    ax.set_ylabel("$y$")
    ax.set_xlabel("$x$")
    ax.set_title(c, y = 1.02)

    ax.legend()
    ax.grid()

    plt.savefig("./results_{}.png".format(c))