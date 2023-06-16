import tensorflow as tf
from tensorflow import keras
import numpy as np

x = np.arange(1, 2, 0.2)
y = np.arange(30, 40, 2)
a = tf.Variable(x, dtype = float)
b = tf.Variable(y, dtype = float)

with tf.GradientTape() as another_outer_tape:
    with tf.GradientTape() as outer_tape:
        with tf.GradientTape() as tape:
            c = a ** 2 - b ** 2 + a * b

            dc = tape.gradient(c, [a, b])
        dcda = outer_tape.gradient(dc[0], [a, b])
    dcdb = another_outer_tape.gradient(dc[1], [a, b])


print(dc)
print(dcda)
print(dcdb)