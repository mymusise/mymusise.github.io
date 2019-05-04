from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow import keras
from matplotlib import pylab as plt
import random
import numpy as np


def make_samples():
    return np.random.randn(150), np.random.randn(150)


def target_sample(x, y):
    labels = []
    for _x, _y in zip(x, y):
        if _x - _y > 0 and _x + _y < 0:
            labels.append(0)
        elif _x - _y < 0 and _x + _y > 0:
            labels.append(1)
        else:
            labels.append(2)
    return labels


def group_sample_with_class(x, y, labels):
    dataset = list(zip(x, y, labels))
    groups = []
    groups.append(list(filter(lambda x: x[2] == 0, dataset)))
    groups.append(list(filter(lambda x: x[2] == 1, dataset)))
    groups.append(list(filter(lambda x: x[2] == 2, dataset)))
    return groups


x, y = make_samples()
labels = target_sample(x, y)
groups = group_sample_with_class(x, y, labels)
train_x = np.array(list(zip(x, y)))
p_x, p_y = make_samples()
predict_x = np.array(list(zip(p_x, p_y)))
predict_y = target_sample(p_x, p_y)


# groups = group_sample_with_class(x, y, labels)
# ax = plt.subplot()
# ax.plot([-3, 3], [-3, 3])
# ax.plot([-3, 3], [3, -3])
# ax.plot([d[0] for d in groups[0]], [d[1] for d in groups[0]], '*', color='green')
# ax.plot([d[0] for d in groups[1]], [d[1] for d in groups[1]], '*', color='red')
# ax.plot([d[0] for d in groups[2]], [d[1] for d in groups[2]], '*', color='black')
# plt.show()


layers = [
    tf.keras.layers.Dense(3, activation=tf.nn.tanh),
    # tf.keras.layers.Dense(2, activation=tf.nn.tanh),
    tf.keras.layers.Dense(3, activation=tf.nn.softmax)
]

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(2, ))
] + layers)
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_x, labels, epochs=2000)

for layer in layers:
    [w, b] = layer.trainable_variables
    print(f"w: {w.numpy()} \t b: {b.numpy()}")


predictions = model.predict_classes(predict_x)
for prediction, expect in zip(predictions, predict_y):
    print(prediction, expect)

p_groups = group_sample_with_class(p_x, p_y, predictions)
# ax = plt.subplot()
# ax.plot([-3, 3], [-3, 3])
# ax.plot([-3, 3], [3, -3])
# ax.plot([d[0] for d in p_groups[0]], [d[1]
#                                       for d in p_groups[0]], '*', color='green')
# ax.plot([d[0] for d in p_groups[1]], [d[1]
#                                       for d in p_groups[1]], '*', color='red')
# ax.plot([d[0] for d in p_groups[2]], [d[1]
#                                       for d in p_groups[2]], '*', color='black')
# plt.show()


from mpl_toolkits.mplot3d import Axes3D


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

high_x = []
[w, b] = layers[0].trainable_variables
for x in train_x:
    _x = []
    for _w, _b in zip(w.numpy().T, b.numpy().T):
        _x.append(np.tanh(_w[0] * x[0] + _w[1] * x[1] + _b))
    high_x.append(_x)

high_x = np.array(high_x)

for data, label in zip(high_x, labels):
    if label == 0:
        color = 'red'
    elif label == 1:
        color = 'green'
    elif label == 2:
        color = 'black'
    ax.plot([data[0]], [data[1]], [data[2]], '*', color=color)

[w, b] = layers[1].trainable_variables
# arrows = []
# for _w, _b in list(zip(w.numpy().T, b.numpy().T))[:1]:
#     ax.quiver([0], [0], [-_b], [1], [1], [(-_b - _w[0] - _w[1]) / _w[2]])
#     print(_w)


def draw_surface(a, b, c, d, ax):
    x = np.linspace(-1,1,10)
    y = np.linspace(-1,1,10)

    X,Y = np.meshgrid(x,y)
    Z = (d - a*X - b*Y) / c

    surf = ax.plot_surface(X, Y, Z)

draw_surface(*w.numpy().T[0], b[0], ax)
draw_surface(*w.numpy().T[1], b[1], ax)

plt.show()
