import matplotlib.pyplot as plt
import tensorflow as tf
import csv
import numpy as np


def normalize(train):
    train = np.array(train)
    mean, std = train.mean(), train.std()
    train = (train - mean) / std
    return train.tolist()


def get_data():
    """
    [月份，价格，火力发电量，成本，产量]
    :return:
    """
    data_x = []
    data_y = []
    with open("data2.csv", "r", encoding="utf-8") as fp:
        content = csv.reader(fp)
        print(content)
        for row in content:
            data_y.append(float(row[1]))
            x = [float(i) for i in row[2:]]
            data_x.append(x)
    data_x = np.array(data_x)
    data_y = np.array(data_y)
    data_y = normalize(data_y)
    for i in range(0, 3):
        data_x[:, i] = normalize(data_x[:, i])
    x = [i for i in range(12, 0, -1)]
    plt.plot(x, data_y)
    return np.array(data_x), np.array(data_y)


class SimpleLayer(tf.keras.layers.Layer):
    def __init__(self, units=3):
        super(SimpleLayer, self).__init__()
        self.units = units

    def build(self, input_shape):
        w_init = tf.random_normal_initializer()
        self.w = tf.Variable(
            initial_value=w_init(shape=(self.units,),
                                 dtype='float32'),
            trainable=True)
        b_init = tf.zeros_initializer()
        self.b = tf.Variable(
            initial_value=b_init(shape=(self.units,), dtype='float32'),
            trainable=True)

    def call(self, inputs, **kwargs):
        return tf.multiply(inputs, self.w) + self.b


def run():
    data_x, data_y = get_data()
    model = tf.keras.Sequential([
        SimpleLayer(3),
        tf.keras.layers.Dense(16, activation=tf.keras.activations.sigmoid),
        tf.keras.layers.Dense(8, activation=tf.keras.activations.sigmoid),
        tf.keras.layers.Dense(1)
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr=1e-3),
        loss=tf.losses.mean_squared_error,
        metrics=['mse']
    )

    model.build(input_shape=data_x.shape)

    model.fit(data_x, data_y, epochs=2000, validation_data=(data_x, data_y))

    x = [i for i in range(12, 0, -1)]
    y = model(data_x)
    plt.plot(x, y)
    plt.show()

    print(tf.keras.activations.softmax(tf.expand_dims(model.layers[0].w, axis=0)))


if __name__ == '__main__':
    run()