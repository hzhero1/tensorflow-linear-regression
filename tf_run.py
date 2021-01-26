import numpy as np
import matplotlib.pyplot as plt
# from sklearn import linear_model
# from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
import pickle
import tensorflow as tf

a = np.array([])
# tf.compat.v1.disable_eager_execution()

np.random.seed(101)
tf.random.set_seed(101)
initializer = tf.random_normal_initializer(seed=1)
X_train = np.random.random((100, 10)) * 100
# Y_train = (2 * np.sum(X_train[:, :1], axis=1) + 3 * np.sum(X_train[:, 2:3], axis=1))[:, tf.newaxis]
Y_train = np.sum(X_train[:, :5], axis=1)[:, tf.newaxis]
X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, test_size=0.2)

print([(x, y) for x, y in zip(X_train[:10], Y_train[:10])])
X_train = tf.constant(X_train, dtype=tf.float32)
Y_train = tf.constant(Y_train, dtype=tf.float32)
X_test = tf.constant(X_test, dtype=tf.float32)
Y_test = tf.constant(Y_test, dtype=tf.float32)
print(X_train.shape)
print(Y_train.shape)

num_features = X_train.shape[1]
weight = tf.Variable(np.random.random((num_features, 1)), dtype=tf.float32)
b = 0


def model(X, W, b):
    return tf.tensordot(X, W, axes=1) + b


def mean_squared_error(y, y_pred):
    # return [tf.reduce_mean(tf.square(y_pred - y)), tf.reduce_mean(tf.abs(y_pred - y))
    return tf.reduce_mean(tf.square(y_pred - y))


def loss(x, y, w, b, alpha):
    # return tf.reduce_mean(tf.square(model(x, w, b) - y))
    return tf.reduce_mean(tf.square(model(x, w, b) - y)) + alpha * tf.norm(w, ord=1)
    # return tf.reduce_mean(tf.square(model(x, w, b) - y))  # + alpha * tf.square(tf.abs(w * 2 - 1) - 1)
    # return tf.reduce_mean(tf.square(model(x, w, b) - y)) + tf.square(tf.norm(2 * w - 1)) + alpha * tf.norm(w, ord=1)


def show_learning_curve(errors_train, errors_test, axis):
    figure = plt.figure()
    ax = figure.add_subplot(1, 1, 1)
    ax.plot(axis, errors_train, color="r", label="Training mse")
    ax.plot(axis, errors_test, color="b", label="Test mse")
    ax.set_xlabel("epoch")
    ax.set_ylabel("mse")
    ax.set_title("Learning curve")
    ax.legend()
    plt.show()


def show_plot(alphas, scores):
    figure = plt.figure()
    ax = figure.add_subplot(1, 1, 1)
    ax.plot(alphas, scores)
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(r"mse")
    ax.set_xscale("log")
    ax.set_title("Ridge")
    plt.show()


num_epochs = 3000
num_samples = X_train.shape[0]
batch_size = 128
learning_rate = 0.01
# alphas = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10]
# alphas.reverse()
alphas = [10]
weights = []
errors = []
errors_train = []
errors_test = []

optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)

for alpha in alphas:
    w = tf.Variable(tf.identity(weight), dtype=tf.float32)
    print("\n{}".format(w))
    for epoch in range(num_epochs):
        # print("\nStart of epoch %d" % (epoch,))
        for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
            with tf.GradientTape() as tape:
                tape.watch(w)
                loss_value = loss(x_batch_train, y_batch_train, w, b, alpha)
            grads = tape.gradient(loss_value, w)
            # print(grads)
            optimizer.apply_gradients(zip([grads], [w]))
            # w.assign_sub(learning_rate * grads)
            train_mse = mean_squared_error(y_batch_train, model(x_batch_train, w, b))
            test_mse = mean_squared_error(Y_test, model(X_test, w, b))
        # print(w)
        print("Training loss at epoch %d: %s" % (epoch, np.array(train_mse)))
        errors_train.append(train_mse)
        errors_test.append(test_mse)
        # print("Testing loss at epoch%d: %s" % (epoch, str(test_mse)))
        # if step % 200 == 0:
        #     print(
        #         "Training loss (for one batch) at step %d: %s"
        #         % (step, str(train_mse))
        #     )
        # print("Seen so far: %s samples" % ((step + 1) * 64))
        # print("\nepoch {}".format(epoch))
        # print("training mse: {}".format(train_mse))
        # print("test mse: {}".format(test_mse))

    print("\nalpha: {}".format(alpha))
    overall_mse = mean_squared_error(Y_train, model(X_train, w, b))
    print("training set mse: {}".format(np.array(overall_mse)))
    print("test set mse: {}".format(np.array(test_mse)))
    print("Weights: \n{}".format(w.value()))
    weights.append(w)
    errors.append(overall_mse)

axis = [i for i in range(1, num_epochs + 1)]
show_learning_curve(errors_train, errors_test, axis)
