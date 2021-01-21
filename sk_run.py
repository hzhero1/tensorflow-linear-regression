import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
import pickle
import tensorflow as tf

np.random.seed(101)
tf.random.set_seed(101)


def show_plot(alphas, scores):
    figure = plt.figure()
    ax = figure.add_subplot(1, 1, 1)
    ax.plot(alphas, scores)
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(r"mse")
    ax.set_xscale("log")
    ax.set_title("Ridge")
    plt.show()


# 正则化系数
alphas = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10]

pl = PolynomialFeatures(degree=2)
X_train = np.random.random((1000, 10))
# X_train = pl.fit_transform(X_train)
Y_train = np.sum(X_train, axis=1)
X_test = np.random.random((100, 9))
Y_test = np.sum(X_train, axis=1)
print("training data size: {}".format(np.shape(X_train)))
print(np.size(Y_train))
scores = []
errors = []
for alpha in alphas:
    lr = linear_model.Lasso(alpha=alpha)
    lr.fit(X_train, Y_train)
    print("alpha: {}".format(alpha))
    print("权重向量:\n%s\nbias: %.2f" % (lr.coef_, lr.intercept_))
    print("mse: %.2f" % np.mean((lr.predict(X_train) - Y_train) ** 2))
    print("预测性能得分: %.2f\n" % lr.score(X_train, Y_train))
    scores.append(lr.score(X_train, Y_train))
    errors.append(np.mean((lr.predict(X_train) - Y_train) ** 2))

show_plot(alphas, errors)

# if __name__ == '__main__':
#     pass;
