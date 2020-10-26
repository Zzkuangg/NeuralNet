import numpy as np


def init_back(A_l, Y,Cache):
    L = len(Cache)
    m = A_l.shape[1]
    Y = Y.reshape(A_l.shape)

    dA_l = - (np.divide(Y, A_l) - np.divide(1 - Y, 1 - A_l))

    return dA_l


def loss(A_l, Y):
    m = Y.shape[1]

    # print("A_l",A_l)
    # print("Y", Y)

    if A_l.shape == Y.shape:
        cost = -np.sum(np.multiply(np.log(A_l), Y) + np.multiply(np.log(1.0 - A_l), 1.0 - Y)) / m
        cost = np.squeeze(cost)
        assert (cost.shape == ())
        return cost
    else:
        print("A_l或Y矩阵维数错误 at init_back")

