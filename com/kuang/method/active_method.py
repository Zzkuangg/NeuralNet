import numpy as np


# np.exp传入向量时是将向量每个值求exp然后返回向量
def sigmoid(Z):
    A = 1 / (1 + np.exp(-Z))
    return A


def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))


def relu(x):
    return np.maximum(0, x)


def leaking_relu(x):
    return np.maximum(0.01 * x, x)


# 下面是激活函数的导数
def d_sigmoid(dA_l, Z_l):
    dZ_l = dA_l * sigmoid(Z_l) * (1.0 - sigmoid(Z_l))
    assert (dZ_l.shape == Z_l.shape)
    return dZ_l

def d_tanh(x):
    return 1.0 - tanh(x) * tanh(x)


def d_relu(dA_l, Z_l):
    dZ_l = np.array(dA_l, copy=True)
    dZ_l[Z_l <= 0] = 0
    assert (dZ_l.shape == Z_l.shape)
    return dZ_l


def d_leaking_relu(dA_l, Z_l):
    dZ_l = np.array(dA_l, copy=True)

    dZ_l[Z_l <= 0] = 0.01*Z_l
    assert (dZ_l.shape == Z_l.shape)
    return dZ_l
