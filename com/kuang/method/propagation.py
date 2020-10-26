import numpy as np
import com.kuang.method.active_method as am
from com.kuang.pojo.forward import Forward
from com.kuang.pojo.backward import Backward


def forward(W_l, B_l, A_l_1, activation):
    """
    本方法是正向传播的方法块

    :param W_l: l层参数W^[l]
    :param A_l_1: l层参数A^[l-1]
    :param B_l: l层参数B^[l]
    :param relu: Relu参数类型
    :return: 正向传播的A^[l]
    """

    if W_l.shape[1] == A_l_1.shape[0]:
        temp = np.dot(W_l, A_l_1)
        # 一矩阵的列数与二矩阵函数相同
        if temp.shape[0] == B_l.shape[0]:
            Z_l = temp + B_l
            if activation == "sigmoid":
                A_l = am.sigmoid(Z_l)
            elif activation == "tanh":
                A_l = am.tanh(Z_l)
            elif activation == "relu":
                A_l = am.relu(Z_l)
            elif activation == "leaking_relu":
                A_l = am.leaking_relu(Z_l)

            # print("temp:", temp)
            # print("W_l:", W_l)
            # print("A_l_1:", A_l_1)
            # print("B_l:", B_l)
            # print("Z_l:", Z_l)
            # print("A_l:", A_l)

            # print("Z_l:", Z_l)
            cache = Forward(Z_l=Z_l, W_l=W_l, B_l=B_l, A_l=A_l)
            return cache
        else:
            print("B_l矩阵维数错误 at forward")
    else:
        print("W_l,A_l_1矩阵维数错误 at forward")


def backward(dA_l, Z_l, A_l_1, activation, W_l):
    # print("dA_l",dA_l.shape)
    # print("A_l_1", A_l_1.shape)
    # print("W_l", W_l.shape)

    m = A_l_1.shape[1]

    # print("===================================================")
    # print("dA_l", dA_l.shape)
    # print("Z_l", Z_l.shape)
    # print("===================================================")
    if activation == "sigmoid":
        dZ_l = am.d_sigmoid(dA_l=dA_l, Z_l=Z_l)
    elif activation == "tanh":
        dZ_l = am.d_tanh(Z_l)
    elif activation == "relu":
        dZ_l = am.d_relu(dA_l=dA_l, Z_l=Z_l)
    elif activation == "leaking_relu":
        dZ_l = am.d_leaking_relu(Z_l)


    temp = np.transpose(A_l_1)
    if dZ_l.shape[1] == temp.shape[0]:
        dW_l = np.dot(dZ_l, temp) / m


    else:
        print("dZ_l或A_l_1矩阵维数错误 at backward")

    dB_l = np.sum(dZ_l, axis=1, keepdims=True) / m

    temp = np.transpose(W_l)
    # print("===================================================")
    # print("temp", temp.shape)
    # print("dZ_l", dZ_l.shape)
    # print("===================================================")
    if temp.shape[1] == dZ_l.shape[0]:
        dA_l_1 = np.dot(temp, dZ_l)
    else:
        print("W_l或dZ_l矩阵维数错误 at backward")

    cache = Backward(dW_l=dW_l, dB_l=dB_l, dA_l_1=dA_l_1)

    # print("dW_l", dW_l)
    # print("dB_l", dB_l)
    # print("dA_l_1", dA_l_1)

    return cache
