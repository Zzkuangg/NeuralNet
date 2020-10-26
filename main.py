import numpy as np
import matplotlib.pyplot as plt
import lr_utils  # 参见资料包，或者在文章底部copy
import com.kuang.method.propagation as prop
import com.kuang.method.deficiency as defi

# 数据加载
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = lr_utils.load_dataset()
train_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
test_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T

# 样本
X_train = train_x_flatten / 255
Y_train = train_set_y
X_test = test_x_flatten / 255  # ????????????????
Y_test = test_set_y

# 参数设置,在这里更改参数
hidden_layer = 4  # 层数
units = (20, 7, 5, 1)  # 各隐藏层的隐藏单元个数,层数和元组长度要相同
learn_rate = 0.0075  # 学习率
iterations = 2500

# 迭代次数
activation = ("relu", "relu", "relu", "sigmoid")  # 指定每个正向传播块的激活函数,层数和元组长度要相同

# 以下参数根据上述参数自动初始化
W_list = []
B_list = []

# 缓存的数据
forward_cache = []
backward_cache = []
costs = []


# 初始化参数
def init(X_train):
    np.random.seed(3)
    W_list.append(np.random.randn(units[0], X_train.shape[0]) / np.sqrt(X_train.shape[0]))  # shape是0行1列
    B_list.append(np.zeros((units[0], 1)))
    if hidden_layer > 1:
        l = 1
        for l in range(1, hidden_layer):
            W_list.append(np.random.randn(units[l], units[l - 1]) / np.sqrt(units[l - 1]))
            B_list.append(np.zeros((units[l], 1)))


def predict(X_test, Y_test, W_list, B_list):
    m = X_test.shape[1]
    prediction = np.zeros((1, m))

    # 第一个正向传播快的计算
    # 因为第一个参数A_l_1来自训练矩阵,所以单独列出来
    # 将计算出来的结果放到缓存列表中
    forward_cache.append(prop.forward(W_l=W_list[0], B_l=B_list[0], A_l_1=X_test, activation=activation[0]))
    # 开始循环进行正向传播
    for i1 in range(1, hidden_layer):
        # 将计算出来的结果放到缓存列表中
        forward_cache.append(
            prop.forward(W_l=W_list[i1], B_l=B_list[i1], A_l_1=forward_cache[i1 - 1].A_l, activation=activation[i1]))
    for i in range(0, forward_cache[hidden_layer - 1].A_l.shape[1]):
        if forward_cache[hidden_layer - 1].A_l[0, i] > 0.5:
            prediction[0, i] = 1
        else:
            prediction[0, i] = 0
    print("准确度为: " + str(float(np.sum((prediction == Y_test)) / m)))


def BPNetWork(flag):
    init(X_train=X_train)

    for i0 in range(0, iterations):
        # 第一个正向传播快的计算
        # 因为第一个参数A_l_1来自训练矩阵,所以单独列出来
        # 将计算出来的结果放到缓存列表中
        forward_cache.append(prop.forward(W_l=W_list[0], B_l=B_list[0], A_l_1=X_train, activation=activation[0]))


        # 开始循环进行正向传播
        for i1 in range(1, hidden_layer):
        # 将计算出来的结果放到缓存列表中
            forward_cache.append(
                prop.forward(W_l=W_list[i1], B_l=B_list[i1], A_l_1=forward_cache[i1 - 1].A_l, activation=activation[i1]))

        # 计算损失
        AL = forward_cache[len(forward_cache) - 1].A_l
        cost = defi.loss(A_l=AL, Y=Y_train)

        # 反向传播初始化
        dA_l = defi.init_back(A_l=AL, Y=Y_train, Cache=forward_cache)

        # 将计算出来的结果放到缓存列表中,不过是逆序
        backward_cache.append(prop.backward(dA_l=dA_l, Z_l=forward_cache[len(forward_cache) - 1].Z_l,
                                            A_l_1=forward_cache[len(forward_cache) - 2].A_l,
                                            activation=activation[len(activation) - 1],
                                            W_l=forward_cache[len(forward_cache) - 1].W_l))

        for i2 in range(1, hidden_layer - 1):
            # 将计算出来的结果放到缓存列表中,不过是逆序
            backward_cache.append(
                prop.backward(dA_l=backward_cache[i2 - 1].dA_l_1, Z_l=forward_cache[len(forward_cache) - 1 - i2].Z_l,
                              A_l_1=forward_cache[len(forward_cache) - 2 - i2].A_l,
                              activation=activation[len(activation) - 1 - i2],
                              W_l=forward_cache[len(forward_cache) - 1 - i2].W_l))

        #反向传播最后一项也要单独列出来
        backward_cache.append(
            prop.backward(dA_l=backward_cache[len(backward_cache) - 1].dA_l_1, Z_l=forward_cache[0].Z_l,
                          A_l_1=X_train,
                          activation=activation[0],
                          W_l=forward_cache[0].W_l))

        # 梯度下降
        for i3 in range(0, hidden_layer):
        # 更新一下W,B参数,可以每次更新完参数写入数据库或文件中
            W_list[i3] = W_list[i3] - learn_rate * backward_cache[hidden_layer - 1 - i3].dW_l
            B_list[i3] = B_list[i3] - learn_rate * backward_cache[hidden_layer - 1 - i3].dB_l


        # 清空缓存
        forward_cache.clear()
        backward_cache.clear()

        # 打印成本值，如果print_cost=False则忽略
        if i0 % 100 == 0:
        # 记录成本
            costs.append(cost)
            # 是否打印成本值
            print("第", i0 + 1, "次迭代，成本值为：", np.squeeze(cost))

    if flag :
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learn_rate))
        plt.show()


BPNetWork(True)

predict(X_test=X_test, Y_test=Y_test, W_list=W_list, B_list=B_list)
