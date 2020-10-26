class Backward:
    """
    存放正向传播计算出来的结果及相应的参数
    """
    def __init__(self,dW_l,dB_l,dA_l_1):
        self.dW_l=dW_l
        self.dB_l=dB_l
        self.dA_l_1=dA_l_1