class Forward:
    """
    存放正向传播计算出来的结果及相应的参数
    """
    def __init__(self,Z_l,W_l,B_l,A_l):
        self.Z_l=Z_l
        self.W_l=W_l
        self.B_l=B_l
        self.A_l=A_l