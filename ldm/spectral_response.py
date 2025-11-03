import pandas as pd
import numpy as np


data = pd.read_csv("/home/root/dataset/spectral-response-function/RGB_Camera_QE.csv", delimiter=",", index_col=0)
P = data.loc[400:701]
def group_by_tens(idx):
    return idx // 10

# 使用 groupby 和 sum 方法每 10 个索引值求和
P = P.groupby(group_by_tens(P.index)).sum()
P = P.to_numpy()
P = P[:, [0, 1, 3]]

P = P / np.sum(P, axis=0, keepdims=True)


def gen_rgb(x):
    return np.einsum('chw, cb -> bhw', x, P)
