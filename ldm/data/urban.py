from torch.utils.data import Dataset
import numpy as np
import cv2
import os


class UrbanDataset(Dataset):
    def __init__(self, datapath="/home/root/dataset/Urben_NTIRE22.npz"):
        data = np.load(datapath)
        self.hsi = data["hsi"]
        self.rgb = data["rgb"]

        
    def __getitem__(self, idx):
        hyper = self.hsi
        rgb = self.rgb
        return {"rgb": rgb, "hsi": hyper}

    def __len__(self):
        return 1
    

if __name__ == "__main__":
    import pandas as pd
    from scipy.io import loadmat
    import matplotlib.pyplot as plt
    
    data = pd.read_csv("/home/root/dataset/spectral-response-function/RGB_Camera_QE.csv", delimiter=",", index_col=0)
    P = data.loc[400:701]
    def group_by_tens(idx):
        return idx // 10

    # 使用 groupby 和 sum 方法每 10 个索引值求和
    P = P.groupby(group_by_tens(P.index)).sum()
    P = P.to_numpy()
    P = P[:, [0, 1, 3]]

    P = P / np.sum(P, axis=0, keepdims=True)
    
    P = loadmat("/home/root/dataset/cave/P.mat")["P"].T
    print(P.shape)
    
    # hsi = np.array(h5py.File("/home/root/dataset/Urban_R162.mat")["hsi"])[:len(P)]
    hsi = np.array(loadmat("/home/root/dataset/Urban_R162.mat")["hsi"])[..., :27]
    hsi = np.pad(hsi, ((0, 0), (0, 0), (4, 0),), mode="edge")
    print(hsi.shape)
    hsi = hsi[-256:, -256:]
    hsi = hsi / np.max(hsi)

    rgb = np.einsum("hwb, bc->hwc", hsi[..., 4:], P[4:])
    # rgb = np.round(rgb * 255) / 255
    plt.imshow(rgb)
    plt.savefig("show.png")
    
    np.savez("/home/root/dataset/Urben_D400.npz", hsi=hsi, rgb=rgb, P=P)
