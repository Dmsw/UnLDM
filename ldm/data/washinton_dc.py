from torch.utils.data import Dataset
import numpy as np
import cv2
import os
import random


class DCTrain(Dataset):
    def __init__(self, data_root="/home/root/dataset/Washinton_DC/train/dc_NTIRE22.npz", crop_size=256, arg=True, stride=8):
        data = np.load(data_root)
        self.crop_size = crop_size
        self.hsi = data["hsi"].transpose(2, 0, 1)
        self.rgb = data["rgb"].transpose(2, 0, 1)
        self.arg = arg
        h,w = self.hsi.shape[1:]
        self.stride = stride
        self.patch_per_line = max((w-crop_size)//stride+1, 1)
        self.patch_per_colum = max((h-crop_size)//stride+1, 1)
        self.patch_per_img = self.patch_per_line*self.patch_per_colum

        self.img_num = 1
        self.length = self.patch_per_img * self.img_num

    def arguement(self, img, rotTimes, vFlip, hFlip):
        # Random rotation
        for j in range(rotTimes):
            img = np.rot90(img.copy(), axes=(1, 2))
        # Random vertical Flip
        for j in range(vFlip):
            img = img[:, :, ::-1].copy()
        # Random horizontal Flip
        for j in range(hFlip):
            img = img[:, ::-1, :].copy()
        return img

    def __getitem__(self, idx):
        stride = self.stride
        crop_size = self.crop_size
        img_idx, patch_idx = idx//self.patch_per_img, idx%self.patch_per_img
        h_idx, w_idx = patch_idx//self.patch_per_line, patch_idx%self.patch_per_line
        bgr = self.rgb
        hyper = self.hsi
        bgr = bgr[:,h_idx*stride:h_idx*stride+crop_size, w_idx*stride:w_idx*stride+crop_size]
        hyper = hyper[:, h_idx * stride:h_idx * stride + crop_size,w_idx * stride:w_idx * stride + crop_size]
        rotTimes = random.randint(0, 3)
        vFlip = random.randint(0, 1)
        hFlip = random.randint(0, 1)
        if self.arg:
            bgr = self.arguement(bgr, rotTimes, vFlip, hFlip)
            hyper = self.arguement(hyper, rotTimes, vFlip, hFlip)
        return {"rgb": np.ascontiguousarray(bgr.transpose(1, 2, 0)), "hsi": np.ascontiguousarray(hyper.transpose(1, 2, 0))}

    def __len__(self):
        return self.patch_per_img*self.img_num

class DCTest(Dataset):
    def __init__(self, datapath="/home/root/dataset/Washinton_DC/test/dc_NTIRE22.npz"):
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
    import rasterio
    
    data = pd.read_csv("/home/root/dataset/spectral-response-function/RGB_Camera_QE.csv", delimiter=",", index_col=0)
    P = data.loc[400:701]
    def group_by_tens(idx):
        return idx // 10

    # 使用 groupby 和 sum 方法每 10 个索引值求和
    P = P.groupby(group_by_tens(P.index)).sum()
    P = P.to_numpy()
    P = P[:, [0, 1, 3]]

    P = P / np.sum(P, axis=0, keepdims=True)
    
    wavelengths = pd.read_csv("/home/root/dataset/Washinton_DC/wavelengths.txt", delimiter=" ", header=None).iloc[:, 1].to_numpy()
    print(wavelengths)
    
    src = rasterio.open("/home/root/dataset/Washinton_DC/dc.tif")
    raw_hsi = src.read()
    raw_hsi = raw_hsi.transpose(1, 2, 0).astype(np.float32)
    H, W, C = raw_hsi.shape
    hsi = np.zeros([H, W, 31])
    count = np.zeros([H, W, 31])
    for i in range(C):
        index = int(np.floor(wavelengths[i] / 10) - 40)
        if index >= 31: break
        hsi[:, :, index] += raw_hsi[:, :, i]
        count[:, :, index] += 1
    
    hsi = hsi / count
    hsi = hsi / np.max(hsi)
    hsi = hsi.clip(0, 1)
    print(hsi.min())
    print(hsi.shape)
    hsi_train = hsi[-256:, -256:].astype(np.float32)

    rgb_train = np.einsum("hwb, bc->hwc", hsi_train, P).astype(np.float32)
    # rgb = np.round(rgb * 255) / 255
    plt.imshow(rgb_train)
    plt.savefig("show_train.png")
    
    np.savez("/home/root/dataset/Washinton_DC/test/dc_NTIRE22.npz", hsi=hsi_train, rgb=rgb_train, P=P)
    
    hsi_test = hsi[:-256, :].astype(np.float32)

    rgb_test = np.einsum("hwb, bc->hwc", hsi_test, P).astype(np.float32)
    # rgb = np.round(rgb * 255) / 255
    plt.imshow(rgb_test)
    plt.savefig("show_test.png")
    
    np.savez("/home/root/dataset/Washinton_DC/train/dc_NTIRE22.npz", hsi=hsi_test, rgb=rgb_test, P=P)

