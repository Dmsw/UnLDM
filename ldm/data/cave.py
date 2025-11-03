import os
import numpy as np
import PIL
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import random
from scipy.io import loadmat


class CAVEBase(Dataset):
    def __init__(self,
                 resolution=256,
                 root_path="/home/root/project/RGB2HSI/PaDM/dataset/cave/test",
                 arg=True,
                 stride=64,
                 verbose=False,
                 ):
        super().__init__()
        P = loadmat("/home/root/dataset/cave/P.mat")["P"]
        self.verbose = verbose
        self.root_path = root_path
        self.files = os.listdir(root_path)
        self.hypers = []
        self.rgbs = []
        self.arg = arg
        h,w = 512,512  # img shape
        self.stride = stride
        self.crop_size = resolution
        self.patch_per_line = (w-self.crop_size)//stride+1
        self.patch_per_colum = (h-self.crop_size)//stride+1
        self.patch_per_img = self.patch_per_line*self.patch_per_colum
        for f in self.files:
            path = os.path.join(self.root_path, f)
            data = np.load(path)
            hsi = data["hsi"]
            rgb = np.einsum("bij,cb->cij", hsi, P)
            self.hypers.append(hsi)
            self.rgbs.append(rgb)
        self.resolution = resolution
        self.arg = arg
        self.img_num = len(self.hypers)
        print(f"load {len(self.files)} files")

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

    def __len__(self):
        return self.patch_per_img*self.img_num

    def __getitem__(self, idx):
        stride = self.stride
        crop_size = self.crop_size
        img_idx, patch_idx = idx//self.patch_per_img, idx%self.patch_per_img
        h_idx, w_idx = patch_idx//self.patch_per_line, patch_idx%self.patch_per_line
        bgr = self.rgbs[img_idx]
        hyper = self.hypers[img_idx]
        bgr = bgr[:,h_idx*stride:h_idx*stride+crop_size, w_idx*stride:w_idx*stride+crop_size]
        hyper = hyper[:, h_idx * stride:h_idx * stride + crop_size,w_idx * stride:w_idx * stride + crop_size]
        rotTimes = random.randint(0, 3)
        vFlip = random.randint(0, 1)
        hFlip = random.randint(0, 1)
        if self.arg:
            bgr = self.arguement(bgr, rotTimes, vFlip, hFlip)
            hyper = self.arguement(hyper, rotTimes, vFlip, hFlip)
        example = {
            "hsi": np.ascontiguousarray(hyper.transpose(1, 2, 0)),
            "rgb": np.ascontiguousarray(bgr.transpose(1, 2, 0))
        }
        if self.verbose:
            example["name"] = self.files[img_idx]
        return example


class CAVETrain(CAVEBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class CAVEValidation(CAVEBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
