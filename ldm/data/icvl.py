import os
import numpy as np
import PIL
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import random
import h5py
from scipy.io import loadmat


def hsi2rgb(hsi, P):
    shape = hsi.shape
    assert P.shape[1] == shape[0]
    X = hsi.reshape(shape[0], shape[1]*shape[2])
    z = P @ X
    rgb = z.reshape(P.shape[0], shape[1], shape[2])
    return rgb


def center_crop(img, crop_size):
    h, w = img.shape[1], img.shape[2]
    start_h = (h - crop_size) // 2
    start_w = (w - crop_size) // 2
    return img[:, start_h:start_h + crop_size, start_w:start_w + crop_size]


class ICVLBase(Dataset):
    def __init__(self,
                 resolution=256,
                 root_path="/home/root/dataset/icvl/mat/",
                 split_file="train_files.txt",
                 P_file="/home/root/dataset/spectral-response-function/P.mat",
                 arg=True,
                 stride=64,
                 verbose=False,
                 ):
        super().__init__()
        self.verbose = verbose
        self.root_path = root_path
        self.split_file = os.path.join(root_path, split_file)
        self.files = np.loadtxt(self.split_file, dtype=str)
        self.hypers = []
        self.rgbs = []
        self.arg = arg
        h,w = 1024, 1024  # img shape
        self.stride = stride
        self.crop_size = resolution
        self.patch_per_line = (w-self.crop_size)//stride+1
        self.patch_per_colum = (h-self.crop_size)//stride+1
        self.patch_per_img = self.patch_per_line*self.patch_per_colum
        P = loadmat(P_file)["P"].astype(np.float32)
        for f in self.files:
            path = os.path.join(self.root_path, f)
            data = h5py.File(path, "r")
            hsi = np.array(data["rad"]).astype(np.float32)
            if hsi.shape[2] == 31:
                hsi = np.transpose(hsi, (2, 0, 1))
            if (np.min(hsi.shape[1:]) < 1024):
                print(f"skip {f} for shape too small")
                self.files = np.delete(self.files, np.where(self.files == f))
                continue
            hsi = center_crop(hsi, h)
            hsi = hsi / np.max(hsi)
            
            rgb = hsi2rgb(hsi, P).astype(np.float32)
            print(f"load {f} with shape {hsi.shape}")

            self.hypers.append(hsi)
            self.rgbs.append(rgb)
            data.close()
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


class ICVLTrain(ICVLBase):
    def __init__(self, **kwargs):
        super().__init__(split_file="train_files.txt", **kwargs)


class ICVLValidation(ICVLBase):
    def __init__(self, **kwargs):
        super().__init__(split_file="val_files.txt", **kwargs)

class ICVLTest(ICVLBase):
    def __init__(self, **kwargs):
        super().__init__(split_file="test_files.txt", **kwargs)
        
