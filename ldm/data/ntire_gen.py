from torch.utils.data import Dataset
import numpy as np
import random
import cv2
import h5py
import os


class NTIRETrain(Dataset):
    def __init__(self, data_root, crop_size, arg=True, bgr2rgb=True, stride=8):
        self.crop_size = crop_size
        self.hypers = []
        self.bgrs = []
        self.arg = arg
        h,w = 482,512  # img shape
        self.stride = stride
        self.patch_per_line = max((w-crop_size)//stride+1, 1)
        self.patch_per_colum = max((h-crop_size)//stride+1, 1)
        self.patch_per_img = self.patch_per_line*self.patch_per_colum

        hyper_data_path = f'{data_root}/Train_spectral/'
        bgr_data_path = f'{data_root}/Train_gen/'

        hyper_list = []
        bgr_list = []
        for f in os.listdir(hyper_data_path):
            if f.endswith('.mat'):
                hyper_list.append(f)
                bgr_list.append(f.replace('mat','png'))
        hyper_list.sort()
        bgr_list.sort()
        print(f'len(hyper) of ntire2022 dataset:{len(hyper_list)}')
        print(f'len(bgr) of ntire2022 dataset:{len(bgr_list)}')
        for i in range(len(hyper_list)):
            hyper_path = hyper_data_path + hyper_list[i]
            if 'mat' not in hyper_path:
                continue
            try:
                with h5py.File(hyper_path, 'r') as mat:
                    hyper =np.float32(np.array(mat['cube']))
                hyper = np.transpose(hyper, [0, 2, 1])
                bgr_path = bgr_data_path + bgr_list[i]
                assert hyper_list[i].split('.')[0] ==bgr_list[i].split('.')[0], 'Hyper and RGB come from different scenes.'
                bgr = cv2.imread(bgr_path)
                if bgr2rgb:
                    bgr = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                bgr = np.float32(bgr) / 255
                bgr = np.transpose(bgr, [2, 0, 1])  # [3,482,512]
                hyper = hyper[:, 1:-1]
                bgr = bgr[:, 1:-1]
                self.hypers.append(hyper)
                self.bgrs.append(bgr)
                mat.close()
                print(f'Ntire2022 scene {i} is loaded.')
            except OSError:
                print(f'Ntire2022 scene {i} is not loaded.')
                
        self.img_num = len(self.hypers)
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
        bgr = self.bgrs[img_idx]
        hyper = self.hypers[img_idx]
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

class NTIREValidate(Dataset):
    def __init__(self, data_root, bgr2rgb=True):
        self.hypers = []
        self.bgrs = []
        hyper_data_path = f'{data_root}/Valid_spectral/'
        bgr_data_path = f'{data_root}/Valid_gen/'
        hyper_list = []
        bgr_list = []
        for f in os.listdir(hyper_data_path):
            if f.endswith('.mat'):
                hyper_list.append(f)
                bgr_list.append(f.replace('mat','png'))
        hyper_list.sort()
        bgr_list.sort()
        print(f'len(hyper_valid) of ntire2022 dataset:{len(hyper_list)}')
        print(f'len(bgr_valid) of ntire2022 dataset:{len(bgr_list)}')
        for i in range(len(hyper_list)):
            hyper_path = hyper_data_path + hyper_list[i]
            if 'mat' not in hyper_path:
                continue
            try:
                with h5py.File(hyper_path, 'r') as mat:
                    hyper = np.float32(np.array(mat['cube']))
                hyper = np.transpose(hyper, [0, 2, 1])
                bgr_path = bgr_data_path + bgr_list[i]
                assert hyper_list[i].split('.')[0] == bgr_list[i].split('.')[0], 'Hyper and RGB come from different scenes.'
                bgr = cv2.imread(bgr_path)
                if bgr2rgb:
                    bgr = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                bgr = np.float32(bgr) / 255
                bgr = np.transpose(bgr, [2, 0, 1])
                hyper = hyper[:, 1:-1]
                bgr = bgr[:, 1:-1]
                self.hypers.append(hyper)
                self.bgrs.append(bgr)
                mat.close()
                print(f'Ntire2022 scene {i} is loaded.')
            except OSError:
                print(f'Ntire2022 scene {i} is not loaded.')

    def __getitem__(self, idx):
        hyper = self.hypers[idx]
        bgr = self.bgrs[idx]
        return {"rgb": np.ascontiguousarray(bgr.transpose(1, 2, 0)), "hsi": np.ascontiguousarray(hyper.transpose(1, 2, 0))}

    def __len__(self):
        return len(self.hypers)