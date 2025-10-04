import pandas as pd
import numpy as np
import json
import os
import torch as th
from scipy.io import loadmat

__SSF__ = None


def load_ssf(path, rgb_format='RGB', normalize=True):
    with open(path) as f:
        data = json.load(f)
        assert 'ssf_bands' in data, 'No ssf_bands in the json file.'
        camera_name = data.pop('camera_name')
        df = pd.DataFrame(data)
        new_bands = np.arange(400, 710 + 10, 10)
        df['group'] = pd.cut(df['ssf_bands'], bins=new_bands, right=False, labels=new_bands[:-1])
        assert df['group'].value_counts(sort=False).min() > 0, 'Some bands are missing.'
        data = df.groupby('group').mean()

        red_ssf = np.array(data['red_ssf'])
        gree_ssf = np.array(data['green_ssf'])
        blue_ssf = np.array(data['blue_ssf'])
        if rgb_format == 'RGB':
            ssf = np.stack([red_ssf, gree_ssf, blue_ssf], axis=0)
        elif rgb_format == 'BGR':
            ssf = np.stack([blue_ssf, gree_ssf, red_ssf], axis=0)
        else:
            raise ValueError('Invalid rgb_format.')
        if normalize:
            ssf = ssf / np.sum(ssf, axis=1, keepdims=True)
        return ssf.astype(np.float32)


def list_all_ssf_path(root):
    for root, _, files in os.walk(root):
        for f in files:
            if f.endswith('.json'):
                yield os.path.join(root, f)
                

def _random_select_ssf_path(root, n=1):
    if __SSF__ is None:
        load_all_ssf_path(root)
    return np.random.choice(__SSF__, n)


def load_all_ssf_path(root):
    global __SSF__
    __SSF__ = list(list_all_ssf_path(root))


def load_random_ssf(root, n=1, rgb_format='RGB', normalize=True):
    paths = _random_select_ssf_path(root, n)
    return [load_ssf(p, rgb_format, normalize) for p in paths]


def load_NTIRE_ssf():
    data = pd.read_csv("/home/root/dataset/spectral-response-function/RGB_Camera_QE.csv", delimiter=",", index_col=0)
    P = data.loc[400:701]
    def group_by_tens(idx):
        return idx // 10

    # 使用 groupby 和 sum 方法每 10 个索引值求和
    P = P.groupby(group_by_tens(P.index)).sum()
    P = P.to_numpy()
    P = P[:, [0, 1, 3]]

    P = P / np.sum(P, axis=0, keepdims=True).astype(np.float32)
    return P.T


def load_D400_ssf():
    P = loadmat("/home/root/dataset/spectral-response-function/P.mat")["P"].astype(np.float32)
    return P

class SSFTool:
    def __init__(self, root="/home/root/dataset/ssf-data-master/", rgb_format='RGB', normalize=True):
        self.root = root
        self.rgb_format = rgb_format
        self.normalize = normalize
        load_all_ssf_path(root)
        self.check_all_ssf()
        self.P = None
        
    def load_ntire_ssf(self):
        self.P = load_NTIRE_ssf()
        
    def load_d400_ssf(self):
        self.P = load_D400_ssf()
        
    def load_ssf(self, name):
        ssf_path = [p for p in __SSF__ if name in p]
        self.P = load_ssf(ssf_path[0], self.rgb_format, self.normalize)
        return self.P
    
    def load_random_ssf(self, n=1):
        self.P = load_random_ssf(self.root, n, self.rgb_format, self.normalize)[0]
        return self.P
    
    def set_ssf(self, P):
        self.P = P
    
    def check_all_ssf(self):
        global __SSF__
        for p in __SSF__:
            try:
                load_ssf(p, self.rgb_format, self.normalize)
            except AssertionError as e:
                print(f'Error: {e} in {p}')
                __SSF__.remove(p)

    def gen_rgb_numpy(self, x, *, max_value=255.0, clip=True, normalize=False, quantize=False, depth=16):
        assert self.P is not None, 'Please load ssf first.'
        rgb = np.einsum('chw, bc -> bhw', x, self.P)
        if normalize:
            rgb -= np.min(rgb)
            rgb /= np.max(rgb)
        if clip:
            rgb = np.clip(rgb, 0, 1)
        if quantize:
            rgb = np.round(rgb * (2 ** depth - 1)) / (2 ** depth - 1)
            
        return rgb.astype(np.float32) * max_value

    def gen_rgb_torch(self, x, *, max_value=255.0, clip=True, normalize=False, quantize=False, depth=16, gamma=1.0):
        assert self.P is not None, 'Please load ssf first.'
        rgb = th.einsum('nchw, bc -> nbhw', x, th.from_numpy(self.P).to(device=x.device, dtype=x.dtype))
        if normalize:
            rgb -= th.min(rgb)
            rgb /= th.max(rgb)
        if clip:
            rgb = th.clip(rgb, 0, 1)
        rgb = rgb ** gamma
        if quantize:
            rgb = th.round(rgb * (2 ** depth - 1)) / (2 ** depth - 1)
            
        return rgb * max_value
    
    def pinv(self, rgb):
        assert self.P is not None, 'Please load ssf first.'
        pinv = th.from_numpy(self.P).to(device=rgb.device, dtype=rgb.dtype)
        pinv = th.pinverse(pinv)
        hsi = th.einsum('nchw, bc -> nbhw', rgb, pinv)            
        return hsi
        
