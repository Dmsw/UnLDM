from __future__ import division

import torch
import torch.nn as nn
import logging
import numpy as np
import os
import piq
import hdf5storage

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def initialize_logger(file_dir):
    logger = logging.getLogger()
    fhandler = logging.FileHandler(filename=file_dir, mode='a')
    formatter = logging.Formatter('%(asctime)s - %(message)s', "%Y-%m-%d %H:%M:%S")
    fhandler.setFormatter(formatter)
    logger.addHandler(fhandler)
    logger.setLevel(logging.INFO)
    return logger

def save_checkpoint(model_path, epoch, iteration, model, optimizer):
    state = {
        'epoch': epoch,
        'iter': iteration,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }

    torch.save(state, os.path.join(model_path, 'net_%depoch.pth' % epoch))

class Loss_MRAE(nn.Module):
    def __init__(self):
        super(Loss_MRAE, self).__init__()

    def forward(self, outputs, label):
        assert outputs.shape == label.shape
        error = torch.abs(outputs - label) / (label + 1e-3)
        mrae = torch.mean(error.reshape(-1))
        return mrae

class Loss_RMSE(nn.Module):
    def __init__(self):
        super(Loss_RMSE, self).__init__()

    def forward(self, outputs, label):
        assert outputs.shape == label.shape
        error = outputs-label
        sqrt_error = torch.pow(error,2)
        rmse = torch.sqrt(torch.mean(sqrt_error.reshape(-1)))
        return rmse

class Loss_PSNR(nn.Module):
    def __init__(self):
        super(Loss_PSNR, self).__init__()

    def forward(self, im_true, im_fake, data_range=1):
        psnr = piq.psnr(im_true, im_fake, data_range=data_range)
        return psnr
    
class Loss_SSIM(nn.Module):
    def __init__(self):
        super(Loss_SSIM, self).__init__()

    def forward(self, im_true, im_fake):
        ssim = piq.ssim(im_true, im_fake, data_range=1.0)
        return ssim
    
class Loss_SAM(nn.Module):
    def __init__(self):
        super(Loss_SAM, self).__init__()

    def forward(self, im_true, im_fake):
        sam = 0
        N, C, H, W = im_true.size()
        for i in range(N):
            im_true_i = im_true[i]
            im_fake_i = im_fake[i]
            im_true_i = im_true_i.reshape(C, -1)
            im_fake_i = im_fake_i.reshape(C, -1)
            sam += torch.acos(torch.clamp(torch.sum(im_fake_i*im_true_i, dim=0) / (torch.norm(im_true_i,dim=0) * torch.norm(im_fake_i,dim=0) + 1e-6), -1, 1)).mean() * 180 / np.pi  
        sam = sam / N
        return sam

def time2file_name(time):
    year = time[0:4]
    month = time[5:7]
    day = time[8:10]
    hour = time[11:13]
    minute = time[14:16]
    second = time[17:19]
    time_filename = year + '_' + month + '_' + day + '_' + hour + '_' + minute + '_' + second
    return time_filename

def record_loss(loss_csv, epoch, iteration, epoch_time, lr, train_loss, test_loss):
    """ Record many results."""
    loss_csv.write('{},{},{},{},{},{}\n'.format(epoch, iteration, epoch_time, lr, train_loss, test_loss))
    loss_csv.flush()
    loss_csv.close
    
    
def tensor_2_mode_product(tensor: torch.Tensor, P):
    return torch.matmul(tensor.permute(0, 2, 3, 1), P.T).permute(0, 3, 1, 2)
    
def tensor_1_mode_product(tensor: torch.Tensor, P):
    return torch.matmul(tensor.permute(1, 2, 0), P.T).permute(2, 0, 1)

def save_matv73(mat_name, var_name, var):
    hdf5storage.savemat(mat_name, {var_name: var}, format='7.3', store_python_metadata=True)
