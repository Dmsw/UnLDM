import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.autograd import Variable
import os
from hsi_dataset import TrainDataset, ValidDataset
from utils import AverageMeter, initialize_logger, save_checkpoint, record_loss, \
    time2file_name, Loss_MRAE, Loss_RMSE, Loss_PSNR
import datetime
import argparse, os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import cv2
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from itertools import islice
from einops import rearrange
from torchvision.utils import make_grid
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import nullcontext
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from ldm.models.diffusion.dpm_solver import DPMSolverSampler
from ldm.models.diffusion.ddpm import disabled_train
from torch.utils.tensorboard import SummaryWriter


def load_model_from_config(config, ckpt, device=torch.device("cuda"), verbose=True):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict({k.replace('first_stage_model.', ''): v for k, v in sd.items() if k.startswith('first_stage_model')}, strict=True)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    if device == torch.device("cuda"):
        model.cuda()
    elif device == torch.device("cpu"):
        model.cpu()
        model.cond_stage_model.device = "cpu"
    else:
        raise ValueError(f"Incorrect device name. Received: {device}")
    model.eval()
    model.train = disabled_train
    return model


def frozen_model(model):
    model.zero_grad()
    for param in model.parameters():
        param.requires_grad = False


parser = argparse.ArgumentParser(description="Spectral Recovery Toolbox")
parser.add_argument("--batch_size", type=int, default=20, help="batch size")
parser.add_argument("--end_epoch", type=int, default=300, help="number of epochs")
parser.add_argument("--init_lr", type=float, default=4e-4, help="initial learning rate")
parser.add_argument("--outf", type=str, default='./exp/spectral_model/', help='path log files')
parser.add_argument("--data_root", type=str, default='../dataset/')
parser.add_argument("--patch_size", type=int, default=128, help="patch size")
parser.add_argument("--stride", type=int, default=8, help="stride")
parser.add_argument("--gpu_id", type=str, default='0', help='path log files')
parser.add_argument("--ckpt", type=str, default="checkpoints/768-v-ema.ckpt", help='checkpoint path')
parser.add_argument("--config", type=str, default="configs/stable-diffusion/first_stage_model.yaml", help='model config path')
parser.add_argument("--spectral_config", type=str, default="configs/spectral_model.yaml", help='spectral model config path')
parser.add_argument("--pretrain", type=str, default=None, help='pretrain model path')
opt = parser.parse_args()
os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id

# load dataset
print("\nloading dataset ...")
train_data = TrainDataset(data_root=opt.data_root, crop_size=opt.patch_size, bgr2rgb=True, arg=True, stride=opt.stride)
print(f"Iteration per epoch: {len(train_data)}")
val_data = ValidDataset(data_root=opt.data_root, bgr2rgb=True)
print("Validation set samples: ", len(val_data))

# iterations
per_epoch_iteration = 1000
total_iteration = per_epoch_iteration*opt.end_epoch

# loss function
criterion_mrae = Loss_MRAE()
criterion_rmse = Loss_RMSE()
criterion_psnr = Loss_PSNR()

# model
config = OmegaConf.load(f"{opt.config}")
device = torch.device("cuda")
first_stage_model = load_model_from_config(config, opt.ckpt, device=device)
# frozen_model(first_stage_model)
spectral_model = instantiate_from_config(OmegaConf.load(f"{opt.spectral_config}").model)
print('Parameters number is ', sum(param.numel() for param in spectral_model.parameters()))

resume_file = opt.pretrain
if resume_file is not None:
    if os.path.isfile(resume_file):
        print("=> loading checkpoint '{}'".format(resume_file))
        checkpoint = torch.load(resume_file)
        spectral_model.load_state_dict(checkpoint['state_dict'])

# output path
date_time = str(datetime.datetime.now())
date_time = time2file_name(date_time)
opt.outf = opt.outf + date_time
if not os.path.exists(opt.outf):
    os.makedirs(opt.outf)

if torch.cuda.is_available():
    spectral_model.cuda()
    criterion_mrae.cuda()
    criterion_rmse.cuda()
    criterion_psnr.cuda()

if torch.cuda.device_count() > 1:
    spectral_model = nn.DataParallel(spectral_model)
    first_stage_model = nn.DataParallel(first_stage_model)

optimizer = optim.Adam(spectral_model.parameters(), lr=opt.init_lr, betas=(0.9, 0.999))
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_iteration, eta_min=1e-6)

# logging
log_dir = os.path.join(opt.outf, 'train.log')
logger = initialize_logger(log_dir)
writter = SummaryWriter(log_dir=opt.outf)

def main():
    cudnn.benchmark = True
    log_interval = 1000
    iteration = 0
    record_mrae_loss = 1000
    while iteration<total_iteration:
        losses = AverageMeter()
        rec_losses = AverageMeter()
        dec_losses = AverageMeter()
        train_loader = DataLoader(dataset=train_data, batch_size=opt.batch_size, shuffle=True, num_workers=2,
                                  pin_memory=True, drop_last=True)
        val_loader = DataLoader(dataset=val_data, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)
        for i, (rgb, hsi) in enumerate(train_loader):
            spectral_model.train()
            rgb = rgb.cuda()
            hsi = hsi.cuda()
            rgb = Variable(rgb)
            hsi = Variable(hsi)
            lr = optimizer.param_groups[0]['lr']
            optimizer.zero_grad()
            first_stage_model.zero_grad()
            z = spectral_model.encode(hsi)
            z_first, _ = first_stage_model(z, sample_posterior=False)
            hsi_de = spectral_model.decode(z_first, rgb)
            rec_loss = criterion_rmse(hsi_de, hsi)
            dec_loss = criterion_rmse(z_first, z)
            loss = rec_loss + dec_loss * 0.1
            loss.backward()
            optimizer.step()
            scheduler.step()
            losses.update(loss.data)
            rec_losses.update(rec_loss.data)
            dec_losses.update(dec_loss.data)
            iteration = iteration+1
            if iteration % 20 == 0:
                writter.add_scalar('Train/loss', losses.avg, iteration)
                print('[iter:%d/%d],lr=%.9f,train_losses.avg=%.9f, rec_loss=%.9f, dec_loss=%.9f'
                      % (iteration, total_iteration, lr, losses.avg, rec_losses.avg, dec_losses.avg))
            if iteration % log_interval == 0:
                mrae_loss, rmse_loss, psnr_loss, z, output = validate(val_loader, spectral_model)
                writter.add_scalar('Test/MRAE', mrae_loss, iteration)
                writter.add_scalar('Test/RMSE', rmse_loss, iteration)
                writter.add_scalar('Test/PSNR', psnr_loss, iteration)
                writter.add_scalar('Test/min_z', torch.amin(z), iteration)
                writter.add_scalar('Test/max_z', torch.amax(z), iteration)
                with torch.no_grad():
                    z = z - torch.amin(z, dim=(2, 3), keepdim=True)
                    z = z / torch.amax(z, dim=(2, 3), keepdim=True)
                    output = output - torch.amin(output, dim=(2, 3), keepdim=True)
                    output = output / torch.amax(output, dim=(2, 3), keepdim=True)
                writter.add_images('Test/Z', z, iteration, dataformats='NCHW')
                writter.add_images('Test/Output', output[0].unsqueeze(1), iteration, dataformats='NCHW')
                print(f'MRAE:{mrae_loss}, RMSE: {rmse_loss}, PNSR:{psnr_loss}')
                # Save model
                if torch.abs(mrae_loss - record_mrae_loss) < 0.01 or mrae_loss < record_mrae_loss or iteration % 5000 == 0:
                    print(f'Saving to {opt.outf}')
                    save_checkpoint(opt.outf, (iteration // log_interval), iteration, spectral_model, optimizer)
                    if mrae_loss < record_mrae_loss:
                        record_mrae_loss = mrae_loss
                # print loss
                print(" Iter[%06d], Epoch[%06d], learning rate : %.9f, Train MRAE: %.9f, Test MRAE: %.9f, "
                      "Test RMSE: %.9f, Test PSNR: %.9f " % (iteration, iteration//log_interval, lr, losses.avg, mrae_loss, rmse_loss, psnr_loss))
                logger.info(" Iter[%06d], Epoch[%06d], learning rate : %.9f, Train Loss: %.9f, Test MRAE: %.9f, "
                      "Test RMSE: %.9f, Test PSNR: %.9f " % (iteration, iteration//log_interval, lr, losses.avg, mrae_loss, rmse_loss, psnr_loss))
    return 0

# Validate
def validate(val_loader, model):
    model.eval()
    losses_mrae = AverageMeter()
    losses_rmse = AverageMeter()
    losses_psnr = AverageMeter()
    for i, (input, target) in enumerate(val_loader):
        input = input.cuda()
        target = target.cuda()
        with torch.no_grad():
            # compute output
            z = model.encode(target)
            z, _ = first_stage_model(z, sample_posterior=False)
            output = model.decode(z, input)
            output = torch.clamp(output, 0, 1)
            loss_mrae = criterion_mrae(output[:, :, 128:-128, 128:-128], target[:, :, 128:-128, 128:-128])
            loss_rmse = criterion_rmse(output[:, :, 128:-128, 128:-128], target[:, :, 128:-128, 128:-128])
            loss_psnr = criterion_psnr(output[:, :, 128:-128, 128:-128], target[:, :, 128:-128, 128:-128])

        # record loss
        losses_mrae.update(loss_mrae.data)
        losses_rmse.update(loss_rmse.data)
        losses_psnr.update(loss_psnr.data)
    return losses_mrae.avg, losses_rmse.avg, losses_psnr.avg, z, output

if __name__ == '__main__':
    main()
    print(torch.__version__)