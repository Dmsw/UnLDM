import argparse, os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import cv2
import torch
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

# from ldm.data.icvl import ICVLTest as TestDataset
# from ldm.data.cave import CAVEValidation as TestDataset
from ldm.data.ntire_gen import NTIREValidate as TestDataset
from torchvision.transforms import ToPILImage

from utils import AverageMeter, initialize_logger, save_checkpoint, record_loss, \
    time2file_name, Loss_MRAE, Loss_RMSE, Loss_PSNR, Loss_SAM, Loss_SSIM, save_matv73

import pandas as pd
import cv2 

from illuminant import Illuminant
from ssf_tools import SSFTool
from awb import gray_world_white_balance

torch.set_grad_enabled(False)

def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def load_model_from_config(config, ckpt, device=torch.device("cuda"), verbose=True, sp_ckpt=None):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)
    
    if sp_ckpt is not None:
        pl_sd = torch.load(sp_ckpt, map_location="cpu")
        model.spectral_model.load_state_dict(pl_sd["state_dict"], strict=True)

    if device == torch.device("cuda"):
        model.cuda()
    elif device == torch.device("cpu"):
        model.cpu()
        model.cond_stage_model.device = "cpu"
    else:
        raise ValueError(f"Incorrect device name. Received: {device}")
    model.eval()
    return model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs/rgb2hsi-samples"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
    )
    parser.add_argument(
        "--plms",
        action='store_true',
        help="use plms sampling",
    )
    parser.add_argument(
        "--dpm",
        action='store_true',
        help="use DPM (2) sampler",
    )
    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/stable-diffusion/v2-inference.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        help="evaluate at this precision",
        choices=["full", "autocast"],
        default="autocast"
    )
    parser.add_argument(
        "--device",
        type=str,
        help="Device on which Stable Diffusion will be run",
        choices=["cpu", "cuda"],
        default="cpu"
    )
    parser.add_argument(
        "--bf16",
        action='store_true',
        help="Use bfloat16",
    )
    parser.add_argument(
        "--data_root", 
        type=str, 
        default='../dataset/')
    parser.add_argument(
        "--n_iter", 
        type=int, 
        default=1)
    parser.add_argument(
        "--sp_ckpt",
        type=str,
        default=None,
    )
    opt = parser.parse_args()
    return opt


def main(opt):
    seed_everything(opt.seed)    
    print("loading ssf")
    ssf = SSFTool()
    ssf.load_ntire_ssf()
    
    print("loading illuminant")
    light = Illuminant(file="CIE_illum_FLs_1nm.csv", light_type=6)
    # light = Illuminant(file="CIE_std_illum_A_1nm.csv", light_type=1, root="/home/root/dataset/illuminant/")
    std_light = Illuminant(file="CIE_std_illum_D65.csv", light_type=1, root="/home/root/dataset/illuminant/")
    
    print(f"loading data from {opt.data_root}")
    dataset = TestDataset(data_root=opt.data_root)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    config = OmegaConf.load(f"{opt.config}")
    device = torch.device("cuda") if opt.device == "cuda" else torch.device("cpu")
    model = load_model_from_config(config, f"{opt.ckpt}", device, sp_ckpt=opt.sp_ckpt)

    if opt.plms:
        sampler = PLMSSampler(model, device=device)
    elif opt.dpm:
        sampler = DPMSolverSampler(model, device=device)
    else:
        sampler = DDIMSampler(model, device=device)

    os.makedirs(opt.outdir, exist_ok=True)
    outpath = opt.outdir

    # loss function
    criterion_mrae = Loss_MRAE()
    criterion_rmse = Loss_RMSE()
    criterion_psnr = Loss_PSNR()
    criterion_ssim = Loss_SSIM()
    criterion_sam = Loss_SAM()
    
    losses_mrae = AverageMeter()
    losses_rmse = AverageMeter()
    losses_sam = AverageMeter()
    losses_ssim = AverageMeter()
    losses_psnr = AverageMeter()
    
    # to device
    if opt.device == "cuda":
        criterion_mrae.cuda()
        criterion_rmse.cuda()
        criterion_psnr.cuda()
        criterion_sam.cuda()
        criterion_ssim.cuda()

    sample_path = os.path.join(outpath, "samples")
    os.makedirs(sample_path, exist_ok=True)
    var_name = "cube"
    with open(os.path.join(sample_path, "results.txt"), "w") as f:
        f.write(str(opt) + '\n')
    precision_scope = autocast if opt.precision=="autocast" or opt.bf16 else nullcontext
    count = 0

    with torch.no_grad(), \
        precision_scope(opt.device), \
        model.ema_scope():
            for data in tqdm(dataloader, desc="data"):
                hsi = data["hsi"].permute(0, 3, 1, 2).float()
                rgb = data["rgb"].permute(0, 3, 1, 2).float()
                name = str(count)
                count += 1
                if opt.device == "cuda":
                    hsi = hsi.cuda()
                    rgb = rgb.cuda()
                assert not torch.any(torch.isnan(hsi))
                lighted_hsi = light.lighting(hsi)
                assert not torch.any(torch.isnan(lighted_hsi))

                lighted_rgb = ssf.gen_rgb_torch(lighted_hsi, max_value=1)
                
                light3d = torch.from_numpy(ssf.gen_rgb_numpy(light.illuminant[0], max_value=1)).to(lighted_rgb.device)
                light3d_rgb = lighted_rgb / light3d
                lighted = torch.from_numpy(ssf.gen_rgb_numpy(std_light.illuminant[0], max_value=1)).to(lighted_rgb.device)
                light3d_rgb = light3d_rgb * lighted
                light3d_rgb -= light3d_rgb.min()
                light3d_rgb /= light3d_rgb.max()
                
                p_rgb = model.spectral_model.encode(lighted_hsi)
                r_hsi = model.spectral_model.decode(p_rgb, lighted_rgb)
                print(f"error {((r_hsi - lighted_hsi)**2).mean()}")
                
                rgb_posterior = model.encode_first_stage(lighted_rgb * 2 - 1)
                z_rgb = model.get_first_stage_encoding(rgb_posterior)
                uc = model.get_unconditional_conditioning(rgb.shape[0])
                c = {'c_concat': [z_rgb], 'c_crossattn': [uc]}
                shape = z_rgb.shape[1:]
                # rgb = rgb_f
                output = torch.zeros_like(hsi)
                for n in range(opt.n_iter):
                    z_hsi, _ = sampler.sample(S=opt.steps,
                                                     conditioning=c,
                                                     batch_size=z_rgb.shape[0],
                                                     shape=shape,
                                                     verbose=False,
                                                     eta=opt.ddim_eta)
                    
                    assert not torch.any(torch.isnan(z_hsi))
                    ps_hsi = model.decode_first_stage_hsi(z_hsi)
                    hsi_est = model.spectral_model.decode(ps_hsi, lighted_rgb)
                    output += hsi_est
                output /= opt.n_iter
                
                hsi_est = output
                hsi = torch.clamp(hsi, 0, 1)
                
                assert not torch.any(torch.isnan(hsi_est))
                
                delight_hsi = light.delighting(hsi_est, threshold=1e-2)
                relight_hsi = std_light.lighting(delight_hsi)
                relight_rgb = ssf.gen_rgb_torch(relight_hsi, max_value=1.0, normalize=True, depth=8).to(torch.float32)
                
                rgb = std_light.lighting(hsi)
                rgb = ssf.gen_rgb_torch(rgb, max_value=1.0, normalize=True, depth=8).to(torch.float32)
                
                hsi_est = torch.clamp(hsi_est, 0, 1)
                lighted_hsi = torch.clamp(lighted_hsi, 0, 1)
                loss_mrae = criterion_mrae(rgb, relight_rgb)
                loss_rmse = criterion_rmse(rgb, relight_rgb)
                loss_psnr = criterion_psnr(rgb, relight_rgb)
                loss_ssim = criterion_ssim(rgb, relight_rgb)
                loss_sam = criterion_sam(rgb, relight_rgb)
                msg = f'name: {name}, mrae:{loss_mrae}, rmse:{loss_rmse}, psnr:{loss_psnr}, ssim:{loss_ssim}, sam:{loss_sam}\n'
                with open(os.path.join(sample_path, "results.txt"), "a") as f:
                    f.write(msg)
                
                losses_mrae.update(loss_mrae.data)
                losses_rmse.update(loss_rmse.data)
                losses_psnr.update(loss_psnr.data)
                losses_ssim.update(loss_ssim.data)
                losses_sam.update(loss_sam.data)
                
                baseline_rgb = gray_world_white_balance(lighted_rgb)
                if True:
                    topil = ToPILImage()
                    # label = os.path.join(sample_path, f"label_{name}.png")
                    # label_pil = topil(rgb[0])
                    # label_pil.save(label)
                    pred = os.path.join(sample_path, f"pred_{name}.png")
                    pred_pil = topil(relight_rgb[0])
                    pred_pil.save(pred)
                    # inp = os.path.join(sample_path, f"input_{name}.png")
                    # inp_pil = topil(lighted_rgb[0])
                    # inp_pil.save(inp)
                    # base = os.path.join(sample_path, f"base_{name}.png")
                    # base_pil = topil(baseline_rgb[0])
                    # base_pil.save(base)
                    # light3d = os.path.join(sample_path, f"light_{name}.png")
                    # light_pil = topil(light3d_rgb[0])
                    # light_pil.save(light3d)
                    # result = hsi_est.cpu().numpy().astype(np.float32)
                    # result = np.transpose(np.squeeze(result), (1, 2, 0))
                    # mat_dir = os.path.join(sample_path, name)
                    # save_matv73(mat_dir, var_name, result)               
    msg = f'mrae:{losses_mrae.avg}, rmse:{losses_rmse.avg}, psnr:{losses_psnr.avg}, ssim:{losses_ssim.avg}, sam:{losses_sam.avg}'
    print(msg)
    with open(os.path.join(sample_path, "results.txt"), "a") as f:
        f.write(msg)
    print(f"Your samples are ready and waiting for you here: \n{outpath} \n"
          f" \nEnjoy.")


if __name__ == "__main__":
    opt = parse_args()
    main(opt)
