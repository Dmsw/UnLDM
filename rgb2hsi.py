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

from utils import AverageMeter, initialize_logger, save_checkpoint, record_loss, \
    time2file_name, Loss_MRAE, Loss_RMSE, Loss_PSNR, Loss_SAM, Loss_SSIM, save_matv73

import pandas as pd
import h5py
import cv2 

from ldm.models.manifold_assemble import ManifoldAssemble

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
    
    data = pd.read_csv("/home/root/dataset/spectral-response-function/RGB_Camera_QE.csv", delimiter=",", index_col=0)
    P = data.loc[400:701]
    def group_by_tens(idx):
        return idx // 10

    # 使用 groupby 和 sum 方法每 10 个索引值求和
    P = P.groupby(group_by_tens(P.index)).sum()
    P = P.to_numpy()
    P = P[:, [0, 1, 3]]

    P = P / np.sum(P, axis=0, keepdims=True)
    P = torch.from_numpy(P.T).float().to(device)
    manifold_assemble = ManifoldAssemble(P)

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
                rgb_posterior = model.encode_first_stage(rgb * 2 - 1)
                z_rgb = model.get_first_stage_encoding(rgb_posterior)
                uc = model.get_unconditional_conditioning(rgb.shape[0])
                c = {'c_concat': [z_rgb], 'c_crossattn': [uc]}
                rgb_f = torch.einsum('b c h w, l c -> b l h w', hsi, P)
                # assert torch.allclose(rgb_t.to(torch.float32), rgb.to(torch.float32), atol=1e-2), f"error: {torch.abs(rgb_t - rgb).max()}"
                shape = z_rgb.shape[1:]
                # rgb = rgb_f
                output = torch.zeros_like(hsi)
                samples = []
                for n in range(opt.n_iter):
                    z_hsi, _ = sampler.sample(S=opt.steps,
                                                     conditioning=c,
                                                     batch_size=z_rgb.shape[0],
                                                     shape=shape,
                                                     verbose=False,
                                                     eta=opt.ddim_eta)
                    
                    ps_hsi = model.decode_first_stage_hsi(z_hsi)
                    hsi_est = model.spectral_model.decode(ps_hsi, rgb)
                    # hsi_est = hsi
                    samples.append(hsi_est)
                    # output.append(hsi_est)
                    output += hsi_est
                output /= opt.n_iter
                
                # manifold_est = manifold_assemble(samples, rgb)
                # output = torch.cat(output, dim=0).cpu()
                # err = torch.abs(output - hsi) ** 2
                # err = torch.sum(err, dim=1)
                # idx = torch.argmin(err, dim=0)
                # hsi_est = output[idx]
                # hsi_est = output / opt.n_iter
                hsi_est = output
                hsi_est = torch.clamp(hsi_est, 0, 1)
                hsi = torch.clamp(hsi, 0, 1)
                rgb_t = torch.einsum('b c h w, l c -> b l h w', hsi_est, P)
                print(f"rgb error: {torch.pow(rgb_t - rgb, 2).mean()}")
                # loss_mrae = criterion_mrae(hsi_est, hsi)
                # loss_rmse = criterion_rmse(hsi_est, hsi)
                # loss_psnr = criterion_psnr(hsi_est, hsi)
                # loss_ssim = criterion_ssim(hsi_est, hsi)
                # loss_sam = criterion_sam(hsi_est, hsi)
                # loss_mrae = criterion_mrae(hsi_est[:, :, :482], hsi[:, :, :482])
                # loss_rmse = criterion_rmse(hsi_est[:, :, :482], hsi[:, :, :482])
                # loss_psnr = criterion_psnr(hsi_est[:, :, :482], hsi[:, :, :482])
                # loss_ssim = criterion_ssim(hsi_est[:, :, :482], hsi[:, :, :482])
                # loss_sam = criterion_sam(hsi_est[:, :, :482], hsi[:, :, :482])
                loss_mrae = criterion_mrae(hsi_est[:, :, 128:-128, 128:-128], hsi[:, :, 128:-128, 128:-128])
                loss_rmse = criterion_rmse(hsi_est[:, :, 128:-128, 128:-128], hsi[:, :, 128:-128, 128:-128])
                loss_psnr = criterion_psnr(hsi_est[:, :, 128:-128, 128:-128], hsi[:, :, 128:-128, 128:-128])
                loss_ssim = criterion_ssim(hsi_est[:, :, 128:-128, 128:-128], hsi[:, :, 128:-128, 128:-128])
                loss_sam = criterion_sam(hsi_est[:, :, 128:-128, 128:-128], hsi[:, :, 128:-128, 128:-128])
                msg = f'name: {name}, mrae:{loss_mrae}, rmse:{loss_rmse}, psnr:{loss_psnr}, ssim:{loss_ssim}, sam:{loss_sam}\n'
                with open(os.path.join(sample_path, "results.txt"), "a") as f:
                    f.write(msg)
                
                losses_mrae.update(loss_mrae.data)
                losses_rmse.update(loss_rmse.data)
                losses_psnr.update(loss_psnr.data)
                losses_ssim.update(loss_ssim.data)
                losses_sam.update(loss_sam.data)
                if True:
                    result = hsi_est.cpu().numpy().astype(np.float32)
                    result = np.transpose(np.squeeze(result), (1, 2, 0))
                    result = np.clip(result, 0, 1)
                    mat_dir = os.path.join(sample_path, name)
                    save_matv73(mat_dir, var_name, result)
                    
    msg = f'mrae:{losses_mrae.avg}, rmse:{losses_rmse.avg}, psnr:{losses_psnr.avg}, ssim:{losses_ssim.avg}, sam:{losses_sam.avg}'
    print(msg)
    with open(os.path.join(sample_path, "results.txt"), "a") as f:
        f.write(msg)
    print(f"Your samples are ready and waiting for you here: \n{outpath} \n"
          f" \nEnjoy.")


if __name__ == "__main__":
    opt = parse_args()
    main(opt)
