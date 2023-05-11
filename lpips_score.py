import torch
import torchvision.transforms as transforms
from loss import perceptual_loss as ps
from PIL import Image
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from pytorch_msssim import ms_ssim


def compute_metrics(
    a: Union[np.array, Image.Image],
    b: Union[np.array, Image.Image],
    max_val: float = 255.0,
) -> Tuple[float, float]:
    """Returns PSNR and MS-SSIM between images `a` and `b`. """
    if isinstance(a, Image.Image):
        a = np.asarray(a)
    if isinstance(b, Image.Image):
        b = np.asarray(b)

    a = torch.from_numpy(a.copy()).float().unsqueeze(0)
    if a.size(3) == 3:
        a = a.permute(0, 3, 1, 2)
    b = torch.from_numpy(b.copy()).float().unsqueeze(0)
    if b.size(3) == 3:
        b = b.permute(0, 3, 1, 2)

    mse = torch.mean((a - b) ** 2).item()
    p = 20 * np.log10(max_val) - 10 * np.log10(mse)
    m = ms_ssim(a, b, data_range=max_val).item()
    return p, m

lpips = ps.PerceptualLoss(model='net-lin', net='vgg',
                               use_gpu=torch.cuda.is_available(),gpu_ids=0)


# img1 = Image.open('/root/workspace/ImageQuality/kodak/kodim14.png')
# img2 = Image.open('/root/workspace/ImageQuality/IQT-main/experiment/elic_char2e6_lp1_sty1e2_gan1_0016/900/kodim14.png')

# psnr, ssim = compute_metrics(img1, img2)

# transform =transforms.Compose([
#     transforms.ToTensor(),
#     ])

# img1_tensor = transform(img1).unsqueeze(0)
# img2_tensor = transform(img2).unsqueeze(0)


# lpips_res = lpips(img1_tensor, img2_tensor)

# print("psnr:%.3f, ms-ssim:%.3f, lpips:%.3f"%(psnr, ssim, lpips_res.item()))
