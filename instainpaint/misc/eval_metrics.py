
from functools import cache
import torch
from lpips import LPIPS
from skimage.metrics import structural_similarity


@torch.no_grad()
def compute_psnr(
    ground_truth,
    predicted,
    mask=None,
):
    ground_truth = ground_truth.clip(min=0, max=1)
    predicted = predicted.clip(min=0, max=1)
    if mask is not None:
        mse = []
        for g,p,m in zip(ground_truth,predicted,mask):
            v = ((g[m > 0.5] - p[m > 0.5]) ** 2).mean()
            mse.append(v)
        mse = torch.Tensor(mse)
    else:
        mse = ((ground_truth - predicted) ** 2).mean([1,2,3])
    return -10 * mse.log10()


@cache
def get_lpips(device: torch.device):
    return LPIPS(net="vgg", spatial=True).to(device)


@torch.no_grad()
def compute_lpips(
    ground_truth,
    predicted,
    mask = None,
):
    if mask is not None:
        ground_truth = ground_truth * mask
        predicted = predicted * mask
    value = get_lpips(predicted.device).forward(ground_truth, predicted, normalize=True)
    if mask is not None:
        lpips = []
        for v,m in zip(value, mask):
            lpips.append(v[m > 0.5].mean())
        lpips = torch.Tensor(lpips)
    else:
        lpips = value.mean([1,2,3])
    return lpips


@torch.no_grad()
def compute_ssim(
    ground_truth,
    predicted,
    mask = None,
):
    assert mask is None, "Not implemented..."
    # # https://github.com/KAIR-BAIR/dycheck/blob/ddf77a4e006fdbc5aed28e0859c216da0de5aff5/dycheck/core/metrics/image.py#L49

    ssim = [
        structural_similarity(
            gt.detach().cpu().numpy(),
            hat.detach().cpu().numpy(),
            win_size=11,
            gaussian_weights=True,
            channel_axis=0,
            data_range=1.0,
        )
        for gt, hat in zip(ground_truth, predicted)
    ]
    return torch.tensor(ssim, dtype=predicted.dtype, device=predicted.device)
