import torch


def apply_feature_mask(
    x: torch.Tensor, coalition_mask: torch.Tensor, baseline: torch.Tensor
) -> torch.Tensor:
    keep = coalition_mask.to(dtype=torch.bool, device=x.device)
    baseline = baseline.to(dtype=x.dtype, device=x.device)
    if keep.ndim == 1:
        keep = keep.unsqueeze(0).expand_as(x)
    if baseline.ndim == 1:
        baseline = baseline.unsqueeze(0).expand_as(x)
    return torch.where(keep, x, baseline)
