from abc import ABC, abstractmethod
from typing import Tuple, Union

import torch
from torch import func, nn


class Contraction(ABC, nn.Module):
    """A space transformation (contraction)"""

    @abstractmethod
    def forward(
        self, x: torch.Tensor, output_jacobian: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Transforms points to contracted space

        Args:
            x: a tensor of 3D points in input space, shape [..., 3]
            output_jacobian: if True, also compute the jacobian matrix
                of the transformation

        Returns:
            x_contr: the 3D points transformed into contracted space, shape [..., 3]
            jacobian: the jacobian of the contraction computed at the
                input points, shape [..., 3, 3]. This is only present if
                output_jacobian is True
        """
        ...


class SphericalContraction(Contraction):
    """Contraction function from MipNeRF-360

    Implements the contraction in eq. 10 of https://arxiv.org/pdf/2111.12077.pdf:

                  { x / b                   if |x| <= b
        x_contr = {
                  { x (2 |x| - b) / |x|^2   if |x| > b

    This is appropriate for scenes where the object of interest is centered w.r.t.
    the model coordinates, all cameras lie close to the unit sphere and the
    (unbounded) background is never intended to be explored at test time.

    Args:
        radius: radius of the linear region (`b` in the formulas)
        norm_ord: order of the norm used in the contraction function, common options are
            `norm_ord=2.0` (MipNeRF-360) and `norm_ord=float("inf")` (hashgrid-based models)
    """

    def __init__(self, radius: float = 1.0, norm_ord: float = 2.0):
        super().__init__()
        self.radius = radius
        self.norm_ord = norm_ord

    def forward(
        self, x: torch.Tensor, output_jacobian: bool = False, get_multiplier: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        eps = torch.finfo(x.dtype).eps

        def contract(_x: torch.Tensor) -> torch.Tensor:
            # Compute normalization factor
            x_mag = torch.linalg.norm(_x, dim=-1, ord=self.norm_ord)
            x_mag = x_mag.clamp(min=eps)
            norm_factor = (2 - self.radius / x_mag) / x_mag

            # Apply contraction
            mask = x_mag <= self.radius
            if get_multiplier:
                return torch.where(
                    mask[..., None], 1 / self.radius, norm_factor[..., None]
                )
            else:
                return torch.where(
                    mask[..., None], _x / self.radius, norm_factor[..., None] * _x
                )

        if output_jacobian:
            jacobian = func.vmap(func.jacrev(contract))(x.view(-1, x.size(-1)))
            jacobian = jacobian.view(x.shape[:-1] + (x.size(-1), x.size(-1)))

            return contract(x), jacobian
        else:
            return contract(x)