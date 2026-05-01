import torch
import torch.nn as nn

from gsplat.rendering import rasterization


class GaussianRenderer(nn.Module):
    def __init__(self, view_dependent=False, znear=0.01, zfar=1000.0):
        super().__init__()
        self.znear = znear
        self.zfar = zfar
        self.view_dependent = view_dependent
        self.register_buffer("bg_color", torch.ones((1, 3), dtype=torch.float32))

    def compute_proj(self, tanfovx, tanfovy):
        top = tanfovy * self.znear
        bottom = -top
        right = tanfovx * self.znear
        left = -right

        P = torch.zeros(4, 4)
        z_sign = 1.0

        P[0, 0] = 2.0 * self.znear / (right - left)
        P[1, 1] = 2.0 * self.znear / (top - bottom)
        P[0, 2] = (right + left) / (right - left)
        P[1, 2] = (top + bottom) / (top - bottom)
        P[3, 2] = z_sign
        P[2, 2] = z_sign * self.zfar / (self.zfar - self.znear)
        P[2, 3] = -(self.zfar * self.znear) / (self.zfar - self.znear)
        return P

    def compute_intrinsic(self, tanfovx, tanfovy, im_height, im_width):
        Ks = torch.eye(3)
        Ks[0, 0] = im_width / tanfovx / 2.0
        Ks[1, 1] = im_height / tanfovy / 2.0
        Ks[0, 2] = im_width / 2.0
        Ks[1, 2] = im_height / 2.0
        return Ks

    def _background(self, viewmats):
        camera_dims = viewmats.shape[:-2]
        return self.bg_color.to(device=viewmats.device, dtype=viewmats.dtype).expand(*camera_dims, -1)

    def forward(self, camera, im_height, im_width, xyz, rgb, opacity, scale, rotation, render_depth=False, feature_render=False):
        # Set up rasterization configuration
        tanfovx = torch.tan(camera[16] * 0.5).item()
        tanfovy = torch.tan(camera[17] * 0.5).item()
        matrix = camera[0:16].reshape(4, 4).clone()
        matrix[:3, 1:3] = -matrix[:3, 1:3]
        extr = torch.inverse(matrix)  # World-to-camera following OpenCV convention
        extr = extr.unsqueeze(0)

        Ks = self.compute_intrinsic(tanfovx, tanfovy, im_height, im_width)
        Ks = Ks.to(device=extr.device, dtype=extr.dtype)
        Ks = Ks.unsqueeze(0)
        background = self._background(extr)

        opacity = opacity.squeeze(-1)

        render_mode = "RGB+ED" if render_depth else "RGB"

        if feature_render:
            render_mode = "RGB"
            out_img, out_alpha, _ = rasterization(
                means=xyz,
                quats=rotation,
                scales=scale,
                opacities=opacity,
                colors=rgb,
                viewmats=extr,
                Ks=Ks,
                width=im_width,
                height=im_height,
                near_plane=self.znear,
                far_plane=self.zfar,
                render_mode=render_mode,
                packed=False,
            )
        else:
            if self.view_dependent:
                rgb_shape = rgb.shape
                if rgb_shape[-1] != 27:
                    raise ValueError(
                        "For view depdent effect, we must use spherical harmonics of order 2."
                    )
                rgb = rgb.reshape(list(rgb_shape[:-1]) + [9, 3])

                out_img, out_alpha, _ = rasterization(
                    means=xyz,
                    quats=rotation,
                    scales=scale,
                    opacities=opacity,
                    colors=rgb,
                    sh_degree=2,
                    viewmats=extr,
                    Ks=Ks,
                    width=im_width,
                    height=im_height,
                    near_plane=self.znear,
                    far_plane=self.zfar,
                    backgrounds=background,
                    render_mode=render_mode,
                    packed=False,
                )
            else:
                out_img, out_alpha, _ = rasterization(
                    means=xyz,
                    quats=rotation,
                    scales=scale,
                    opacities=opacity,
                    colors=rgb,
                    viewmats=extr,
                    Ks=Ks,
                    width=im_width,
                    height=im_height,
                    near_plane=self.znear,
                    far_plane=self.zfar,
                    backgrounds=background,
                    render_mode=render_mode,
                    packed=False,
                )

        out_img = out_img.permute(0, 3, 1, 2)
        if not feature_render:
            if out_img.shape[1] == 4:
                out_img[:, :-1] = torch.clamp(2 * out_img[:, :-1] - 1, -1, 1)
            else:
                out_img = torch.clamp(2 * out_img - 1, -1, 1)
        out_alpha = out_alpha.permute(0, 3, 1, 2)

        return {"rgb": out_img, "mask": out_alpha}
