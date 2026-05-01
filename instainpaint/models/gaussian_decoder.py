import torch
from torch import nn

from ..geometry.utils import build_pytorch_mlp


class GaussianMlpUpsampler(nn.Module):
    def __init__(
        self,
        token_dim=1024,
        mlp_dim=1024,
        mlp_depth=1,
        patch_size=8,
        norm_layer=nn.LayerNorm,
        depth_near=0.0,
        depth_far=500,
        depth_bias=-4.0,
        scale_bias=-2.3,
        opacity_bias=-2.0,
        norm_use_bias=False,
        norm_use_affine=False,
        use_weight_norm=True,
        input_image_num=4,
        color_space="rgb",
        **unused,
    ):
        super().__init__()
        if color_space != "rgb":
            raise ValueError("Only RGB Gaussian color is supported in this runtime.")
        self.patch_size = patch_size
        self.input_image_num = input_image_num
        self.depth_near = depth_near
        self.depth_far = depth_far
        self.depth_bias = depth_bias
        self.scale_bias = scale_bias
        self.opacity_bias = opacity_bias
        self.color_dim = 3
        self.vary_view_joint_train = False
        self.norm = norm_layer(token_dim, elementwise_affine=norm_use_affine, bias=norm_use_bias)

        patch_area = patch_size * patch_size
        self.mlp_depth = self._head(token_dim, mlp_dim, patch_area, mlp_depth, use_weight_norm)
        self.mlp_rgb = self._head(token_dim, mlp_dim, patch_area * self.color_dim, mlp_depth, use_weight_norm)
        self.mlp_opacity = self._head(token_dim, mlp_dim, patch_area, mlp_depth, use_weight_norm)
        self.mlp_scale = self._head(token_dim, mlp_dim, patch_area * 3, mlp_depth, use_weight_norm)
        self.mlp_rotation = self._head(token_dim, mlp_dim, patch_area * 4, mlp_depth, use_weight_norm)

    @staticmethod
    def _head(token_dim, mlp_dim, out_dim, depth, use_weight_norm):
        return build_pytorch_mlp(
            token_dim,
            mlp_dim,
            out_dim,
            depth=depth,
            bias=False,
            use_weight_norm=use_weight_norm,
        )

    def _tokens_to_frames(self, tokens, num_views, height, width):
        batch = tokens.shape[0]
        patch = self.patch_size
        frames = tokens.reshape(batch, num_views, height // patch, width // patch, patch, patch, -1)
        frames = frames.permute(0, 1, 6, 2, 4, 3, 5)
        return frames.reshape(batch, num_views, -1, height, width)

    def _range_map(self, depth, opacity, scale):
        depth = torch.sigmoid(depth + self.depth_bias)
        depth = (1.0 - depth) * self.depth_near + depth * self.depth_far
        opacity = torch.sigmoid(opacity + self.opacity_bias)
        scale = (scale + self.scale_bias).exp().clamp(min=1e-4, max=0.3)
        return depth, opacity, scale

    def forward(
        self,
        token,
        initial_token=None,
        cam=None,
        im=None,
        *unused,
        mode=None,
        **unused_kwargs,
    ):
        if im is None:
            raise ValueError("DynamicGaussianMlpUpsampler.forward requires the input image tensor.")
        _, num_views, _, height, width = im.shape
        token = self.norm(token)
        depth = self._tokens_to_frames(self.mlp_depth(token), num_views, height, width)
        rgb = self._tokens_to_frames(self.mlp_rgb(token), num_views, height, width)
        opacity = self._tokens_to_frames(self.mlp_opacity(token), num_views, height, width)
        scale = self._tokens_to_frames(self.mlp_scale(token), num_views, height, width)
        rotation = self._tokens_to_frames(self.mlp_rotation(token), num_views, height, width)
        depth, opacity, scale = self._range_map(depth, opacity, scale)
        return {
            "rgb": rgb,
            "depth": depth,
            "opacity": opacity,
            "scale": scale,
            "rotation": nn.functional.normalize(rotation, dim=2, eps=1e-5),
        }
