import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class MSDSA(nn.Module):
    """
    Multi-scale Sub-band Deformable Self-Attention (MSDSA)
    Implementation of Equation (7) to (10) in the paper.
    Supports dynamic resolution caching for Manhattan Decay matrix.
    """

    def __init__(self, in_channels, n_heads=4, offset_range_factor=2.0):
        super(MSDSA, self).__init__()
        self.in_channels = in_channels
        self.n_heads = n_heads
        self.head_dim = in_channels // n_heads
        self.scale = self.head_dim ** -0.5
        self.offset_range_factor = offset_range_factor

        # 1. Multi-scale Offset Network
        self.offset_weights = nn.Parameter(torch.ones(3))  # Learnable weights for 3x3, 5x5, 7x7
        self.offset_conv3 = nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels, bias=False)
        self.offset_conv5 = nn.Conv2d(in_channels, in_channels, 5, padding=2, groups=in_channels, bias=False)
        self.offset_conv7 = nn.Conv2d(in_channels, in_channels, 7, padding=3, groups=in_channels, bias=False)
        self.offset_proj = nn.Conv2d(in_channels, 2, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

        # 2. Projections for Attention
        self.proj_q = nn.Conv2d(in_channels, in_channels, 1, bias=False)
        self.proj_k = nn.Conv2d(in_channels, in_channels, 1, bias=False)
        self.proj_v = nn.Conv2d(in_channels, in_channels, 1, bias=False)
        self.proj_out = nn.Conv2d(in_channels, in_channels, 1, bias=False)

        # 3. Manhattan Decay Parameters
        self.beta = nn.Parameter(torch.tensor(0.5))
        self._dist_cache = {}  # Cache for different resolutions (H, W)

    def _get_manhattan_dist(self, H, W, device):
        """
        Dynamically generates or retrieves Manhattan distance matrix from cache.
        """
        cache_key = (H, W, device)
        if cache_key not in self._dist_cache:
            # Standard Manhattan Grid Calculation: |dy| + |dx|
            y = torch.arange(H, device=device)
            x = torch.arange(W, device=device)
            grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')
            grid = torch.stack([grid_y, grid_x], dim=-1).view(-1, 2)  # (HW, 2)

            # Efficient broadcasting: (HW, 1, 2) - (1, HW, 2) -> (HW, HW)
            dist = torch.abs(grid.unsqueeze(1) - grid.unsqueeze(0)).sum(-1)
            self._dist_cache[cache_key] = dist
        return self._dist_cache[cache_key]

    def _get_ref_points(self, H, W, B, device):
        """ Generates normalized reference points for grid sampling. """
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, H - 0.5, H, device=device),
            torch.linspace(0.5, W - 0.5, W, device=device),
            indexing='ij'
        )
        ref = torch.stack((ref_y, ref_x), -1)  # (H, W, 2)
        ref[..., 1].div_(W - 1.0).mul_(2.0).sub_(1.0)  # Normalize to [-1, 1]
        ref[..., 0].div_(H - 1.0).mul_(2.0).sub_(1.0)
        return ref[None, ...].expand(B, -1, -1, -1)

    def forward(self, x):
        B, C, H, W = x.size()
        device = x.device

        # --- Step 1: Offset Generation ---
        off_feat = self.offset_conv3(x) * self.offset_weights[0] + \
                   self.offset_conv5(x) * self.offset_weights[1] + \
                   self.offset_conv7(x) * self.offset_weights[2]

        offset = self.offset_proj(self.relu(off_feat))  # (B, 2, H, W)
        offset = rearrange(offset, 'b c h w -> b h w c')

        # --- Step 2: Deformable Sampling  ---
        ref_points = self._get_ref_points(H, W, B, device)
        # Apply tanh to restrict offset range and scale by range_factor
        sampling_grid = (ref_points + offset.tanh() * self.offset_range_factor).clamp(-1, 1)

        # Grid sample expects (x, y) coordinates
        x_sampled = F.grid_sample(x, sampling_grid[..., [1, 0]], mode='bilinear', align_corners=True)

        # --- Step 3: Manhattan Decay ---
        dist_matrix = self._get_manhattan_dist(H, W, device)
        # Safe exponential decay with learnable beta
        decay_matrix = torch.exp(-dist_matrix / (self.beta.abs() + 1e-6))

        # --- Step 4: Refined Attention ---
        q = self.proj_q(x).view(B * self.n_heads, self.head_dim, H * W)
        k = self.proj_k(x_sampled).view(B * self.n_heads, self.head_dim, H * W)
        v = self.proj_v(x_sampled).view(B * self.n_heads, self.head_dim, H * W)

        # q.T * k: (B*h, HW, HW)
        attn = torch.matmul(q.transpose(-2, -1), k) * self.scale
        # Add Manhattan decay matrix M
        attn = attn + decay_matrix.unsqueeze(0)
        attn = attn.softmax(dim=-1)

        out = torch.matmul(v, attn.transpose(-2, -1))
        out = rearrange(out, '(b h) d (height width) -> b (h d) height width', h=self.n_heads, height=H, width=W)

        return self.proj_out(out)