import torch
import torch.nn as nn
from MSDSA import MSDSA
from Fusion import ThreeFeatureFusion
from CrossAttention import AttentionBlock


class MDWA(nn.Module):
    """
    Multi-directional Wavelet Attention (MDWA)
    Consists of Multi-scale Sub-band Deformable Self-Attention (MSDSA)
    and Directional Sub-band Cross-Attention (DSCA) in parallel.
    Reference: Section 3.3 in the paper.
    """

    def __init__(self, in_channels):
        super(MDWA, self).__init__()
        # MSDSA operates on concatenated sub-bands (LL, LH, HL, HH)
        self.msdsa = MSDSA(in_channels * 4)

        # DSCA Branch (Cross-attention blocks)
        self.cross_attention = AttentionBlock(dim=in_channels, num_heads=4, bias=False)
        self.fusion_high = ThreeFeatureFusion(in_channels)  # Fuses LH, HL, HH

        # Learnable weighting parameters (alpha1, alpha2 in Eq. 6)
        self.alphas = nn.Parameter(torch.ones(2) * 0.5)

    def forward(self, ll, lh, hl, hh):
        """
        Args:
            ll, lh, hl, hh (Tensor): Sub-bands each of shape (B, C, H, W)
        Returns:
            Tuple: Four refined sub-bands (B, C, H, W)
        """
        B, C, H, W = ll.shape

        # --- Branch A: MSDSA ---
        combined_in = torch.cat([ll, lh, hl, hh], dim=1)  # (B, 4C, H, W)
        feat_msdsa = self.msdsa(combined_in)

        # --- Branch B: DSCA (Directional Cross-Attention) ---
        # 1. LH/HL/HH attend to LL (Sub-band specific relationships)
        out_lh = self.cross_attention(lh, ll)
        out_hl = self.cross_attention(hl, ll)
        out_hh = self.cross_attention(hh, ll)

        # 2. LL attends to the fused high-frequency information (Eq. Fig 2)
        high_freq_fused = self.fusion_high(lh, hl, hh)
        out_ll = self.cross_attention(ll, high_freq_fused)

        feat_dsca = torch.cat([out_ll, out_lh, out_hl, out_hh], dim=1)

        # --- Branch Fusion  ---
        # Final output is a weighted sum of both branches
        fused_output = self.alphas[0] * feat_msdsa + self.alphas[1] * feat_dsca

        # Split the fused 4C tensor back into four C tensors
        ll_hat, lh_hat, hl_hat, hh_hat = torch.chunk(fused_output, 4, dim=1)

        return ll_hat, lh_hat, hl_hat, hh_hat