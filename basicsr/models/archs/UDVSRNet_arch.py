"""
Video Super-Resolution Architecture - v5 (Modified per Architecture Diagrams)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math

##########################################################################
## Layer Normalization
class LayerNorm(nn.Module):
    def __init__(self, dim):
        super(LayerNorm, self).__init__()
        self.body = nn.LayerNorm(dim)

    def forward(self, x):
        # Expects (B, C, H, W)
        # Permute to (B, H, W, C) for LayerNorm, then back
        return self.body(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class GDFN(nn.Module):
    def __init__(self, dim, ffn_expansion_factor=2.66, bias=False):
        super(GDFN, self).__init__()
        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, 
                                stride=1, padding=1, groups=hidden_features * 2, bias=bias)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


##########################################################################
## Phase Extraction Module (PEM) - FFT Based
class PhaseExtractionModule(nn.Module):
    def __init__(self, dim):
        super(PhaseExtractionModule, self).__init__()
        self.post_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=False)
        
    def forward(self, x):
        x = x.contiguous()
        fft_x = torch.fft.rfft2(x, norm='backward')
        phase = torch.angle(fft_x)
        # Set magnitude M_z = |C| (1.0), retain phase
        magnitude = torch.ones_like(phase)
        complex_spec = torch.polar(magnitude, phase)
        x_phase = torch.fft.irfft2(complex_spec.contiguous(), s=x.shape[-2:], norm='backward')        
        out = self.post_conv(x_phase)
        return out, phase


##########################################################################
## Phase-based Multi-head Self Attention (PMSA)
class PMSA(nn.Module):
    def __init__(self, dim, num_heads=8, bias=False):
        super(PMSA, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.pem = PhaseExtractionModule(dim)
        
        # Q and K use the extracted phase (F_p), V uses standard feature (F)
        self.q_proj = nn.Sequential(nn.Conv2d(dim, dim, 1, bias=bias), 
                                    nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=bias))
        self.k_proj = nn.Sequential(nn.Conv2d(dim, dim, 1, bias=bias), 
                                    nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=bias))
        self.v_proj = nn.Sequential(nn.Conv2d(dim, dim, 1, bias=bias), 
                                    nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=bias))
                                     
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape
        x_phase, _ = self.pem(x)
        
        q = self.q_proj(x_phase)
        k = self.k_proj(x_phase)
        v = self.v_proj(x)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


##########################################################################
## Phase-Based Transformer Block (PBTB)
class PhaseBasedTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=8, ffn_expansion_factor=2.66, bias=False):
        super(PhaseBasedTransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim)
        self.attn = PMSA(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim)
        self.ffn = GDFN(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


##########################################################################
## Temporal Haar Wavelet Transform (Forward & Inverse)
class TemporalHaarDWT(nn.Module):
    def __init__(self):
        super(TemporalHaarDWT, self).__init__()
        self.sqrt2 = 1.4142135623730951
        
    def forward_step(self, frame_list):
        lows, highs = [], []
        for i in range(0, len(frame_list), 2):
            f1, f2 = frame_list[i], frame_list[i + 1]
            L = (f1 + f2) / self.sqrt2
            H = (f1 - f2) / self.sqrt2
            lows.append(L)
            highs.append(H)
        return lows, highs
        
    def forward(self, x):
        b, t, c, h, w = x.shape
        frames = [x[:, i] for i in range(t)]
        
        # Level 1 to 3
        L1, H1 = self.forward_step(frames)
        LL, LH = self.forward_step(L1)
        HL, HH = self.forward_step(H1)
        LLL, LLH = self.forward_step(LL)
        LHL, LHH = self.forward_step(LH)
        HLL, HLH = self.forward_step(HL)
        HHL, HHH = self.forward_step(HH)
        
        # Stack bands: Index 0 is LLL, 1-7 are High freqs
        bands = torch.stack([LLL[0], LLH[0], LHL[0], LHH[0], 
                             HLL[0], HLH[0], HHL[0], HHH[0]], dim=1)
        return bands

class InverseTemporalHaarDWT(nn.Module):
    def __init__(self):
        super(InverseTemporalHaarDWT, self).__init__()
        self.sqrt2 = 1.4142135623730951
        
    def inverse_step(self, lows, highs):
        recon_frames = []
        for L, H in zip(lows, highs):
            f1 = (L + H) / self.sqrt2
            f2 = (L - H) / self.sqrt2
            recon_frames.extend([f1, f2])
        return recon_frames

    def forward(self, bands):
        LLL, LLH = [bands[:, 0]], [bands[:, 1]]
        LHL, LHH = [bands[:, 2]], [bands[:, 3]]
        HLL, HLH = [bands[:, 4]], [bands[:, 5]]
        HHL, HHH = [bands[:, 6]], [bands[:, 7]]
        
        # Reconstruction Levels
        LL = self.inverse_step(LLL, LLH) 
        LH = self.inverse_step(LHL, LHH)
        HL = self.inverse_step(HLL, HLH)
        HH = self.inverse_step(HHL, HHH)
        
        L1 = self.inverse_step(LL, LH)   
        H1 = self.inverse_step(HL, HH)
        frames = self.inverse_step(L1, H1) 
        
        return torch.stack(frames, dim=1)


##########################################################################
## Temporal DWT Block
class TemporalDWTBlock(nn.Module):
    def __init__(self, dim):
        super(TemporalDWTBlock, self).__init__()
        
        self.dwt = TemporalHaarDWT()
        self.idwt = InverseTemporalHaarDWT()
        
        # Processing for High-Frequency Bands 
        self.norm_all = LayerNorm(dim)
        
        self.norm1 = LayerNorm(dim)
        self.conv1 = nn.Sequential(nn.Conv2d(dim, 16, kernel_size=1), nn.PReLU())
        
        self.norm2 = LayerNorm(16)
        self.conv2 = nn.Sequential(nn.Conv2d(16, 8, kernel_size=3, padding=1), nn.PReLU())
        
        self.norm3 = LayerNorm(8)
        self.conv3 = nn.Sequential(nn.Conv2d(8, dim, kernel_size=1), nn.PReLU())
        
        # Aggregation Block (After IDWT)
        self.norm_agg = LayerNorm(dim * 8) 
        self.agg_reduce = nn.Conv2d(dim * 8, dim, kernel_size=1)
        self.agg_dw = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.agg_act = nn.Sigmoid()

    def forward(self, x):
        b, t, c, h, w = x.shape
        
        # 1. Temporal DWT: (B, 8, C, H, W)
        bands = self.dwt(x)
        
        # 2. Global LayerNorm on all bands
        bands_flat = bands.contiguous().view(b * 8, c, h, w)
        bands_normed = self.norm_all(bands_flat).view(b, 8, c, h, w)
        
        lll = bands_normed[:, 0:1] 
        highs = bands_normed[:, 1:] 
        
        # 3. Process High Freqs
        highs_reshaped = highs.contiguous().reshape(b * 7, c, h, w)
        
        out1 = self.conv1(self.norm1(highs_reshaped))
        out2 = self.conv2(self.norm2(out1))
        out3 = self.conv3(self.norm3(out2))
        
        highs_processed = out3.reshape(b, 7, c, h, w)
        
        # 4. Concatenate and Inverse DWT
        bands_processed = torch.cat([lll, highs_processed], dim=1)
        x_recon = self.idwt(bands_processed) 
        
        # 5. Temporal Aggregation / Attention Map Generation
        x_flat = x_recon.reshape(b, t * c, h, w)
        x_agg = self.norm_agg(x_flat)
        x_agg = self.agg_reduce(x_agg)
        x_agg = self.agg_dw(x_agg)
        x_agg = self.agg_act(x_agg) 
        
        out = x_agg.unsqueeze(1) # (B, 1, C, H, W)
        return out 


##########################################################################
## Spatial Phase Block
class SpatialPhaseBlock(nn.Module):
    def __init__(self, dim, num_blocks=4, num_heads=8, ffn_expansion_factor=2.66):
        super(SpatialPhaseBlock, self).__init__()
        
        self.blocks = nn.ModuleList([
            PhaseBasedTransformerBlock(dim, num_heads, ffn_expansion_factor)
            for _ in range(num_blocks)
        ])
        
    def forward(self, x):
        identity = x
        b, t, c, h, w = x.shape
        x = x.reshape(b * t, c, h, w)
        for block in self.blocks:
            x = block(x)
        x = x.reshape(b, t, c, h, w)
        return identity + x


##########################################################################
## Spatio Temporal Block (Parallel combination)
class SpatioTemporalBlock(nn.Module):
    def __init__(self, dim, num_spatial_blocks=4, num_heads=8, ffn_expansion_factor=2.66):
        super(SpatioTemporalBlock, self).__init__()
        self.temporal_path = TemporalDWTBlock(dim)
        self.spatial_path = SpatialPhaseBlock(dim, num_spatial_blocks, num_heads, ffn_expansion_factor)

    def forward(self, x):
        t_out = self.temporal_path(x) # (B, 1, C, H, W)
        s_out = self.spatial_path(x)  # (B, T, C, H, W)
        
        # Fusion via Broadcast Element-wise Multiplication (Circle Dot)
        return s_out * t_out


##########################################################################
## Refinement Block
class RefinementBlock(nn.Module):
    def __init__(self, dim, num_blocks=3, num_heads=8, ffn_expansion_factor=2.66):
        super(RefinementBlock, self).__init__()
        
        self.blocks = nn.ModuleList([
            PhaseBasedTransformerBlock(dim, num_heads, ffn_expansion_factor)
            for _ in range(num_blocks)
        ])
        
    def forward(self, x):
        identity = x
        b, t, c, h, w = x.shape
        x = x.reshape(b * t, c, h, w)
        for block in self.blocks:
            x = block(x)
        x = x.reshape(b, t, c, h, w)
        return identity + x


##########################################################################
## Learnable Spectral Upsampling (LSU)
class LearnableSpectralUpsampler(nn.Module):
    def __init__(self, channels, scale_factor):
        super(LearnableSpectralUpsampler, self).__init__()
        self.scale = scale_factor
        
        # Fusion Block: 1x1 -> LReLU -> 1x1
        self.fusion = nn.Sequential(
            nn.Conv2d(channels * 2, channels * 2, kernel_size=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels * 2, channels * 2, kernel_size=1)
        )
        
        self.post = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x):
        b, c, h, w = x.shape
        target_h, target_w = h * self.scale, w * self.scale
        target_w_half = target_w // 2 + 1
        
        # 1. Transform to Frequency Domain
        fft_x = torch.fft.rfft2(x, norm='ortho')
        mag_x = torch.abs(fft_x)
        pha_x = torch.angle(fft_x)
        
        # 2A. Smoothly Interpolate Phase
        pha_up = F.interpolate(pha_x, size=(target_h, target_w_half), mode='bicubic', align_corners=False)
        
        # 2B. Spectral Zero-Pad (SZP) Magnitude
        pad_w = target_w_half - mag_x.shape[-1]
        pad_h = target_h - h
        mag_up = F.pad(mag_x, (0, pad_w, 0, pad_h), mode='constant', value=0)
        
        # 3. Concatenate and Refine
        fused = torch.cat([mag_up, pha_up], dim=1)
        res = self.fusion(fused)
        res_mag, res_pha = torch.chunk(res, 2, dim=1)
        
        # Ensure magnitude remains non-negative
        mag_hr = F.relu(mag_up + res_mag)
        pha_hr = pha_up + res_pha  
        
        # 4. Complex Reconstruction
        fft_hr = torch.polar(mag_hr, pha_hr)
        output = torch.fft.irfft2(fft_hr, s=(target_h, target_w), norm='ortho')
        
        return self.post(output)
    
##########################################################################
## Spatial Upsampling 
class SpatialUpsampling(nn.Module):
    def __init__(self, channels, scale_factor):
        super(SpatialUpsampling, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels * (scale_factor ** 2), kernel_size=3, padding=1, bias=False),
            nn.PixelShuffle(scale_factor)
        )

    def forward(self, x):
        return self.body(x)

##########################################################################
## Hybrid Fourier-Spatial Upsampling Block
class HybridFourierSpatialUpsampling(nn.Module):
    def __init__(self, channels, scale_factor):
        super(HybridFourierSpatialUpsampling, self).__init__()
        self.su = SpatialUpsampling(channels, scale_factor)
        self.dfu = LearnableSpectralUpsampler(channels, scale_factor) 
        self.reduce = nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False)
        
    def forward(self, x):
        out = torch.cat([self.su(x), self.dfu(x)], dim=1)
        return self.reduce(out)

##########################################################################
## Main Upsample Block
class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, scale_factor):
        super(UpsampleBlock, self).__init__()
        self.scale = scale_factor
        
        if scale_factor in [2, 3]:
            self.up_block = HybridFourierSpatialUpsampling(in_channels, scale_factor)
        elif scale_factor == 4:
            self.up_block1 = HybridFourierSpatialUpsampling(in_channels, 2)
            self.up_block2 = HybridFourierSpatialUpsampling(in_channels, 2)
        else:
            raise ValueError("Scale factor not supported")
            
    def forward(self, x):
        b, t, c, h, w = x.shape
        x = x.reshape(b * t, c, h, w)
        
        if self.scale in [2, 3]:
            x = self.up_block(x)
        elif self.scale == 4:
            x = self.up_block1(x)
            x = self.up_block2(x)
            
        _, c_out, h_out, w_out = x.shape
        x = x.reshape(b, t, c_out, h_out, w_out)
        return x


##########################################################################
## Main Video Super-Resolution Network v5 (Dual-Stream Architecture)
class VideoSRNet(nn.Module):
    def __init__(self, 
                 in_channels=3,
                 out_channels=3,
                 embed_dim=24,   # Updated to match 24ch layout
                 scale_factor=4, # Overall scale factor 's'
                 num_spatial_blocks=4,
                 num_temporal_blocks=1, 
                 num_refinement_blocks=3,
                 num_heads=8,
                 ffn_expansion_factor=2.66):
        super(VideoSRNet, self).__init__()
        
        self.scale = scale_factor
        
        # --- Top Stream ---
        self.conv_in_top = nn.Conv2d(in_channels, embed_dim, kernel_size=3, padding=1)
        self.st_block_top = SpatioTemporalBlock(embed_dim, num_spatial_blocks, num_heads, ffn_expansion_factor)
        
        # --- Bottom Stream ---
        self.conv_in_bottom = nn.Conv2d(in_channels, embed_dim * 2, kernel_size=3, padding=1)
        self.st_block_bottom = SpatioTemporalBlock(embed_dim * 2, num_spatial_blocks, num_heads, ffn_expansion_factor)
        self.refine_bottom = RefinementBlock(embed_dim * 2, num_refinement_blocks, num_heads, ffn_expansion_factor)
        self.upsample_bottom = UpsampleBlock(embed_dim * 2, scale_factor=2)
        
        
        # --- Fusion & Main Upsampling Path ---
        # 1x1 conv to reduce concatenated channels (24+24=48) back to 24 before refinement
        self.fusion_reduce = nn.Conv2d(embed_dim * 3, embed_dim, kernel_size=1)
        self.refine_main = RefinementBlock(embed_dim, num_refinement_blocks, num_heads, ffn_expansion_factor)
        self.upsample_main = UpsampleBlock(embed_dim, scale_factor=scale_factor)
        
        # --- Skip Path Upsampling (Bicubic) ---
        self.upsample_skip = nn.Upsample(scale_factor=scale_factor, mode='bicubic', align_corners=False)
        
        # --- Final 3x3 Conv Output ---
        self.conv_out = nn.Conv2d(embed_dim, out_channels, kernel_size=3, padding=1)
        
    def forward(self, x):
        # x: (B, T, 3, 256, 256)
        b, t, c, h, w = x.shape
        x_flat = x.reshape(b * t, c, h, w)
        
        # --- Bottom Input Generation (Downsample to 128x128) ---
        x_down = F.interpolate(x_flat, scale_factor=0.5, mode='bicubic', align_corners=False)
        x_down = x_down.reshape(b, t, c, h // 2, w // 2)
        
        # --- Top Stream ---
        feat_top_flat = self.conv_in_top(x_flat)
        feat_top = feat_top_flat.reshape(b, t, -1, h, w) # (B, T, 24, 256, 256)
        st_top = self.st_block_top(feat_top)             # (B, T, 24, 256, 256)
        
        # --- Bottom Stream ---
        feat_bottom = self.conv_in_bottom(x_down.reshape(b * t, c, h // 2, w // 2)).reshape(b, t, -1, h // 2, w // 2)
        st_bottom = self.st_block_bottom(feat_bottom)
        ref_bottom = self.refine_bottom(st_bottom)
        up_bottom = self.upsample_bottom(ref_bottom)
              
        
        # --- Fusion ---
        fused = torch.cat([st_top, up_bottom], dim=2)
        fused_flat = fused.reshape(b * t, -1, h, w)
        fused_reduced = self.fusion_reduce(fused_flat).reshape(b, t, -1, h, w) # (B, T, 24, 256, 256)
        
        ref_main = self.refine_main(fused_reduced)
        main_up = self.upsample_main(ref_main) # (B, T, 24, sH, sW)
        
        # --- Skip Path Upsample ---
        # Starts from the 24ch top feature right before ST Block as drawn
        skip_up_flat = self.upsample_skip(feat_top_flat)
        skip_up = skip_up_flat.reshape(b, t, -1, h * self.scale, w * self.scale)
        
        # --- Global Addition & Final Projection ---
        out_feat = main_up + skip_up
        out_feat_flat = out_feat.reshape(b * t, -1, h * self.scale, w * self.scale)
        out = self.conv_out(out_feat_flat).reshape(b, t, -1, h * self.scale, w * self.scale)
        
        return out

r'''
if __name__ == '__main__':
    print("Testing Updated VideoSRNet v5 Architecture...")
    # Matches original overall architecture diagram test shapes
    model = VideoSRNet(embed_dim=24, scale_factor=4) 
    x = torch.randn(1, 8, 3, 256, 256) 
    y = model(x)
    print(f"Input: {x.shape}")
    print(f"Output: {y.shape}")'''