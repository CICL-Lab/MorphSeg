import torch
import torch.nn as nn
import torch.nn.functional as F


import torch
import torch.nn as nn
import torch.nn.functional as F

class CoordAttBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(CoordAttBlock, self).__init__()
        self.channels = channels
        self.reduction = reduction

        self.fusion_conv = nn.Sequential(
            nn.Conv3d(channels, channels // reduction, kernel_size=1),  # C->C/r
            nn.InstanceNorm3d(channels // reduction),
            nn.LeakyReLU(inplace=True),
            # nn.Conv3d(channels // reduction, channels, kernel_size=1)  # C通道
        )

        self.depth_conv = nn.Conv3d(channels // reduction, channels, kernel_size=1)  # C/r -> C
        self.height_conv = nn.Conv3d(channels // reduction, channels, kernel_size=1)
        self.width_conv = nn.Conv3d(channels // reduction, channels, kernel_size=1)

    def forward(self, x):
        b, c, d, h, w = x.shape
        # print(f"Input shape: {x.shape}")  # (B, C, D, H, W)

        # --- (B, C, D, 1, 1), (B, C, 1, H, 1), (B, C, 1, 1, W) ---
        depth_feat = F.adaptive_avg_pool3d(x, (d, 1, 1))
        height_feat = F.adaptive_avg_pool3d(x, (1, h, 1))
        width_feat = F.adaptive_avg_pool3d(x, (1, 1, w))
        # print(f"Depth pooled shape: {depth_feat.shape}")  # (B, C, D, 1, 1)
        # print(f"Height pooled shape: {height_feat.shape}")  # (B, C, 1, H, 1)
        # print(f"Width pooled shape: {width_feat.shape}")  # (B, C, 1, 1, W)

        # --- (B, C, 1, 1, D), (B, C, 1, 1, H), (B, C, 1, 1, W) ---
        depth_seq = depth_feat.permute(0, 1, 4, 3, 2)  # 交换维度
        height_seq = height_feat.permute(0, 1, 4, 2, 3)
        width_seq = width_feat.permute(0, 1, 2, 3, 4)
        # print(f"Depth seq shape: {depth_seq.shape}")  # (B, C, 1, 1, D)
        # print(f"Height seq shape: {height_seq.shape}")  # (B, C, 1, 1, H)
        # print(f"Width seq shape: {width_seq.shape}")  # (B, C, 1, 1, W)

        concat_feat = torch.cat([depth_seq, height_seq, width_seq], dim=-1)  # (B,C,1,1,D+H+W)
        # print(f"Concatenated feature shape: {concat_feat.shape}")  # (B, C, 1, 1, D+H+W)

        fused = self.fusion_conv(concat_feat)  # (B, C, 1, 1, D+H+W)
        # print(f"Fused shape: {fused.shape}")  # (B, C, 1, 1, D+H+W)

        depth_fused, height_fused, width_fused = torch.split(fused, [d, h, w], dim=-1)
        # print(f"Depth fused shape: {depth_fused.shape}")  # (B, C, 1, 1, D)
        # print(f"Height fused shape: {height_fused.shape}")  # (B, C, 1, 1, H)
        # print(f"Width fused shape: {width_fused.shape}")  # (B, C, 1, 1, W)

        # --- (B, C, D, 1, 1), (B, C, 1, H, 1), (B, C, 1, 1, W) ---
        depth_fused = depth_fused.permute(0, 1, 4, 3, 2)
        height_fused = height_fused.permute(0, 1, 3, 4, 2)
        width_fused = width_fused.permute(0, 1, 2, 3, 4)
        # print(f"Depth fused (reordered) shape: {depth_fused.shape}")  # (B, C, D, 1, 1)
        # print(f"Height fused (reordered) shape: {height_fused.shape}")  # (B, C, 1, H, 1)
        # print(f"Width fused (reordered) shape: {width_fused.shape}")  # (B, C, 1, 1, W)

        depth_weight = self.depth_conv(depth_fused).sigmoid()
        height_weight = self.height_conv(height_fused).sigmoid()
        width_weight = self.width_conv(width_fused).sigmoid()
        # print(f"Depth weight shape: {depth_weight.shape}")  # (B, C, D, 1, 1)
        # print(f"Height weight shape: {height_weight.shape}")  # (B, C, 1, H, 1)
        # print(f"Width weight shape: {width_weight.shape}")  # (B, C, 1, 1, W)

        # out = x * depth_weight * height_weight * width_weight
        # print(f"Output shape: {out.shape}")  # (B, C, D, H, W)
        return width_weight, height_weight, depth_weight


class GlobalAgg(nn.Module):
    def __init__(self, in_channels,num_heads=8):
        super(GlobalAgg, self).__init__()
        self.CA = CoordAttBlock(in_channels)

        self.cross_attn_x = nn.MultiheadAttention(in_channels, num_heads, batch_first=True)
        self.cross_attn_y = nn.MultiheadAttention(in_channels, num_heads, batch_first=True)
        self.cross_attn_z = nn.MultiheadAttention(in_channels, num_heads, batch_first=True)

    def forward(self, f):
        """
               x: (b, c, d, 1, 1) -> Query
               y: (b, c, 1, h, 1) -> Key
               z: (b, c, 1, 1, w) -> Value
               """
        z, y, x = self.CA(f)
        # print(z.shape)
        # print(y.shape)
        # print(x.shape)


        # print(x.shape)
        # print(y.shape)
        # print(z.shape)

        b, c, d, _, _ = x.shape
        _, _, _, h, _ = y.shape
        _, _, _, _, w = z.shape


        # Reshape to (b, seq_len, embed_dim)
        x = x.view(b, c, d).permute(0, 2, 1)  # (b, d, c)
        y = y.view(b, c, h).permute(0, 2, 1)  # (b, h, c)
        z = z.view(b, c, w).permute(0, 2, 1)  # (b, w, c)

        # Compute cross-attention
        attn_x, _ = self.cross_attn_x(x, y, y)  # (b, d, c)
        attn_x, _ = self.cross_attn_x(attn_x, z, z)  # (b, h, c)
        attn_y, _ = self.cross_attn_y(y, x, x)  # (b, h, c)
        attn_y, _ = self.cross_attn_y(attn_y, z, z)  # (b, d, c)
        attn_z, _ = self.cross_attn_z(z, x, x)  # (b, w, c)
        attn_z, _ = self.cross_attn_z(attn_z, y, y)  # (b, w, c)

        # print(attn_x.shape)
        # print(attn_y.shape)
        # print(attn_z.shape)

        # Restore shape if needed
        attn_x = attn_x.permute(0, 2, 1).view(b, c, d, 1, 1)
        attn_y = attn_y.permute(0, 2, 1).view(b, c, h, 1, 1).permute(0, 1, 3, 2, 4)
        attn_z = attn_z.permute(0, 2, 1).view(b, c, w, 1, 1).permute(0, 1, 4, 3, 2)

        # print(attn_x.shape)
        # print(attn_y.shape)
        # print(attn_z.shape)

        return f * attn_x * attn_y * attn_z



if __name__ == '__main__':

    B, C, D, W, H = 4, 16, 32, 64, 128
    x = torch.randn(B, C, D, W, H)

    model = GlobalAgg(in_channels=16)
    output = model(x)



