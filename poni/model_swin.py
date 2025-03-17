import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
# =========================== Sub-parts of the U-Net model ============================


class double_conv(nn.Module):
    """(conv => BN => ReLU) * 2"""

    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class embedconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        return self.conv(x)


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(nn.MaxPool2d(2), double_conv(in_ch, out_ch))

    def forward(self, x):
        x = self.mpconv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(up, self).__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)

        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2))

        # for padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x





##### module from swinunet
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH
        # print(" self.relative_position_bias_table.shape=", self.relative_position_bias_table.shape)


        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        # print("attn = ",attn.shape)
        # print("relative_position_bias=",relative_position_bias.shape,relative_position_bias.unsqueeze(0).shape)

        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops


class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops


class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.dim
        flops += (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        return flops

class PatchExpand(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.expand = nn.Linear(dim, 2*dim, bias=False) if dim_scale==2 else nn.Identity()
        self.norm = norm_layer(dim // dim_scale)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        # print("H,W,L=",H,W,L)
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=2, p2=2, c=C//4)
        x = x.view(B,-1,C//4)
        x= self.norm(x)

        return x
    
# class PatchExpand2(nn.Module):
#     def __init__(self, input_resolution, dim, dim_scale=2, norm_layer=nn.LayerNorm):
#         super().__init__()
#         self.input_resolution = input_resolution
#         self.dim = dim
#         self.expand = nn.Linear(dim, 2*dim, bias=False) if dim_scale==2 else nn.Identity()
#         self.norm = norm_layer(dim // dim_scale)

#     def forward(self, x):
#         """
#         x: B, H*W, C
#         """
#         H, W = self.input_resolution
#         x = self.expand(x)
#         B, L, C = x.shape
#         print("H,W,L=",H,W,L)
#         assert L == H * W, "input feature has wrong size"

#         x = x.view(B, H, W, C)
#         x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=2, p2=2, c=C//4)
#         x = x.view(B,-1,C//4)
#         x= self.norm(x)

#         return x



class FinalPatchExpand_X4(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(dim, 16*dim, bias=False)
        self.output_dim = dim 
        self.norm = norm_layer(self.output_dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.dim_scale, p2=self.dim_scale, c=C//(self.dim_scale**2))
        x = x.view(B,-1,self.output_dim)
        x= self.norm(x)

        return x

class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops


class BasicLayer_up(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        upsample (nn.Module | None, optional): upsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, upsample=None, use_checkpoint=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer
        if upsample is not None:
            self.upsample = PatchExpand(input_resolution, dim=dim, dim_scale=2, norm_layer=norm_layer)
        else:
            self.upsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.upsample is not None:
            x = self.upsample(x)
        return x



class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=448, patch_size=4, in_chans=17, embed_dim=192, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        # print("patches_resolution=",patches_resolution)
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]
        # print("num_patches=",self.num_patches)

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops
    
# ##### channel position encoding
# class ChannelPositionalEncoding(nn.Module):
#     def __init__(self, num_channels):
#         super(ChannelPositionalEncoding, self).__init__()
#         # 初始化一个可学习的位置编码权重
#         self.channel_weights = nn.Parameter(torch.ones(num_channels))

#     def forward(self,x):
#         # 将位置编码权重应用于输入的通道维度
#         return  x + self.channel_weights[None, :, None, None]
class ChannelPositionalEmbedding(nn.Module):
    def __init__(self, num_channels, embedding_size):
        super(ChannelPositionalEmbedding, self).__init__()
        # 创建一个位置编码为每个channel
        self.position_embed = nn.Parameter(torch.randn(1, num_channels, embedding_size))

    def forward(self, x):
        # x shape: [batch, num_patches, embedding_size]
        
        # 计算每个channel应该重复的次数
        repeat_times = x.shape[1] // self.position_embed.shape[1]
        
        # 扩展位置编码以匹配x的形状
        position_embed_expanded = self.position_embed.repeat(x.shape[0], repeat_times, 1)
        
        # 将位置编码添加到x上
        return x + position_embed_expanded


# =================================== Component modules ===============================


class UNetEncoder(nn.Module):
    def __init__(
        self,
        n_channels,
        nsf=16,
        embedding_size=64,
        map_size=400,

        img_size=448, patch_size=4, in_chans=17, num_classes=17,
        embed_dim=192, depths=[2, 2, 2, 2], depths_decoder=[2, 2, 2, 1], num_heads=[6, 12, 24, 48],
        window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
        drop_rate=0.2, attn_drop_rate=0., drop_path_rate=0.1,
        norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
        use_checkpoint=False, final_upsample="expand_first", **kwargs



    ):
        super().__init__()
        print("SwinTransformerSys expand initial----depths:{};depths_decoder:{};drop_path_rate:{};num_classes:{}".format(depths,
        depths_decoder,drop_path_rate,num_classes))

        # adding positional embeeding in layer

        self.embed = embedconv(n_channels, embedding_size)
        self.nsf = nsf
        self.inc = inconv(embedding_size, nsf)
        self.down1 = down(nsf, nsf * 2)
        self.down2 = down(nsf * 2, nsf * 4)
        self.down3 = down(nsf * 4, nsf * 8)
        self.down4 = down(nsf * 8, nsf * 8)
        self.map_size = map_size
        self.n_channels = n_channels


         # new self.
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.num_features_up = int(embed_dim * 2)
        self.mlp_ratio = mlp_ratio
        self.final_upsample = final_upsample

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # absolute position embedding (in spatial)
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        # absolute position embedding (in channel)
        self.channel_position_embed_original = nn.Parameter(torch.zeros(1, in_chans))
        trunc_normal_(self.channel_position_embed_original, std=.02)
        self.linear_expand = nn.Linear(in_chans, embed_dim)


        self.pos_drop = nn.Dropout(p=drop_rate)

        # pose embedding
        self.gps_layer = nn.Sequential(
            nn.Linear(2, 512), # (location: x, y)
            nn.LayerNorm(512)
        )
        self.compass_layer = nn.Sequential(
            nn.Linear(2, 512), # (heading: sin, cos)
            nn.LayerNorm(512)
        )

        #world_xyz embedding
        self.linear_xyz= nn.Linear(3, 480 * 480)

        self.linear_heading= nn.Linear(1, 480 * 480)




        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build encoder and bottleneck layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                               input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                 patches_resolution[1] // (2 ** i_layer)),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                               use_checkpoint=use_checkpoint)
            self.layers.append(layer)
        
        # build decoder layers for object PF
        self.layers_up = nn.ModuleList()
        self.concat_back_dim = nn.ModuleList()
        for i_layer in range(self.num_layers):
            concat_linear = nn.Linear(2*int(embed_dim*2**(self.num_layers-1-i_layer)),
            int(embed_dim*2**(self.num_layers-1-i_layer))) if i_layer > 0 else nn.Identity()
            if i_layer ==0 :
                # print(patches_resolution[0],self.num_layers,int(embed_dim * 2 ** (self.num_layers-1-i_layer)))
                layer_up = PatchExpand(input_resolution=(patches_resolution[0] // (2 ** (self.num_layers-1-i_layer)),
                patches_resolution[1] // (2 ** (self.num_layers-1-i_layer))), dim=int(embed_dim * 2 ** (self.num_layers-1-i_layer)), dim_scale=2, norm_layer=norm_layer)
            else:
                layer_up = BasicLayer_up(dim=int(embed_dim * 2 ** (self.num_layers-1-i_layer)),
                                input_resolution=(patches_resolution[0] // (2 ** (self.num_layers-1-i_layer)),
                                                    patches_resolution[1] // (2 ** (self.num_layers-1-i_layer))),
                                depth=depths[(self.num_layers-1-i_layer)],
                                num_heads=num_heads[(self.num_layers-1-i_layer)],
                                window_size=window_size,
                                mlp_ratio=self.mlp_ratio,
                                qkv_bias=qkv_bias, qk_scale=qk_scale,
                                drop=drop_rate, attn_drop=attn_drop_rate,
                                drop_path=dpr[sum(depths[:(self.num_layers-1-i_layer)]):sum(depths[:(self.num_layers-1-i_layer) + 1])],
                                norm_layer=norm_layer,
                                upsample=PatchExpand if (i_layer < self.num_layers - 1) else None,
                                use_checkpoint=use_checkpoint)
            self.layers_up.append(layer_up)
            self.concat_back_dim.append(concat_linear)

        self.norm = norm_layer(self.num_features)
        self.norm_up= norm_layer(self.embed_dim)

        # build decoder layers for area PF
        self.layers_up2 = nn.ModuleList()
        self.concat_back_dim2 = nn.ModuleList()
        for i_layer2 in range(self.num_layers):
            concat_linear2 = nn.Linear(2*int(embed_dim*2**(self.num_layers-1-i_layer2)),
            int(embed_dim*2**(self.num_layers-1-i_layer2))) if i_layer2 > 0 else nn.Identity()
            if i_layer2 ==0 :
                # print(patches_resolution[0],self.num_layers,int(embed_dim * 2 ** (self.num_layers-1-i_layer)))

                layer_up2 = PatchExpand(input_resolution=(patches_resolution[0] // (2 ** (self.num_layers-1-i_layer2)),
                patches_resolution[1] // (2 ** (self.num_layers-1-i_layer2))), dim=int(embed_dim * 2 ** (self.num_layers-1-i_layer2)), dim_scale=2, norm_layer=norm_layer)
            else:
                layer_up2 = BasicLayer_up(dim=int(embed_dim * 2 ** (self.num_layers-1-i_layer2)),
                                input_resolution=(patches_resolution[0] // (2 ** (self.num_layers-1-i_layer2)),
                                                    patches_resolution[1] // (2 ** (self.num_layers-1-i_layer2))),
                                depth=depths[(self.num_layers-1-i_layer2)],
                                num_heads=num_heads[(self.num_layers-1-i_layer2)],
                                window_size=window_size,
                                mlp_ratio=self.mlp_ratio,
                                qkv_bias=qkv_bias, qk_scale=qk_scale,
                                drop=drop_rate, attn_drop=attn_drop_rate,
                                drop_path=dpr[sum(depths[:(self.num_layers-1-i_layer2)]):sum(depths[:(self.num_layers-1-i_layer2) + 1])],
                                norm_layer=norm_layer,
                                upsample=PatchExpand if (i_layer2 < self.num_layers - 1) else None,
                                use_checkpoint=use_checkpoint)
            self.layers_up2.append(layer_up2)
            self.concat_back_dim2.append(concat_linear2)

        self.norm_up2= norm_layer(self.embed_dim)




        if self.final_upsample == "expand_first":
            print("---final upsample expand_first for object pf---")
            self.up = FinalPatchExpand_X4(input_resolution=(img_size//patch_size,img_size//patch_size),dim_scale=4,dim=embed_dim)
            self.up_obj_var = FinalPatchExpand_X4(input_resolution=(img_size//patch_size,img_size//patch_size),dim_scale=4,dim=embed_dim)

            self.output = nn.Conv2d(in_channels=embed_dim,out_channels=17,kernel_size=1,bias=False)
            self.output_uncer_obj = nn.Conv2d(in_channels=embed_dim,out_channels=17,kernel_size=1,bias=False)

            # print("---final upsample expand_first for area pf---")
            self.up2 = FinalPatchExpand_X4(input_resolution=(img_size//patch_size,img_size//patch_size),dim_scale=4,dim=embed_dim)
            self.up_area_var = FinalPatchExpand_X4(input_resolution=(img_size//patch_size,img_size//patch_size),dim_scale=4,dim=embed_dim)

            self.output2 = nn.Conv2d(in_channels=embed_dim,out_channels=1,kernel_size=1,bias=False)
            self.output_uncer_area = nn.Conv2d(in_channels=embed_dim,out_channels=1,kernel_size=1,bias=False)


        self.apply(self._init_weights)
        
        # end self.

      ###new def from class SwinTransformerSys

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    #Encoder and Bottleneck
    def forward_features(self, x):
        x = self.patch_embed(x)
        # print("x after patch_embed=",x.shape)
        # x after patch_embed= torch.Size([1, 12544, 192])
        if self.ape:
            # print("into ape")
            # print('x=',x.shape)
            # print("absolute_pos_embed=",self.absolute_pos_embed.shape)
            x = x + self.absolute_pos_embed

        # add channel posotonal embedding
        channel_position_embed_expanded = self.linear_expand(self.channel_position_embed_original)
        # print(channel_position_embed_expanded.shape)
        x = x + channel_position_embed_expanded[:, None, :]
        
        
        # print("x after channel pos embed=",x.shape)


        x = self.pos_drop(x)
        # print("x after dropout=",x.shape)
        x_downsample = []

        for layer in self.layers:
            x_downsample.append(x)
            x = layer(x)
        # print("x after layer=",x.shape)
        # x after layer= torch.Size([1, 196, 1536])
        x = self.norm(x)  # B L C
        # print("x after norm=",x.shape)
        # print("x_downsample=",len(x_downsample),x_downsample[0].shape)
        # x after norm= torch.Size([1, 196, 1536])
        # x_downsample= 4 torch.Size([1, 12544, 192])
        return x, x_downsample

    #Dencoder and Skip connection
    def forward_up_features(self, x, x_downsample):
        for inx, layer_up in enumerate(self.layers_up):
            if inx == 0:
                x = layer_up(x)
            else:
                x = torch.cat([x,x_downsample[3-inx]],-1)
                x = self.concat_back_dim[inx](x)
                x = layer_up(x)

        x = self.norm_up(x)  # B L C
  
        return x
    
    # Decoder for area PF
    def forward_up_features2(self, x, x_downsample):
        for inx2, layer_up2 in enumerate(self.layers_up2):
            if inx2 == 0:
                x = layer_up2(x)
            else:
                x = torch.cat([x,x_downsample[3-inx2]],-1)
                x = self.concat_back_dim2[inx2](x)
                x = layer_up2(x)

        x = self.norm_up2(x)  # B L C
  
        return x



    def up_x4(self, x):
        H, W = self.patches_resolution
        B, L, C = x.shape
        assert L == H*W, "input features has wrong size"

        if self.final_upsample=="expand_first":
            x1 = self.up(x)
            x2 = self.up_obj_var(x)
            x1 = x1.view(B,4*H,4*W,-1)
            x2 = x2.view(B,4*H,4*W,-1)
            x1 = x1.permute(0,3,1,2) #B,C,H,W
            x2 = x2.permute(0,3,1,2)
            x1 = self.output(x1)
            x_var_obj = self.output_uncer_obj(x2)
            
        return x1, x_var_obj
    

    def up_x42(self, x):
        H, W = self.patches_resolution
        B, L, C = x.shape
        assert L == H*W, "input features has wrong size"

        if self.final_upsample=="expand_first":
            x_1 = self.up2(x)
            x3 = self.up_area_var(x)
            x_1 = x_1.view(B,4*H,4*W,-1)
            x3 = x3.view(B,4*H,4*W,-1)
            x_1 = x_1.permute(0,3,1,2) #B,C,H,W
            x3 = x3.permute(0,3,1,2)
            x_1 = self.output2(x_1)
            x_var_area = self.output_uncer_area(x3)
        return x_1,x_var_area

    ##end new def


    def forward(self, x, dirs, locs, world_xyz, world_heading):

        # print("dirs, locs, world_xyz, world_heading=",dirs, locs, world_xyz, world_heading)
        # dirs, locs, world_xyz, world_heading= 
        # tensor([[1, 1, 3, 2, 2, 0, 1, 8, 8, 4, 6, 5, 2, 8, 6, 7, 6]], device='cuda:0') 
        # tensor([[[ 0.6594,  0.6260],
        #  [ 0.5302,  0.5177],
        #  [ 0.4360,  0.5384],
        #  [ 0.4845,  0.6445],
        #  [ 0.4753,  0.5962],
        #  [ 0.7698,  0.5657],
        #  [ 0.6164,  0.6378],
        #  [-1.0000, -1.0000],
        #  [-1.0000, -1.0000],
        #  [ 0.4374,  0.5077],
        #  [ 0.5035,  0.3890],
        #  [ 0.4056,  0.4115],
        #  [ 0.4853,  0.6082],
        #  [-1.0000, -1.0000],
        #  [ 0.5010,  0.3719],
        #  [ 0.5894,  0.3735],
        #  [ 0.5402,  0.3737]]], device='cuda:0') 
        # tensor([[-12.8232,  -0.3777,  10.8398]], device='cuda:0', dtype=torch.float64) 
        # tensor([[3.0391]], device='cuda:0', dtype=torch.float64)
        
        # print("dirs shape=",dirs.shape)
        # print("locs.shape=",locs.shape)
        # print("world_xyz=",world_xyz.shape)
        # print("world_heading=",world_heading.shape)
        # dirs shape= torch.Size([1, 17])
        # locs.shape= torch.Size([1, 17, 2])
        # world_xyz= torch.Size([1, 3])
        # world_heading= torch.Size([1, 1])

        dirs = dirs.float().unsqueeze(-1).unsqueeze(-1)
        dirs = dirs.expand(-1,-1,480,480)

        locs = torch.norm(locs.float(), dim=-1, keepdim=True).unsqueeze(-1)
        locs = locs.expand(-1,-1,480,480)

        world_xyz = self.linear_xyz(world_xyz.float())
        # print("world_xyz=",world_xyz.shape)
        world_xyz = world_xyz.view(world_xyz.size(0),1,480,480)
        world_xyz = world_xyz.repeat(1,17,1,1)

        world_heading = self.linear_heading(world_heading.float())
        world_heading = world_heading.view(world_heading.size(0),1,480,480)
        world_heading = world_heading.repeat(1,17,1,1)
        

            

        # print("x into encoder=",x.shape)
        x_reshape = F.interpolate(x, size=(448, 448), mode='bilinear', align_corners=True)
        dirs = F.interpolate(dirs, size=(448, 448), mode='bilinear', align_corners=True)
        locs = F.interpolate(locs, size=(448, 448), mode='bilinear', align_corners=True)
        world_heading = F.interpolate(world_heading, size=(448, 448), mode='bilinear', align_corners=True)
        world_xyz = F.interpolate(world_xyz, size=(448, 448), mode='bilinear', align_corners=True)


        # print("x_reshape=",x_reshape.shape)
        x = x_reshape + dirs + locs + world_xyz + world_heading
        # x = x_reshape 

        
        
        # x = self.channel_pos_enc(x_reshape)
        # print("x after channel PE=",x.shape)

        x, x_downsample = self.forward_features(x)


        x1 = self.forward_up_features(x,x_downsample)
        x2 = self.forward_up_features2(x,x_downsample)

        # print("x1 after forward_up_features=",x1.shape)
        # print("x2 after forward_up_features=",x2.shape)

        x_obj,x_var_obj = self.up_x4(x1)
        # print("x after up_x4=",x_obj.shape)
        # print("x_var_obj=",x_var_obj.shape)

        x_area, x_var_area = self.up_x42(x2)
        # print("x_var_area=",x_var_area.shape)
        # print("x_obj,x_area=",x_obj.shape,x_area.shape)



        x_obj = F.interpolate(x_obj, size=(480, 480), mode='bilinear', align_corners=True)
        x_area = F.interpolate(x_area, size=(480, 480), mode='bilinear', align_corners=True)

        x_obj_var = F.interpolate(x_var_obj, size=(480, 480), mode='bilinear', align_corners=True)
        x_area_var = F.interpolate(x_var_area, size=(480, 480), mode='bilinear', align_corners=True)

    
        # x = self.embed(x)
        # x1 = self.inc(x)  # (bs, nsf, ..., ...)
        # x2 = self.down1(x1)  # (bs, nsf*2, ... ,...)
        # x3 = self.down2(x2)  # (bs, nsf*4, ..., ...)
        # x4 = self.down3(x3)  # (bs, nsf*8, ..., ...)
        # x5 = self.down4(x4)  # (bs, nsf*8, ..., ...)
        return x_obj, x_area, x_obj_var, x_area_var
        # return {"x1": x1, "x2": x2, "x3": x3, "x4": x4, "x5": x5}

    def get_feature_map_shape(self):
        x = torch.randn(1, self.n_channels, self.map_size, self.map_size)
        x = self.inc(x)  # (bs, nsf, ..., ...)
        x = self.down1(x)  # (bs, nsf*2, ... ,...)
        x = self.down2(x)  # (bs, nsf*4, ..., ...)
        x = self.down3(x)  # (bs, nsf*8, ..., ...)
        x = self.down4(x)  # (bs, nsf*8, ..., ...)
        return x.shape


class UNetDecoder(nn.Module):
    def __init__(
        self,
        n_classes,
        nsf=16,
        bilinear=True,
    ):
        super().__init__()
        self.up1 = up(nsf * 16, nsf * 4, bilinear=bilinear)
        self.up2 = up(nsf * 8, nsf * 2, bilinear=bilinear)
        self.up3 = up(nsf * 4, nsf, bilinear=bilinear)
        self.up4 = up(nsf * 2, nsf, bilinear=bilinear)
        self.outc = outconv(nsf, n_classes)

    def forward(self, xin):
        """
        xin is a dictionary that consists of x1, x2, x3, x4, x5 keys
        from the UNetEncoder
        """
        x1 = xin["x1"]  # (bs, nsf, ..., ...)
        x2 = xin["x2"]  # (bs, nsf*2, ..., ...)
        x3 = xin["x3"]  # (bs, nsf*4, ..., ...)
        x4 = xin["x4"]  # (bs, nsf*8, ..., ...)
        x5 = xin["x5"]  # (bs, nsf*8, ..., ...)

        x = self.up1(x5, x4)  # (bs, nsf*4, ..., ...)
        x = self.up2(x, x3)  # (bs, nsf*2, ..., ...)
        x = self.up3(x, x2)  # (bs, nsf, ..., ...)
        x = self.up4(x, x1)  # (bs, nsf, ..., ...)
        x = self.outc(x)  # (bs, n_classes, ..., ...)

        return x


class ConfidenceDecoder(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.encoder = UNetEncoder(n_classes * 2, nsf=16)
        self.decoder = UNetDecoder(n_classes, nsf=16)

    def forward(self, xin, xpf):
        """
        xin - (bs, n_classes, H, W) semantic map
        xpf - (bs, n_classes, H, W) potential fields prediction
        """
        x_enc = self.encoder(torch.cat([xin, xpf], dim=1))
        x_dec = self.decoder(x_enc)
        return x_dec


class DirectionDecoder(nn.Module):
    def __init__(self, n_classes, n_dirs, nsf=16):
        super().__init__()
        self.n_classes = n_classes
        self.n_dirs = n_dirs
        self.conv1 = double_conv(8 * nsf, 16 * nsf)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = double_conv(16 * nsf, 32 * nsf)
        self.pool2 = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(32 * nsf, n_classes * n_dirs)

    def forward(self, xin):
        """
        xin is a dictionary that consists of x1, x2, x3, x4, x5 keys
        from the UNetEncoder
        """
        x5 = xin["x5"]  # (bs, nsf*8, ..., ...)

        x = self.conv1(x5)  # (bs, nsf*16, ..., ...)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)  # (bs, nsf*32, 1, 1)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)  # (bs, N * D)
        x = x.view(x.shape[0], self.n_classes, self.n_dirs)

        return x


class PositionDecoder(nn.Module):
    def __init__(self, n_classes, nsf=16):
        super().__init__()
        self.n_classes = n_classes
        self.conv1 = double_conv(8 * nsf, 16 * nsf)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = double_conv(16 * nsf, 32 * nsf)
        self.pool2 = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(32 * nsf, n_classes * 2)

    def forward(self, xin):
        """
        xin is a dictionary that consists of x1, x2, x3, x4, x5 keys
        from the UNetEncoder
        """
        x5 = xin["x5"]  # (bs, nsf*8, ..., ...)

        x = self.conv1(x5)  # (bs, nsf*16, ..., ...)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)  # (bs, nsf*32, 1, 1)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)  # (bs, N * 2)
        x = x.view(x.shape[0], self.n_classes, 2)

        return x


class ActionDecoder(nn.Module):
    def __init__(self, n_classes, nsf=16, num_actions=4):
        super().__init__()
        self.n_classes = n_classes
        self.num_actions = num_actions
        self.conv1 = double_conv(8 * nsf, 16 * nsf)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = double_conv(16 * nsf, 32 * nsf)
        self.pool2 = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(32 * nsf, n_classes * num_actions)

    def forward(self, xin):
        """
        xin is a dictionary that consists of x1, x2, x3, x4, x5 keys
        from the UNetEncoder
        """
        x5 = xin["x5"]  # (bs, nsf*8, ..., ...)

        x = self.conv1(x5)  # (bs, nsf*16, ..., ...)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)  # (bs, nsf*32, 1, 1)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)  # (bs, N * 2)
        x = x.view(x.shape[0], self.n_classes, self.num_actions)

        return x


def get_semantic_encoder_decoder(cfg):
    model_cfg = cfg.MODEL
    data_cfg = cfg.DATASET

    encoder, object_decoder, area_decoder = None, None, None
    output_type = model_cfg.output_type
    assert output_type in ["map", "dirs", "locs", "acts"]
    encoder = UNetEncoder(
        model_cfg.num_categories,
        model_cfg.nsf,
        model_cfg.embedding_size,

        img_size=448,
        patch_size=4,
        in_chans=17,
        num_classes=17,
        embed_dim=192,
        depths=[2,2,2,2],
        num_heads=[6,12,24,48],
        window_size=7,
        mlp_ratio=4.,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.2,
        drop_path_rate=0.1,
        ape=False,
        patch_norm=True,
        use_checkpoint=False


    )
    if output_type == "map":
        object_decoder = UNetDecoder(
            model_cfg.num_categories,
            model_cfg.nsf,
            bilinear=model_cfg.unet_bilinear_interp,
        )
    elif output_type == "dirs":
        object_decoder = DirectionDecoder(
            model_cfg.num_categories, model_cfg.ndirs, model_cfg.nsf
        )
    elif output_type == "locs":
        object_decoder = PositionDecoder(model_cfg.num_categories, model_cfg.nsf)
    elif output_type == "acts":
        object_decoder = ActionDecoder(
            model_cfg.num_categories, model_cfg.nsf, data_cfg.num_actions
        )
    if model_cfg.enable_area_head:
        area_decoder = UNetDecoder(
            1,
            model_cfg.nsf,
            bilinear=model_cfg.unet_bilinear_interp,
        )

    # return encoder, object_decoder, area_decoder
    return encoder

