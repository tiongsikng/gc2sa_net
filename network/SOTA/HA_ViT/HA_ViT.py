import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from itertools import repeat
import collections.abc
import numpy as np
import torch
import torch.nn as nn
import sys, os
sys.path.insert(0, os.path.abspath('.'))
from network import load_model


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


try:
    from torch import _assert
except ImportError:
    def _assert(condition: bool, message: str):
        assert condition, message


class Mlp(nn.Module):
    """
        Multi-layer Perceptron
    """
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


class Conv_Attention(nn.Module):
    """

    """
    def __init__(self, dim, depth_block_channel, num_heads, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q, self.k, self.v = DepthwiseConv2D(depth_block_channel), DepthwiseConv2D(depth_block_channel), \
                                 DepthwiseConv2D(depth_block_channel)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        x = torch.reshape(x, (B, N, int(np.sqrt(C)), int(np.sqrt(C))))
        q, k, v = self.q(x), self.k(x), self.v(x)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Conv_Block(nn.Module):
    """

    """
    def __init__(self, dim, depth_block_channel, num_heads, mlp_ratio=4., drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Conv_Attention(dim, depth_block_channel=depth_block_channel, num_heads=num_heads,
                                   attn_drop=attn_drop, proj_drop=drop)
        # # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    """
        2D Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, view=1, in_chans=3, embed_dim=768, norm_layer=None, flatten=True):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = nn.Conv2d(view * in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, V, C, H, W = x.shape
        _assert(H == self.img_size[0], f"Input image height ({H}) doesn't match model ({self.img_size[0]}).")
        _assert(W == self.img_size[1], f"Input image width ({W}) doesn't match model ({self.img_size[1]}).")
        x = x.reshape(B, V * C, H, W)  # shape => [B, N, H, W]
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BNHW -> BNC
        x = self.norm(x)

        return x


# From PyTorch internals
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse


to_2tuple = _ntuple(2)


class DepthwiseConv2D(nn.Module):
    """

    """
    def __init__(self, channels):
        super(DepthwiseConv2D, self).__init__()
        self.depthwise = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels)

    def forward(self, x):
        out = self.depthwise(x)
        return out


class Conv_Attention_Block(nn.Module):
    """

    """
    def __init__(self, dim, num_heads, depth_block_channel, mlp_ratio=4., drop=0.,
                 attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.transformer_block = Conv_Block(
            dim=dim, depth_block_channel=depth_block_channel, num_heads=num_heads, mlp_ratio=mlp_ratio, drop=drop,
            attn_drop=attn_drop, drop_path=drop_path, norm_layer=norm_layer, act_layer=act_layer)
        self.depthwise_conv = DepthwiseConv2D(channels=depth_block_channel)
        self.depthwise_conv2 = DepthwiseConv2D(channels=depth_block_channel)
        self.conv_1x1 = nn.Conv2d(depth_block_channel * 2, depth_block_channel, kernel_size=1, padding=0)
        self.conv_3x3 = nn.Conv2d(depth_block_channel, depth_block_channel, kernel_size=3, padding=1)
        self.relu = nn.LeakyReLU()
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        if len(x.shape) == 3:
            # Transformer Block
            x_trans = self.transformer_block(x)

            # Conv Block
            B, C, F = x.shape
            x = torch.reshape(x, (B, C, int(np.sqrt(F)), int(np.sqrt(F))))
            x_conv = self.depthwise_conv(x)
            x_trans = torch.reshape(x_trans, (B, C, int(np.sqrt(F)), int(np.sqrt(F))))

            x = torch.cat((x_conv, x_trans), dim=1)
            x = self.drop(self.relu(self.conv_3x3(self.conv_1x1(x))))
        else:
            # Conv Block
            x_conv = self.depthwise_conv2(x)

            # Transformer Block
            B, C, H, W = x.shape
            x = torch.reshape(x, (B, C, H * W))
            x_trans = self.transformer_block(x)
            x_trans = torch.reshape(x_trans, (B, C, H, W))

            x = torch.cat((x_conv, x_trans), dim=1)
            x = self.drop(self.relu(self.conv_3x3(self.conv_1x1(x))))

        return x


class HA_ViT(nn.Module):
    """

    """
    def __init__(self, img_size=112, patch_size=8, in_chans=3, embed_dim=1024, num_classes_list=(2239, 7, 2),
                 layer_depth=12, num_heads=12, mlp_ratio=4., norm_layer=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0.):
        super().__init__()
        self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = len(num_classes_list)
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        self.num_patches = int(img_size / patch_size) ** 2

        self.patch_embed_face = PatchEmbed(
            img_size=img_size, view=1, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)

        # shared token for teacher and student
        self.cls_token = nn.Parameter(torch.zeros(1, self.num_tokens, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + self.num_tokens, embed_dim))

        self.pos_drop = nn.Dropout(p=drop_rate)
        act_layer = nn.GELU

        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, layer_depth)]

        self.conv_atten_block_1 = nn.Sequential(*[
            Conv_Attention_Block(dim=self.embed_dim, num_heads=num_heads,
                                 depth_block_channel=(self.num_patches + self.num_tokens), mlp_ratio=mlp_ratio,
                                 drop=drop_rate, attn_drop=attn_drop_rate,
                                 drop_path=dpr[i], act_layer=act_layer, norm_layer=norm_layer)
            for i in range(layer_depth)])
        self.conv_atten_block_2 = nn.Sequential(*[
            Conv_Attention_Block(dim=self.embed_dim, num_heads=num_heads,
                                 depth_block_channel=(self.num_patches + self.num_tokens), mlp_ratio=mlp_ratio,
                                 drop=drop_rate, attn_drop=attn_drop_rate,
                                 drop_path=dpr[i], act_layer=act_layer, norm_layer=norm_layer)
            for i in range(layer_depth)])

        # Face Classifier head(s)
        self.head_face = nn.ModuleList([
            nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
            for num_classes in num_classes_list])

        # # Ocular Classifier head(s)
        self.head_ocu = nn.ModuleList([
            nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
            for num_classes in num_classes_list])

    def forward_patch_embed(self, x, mode):  # mode = "face" or "ocular"
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        if mode == "face":
            x = self.patch_embed_face(x)
        elif mode == "ocular":
            x = self.patch_embed_face(x)
        x = torch.cat((cls_token, x), dim=1)
        x = self.pos_drop(x + self.pos_embed)

        return x

    def forward_features(self, x):
        x1 = self.conv_atten_block_1(x)
        x2 = self.conv_atten_block_2(x1)
        x = torch.cat((x1, x2), dim=1)

        B, N, H, W = x.shape
        x = torch.reshape(x, (B, N, -1))

        return x

    def forward(self, x, peri_flag=False):       
        x_emb = self.forward_patch_embed(x, mode="ocular") if peri_flag else self.forward_patch_embed(x, mode="face")
        x_emb = self.forward_features(x_emb)

        y_face, y_ocular = [], []

        if peri_flag == False:
            for i, clf_head in enumerate(self.head_face):
                y_face.append(clf_head(x_emb[:, i]))
            
            return F.normalize(x_emb[:, :self.num_tokens][:,0], p=2, dim=1), y_face[0]
        else:
            for i, clf_head in enumerate(self.head_ocu):
                y_ocular.append(clf_head(x_emb[:, i]))
            
            return F.normalize(x_emb[:, :self.num_tokens][:,0], p=2, dim=1), y_ocular[0]

if __name__ == '__main__':
    device = torch.device('cuda:0')
    load_model_path = './models/sota/HA-ViT.pth'
    model = HA_ViT(img_size=112, patch_size=8, in_chans=3, embed_dim=1024, num_classes_list=(1054,),
                   layer_depth=3, num_heads=8, mlp_ratio=4., norm_layer=None, drop_rate=0.1, attn_drop_rate=0.1,
                   drop_path_rate=0.).to(device)
    model = load_model.load_pretrained_network(model, load_model_path, device = device)   
