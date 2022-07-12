# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------
# Vision Transformer with Deformable Attention
# Modified by Zhuofan Xia 
# --------------------------------------------------------

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import DropPath, to_2tuple

from .base_model import BaseModel

from .dat_blocks import *

from .blocks import (
    FeatureFusionBlock,
    FeatureFusionBlock_custom,
    Interpolate,
    _make_encoder,
    forward_vit,
)
from .base_model import BaseModel


def _make_fusion_block(features, use_bn,expand=False):
    return FeatureFusionBlock_custom(
        features,
        nn.ReLU(False),
        deconv=False,
        bn=use_bn,
        expand=expand,
        align_corners=True,
    )


class TransformerStage(nn.Module):

    def __init__(self, fmap_size, window_size, ns_per_pt,
                 dim_in, dim_embed, depths, stage_spec, n_groups, 
                 use_pe, sr_ratio, 
                 heads, stride, offset_range_factor, stage_idx,
                 dwc_pe, no_off, fixed_pe,
                 attn_drop, proj_drop, expansion, drop, drop_path_rate, use_dwc_mlp):

        super().__init__()
        fmap_size = to_2tuple(fmap_size)
        self.depths = depths
        hc = dim_embed // heads
        assert dim_embed == heads * hc
        self.proj = nn.Conv2d(dim_in, dim_embed, 1, 1, 0) if dim_in != dim_embed else nn.Identity()

        self.layer_norms = nn.ModuleList(
            [LayerNormProxy(dim_embed) for _ in range(2 * depths)]
        )
        self.mlps = nn.ModuleList(
            [
                TransformerMLPWithConv(dim_embed, expansion, drop) 
                if use_dwc_mlp else TransformerMLP(dim_embed, expansion, drop)
                for _ in range(depths)
            ]
        )
        self.attns = nn.ModuleList()
        self.drop_path = nn.ModuleList()
        for i in range(depths):
            if stage_spec[i] == 'L':
                self.attns.append(
                    LocalAttention(dim_embed, heads, window_size, attn_drop, proj_drop)
                )
            elif stage_spec[i] == 'D':
                self.attns.append(
                    DAttentionBaseline(fmap_size, fmap_size, heads, 
                    hc, n_groups, attn_drop, proj_drop, 
                    stride, offset_range_factor, use_pe, dwc_pe, 
                    no_off, fixed_pe, stage_idx)
                )
            elif stage_spec[i] == 'S':
                shift_size = math.ceil(window_size / 2)
                self.attns.append(
                    ShiftWindowAttention(dim_embed, heads, window_size, attn_drop, proj_drop, shift_size, fmap_size)
                )
            else:
                raise NotImplementedError(f'Spec: {stage_spec[i]} is not supported.')
            
            self.drop_path.append(DropPath(drop_path_rate[i]) if drop_path_rate[i] > 0.0 else nn.Identity())
        
    def forward(self, x):
        
        x = self.proj(x)
        
        positions = []
        references = []
        for d in range(self.depths):

            x0 = x
            x, pos, ref = self.attns[d](self.layer_norms[2 * d](x))
            x = self.drop_path[d](x) + x0
            x0 = x
            x = self.mlps[d](self.layer_norms[2 * d + 1](x))
            x = self.drop_path[d](x) + x0
            positions.append(pos)
            references.append(ref)

        return x, positions, references

class DAT(nn.Module):
    #### Need to set configurations manually / will be fixed in the near future
    #### Especially, check carefully 'window_size' & 'img_size'
    
    def __init__(self, img_size=256, patch_size=4, num_classes=1000, expansion=4,
                 dim_stem=128, dims=[64, 128, 256, 512], depths=[2, 2, 18, 2], 
                 heads=[4, 8, 16, 32], 
                 window_sizes=[8, 8, 8, 8],
                 drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.0, 
                 strides=[-1,-1,1,1], offset_range_factor=[-1, -1, 2, 2], 
                 stage_spec = [['L', 'S'], ['L', 'S'], ['L', 'D', 'L', 'D', 'L', 'D', 'L', 'D', 'L', 'D', 'L', 'D', 'L', 'D', 'L', 'D', 'L', 'D'], ['L', 'D']],
                 groups=[-1, -1, 4, 8],
                 use_pes=[False, False, False, False], 
                 dwc_pes=[False, False, False, False],
                 sr_ratios=[-1, -1, -1, -1], 
                 fixed_pes=[False, False, False, False],
                 no_offs=[False, False, False, False],
                 ns_per_pts=[4, 4, 4, 4],
                 use_dwc_mlps=[False, False, False, False],
                 use_conv_patches=False,
                 **kwargs):

        super().__init__()

        self.patch_proj = nn.Sequential(
            nn.Conv2d(3, dim_stem, 7, patch_size, 3),
            LayerNormProxy(dim_stem)
        ) if use_conv_patches else nn.Sequential(
            nn.Conv2d(3, dim_stem, patch_size, patch_size, 0),
            LayerNormProxy(dim_stem)
        ) 

        img_size = img_size // patch_size
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        
        self.stages = nn.ModuleList()
        for i in range(4):
            dim1 = dim_stem if i == 0 else dims[i - 1] * 2
            dim2 = dims[i]

            self.stages.append(
                TransformerStage(img_size, window_sizes[i], ns_per_pts[i],
                dim1, dim2, depths[i], stage_spec[i], groups[i], use_pes[i], 
                sr_ratios[i], heads[i], strides[i], 
                offset_range_factor[i], i,
                dwc_pes[i], no_offs[i], fixed_pes[i],
                attn_drop_rate, drop_rate, expansion, drop_rate, 
                dpr[sum(depths[:i]):sum(depths[:i + 1])],
                use_dwc_mlps[i])
            )

            img_size = img_size // 2

        self.down_projs = nn.ModuleList()
        for i in range(3):
            self.down_projs.append(
                nn.Sequential(
                    nn.Conv2d(dims[i], dims[i + 1], 3, 2, 1, bias=False),
                    LayerNormProxy(dims[i + 1])
                ) if use_conv_patches else nn.Sequential(
                    nn.Conv2d(dims[i], dims[i + 1], 2, 2, 0, bias=False),
                    LayerNormProxy(dims[i + 1])
                )
            )
           
        self.cls_norm = LayerNormProxy(dims[-1]) 
        self.cls_head = nn.Linear(dims[-1], num_classes)
        
        self.reset_parameters()
    
    def reset_parameters(self):

        for m in self.parameters():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)
                
    @torch.no_grad()
    def load_pretrained(self, state_dict):
        
        new_state_dict = {}
        for state_key, state_value in state_dict.items():
            keys = state_key.split('.')
            m = self
            for key in keys:
                if key.isdigit():
                    m = m[int(key)]
                else:
                    m = getattr(m, key)
            if m.shape == state_value.shape:
                new_state_dict[state_key] = state_value
            else:
                # Ignore different shapes
                if 'relative_position_index' in keys:
                    new_state_dict[state_key] = m.data
                if 'q_grid' in keys:
                    new_state_dict[state_key] = m.data
                if 'reference' in keys:
                    new_state_dict[state_key] = m.data
                # Bicubic Interpolation
                if 'relative_position_bias_table' in keys:
                    n, c = state_value.size()
                    l = int(math.sqrt(n))
                    assert n == l ** 2
                    L = int(math.sqrt(m.shape[0]))
                    pre_interp = state_value.reshape(1, l, l, c).permute(0, 3, 1, 2)
                    post_interp = F.interpolate(pre_interp, (L, L), mode='bicubic')
                    new_state_dict[state_key] = post_interp.reshape(c, L ** 2).permute(1, 0)
                if 'rpe_table' in keys:
                    c, h, w = state_value.size()
                    C, H, W = m.data.size()
                    pre_interp = state_value.unsqueeze(0)
                    post_interp = F.interpolate(pre_interp, (H, W), mode='bicubic')
                    new_state_dict[state_key] = post_interp.squeeze(0)
        
        self.load_state_dict(new_state_dict, strict=False)
    
    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table', 'rpe_table'}
    
    def forward(self, x):
        
        x = self.patch_proj(x)
        positions = []
        references = []
        features = []
        for i in range(4):
            x, pos, ref = self.stages[i](x)
##############################################################
            #### The purpose of this block?? ####
            # Need to check this part #
            if i < 3:
                x = self.down_projs[i](x)
#############################################################
            features.append(x)
            positions.append(pos)
            references.append(ref)

        return features, positions, references



##### Decoder model (refer to Dense prediction transformer) #####
class Conv_Decoder(BaseModel):
    def __init__(
        self,
        features=512,
        readout="project",
        channels_last=False,
        use_bn=False,
        enable_attention_hooks=False,
    ):

        super(Conv_Decoder, self).__init__()

        self.channels_last = channels_last
        self.scale =1.0
        self.shift = 1e-8

        # Instantiate fusion blocks

        self.refinenet1 = _make_fusion_block(features//4, use_bn,expand=True)
        self.refinenet2 = _make_fusion_block(features//2, use_bn,expand=True)
        self.refinenet3 = _make_fusion_block(features, use_bn,expand=True)
        self.refinenet4 = _make_fusion_block(features, use_bn)

##################### Need to check this part ########################
        non_negative = True

        self.output_conv = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True) if non_negative else nn.Identity(),
            nn.Identity(),
        )



    def forward(self, features):
        if self.channels_last == True:
            x.contiguous(memory_format=torch.channels_last)
        
        path_4 = self.refinenet4(features[3])

        path_3 = self.refinenet3(path_4, features[2])
        
        path_2 = self.refinenet2(path_3, features[1])

        path_1 = self.refinenet1(path_2, features[0])

        out = self.output_conv(path_1)

        ## Crop depth value from 0 to 1 ##
        out = self.scale * out + self.shift
        out[out < 1e-8] = 1e-8
        out = 1.0 / out
        out[out>1] = 1


        return out



