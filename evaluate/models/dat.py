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
import timm

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
import PixelUnshuffle

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
    '''
    Args:
        fmap_size   : size of feature
        window_size : size of window to get local/shift-window attention
        ns_per_pt   : Unknown parameter (Not implemented)
        dim_in      : dimension size of input feature
        dim_embed   : dimension size of embedded input feature
        depths      : number of attention blocks in the stage (2*N_i in Table 1.)
        stage_spec  : determine which attention moudle to use. L : Local, S : Shift-Window, D : Deformable Attention
        n_groups    : number of offset groups in DMHA (Deformable Multi-Head Attention)
        use_pe      : whether to use position embedding (Boolean)
        sr_ratio    : Unknown parameter (Not implemented)
        heads       : number of heads at each stage (Table 1.)
        stride      : stride used in conv2D to get Q, K, V projection from feature, a.k.a. downsample factor r in paper.
        offset_range_factor : offset range scale factor used in offset network, a.k.a. scale factor 's' in paper.
        stage_idx   : index of stages. (Stage 1 ~ Stage 4)
        dwc_pe      : whether to use depth-wise convolution as position embedding
        no_off      : whether to use offset. (True : Offset = 0)
        fixed_pe    : whether to use fixed values (zero) as position embedding
        attn_drop   : dropout ratio after softamx in Attention block
        proj_drop   : dropout ratio after DMHA (Deformable Multi-Head Attention)
        expansion   : expansion ratio of channel size in MLP layer at Encoder attention block
        drop        : dropout ratio at MLP after Attention block
        drop_path_rate : DropPath ratio. Similar to dropout, but Dropout randomly zeros the whole target batch. Refer to :
                            1. https://stackoverflow.com/questions/69175642/droppath-in-timm-seems-like-a-dropout
                            2. https://github.com/rwightman/pytorch-image-models/blob/c5e0d1c700de2e39441af9b93f745aadf34be878/timm/models/layers/drop.py#L157
        use_dwc_mlp : whether to use depth-wise convolution instead of MLP in attention block
    '''
    def __init__(self, fmap_size, window_size, ns_per_pt,
                 dim_in, dim_embed, depths, stage_spec, n_groups, 
                 use_pe, sr_ratio, 
                 heads, stride, offset_range_factor, stage_idx,
                 dwc_pe, no_off, fixed_pe,
                 attn_drop, proj_drop, expansion, drop, drop_path_rate, use_dwc_mlp):

        super().__init__()
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
    ''' Deformable Attention Transformer
    Args:
        img_size    : size of input image
        patch_size  : patch size at (first) patch embeddings
        num_classes : number of classes used in classification task
        expansion   : expansion ratio of channel size in MLP layer at Encoder attention block
        dim_stem    : parameter C in Figure 3.
        dims        : channel dimension of each stage (Table 1.)
        depths      : number of attention blocks at each stage (2*N_i in Table 1.)
        heads       : number of heads at each stage (Table 1.)
        window_sizes: window size of Local Attention and Shift-Window Attention
        drop_rate   : dropout ratio after MLP layer in Attention block
        attn_drop_rate : dropout ratio after softamx in Attention block
        drop_path_rate : DropPath ratio. Similar to dropout, but Dropout randomly zeros the whole batch unit. Refer to :
                            1. https://stackoverflow.com/questions/69175642/droppath-in-timm-seems-like-a-dropout
                            2. https://github.com/rwightman/pytorch-image-models/blob/c5e0d1c700de2e39441af9b93f745aadf34be878/timm/models/layers/drop.py#L157
        strides     : stride used in conv2D to get Q, K, V projection from feature. A.K.A. downsample factor 'r' in paper.
        offset_range_factor : offset range scale factor used in offset network. A.K.A. scale factor 's' in paper.
        stage_spec  : determine which attention moudle to use. L : Local, S : Shift-Window, D : Deformable Attention
        groups      : number of offset groups in DMHA (Deformable Multi-Head Attention)
        use_pes     : whether to use position embedding (Boolean)
        dwc_pes     : whether to use depth-wise convolution as position embedding
        sr_ratios   : Unknown parameter (Not implemented)
        fixed_pes   : whether to use fixed values (zero) as position embedding
        no_offs     : whether to use offset
        ns_per_pts  : Unknown parameter (Not implemented)
        use_dwc_mlps: whether to use depth-wise convolution instead of MLP in attention block
        use_conv_patches : whether to use convolutional patch embeddings. Refer to Table 9. (Appendix B.)
    '''    
    def __init__(self, img_size=(512, 1024), patch_size=4, num_classes=1000, expansion=4,
                 dim_stem=256, dims=[256, 256, 256, 256], depths=[2, 2, 18, 2],
                 heads=[4, 8, 16, 32], 
                 window_sizes=[8, 8, 8, 8],
                 drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.0, 
                 strides=[-1,-1,1,1], offset_range_factor=[-1, -1, 2, 2], 
                 stage_spec = [['L', 'S'], ['L', 'S'], ['L', 'D', 'L', 'D', 'L', 'D', 'L', 'D', 'L', 'D', 'L', 'D', 'L', 'D', 'L', 'D', 'L', 'D'], ['L', 'D']],
                 groups=[-1, -1, 4, 8],
                 use_pes=[False, False, True, True], 
                 dwc_pes=[False, False, False, False],
                 sr_ratios=[-1, -1, -1, -1], 
                 fixed_pes=[False, False, False, False],
                 no_offs=[False, False, False, False],
                 ns_per_pts=[4, 4, 4, 4],
                 use_dwc_mlps=[False, False, False, False],
                 use_conv_patches=False,
                 hybrid=True,
                 **kwargs):

        super().__init__()

        if hybrid:  # Use hybrid patch embedding
            self.patch_proj = self.hybrid_patch_embedding()
            img_size = (img_size[0] // 4, img_size[1] // 4)

        else:   # Use convolutional patch embedding
            self.patch_proj = nn.Sequential(
                nn.Conv2d(3, dim_stem, 7, patch_size, 3),
                LayerNormProxy(dim_stem)
            ) if use_conv_patches else nn.Sequential(
                nn.Conv2d(3, dim_stem, patch_size, patch_size, 0),
                LayerNormProxy(dim_stem)
            ) 
            img_size = (img_size[0] // patch_size, img_size[1] // patch_size)   


        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        
        self.stages = nn.ModuleList()
        for i in range(4):
            # dim1 = dim_stem if i == 0 else dims[i - 1] * 2  # Baseline
            dim1 = dim_stem if i == 0 else dims[i - 1] # Baseline
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

            img_size = (img_size[0] // 2, img_size[1] // 2)

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
                
    def hybrid_patch_embedding(self, name='resnet50'):
        model = timm.create_model(name, pretrained=False)
        
        new_model = []
        for k, v in model._modules.items(): # Keys : ['conv1', 'bn1', 'act1', 'maxpool', 'layer1', 'layer2', 'layer3', 'layer4', 'global_pool', 'fc']
            if k in ['conv1', 'bn1', 'act1', 'maxpool', 'layer1']:
                new_model.append(v)

        new_model = nn.Sequential(*new_model)
        return new_model
                
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
 
            features.append(x)
            if i < 3:
                x = self.down_projs[i](x)
            
            positions.append(pos)
            references.append(ref)

        return features, positions, references

##### Decoder model (refer to Dense prediction transformer) #####
class Conv_Decoder(BaseModel):
    def __init__(
        self,
        features=256,
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

        self.refinenet1 = _make_fusion_block(features, use_bn,expand=False)
        self.refinenet2 = _make_fusion_block(features, use_bn,expand=False)
        self.refinenet3 = _make_fusion_block(features, use_bn,expand=False)
        self.refinenet4 = _make_fusion_block(features, use_bn,expand=False)

        non_negative = True

        self.output_conv = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(128, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True) if non_negative else nn.Identity(),
#            nn.Identity(),
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



