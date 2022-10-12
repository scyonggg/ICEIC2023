import torch
import torch.nn as nn
import torch.nn.functional as F
from NLNN.non_local_embedded_gaussian import NONLocalBlock2D
from .base_model import BaseModel
from .blocks import (
    FeatureFusionBlock,
    FeatureFusionBlock_custom,
    Interpolate,
    _make_encoder,
    forward_vit,
)

import torch.distributions as D
import Resblock

def _make_fusion_block(features, use_bn):
    return FeatureFusionBlock_custom(
        features,
        nn.ReLU(False),
        deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=True,
    )

class MDDPT(BaseModel):
    def __init__(
        self,
        pi_head,
        mu_head,
        var_head,
        features=256,
        backbone="vitb_rn50_384",
        readout="project",
        channels_last=False,
        use_bn=False,
        enable_attention_hooks=False,
    ):

        super(MDDPT, self).__init__()

        self.channels_last = channels_last

        hooks = {
            "vitb_rn50_384": [0, 1, 8, 11],
            "vitb16_384": [2, 5, 8, 11],
            "vitl16_384": [5, 11, 17, 23],
        }

        # Instantiate backbone and reassemble blocks
        self.pretrained, self.scratch = _make_encoder(
            backbone,
            features,
            False,  # Set to true of you want to train from scratch, uses ImageNet weights
            groups=1,
            expand=False,
            exportable=False,
            hooks=hooks[backbone],
            use_readout=readout,
            enable_attention_hooks=enable_attention_hooks,
        )

        self.scratch.refinenet1 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet2 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet3 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet4 = _make_fusion_block(features, use_bn)

        self.nl_1 = NONLocalBlock2D(in_channels=features)
        self.nl_2 = NONLocalBlock2D(in_channels=features)
        self.nl_3 = NONLocalBlock2D(in_channels=features)
        self.nl_4 = NONLocalBlock2D(in_channels=features)

        self.pi_head2 = pi_head
        self.mu_head = mu_head
        self.var_head2 = var_head 
#        self.map_head = map_head
    def forward(self, x):
        if self.channels_last == True:
            x.contiguous(memory_format=torch.channels_last)

        layer_1, layer_2, layer_3, layer_4 = forward_vit(self.pretrained, x)

        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        layer_4_rn = self.nl_4(layer_4_rn)
        path_4 = self.scratch.refinenet4(layer_4_rn)

        path_4 = self.nl_3(path_4)
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn)


        path_3 = self.nl_2(path_3)
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn)


        path_2 = self.nl_1(path_2)
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)

        pi  = self.pi_head2(path_1)
        mu = self.mu_head(path_1)
        var = self.var_head2(path_1)

        return pi,mu,var


class DPT(BaseModel):
    def __init__(
        self,
        head,
        features=256,
        backbone="vitb_rn50_384",
        readout="project",
        channels_last=False,
        use_bn=False,
        enable_attention_hooks=False,
    ):

        super(DPT, self).__init__()

        self.channels_last = channels_last

        hooks = {
            "vitb_rn50_384": [0, 1, 8, 11],
            "vitb16_384": [2, 5, 8, 11],
            "vitl16_384": [5, 11, 17, 23],
        }

        # Instantiate backbone and reassemble blocks
        self.pretrained, self.scratch = _make_encoder(
            backbone,
            features,
            False,  # Set to true of you want to train from scratch, uses ImageNet weights
            groups=1,
            expand=False,
            exportable=False,
            hooks=hooks[backbone],
            use_readout=readout,
            enable_attention_hooks=enable_attention_hooks,
        )

        self.scratch.refinenet1 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet2 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet3 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet4 = _make_fusion_block(features, use_bn)

        self.nl_1 = NONLocalBlock2D(in_channels=features)
        self.nl_2 = NONLocalBlock2D(in_channels=features)
        self.nl_3 = NONLocalBlock2D(in_channels=features)
        self.nl_4 = NONLocalBlock2D(in_channels=features)
 
#        self.pi_head = pi_head
#        self.mu_head = mu_head
#        self.var_head = var_head
#        self.map_head = map_head

    def forward(self, x):
        if self.channels_last == True:
            x.contiguous(memory_format=torch.channels_last)

        layer_1, layer_2, layer_3, layer_4 = forward_vit(self.pretrained, x)

        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        layer_4_rn = self.nl_4(layer_4_rn)
        path_4 = self.scratch.refinenet4(layer_4_rn)

        path_4 = self.nl_3(path_4)
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn)


        path_3 = self.nl_2(path_3)
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn)


        path_2 = self.nl_1(path_2)
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)

        out = self.scratch.output_conv(path_1)

        return out

class DPTDepthModel(MDDPT):
    def __init__(
        self, path=None, non_negative=True, scale=1.0, shift=1e-8,  **kwargs
    ):
        features = kwargs["features"] if "features" in kwargs else 256

        self.scale = scale
        self.shift = shift
        self.k = 3

        pi_head = nn.Sequential(
            nn.Conv2d(features, features // 2, kernel_size=3, stride=1, padding=1),
            Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(features // 2, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, self.k, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True) if non_negative else nn.Identity(),
           #nn.Identity(),
        )

        mu_head = nn.Sequential(
            nn.Conv2d(features, features // 2, kernel_size=3, stride=1, padding=1),
            Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(features // 2, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, self.k, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True) if non_negative else nn.Identity(),
            nn.Identity(),
        )

        var_head = nn.Sequential(
            nn.Conv2d(features, features // 2, kernel_size=3, stride=1, padding=1),
            Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(features // 2, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, self.k, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True) if non_negative else nn.Identity(),
#            nn.Identity(),
        )

        super().__init__(pi_head,mu_head,var_head, **kwargs)
        self.softmax = nn.Softmax(dim = 1)
        self.sigmoid = nn.Sigmoid()

        if path is not None:
            self.load(path)

    def forward(self, x):
        pi,mu,var = super().forward(x)
        # 1x1xHxW

        pi = self.softmax(pi)
        
        mu = self.scale * mu + self.shift
        mu[mu < 1e-8] = 1e-8
        mu = 1.0 / mu
        mu[mu>1] = 1
            
        var = self.scale * var + self.shift
        var[var < 1e-8] = 1e-8
        var = 1 / var
        var[var>1] = 1

        argmax = pi.argmax(1).cuda()
        one_hot = torch.zeros(pi.shape).cuda().scatter(1,argmax.unsqueeze(1),1.0).cuda()
        inverse = 1 - one_hot
        
        var_f = var
        AU= torch.mean(var_f * pi, dim = 1)

#        mu = depth
#
        
        
        mu_avg = torch.sum(mu * pi,dim=1)
        mu_sq = torch.square(mu - mu_avg.unsqueeze(1).repeat((1,3,1,1))) 
        EU = torch.mean(mu_sq * pi , dim = 1)

        if True:
            MAP= torch.mean((one_hot * mu), dim = 1) 

        argmax = pi.argmax(1).cuda()
        one_hot = torch.zeros(pi.shape).cuda().scatter(1,argmax.unsqueeze(1),1.0).cuda()
        MAP= torch.mean((one_hot * mu), dim = 1) 

        return var,pi,MAP,EU,AU,mu

class DPTVarModel(BaseModel):
    def __init__(
        self,
    ):
        super(DPTVarModel, self).__init__()

        self.k = 3
        features=256
        self.scale =1.0
        self.shift = 1e-8
 
 
        self.res_block = Resblock.__dict__['MultipleBasicBlock_2'](features // 2, features // 2, intermediate_feature = 64)

        self.var_head2 = nn.Sequential(
            nn.Conv2d(features + 6, features // 2, kernel_size=3, stride=1, padding=1),
            Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
            )
 
        self.var_out = nn.Sequential(
            nn.Conv2d(128, self.k, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True))
        
        self.res_block_pi = Resblock.__dict__['MultipleBasicBlock_2'](features // 2, features // 2, intermediate_feature = 64)

        self.pi_head2 = nn.Sequential(
            nn.Conv2d(features + 6, features // 2, kernel_size=3, stride=1, padding=1),
            Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
            )
 
        self.pi_out = nn.Sequential(
            nn.Conv2d(128, self.k, kernel_size=1, stride=1, padding=0),
            nn.ReLU())

        self.softmax = nn.Softmax(dim = 1)
        # Instantiate backbone and reassemble blocks
        
    def forward(self, image,depth,features):

        x = torch.cat((F.interpolate(image,scale_factor=0.5),F.interpolate(depth,scale_factor=0.5),features),dim=1)

#        x = features       

        var = self.var_head2(x)
        var = self.res_block(var)
        var = self.var_out(var)

        pi = self.pi_head2(x)
        pi = self.res_block_pi(pi)
        pi = self.pi_out(pi)

        var = self.scale * var + self.shift
        var[var < 1e-8] = 1e-8
        var = 1.0 / var
        var[var>1] = 1
        var = var

        pi = self.softmax(pi)

        argmax = pi.argmax(1).cuda()
        one_hot = torch.zeros(pi.shape).cuda().scatter(1,argmax.unsqueeze(1),1.0).cuda()
        inverse = 1 - one_hot
        
        var_f = var
        AU = var_f[:,1,:,:]
#        AU= torch.mean(var_f * pi, dim = 1)
        mu = depth
#        mu_f = mu[:,1,:,:]
        
        
        mu_avg = torch.sum(mu * pi,dim=1)
        mu_sq = torch.square(mu - mu_avg.unsqueeze(1).repeat((1,3,1,1))) 
        EU = torch.mean(mu_sq * pi , dim = 1) 
#        EU = pi[:,1,:,:]
        if True:
            MAP= torch.mean((one_hot * mu), dim = 1) 
#            mu_avg = torch.mean(mu * pi, dim = 1)
#            EU = torch.abs(MAP - mu_avg)

        if False:
            pi_uniform = torch.ones_like(depth).cuda().detach() / 3.
            AU = torch.mean(var * pi_uniform, dim = 1)
            mu = depth

            mu_ex = mu
            mu_avg = torch.mean(mu_ex * pi_uniform, dim = 1)
 
            mu_sq = torch.square(mu_ex - mu_avg.unsqueeze(1).repeat((1,3,1,1))) 
            EU = torch.mean(mu_sq * pi_uniform, dim = 1)

        argmax = pi.argmax(1).cuda()
        one_hot = torch.zeros(pi.shape).cuda().scatter(1,argmax.unsqueeze(1),1.0).cuda()
        MAP= torch.mean((one_hot * mu), dim = 1) 

        return var,pi,MAP,EU,AU
