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
        e1_head,
        e2_head,
        pi_head,
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

        self.e1m_head = e1_head
        self.e1v_head = e2_head
        self.e1mp_head = pi_head


#        self.e3_head = e3_head
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

        e1m = self.e1m_head(path_1)
        e1v = self.e1v_head(path_1)
        e1mp = self.e1mp_head(path_1)
        
#        var = torch.cat((e1[:,1,:,:].unsqueeze(1),e2[:,1,:,:].unsqueeze(1)),dim=1)
#        pi = self.pi_head2(var)

#        e3 = self.e3_head(path_1)
#        pi  = self.pi_head(path_1)
#        mu = self.mu_head(path_1)
#        var = self.var_head(path_1)
#        map = self.map_head(path_1)

        return e1m,e1v,e1mp


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
        self.k = 2

        e1_m = nn.Sequential(
            nn.Conv2d(features, features // 2, kernel_size=3, stride=1, padding=1),
            Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(features // 2, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True) if non_negative else nn.Identity(),
           #nn.Identity(),
        )
        
        e1_var = nn.Sequential(
            nn.Conv2d(features, features // 2, kernel_size=3, stride=1, padding=1),
            Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(features // 2, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True) if non_negative else nn.Identity(),
           #nn.Identity(),
        )


        e1_mp = nn.Sequential(
            nn.Conv2d(features, features // 2, kernel_size=3, stride=1, padding=1),
            Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(features // 2, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True) if non_negative else nn.Identity(),
           #nn.Identity(),
        )
        pi_head = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(16, 2, kernel_size=1, stride=1, padding=0),
#            nn.ReLU(True) if non_negative else nn.Identity(),
           #nn.Identity(),
        )

        super().__init__(e1_m,e1_var,e1_mp, **kwargs)
        self.softmax = nn.Softmax(dim = 1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(True)
        if path is not None:
            self.load(path)

    def forward(self, x, training):
        e1_m,e1_v,e1_mp = super().forward(x)

        mu = torch.cat((e1_m,e1_mp),dim=1)
        
        var_2 = torch.max(e1_v) - e1_v + 1e-6
        var = torch.cat((e1_v,var_2),dim=1)

        
        mu = self.scale * mu + self.shift
        mu[mu < 1e-8] = 1e-8
        mu = 1.0 / mu
        mu[mu>1] = 1

        var = self.scale * var + self.shift
        var[var < 1e-8] = 1e-8
        var = 1 / var
        var[var>1] = 1
        

        pi = torch.ones_like(mu).cuda().detach() / 2.

        pi_norand = pi
 
        # Aleatoric Uncertainty #
        AU = torch.mean(var * pi_norand, dim = 1)

        # Epistemic Uncertainty #

        mu_avg = torch.mean(mu * pi_norand, dim = 1)
        mu_sq = torch.square(mu - mu_avg.unsqueeze(1).repeat((1,self.k,1,1))) 
        EU = torch.mean(mu_sq * pi_norand, dim = 1)

##        argmax = pi_norand.argmax(1).cuda()
#        one_hot = torch.zeros(pi.shape).cuda().scatter(1,argmax.unsqueeze(1),1.0).cuda()
#        inverse = 1 - one_hot

########################################################################################

#        var_f = inverse * var
#        AU= torch.mean(var_f * pi_norand, dim = 1)

#        mu_f = inverse * mu
#        mu_avg = torch.mean(mu_f * pi, dim = 1)
#        mu_sq = torch.square(mu_f - mu_avg.unsqueeze(1).repeat((1,self.k,1,1))) 
#        EU = torch.mean(mu_sq * pi_norand, dim = 1)
        

#        argmax = pi_norand.argmax(1).cuda()
#        one_hot = torch.zeros(pi.shape).cuda().scatter(1,argmax.unsqueeze(1),1.0).cuda()
#        MAP= torch.mean((one_hot * mu), dim = 1).unsqueeze(1) 
#        print(MAP.size())

#        mu_pi = mu*pi
#        mu_avg_except = torch.mean(torch.cat((mu_pi[:,0,:,:].unsqueeze(1),mu_pi[:,1,:,:].unsqueeze(1),mu_pi[:,2,:,:].unsqueeze(1)),dim=1),dim=1)
#        mu_except =  torch.cat((mu[:,0,:,:].unsqueeze(1),mu[:,1,:,:].unsqueeze(1),mu[:,2,:,:].unsqueeze(1)),dim=1)

#        mu_sq = torch.square(mu_except - mu_avg_except.unsqueeze(1).repeat((1,self.k - 1,1,1))) 
#        pi_except =  torch.cat((pi[:,0,:,:].unsqueeze(1),pi[:,1,:,:].unsqueeze(1),pi[:,2,:,:].unsqueeze(1)),dim=1)
#        EU = torch.mean(mu_sq * pi_except, dim = 1)

 
        if training:
            return pi, mu, var
        else:
            ### MAP ###
            argmax = pi_norand.argmax(1).cuda()
            one_hot = torch.zeros(pi.shape).cuda().scatter(1,argmax.unsqueeze(1),1.0).cuda()
            MAP= torch.mean((one_hot * mu), dim = 1) 
            MAP_laplace = MAP 

            var_laplace = torch.mean(var * one_hot, dim=1) 
 
            return MAP_laplace,AU,EU,pi,mu,MAP,var
