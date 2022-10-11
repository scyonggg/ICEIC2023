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
        mu_head1,
#        mu_head2,
#        mu_head3,
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

        self.mu_head1 = mu_head1
#        self.mu_head2 = mu_head2
#        self.mu_head3 = mu_head3

#    def init_mu(self):
#        self.mu_head2.load_state_dict(self.mu_head1.state_dict())
#        self.mu_head3.load_state_dict(self.mu_head1.state_dict())


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

        path_1 = path_1.detach()

#        mu1 = self.mu_head1(path_1)
#        mu2 = self.mu_head2(path_1)
#        mu3 = self.mu_head3(path_1)

        return path_1
#        return mu1,mu2,mu3,path_1


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
 
        self.scratch.output_conv = head

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

        return out,path_1

class DPTDepthModel(DPT):
    def __init__(
        self, path=None, non_negative=True, scale=1.0, shift=0.00001, invert=True, **kwargs
    ):
        features = kwargs["features"] if "features" in kwargs else 256

        self.scale = scale
        self.shift = shift
        self.invert = invert

        head = nn.Sequential(
            nn.Conv2d(features, features // 2, kernel_size=3, stride=1, padding=1),
            Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(features // 2, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
#            nn.Tanh(),
#            nn.Sigmoid(),
            nn.ReLU(True) if non_negative else nn.Identity(),
            nn.Identity(),
        )

        super().__init__(head, **kwargs)


        if path is not None:
            self.load(path)

    def forward(self, x):
        inv_depth, features = super().forward(x)

        if self.invert:
            depth = self.scale * inv_depth + self.shift
            depth[depth < 1e-8] = 1e-8
            depth = 1.0 / depth
            
############ check this part ##############
            depth[depth>1] = 1
            return depth , features
        else:
            inv_depth = 0.000001 + inv_depth
            return inv_depth, features

class MDDPTDepthModel(MDDPT):
    def __init__(
        self, path=None, non_negative=True, scale=1.0, shift=1e-8,  **kwargs
    ):
        features = kwargs["features"] if "features" in kwargs else 256

        self.scale = scale
        self.shift = shift
        self.k = 3

        super().__init__( **kwargs)
        self.softmax = nn.Softmax(dim = 1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(True)
        if path is not None:
            self.load(path)

    def forward(self, x, training):
        features = super().forward(x)
        
        if training:
            return features

        else:
            return mu,features

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

        self.mu_head = nn.Sequential(
            nn.Conv2d(features + 4, features // 2, kernel_size=3, stride=1, padding=1),
            Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(features // 2, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, self.k, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True) ,
            nn.Identity(),
        )


        self.mu_head2 = nn.Sequential(
            nn.Conv2d(features + 4, features // 2, kernel_size=3, stride=1, padding=1),
            Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
            )
 
        self.mu_out = nn.Sequential(
            nn.Conv2d(128, self.k, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True))
 
        self.res_block_mu = Resblock.__dict__['MultipleBasicBlock_2'](features // 2, features // 2, intermediate_feature = 64)

       
        self.var_head2 = nn.Sequential(
            nn.Conv2d(features + 4, features // 2, kernel_size=3, stride=1, padding=1),
            Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
            )
 
        self.var_out = nn.Sequential(
            nn.Conv2d(128, self.k, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True))
        
        self.res_block_pi = Resblock.__dict__['MultipleBasicBlock_2'](features // 2, features // 2, intermediate_feature = 64)

        self.pi_head2 = nn.Sequential(
            nn.Conv2d(features + 4, features // 2, kernel_size=3, stride=1, padding=1),
            Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
            )
 
        self.pi_out = nn.Sequential(
            nn.Conv2d(128, self.k, kernel_size=1, stride=1, padding=0),
            nn.ReLU())

        self.softmax = nn.Softmax(dim = 1)
        
        # Instantiate backbone and reassemble blocks
        
    def forward(self, image,depth,features):

        x = torch.cat((F.interpolate(image,scale_factor=0.5),F.interpolate(depth,scale_factor=0.5),features),dim=1)
        
        var = self.var_head2(x)
        var = self.res_block(var)
        var = self.var_out(var)

        pi = self.pi_head2(x)
        pi = self.res_block_pi(pi)
        pi = self.pi_out(pi)

        mu = self.mu_head2(x)
        mu = self.res_block_mu(mu)
        mu = self.mu_out(mu)

#        pi = self.pi_out(x)
#        pi = pi - torch.max(pi,1).repeat(1,3,1,1)
#        pi = pi.clamp(0,1)
#        pi_uniform = torch.ones_like(pi).cuda().detach()/3.
#        x = (1 + x) / 2. + 1e-8

        var = self.scale * var + self.shift
        var[var < 1e-8] = 1e-8
        var = 1.0 / var
        var[var>1] = 1
#        var = var

        mu = self.scale * mu + self.shift
        mu[mu < 1e-8] = 1e-8
        mu = 1.0 / mu
        mu[mu>1] = 1

#        mu[:,0,:,:] = depth

        pi = self.softmax(pi)

        argmax = pi.argmax(1).cuda()
        one_hot = torch.zeros(pi.shape).cuda().scatter(1,argmax.unsqueeze(1),1.0).cuda()
       
#        inverse = 1 - one_hot
        
        
        var_f = var
        AU= torch.mean(var_f * pi, dim = 1)

#        mu_f = inverse * mu
        mu_f = mu

        mu_avg = torch.mean(mu_f * pi, dim = 1)
        mu_sq = torch.square(mu_f - mu_avg.unsqueeze(1).repeat((1,self.k,1,1))) 
        EU = torch.mean(mu_sq * pi, dim = 1)

        if True:
            MAP= torch.mean((one_hot * mu), dim = 1) 
            mu_avg = torch.mean(mu * pi, dim = 1)

            EU = torch.abs(MAP - mu_avg)

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

        return mu,var,pi,MAP,EU,AU


