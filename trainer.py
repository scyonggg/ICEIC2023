import math
import torch
import torchvision
import torch.nn.functional as F
from torchvision import transforms
import os
from PIL import Image
import numpy as np
import torch.nn as nn
import scipy.misc
import matplotlib.pyplot as plt
import cv2
from bilinear import *
from torch import optim
from torch.autograd import Variable
import OpenEXR
import Imath
import array
import matplotlib as mpl
import matplotlib.cm as cm
import argparse
import random
from imageio import imread
import skimage
import skimage.transform

from midas_loss import ScaleAndShiftInvariantLoss
from models.dat import DAT, Conv_Decoder
from models.cswin import CSWinDepthModel

class Train(object):
    def __init__(self,config,s3d_loader, gpu, train_sampler):
        self.posenet = None
        self.checkpoint_path = config.checkpoint_path
        self.eval_data_path = config.val_path
        self.model_name = config.model_name
        self.model_path = os.path.join(config.model_name,config.model_path)
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.lr = config.lr 
        self.log_step = config.log_step
        self.sample_step = config.sample_step
        self.sample_path = os.path.join(self.model_name,config.sample_path)
        self.log_path = os.path.join(self.model_name,'log.txt')
        self.eval_path = os.path.join(self.model_name, config.eval_path)
        self.test_path = os.path.join(self.model_name, config.test_path)
        self.s3d_loader = s3d_loader
        self.num_epochs = config.num_epochs
        self.max_depth = 255.0
        self.batch_size = config.batch_size
        self.config = config
        self.input_width = config.input_width
        self.input_height = config.input_height
        self.parameters_to_train = []
        self.max_norm = 1
        self.use_hybrid = self.config.use_hybrid
        self.backbone = self.config.backbone

        self.encoder = None
        self.conv_decoder = None 
        
        self.enc_path = config.enc_path
        self.dec_path = config.dec_path
        
        self.scale_loss = ScaleAndShiftInvariantLoss().cuda(gpu)
        self.l1_loss = nn.L1Loss() 
        self.mse_loss = nn.MSELoss()
        self.crop_ratio = config.eval_crop_ratio
        self.models = {}

        if not os.path.exists(self.model_name):
            os.mkdir(self.model_name)
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)
        if not os.path.exists(self.sample_path):
            os.mkdir(self.sample_path)
        if not os.path.exists(self.eval_path):
            os.mkdir(self.eval_path)
        if not os.path.exists(self.test_path):
            os.mkdir(self.test_path)

        self.num_samples = 5000
        # DDP settings
        self.gpu = gpu
        self.distributed = config.distributed
        self.train_sampler = train_sampler

        self.build_model()

    def build_model(self):
        if self.backbone == 'CSwin':
            self.encoder = CSWinDepthModel(split_size=[4,4,8,8], num_heads=[8,16,32,32], hybrid=self.use_hybrid)
        elif self.backbone == 'DAT':
            self.encoder = DAT(hybrid=self.use_hybrid)
        elif self.backbone == 'Swin':
            self.encoder = DAT(strides=[-1,-1,-1,-1], offset_range_factor=[-1, -1, -1, -1], 
                 stage_spec = [['L', 'S'], ['L', 'S'], ['L', 'S', 'L', 'S', 'L', 'S', 'L', 'S', 'L', 'S', 'L', 'S', 'L', 'S', 'L', 'S', 'L', 'S'], ['L', 'S']], groups=[-1, -1, -1,-1], hybrid=self.use_hybrid)
        
        else:
            print("Error")
            break


        self.conv_decoder = Conv_Decoder() 
        self.g_optimizer = optim.AdamW([{"params": list(self.encoder.parameters())},
                                         {"params": list(self.conv_decoder.parameters())}],
                                        self.lr,[self.beta1,self.beta2])

        self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.g_optimizer, 0.95)
  
        if not torch.cuda.is_available():
            print(f'Using CPU')
        elif self.distributed:  # Using multi-GPUs
            if self.gpu is not None:
                torch.cuda.set_device(self.gpu)
                self.encoder.cuda(self.gpu)
                self.encoder = torch.nn.parallel.DistributedDataParallel(self.encoder, device_ids=[self.gpu], find_unused_parameters=True)
                self.conv_decoder.cuda(self.gpu)
                self.conv_decoder = torch.nn.parallel.DistributedDataParallel(self.conv_decoder, device_ids=[self.gpu], find_unused_parameters=True)
            else:
                self.encoder.cuda()
                self.encoder = torch.nn.parallel.DistributedDataParallel(self.encoder, find_unused_parameters=True)
                self.conv_decoder.cuda()
                self.conv_decoder = torch.nn.parallel.DistributedDataParallel(self.conv_decoder, find_unused_parameters=True)

        elif self.gpu is not None:  # Not using multi-GPUs
            torch.cuda.set_device(self.gpu)
            self.encoder = self.encoder.cuda(self.gpu)
            self.conv_decoder = self.conv_decoder.cuda(self.gpu)

    def to_variable(self,x):
        if torch.cuda.is_available():
            x = x.cuda()
        return Variable(x)

    def transform(self,input):
        transform = transforms.Compose([
                    transforms.ToTensor()])
        return transform(input)
 
    def reset_grad(self):
        self.encoder.zero_grad()
        self.conv_decoder.zero_grad()
        
    def freeze(self, model : nn.Module):
        for param in model.module.parameters():
            param.requires_grad=False

    def load_encoder(self):
        '''
        Load pretrained weights of encoder stages from DAT
        '''
        pretrained_enc_dict = torch.load(self.enc_path, map_location=torch.device("cpu"))
        pretrained_enc_dict = pretrained_enc_dict['model']
        # clean up below
        pretrained_enc_dict = {k: v for k, v in pretrained_enc_dict.items() if 'relative_position_bias_table' not in k}
        pretrained_enc_dict = {k: v for k, v in pretrained_enc_dict.items() if 'relative_position_index' not in k}
        pretrained_enc_dict = {k: v for k, v in pretrained_enc_dict.items() if 'attn_mask' not in k}
        pretrained_enc_dict = {k: v for k, v in pretrained_enc_dict.items() if 'rpe_table' not in k}
        self.encoder.load_state_dict(pretrained_enc_dict, strict=False)

    def load_decoder(self):
        '''
        Load pretrained weights of fusion blocks from DPT
        '''
        pretrained_dec_dict = torch.load(self.dec_path, map_location=torch.device("cpu"))
        pretrained_dec_scratch = { 'module.' + k[8:]: v for k, v in pretrained_dec_dict.items() if k.startswith('scratch')}
        self.conv_decoder.load_state_dict(pretrained_dec_scratch, strict=False)

    def load_checkpoint(self):
        '''
        Load checkpoint models. (Not validated)
        '''
        pretrained_dict = torch.load(self.checkpoint_path , map_location=torch.device("cpu"))
        
        enc_dict = self.encoder.state_dict()
        dec_dict = self.conv_decoder.state_dict()

        enc_pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in list(enc_dict)}
        dec_pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in list(dec_dict)}

        enc_dict.update(enc_pretrained_dict)
        dec_dict.update(dec_pretrained_dict)
        
        self.encoder.load_state_dict(enc_dict)
        self.conv_decoder.load_state_dict(dec_dict)


    def resize(self,input,scale):
        input = nn.functional.interpolate(
                input, scale_factor=scale, mode="bilinear", align_corners=True)   
        return input

    def train(self):
        if os.path.isfile(self.log_path):
            os.remove(self.log_path)  
 
        if self.config.Pretrained:
            '''
            Load pretrained weights of DPT fusion blocks into self.conv_decoder
            '''
            # self.load_encoder()
            self.load_decoder()
            print('pretrained weights loaded')
            if self.config.freeze:
                self.freeze(self.conv_decoder)
                print(f'Freeze decoder')

        elif self.config.Continue:
            '''
            Load checkpoint of model
            '''
            dat_dict = self.encoder.state_dict()
            pretrained_dict = torch.load(self.config.enc_path, map_location=torch.device("cpu"))
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in dat_dict}
            dat_dict.update(pretrained_dict)
            self.encoder.load_state_dict(dat_dict)
            
            self.conv_decoder.load_state_dict(torch.load(self.dec_path), strict=False)  # strict=False option ignores error when size mismatchs
            print('checkpoint weights loaded')


        self.max_depth = 255.0
        max_batch_num = len(self.s3d_loader) - 1

############################# Evaluation code ##########################
        with torch.no_grad(): 
            eval_name = '3d60_%d' %(0)
            self.sample(self.eval_data_path,'test',eval_name,self.crop_ratio)
########################################################################

        for epoch in range(self.num_epochs):
            if self.distributed:
                self.train_sampler.set_epoch(epoch)

            for batch_num, data in enumerate(self.s3d_loader):
                if True: 
                    inputs = self.to_variable(data[0])
                    gt = self.to_variable(data[1])
                    mask = self.to_variable(torch.ones_like(gt)).detach()
                  
                    gt = gt / 32768.
                    if self.backbone == 'Cswin':
                        features = self.encoder(inputs)
                    else:
                        features, _, _ = self.encoder(inputs)

                    depth = self.conv_decoder(features)

                    ######### scale_loss 가 column-wise manner 로 계산하는게 맞는지 check ########
                    gen_loss = self.scale_loss(depth,gt,mask)

                    try: 
                        gen_loss.backward()
                        self.g_optimizer.step()
                    except:
                        print('skip_genbackward')
                        self.g_optimizer.zero_grad()                           

                self.reset_grad()                   


                if (batch_num) % self.log_step == 0:
                    print('Epoch [%d/%d], Step[%d/%d], image_loss: %.5f,dis_loss: %.7f' 
                          %(epoch, self.num_epochs, batch_num, max_batch_num, 
                            gen_loss.item(), gen_loss.item()))
                    
                if (batch_num) % self.sample_step == 0:
                    g_path = os.path.join(self.model_path,'generator-%d.pkl' % (epoch ))
                    d_path =  os.path.join(self.model_path,'dis-%d.pkl' % (epoch +1 ))
                    
                    eval_name = '3d60_%d' %(epoch)

 
                    with torch.no_grad():
                        self.sample(self.eval_data_path,g_path,eval_name,self.crop_ratio)

            
            e_path = os.path.join(self.model_path,'encoder-%d.pkl' % (epoch))
            d_path =  os.path.join(self.model_path,'decoder-%d.pkl' % (epoch ))
                         
            torch.save(self.encoder.state_dict(),e_path)
            torch.save(self.conv_decoder.state_dict(),d_path)
            
            with torch.no_grad():
                self.lr_scheduler.step()
                eval_name = '3d60_%d' %(epoch)
                self.sample(self.eval_data_path,g_path,eval_name,self.crop_ratio)
    def post_process_disparity(self,disp):
        
        disp = disp.cpu().detach().numpy() 
        _, h, w = disp.shape
        l_disp = disp[0,:,:]
        
        return l_disp


    def process_sample(self,sample,height,width):
        map = sample

        original_height = height
        original_width = width

        map = self.post_process_disparity(map.squeeze(1)).astype(np.float32)

        pred_width = map.shape[1]
        map = cv2.resize(map.squeeze(), (original_width, original_height))
        map = map.squeeze()

        vmax = np.percentile(map, 95)
        normalizer = mpl.colors.Normalize(vmin=map.min(), vmax=vmax)
        mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
        
        map = (mapper.to_rgba(map)[:, :, :3] * 255).astype(np.uint8)

        return map

    def sample(self,root,checkpoint_path,eval_name,crop_ratio):
        image_list = os.listdir(root)
        eval_image = []
        for image_name in image_list:
            eval_image.append(os.path.join(root,image_name))
        
        self.sigmoid = nn.Sigmoid()
 
        index = 0  
        for image_path in eval_image:
            stereo_baseline = 0.472
            index = index + 1
 
            input_image = (imread(image_path).astype("float32")/255.0)
            original_height, original_width, num_channels = input_image.shape
        
            input_height = 512
            input_width = 1024

            input_image = skimage.transform.resize(input_image, [input_height-2 * crop_ratio , input_width])
            input_image = np.pad(input_image, ((crop_ratio,crop_ratio),(0,0),(0,0)), mode='constant')
            input_image = input_image.astype(np.float32)
            
            left = torch.from_numpy(input_image).unsqueeze(0).float().permute(0,3,1,2).cuda()
        
            
            if self.backbone == 'CSwin':
                features = self.encoder(left)
            else:
                features,_,_ = self.encoder(left)

            depth = self.conv_decoder(features)

            if True:
                max_value = torch.tensor([0.000005]).cuda()
                depth = depth
                depth =1. / torch.max(depth,max_value)


            depth = self.process_sample(depth, original_height, original_width)


            save_name = eval_name + '_'+str(index)+'.png'        

            plt.imsave(os.path.join(self.eval_path,save_name ), depth, cmap='magma')

