import argparse
import os
from trainer import Train
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torch.utils.data import ConcatDataset
from data_load import EQUI_loader, S3D_loader
from torchvision import transforms
import torch
def main(config):
    cudnn.benchmark = True
#    torch.manual_seed(1593665876)
#    torch.cuda.manual_seed_all(4099049913103886)    

    torch.manual_seed(159111236)
    torch.cuda.manual_seed_all(4099049123103886)    
#    torch.manual_seed(159111235)
#    torch.cuda.manual_seed_all(4099049123103885)    



    transform = transforms.Compose([
                    transforms.Resize((config.input_height,config.input_width)),
#                    transforms.CenterCrop((config.input_height * 3 //4 ,config.input_width)),
                    transforms.ToTensor()
                    ])

    transform_s3d = transforms.Compose([
                    transforms.ToTensor()
                    ])
     
    if True:
        S3D_data = S3D_loader(config.S3D_path,transform = transform_s3d,transform_t = transform_s3d)
        S3D_dataloader = DataLoader(S3D_data,batch_size=config.batch_size,shuffle=True,num_workers=config.num_workers,pin_memory=True)

    
    if config.mode == 'train':
        train = Train(config,S3D_dataloader)
        train.train()

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--mode', type=str, default='train')

    parser.add_argument('--ThreeD_path',help='Text file containing the image list', type=str,default='../3D60/splits/filtered/final_train.txt') # train image list file for supervised learning.
    parser.add_argument('--Video_path',help='Folder containing video frames',type=str,default='../Video_folder') # train image list file for supervised learning
    parser.add_argument('--S3D_path',help='Folder containing Structure3D dataset', type=str,default='../Structure3D/Structured3D') # train image list file for supervised learning.
 
    parser.add_argument('--val_path',type=str,default='./SAMPLE') # file path which contains images to be sampled
    parser.add_argument('--test_path',type=str,default='test') # text which contains test image list -> not used here

    parser.add_argument('--checkpoint_path',type=str,default='./pretrained_BTS/generator_best.pkl') # detphnet checkpoint path
    parser.add_argument('--posenet_path',type=str,default='./pretrained_BTS/generator_best.pkl') # posenet checkpoint path
 
    ##### hyper-parameters #####
    parser.add_argument('--lr_loss_weight',            type=float, help='LR Depth consistency weight', default=0.5)
    parser.add_argument('--num_scales',type=int, help='number of scales', default=1)
    parser.add_argument('--alpha_image_loss',          type=float, help='weight between SSIM and L1 in the image loss', default=0.85)
    parser.add_argument('--disp_gradient_loss_weight', type=float, help='disparity smoothness weigth', default=0) # not used in the experiments. 
    
#    parser.add_argument('--supervised', help='Set True to train via supervised learning', action='store_true')
 #   parser.add_argument('--self_supervised', help='Set True to train via self-supervised learning', action='store_true')
    parser.add_argument('--Crop_data', help='Crop Top-bottom part of depth & image additionaly for self-supervised learning ', action='store_true')
    parser.add_argument('--WILD', help='USE Wild video for self-supervised learning', action='store_true') # Due to review policy, not implemented for the AAAI22 submitted version.
    parser.add_argument('--pose_only', help='Train pose network only', action='store_true')
    parser.add_argument('--DPT', help='Set True', action='store_true')
    parser.add_argument('--Occlusion', help='USE Occlusion-aware loss. not tested yet', action='store_true')
    parser.add_argument('--depth_sample', help=' Set True as default', action='store_true')
    parser.add_argument('--Continue', help=' Strat training from the pre-trained model', action='store_true')
    
    parser.add_argument('--input_height', type=int, help='input height', default=256)
    parser.add_argument('--input_width', type=int, help='input width', default=512)
    parser.add_argument('--fovy_ratio', type=float, help='crop fovy ratio when training network', default=1)
 
    parser.add_argument('--S3D', help='Use Structure3D data for trianing', action='store_true')
    parser.add_argument('--ThreeD', help='Use ThreeD data for training ', action='store_true')
    parser.add_argument('--Video', help='Use Video data for training' , action='store_true')


    ##### training environment #####
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--beta1', type=float, default=0.5)        # momentum1 in Adam
    parser.add_argument('--beta2', type=float, default=0.999)      # momentum2 in Adam
    
    ############## Directory ############## 
    parser.add_argument('--model_name',help='path where models are to be saved' , type=str, default='./checkpoints/default') 
    
    parser.add_argument('--model_path', type=str, default='models')
    parser.add_argument('--sample_path', type=str, default='samples')
    parser.add_argument('--eval_path', type=str, default='evaluate')

    ############ Set log step ############
    parser.add_argument('--log_step', type=int , default=10)
    parser.add_argument('--sample_step', type=int , default=500)
    parser.add_argument('--checkpoint_step', type=int , default=10000)
    parser.add_argument('--eval_crop_ratio', type=int , default=0)
    
    config = parser.parse_args()
    
    config_path = os.path.join(config.model_name,'config.txt')
    f = open(config_path,'w')
    print(config,file=f)
    f.close()
 
    print(config)
    main(config)
