import torch
import torch.nn as nn
import argparse
from evaluate import Evaluation
from network import *
from util import mkdirs, set_caffe_param_mult
from data_loader import S3D_loader,OmniDepthDataset,Stanford_loader,Sample_loader
import os.path as osp
from torch.utils.data import DataLoader
from torch.utils.data import ConcatDataset
from torchvision import transforms

from pano_loader.pano_loader import Pano3D

def evaluate_main(config):
    
    transform = transforms.Compose([
            transforms.ToTensor()])
    
    input_dir = ""
    device_ids = [0]


    if config.eval_data == "3D60":
        val_loader = OmniDepthDataset(root_path=input_dir, path_to_img_list=config.data_path)

    elif config.eval_data == 'Structure3D':
        val_loader = S3D_loader(config.data_path,transform = transform,transform_t = transform)

    elif config.eval_data == 'Stanford':
        val_loader = Stanford_loader(config.data_path,transform = transform,transform_t = transform)
 
    elif config.eval_data == 'Pano3D':
        val_loader = Pano3D(
        root=config.pano3d_root,
        part=config.pano3d_part,
        split=config.pano3d_split,
        types=config.pano3d_types,
    )
    elif config.eval_data == 'Inference':
        val_loader = Sample_loader(config.data_path,transform = transform,transform_t = transform)
 
    else:
        print("Check the command option")
# -------------------------------------------------------
    device = torch.device('cuda', device_ids[0])

    val_dataloader = torch.utils.data.DataLoader(
    	val_loader,
        batch_size=1,
	    shuffle=False,
    	num_workers=config.num_workers,
	    drop_last=False)

    evaluation = Evaluation(
    	config,
        val_dataloader, 
	    device)
    
    if config.method == "EBS":
        evaluation.evaluate_EBS()
    elif config.method == "SvSyn":
        evaluation.evaluate_svsyn()
    elif config.method == "JointDepth":
        evaluation.evaluate_jointdepth()
    elif config.method == "Bifuse":
        evaluation.evaluate_bifuse()
    elif config.method == "Hohonet":
        evaluation.evaluate_hohonet()
    elif config.method == "Omnidepth":
        evaluation.evaluate_omnidepth()
    elif config.method == "MD":
        evaluation.evaluate_MD()
    elif config.method == "enc_dec":
        evaluation.evaluate_backbone()
    else:
        print("Check Command options")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    ## JointDepth -> Proposed model 
    parser.add_argument("--method",
                                 type=str,
                                 help="method to be evaluated",
                                 choices=["JointDepth", "Bifuse", "Hohonet", "Omnidepth", "SvSyn", "EBS","MD","enc_dec"],
                                 default="JointDepth")
    
    parser.add_argument("--eval_data",
                                 type=str,
                                 help="data category to be evaluated",
                                 choices=["3D60", "Structure3D", "Stanford","Pano3D","Inference"],
                                 default="3D60")
  
    parser.add_argument("--backbone",
                                 type=str,
                                 help="data category to be evaluated",
                                 choices=["CSwin", "Swin", "DAT"],
                                 default="CSwin")


    
    # For 3D60 testset -> data_path should be text file containing image list
    # For Structure3D and Stanford testset -> data_path should be folder containing image/video frames
    parser.add_argument('--data_path', help = 'Data_path' , type=str, default='')

    parser.add_argument('--num_workers' , type=int, default=1)
    parser.add_argument('--checkpoint_path', type=str, default='rectnet.pth')
    parser.add_argument('--enc_path', type=str, default='')
    parser.add_argument('--dec_path', type=str, default='')

    parser.add_argument('--save_sample', help= 'save sampled results', action='store_true')
    parser.add_argument('--output_path', help = 'path where inferenced samples saved' , type=str, default='output')
    parser.add_argument('--pre_crop', help= 'crop image before input to the network ', action='store_true') # Used for 'Hres' pre-trained model 
    
    #### Used for HoHoNet evaluatoin ####
    parser.add_argument('--cfg',default='./HoHoNet/config/mp3d_depth/HOHO_depth_dct_efficienthc_TransEn1.yaml')
    parser.add_argument('opts',
                        help='Modify config options using the command-line',
                        default=None, nargs=argparse.REMAINDER)

    parser.add_argument("--pano3d_root", type=str, help="Path to the root folder containing the Pano3D extracted data.",default='/hdd/hklee/yuniw/Pano3D_dataset')
    parser.add_argument("--pano3d_part", type=str, help="The Pano3D subset to load.",default='M3D_high')
    parser.add_argument("--pano3d_split", type=str, help="The Pano3D split corresponding to the selected subset that will be loaded.",default='./pano_loader/Pano3D/splits/M3D_v1_test.yaml')
    parser.add_argument('--pano3d_types', default=['color','depth'], nargs='+',
            choices=[
                'color', 'depth', 'normal', 'semantic', 'structure', 'layout',
                'color_up', 'depth_up', 'normal_up', 'semantic_up', 'structure_up', 'layout_up'
                'color_down', 'depth_down', 'normal_down', 'semantic_down', 'structure_down', 'layout_down'
                'color_left', 'depth_left', 'normal_left', 'semantic_left', 'structure_left', 'layout_left'
                'color_right', 'depth_right', 'normal_right', 'semantic_right', 'structure_right', 'layout_right'
            ],
            help='The Pano3D data types that will be loaded, one of [color, depth, normal, semantic, structure, layout], potentially suffixed with a stereo placement from [up, down, left, right].'
        )
  
    config = parser.parse_args()
    evaluate_main(config)


