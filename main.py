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
    os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu
    cudnn.benchmark = True
#    torch.manual_seed(1593665876)
#    torch.cuda.manual_seed_all(4099049913103886)    

    torch.manual_seed(159111236)
    torch.cuda.manual_seed_all(4099049123103886)    
#    torch.manual_seed(159111235)
#    torch.cuda.manual_seed_all(4099049123103885)    

    config.distributed = config.world_size > 1 or config.multiprocessing_distributed
    ngpus_per_node = torch.cuda.device_count()

    if config.multiprocessing_distributed:
        config.world_size = ngpus_per_node * config.world_size
        torch.multiprocessing.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, config))
    else:
        main_worker(config.gpu, ngpus_per_node, config)

def main_worker(gpu, ngpus_per_node, config):
    if config.gpu is not None:
        print(f'Use GPU: {gpu} for training')

    if config.distributed:
        if config.dist_url == "envs://" and config.rank == -1:
            config.rank = int(os.environ["RANK"])
        if config.multiprocessing_distributed:
            config.rank = config.rank * ngpus_per_node + gpu
        torch.distributed.init_process_group(backend=config.dist_backend, init_method=config.dist_url, world_size=config.world_size, rank=config.rank)

    transform = transforms.Compose([
                    transforms.Resize((config.input_height,config.input_width)),
#                    transforms.CenterCrop((config.input_height * 3 //4 ,config.input_width)),
                    transforms.ToTensor()
                    ])

    transform_s3d = transforms.Compose([
                    transforms.ToTensor()
                    ])


    S3D_data = S3D_loader(config.S3D_path,transform = transform_s3d,transform_t = transform_s3d)
    if config.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(S3D_data)
    else:
        train_sampler = None

    config.batch_size = int(config.batch_size / ngpus_per_node)
    config.num_workers = int((config.num_workers + ngpus_per_node - 1) / ngpus_per_node)

    S3D_dataloader = DataLoader(S3D_data,batch_size=config.batch_size,num_workers=config.num_workers,pin_memory=True, sampler=train_sampler)

    if config.mode == 'train':
        train = Train(config,S3D_dataloader, gpu, train_sampler)
        train.train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--mode', type=str, default='train')

    parser.add_argument('--S3D_path',help='Folder containing Structure3D dataset', type=str,default='../Structure3D/Structured3D') # train image list file for supervised learning.
 
    parser.add_argument('--val_path',type=str,default='./SAMPLE') # file path which contains images to be sampled
    parser.add_argument('--test_path',type=str,default='test') # text which contains test image list -> not used here

    parser.add_argument('--checkpoint_path',type=str,default='./pretrained_BTS/generator_best.pkl') # detphnet checkpoint path
 
    ##### hyper-parameters #####
    parser.add_argument('--Continue', help=' Strat training from the pre-trained model', action='store_true')
    parser.add_argument('--Pretrained', help=' Strat training from the pre-trained model', action='store_true')
    parser.add_argument('--freeze', help=' Freeze decoder weight', action='store_true')


    parser.add_argument('--input_height', type=int, help='input height', default=256)
    parser.add_argument('--input_width', type=int, help='input width', default=512)
    parser.add_argument('--fovy_ratio', type=float, help='crop fovy ratio when training network', default=1)
 
    parser.add_argument('--S3D', help='Use Structure3D data for trianing', action='store_true')
    parser.add_argument('--ThreeD', help='Use ThreeD data for training ', action='store_true')
    parser.add_argument('--Video', help='Use Video data for training' , action='store_true')


    ##### training environment #####
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--beta1', type=float, default=0.5)        # momentum1 in Adam
    parser.add_argument('--beta2', type=float, default=0.999)      # momentum2 in Adam
    
    ############## Directory ############## 
    parser.add_argument('--use_hybrid', help='Use hybrid patch embedding', action='store_true')
    parser.add_argument("--backbone",
                                 type=str,
                                 help="backbone to be used",
                                 choices=["Swin", "DAT", "CSwin"],
                                 default="Cswin")
 
    parser.add_argument('--model_name',help='path where models are to be saved' , type=str, default='./checkpoints/default') 
    parser.add_argument('--model_path', type=str, default='models')
    parser.add_argument('--sample_path', type=str, default='samples')
    parser.add_argument('--eval_path', type=str, default='evaluate')
    parser.add_argument('--enc_path', type=str, help='path to pretrained encoder weight', default='./dat_base_in1k_384.pth')
    parser.add_argument('--dec_path', type=str, help='path to pretrained decoder weight', default='./dpt_large-midas-2f21e586.pt')

    ############ Set log step ############
    parser.add_argument('--log_step', type=int , default=10)
    parser.add_argument('--sample_step', type=int , default=500)
    parser.add_argument('--checkpoint_step', type=int , default=10000)
    parser.add_argument('--eval_crop_ratio', type=int , default=0)
    
    ############ Distributed Data Parallel (DDP) ############
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--rank', type=int, default=-1)
    parser.add_argument('--gpu', type=str, default="0,1,2,3")
    parser.add_argument('--dist-url', type=str, default="tcp://127.0.0.1:7777")
    parser.add_argument('--dist-backend', type=str, default="nccl")
    parser.add_argument('--multiprocessing_distributed', default=True)

    config = parser.parse_args()
    
    if not os.path.exists(config.model_name):
        os.mkdir(config.model_name)
    config_path = os.path.join(config.model_name,'config.txt')
    f = open(config_path,'w')
    print(config,file=f)
    f.close()
 
    print(config)
    main(config)
