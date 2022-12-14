# System / Python
import pandas as pd
import os
import argparse
import logging
import random
import shutil
import time
import numpy as np
from tqdm import tqdm
# PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler#并行网路
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
# Custom
from Networks.network import ParallelNetwork as Network#导入网络
# from IXI_dataset import IXIData as Dataset
from mri_tools import rA, rAtA, rfft2
from data.utils import *
# from preprocessing import *
from mask.gen_mask import *
#导入fastmri数据集文件 导入方式可能有问题
# from .data.dataset import FASTMRIDataset as Dataset#引用包问题
from PIL import Image
# import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from torchvision.utils import save_image
# from utils import *
from data.dataset import *
from loss import cal_loss

from data.ploting import imsshow

results_save_path='./results'

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'#单GPU进行训练

parser = argparse.ArgumentParser()
#save images
parser.add_argument('--strain', type=str, default=None, help='file_name_train')
parser.add_argument('--stest', type=str, default=None, help='file_name_test')

parser.add_argument('--exp-name', type=str, default='self-supervised MRI reconstruction', help='name of experiment')
# parameters related to distributed training
parser.add_argument('--init-method', default=f'tcp://localhost:{np.random.randint(1000,2000)}', help='initialization method')
# parser.add_argument('--init-method', default=f'tcp://localhost:1883', help='initialization method')
parser.add_argument('--nodes', type=int, default=1, help='number of nodes')
parser.add_argument('--gpus', type=int, default=1, help='number of gpus per node')#放到一块GPU上进行训练
parser.add_argument('--world-size', type=int, default=None, help='world_size = nodes * gpus')
# parameters related to model
parser.add_argument('--use-init-weights', '-uit', type=bool, default=True, help='whether initialize model weights with defined types')
parser.add_argument('--init-type', type=str, default='xavier', help='type of initialize model weights')
parser.add_argument('--gain', type=float, default=1.0, help='gain in the initialization of model weights')
parser.add_argument('--num-layers', type=int, default=9, help='number of iterations')
# learning rate, batch size, and etc
parser.add_argument('--seed', type=int, default=30, help='random seed number')
parser.add_argument('--lr', '-lr', type=float, default=1e-4, help='initial learning rate')
parser.add_argument('--batch-size', type=int, default=1, help='batch size of single gpu')

parser.add_argument('--num-workers', type=int, default=8, help='number of workers')
parser.add_argument('--warmup-epochs', type=int, default=10, help='number of warmup epochs')
parser.add_argument('--num-epochs', type=int, default=500, help='maximum number of epochs')
# parameters related to data and masks
parser.add_argument('--train-path', type=str, default='/data/lc/3Dunet/fastmri_train_3c.npz', help='path of training data')
parser.add_argument('--val-path', type=str, default='/data/lc/3Dunet/fastmri_val_3c.npz', help='path of validation data')
parser.add_argument('--test-path', type=str, default='/data/lc/3Dunet/fastmri_test_3c.npz', help='path of test data')
#改变了数据集路径
parser.add_argument('--u-mask-path', type=str, default='./mask/undersampling_mask/mask_8.00x_acs24.mat', help='undersampling mask')
#欠采样问题
parser.add_argument('--s-mask-up-path', type=str, default='./mask/selecting_mask/mask_2.00x_acs16.mat', help='selection mask in up network')
parser.add_argument('--s-mask-down-path', type=str, default='./mask/selecting_mask/mask_2.50x_acs16.mat', help='selection mask in down network')
parser.add_argument('--train-sample-rate', '-trsr', type=float, default=0.06, help='sampling rate of training data')
parser.add_argument('--val-sample-rate', '-vsr', type=float, default=0.02, help='sampling rate of validation data')
parser.add_argument('--test-sample-rate', '-tesr', type=float, default=0.02, help='sampling rate of test data')
# save path
# parser.add_argument('--model-save-path', type=str, default='./checkpoints/', help='save path of trained model')
parser.add_argument('--model-save-path', type=str, default='./checkpoints', help='save path of trained model')
parser.add_argument('--loss-curve-path', type=str, default='./runs/loss_curve/', help='save path of loss curve in tensorboard')
# others
parser.add_argument('--mode', '-m', type=str, default='train', help='whether training or test model, value should be set to train or test')
parser.add_argument('--pretrained', '-pt', type=bool, default=False, help='whether load checkpoint')

#新网络参数
parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')

def create_logger():
    logger = logging.getLogger()
    logger.setLevel(level=logging.DEBUG)
    file_formatter = logging.Formatter('%(asctime)s %(filename)s [line:%(lineno)d] %(levelname)s:\t%(message)s')
    stream_formatter = logging.Formatter('%(levelname)s:\t%(message)s')

    file_handler = logging.FileHandler(filename='logger.txt', mode='a+', encoding='utf-8')
    file_handler.setLevel(level=logging.DEBUG)
    file_handler.setFormatter(file_formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(level=logging.INFO)
    stream_handler.setFormatter(stream_formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger

#初始化是否去掉
def init_weights(net, init_type='xavier', gain=1.0):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                nn.init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('Initialization method {} is not implemented.'.format(init_type))
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            nn.init.normal_(m.weight.data, 1.0, gain)
            nn.init.constant_(m.bias.data, 0.0)
    net.apply(init_func)


class EarlyStopping:
    def __init__(self, patience=50, delta=0.0):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta

    def __call__(self, metrics, loss=True):
        score = -metrics if loss else metrics
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0


def forward(mode, rank, model, dataloader, criterion, optimizer, log, args, epoch):
    writer = SummaryWriter('test2_dc')
    assert mode in ['train', 'val', 'test']
    loss, psnr, ssim = 0.0, 0.0, 0.0
    print('dataloader:',len(dataloader))
    t = tqdm(dataloader, desc=mode + 'ing', total=int(len(dataloader))) if rank == 0 else dataloader
    save_excel=[]
    for iter_num, data_batch in enumerate(t):
        k_und=data_batch[0].to(rank, non_blocking=True)
        und_mask=data_batch[1].to(rank, non_blocking=True)  #mask不用参与运算？？不用是tensor?
        im_gt = data_batch[2].to(rank, non_blocking=True)
        select_mask_up = data_batch[3].to(rank, non_blocking=True)
        select_mask_down = data_batch[4].to(rank, non_blocking=True)
        
        # ks=np.ones((256,256,1))

        # select_mask_up,select_mask_down=uniform_selection(ks,und_mask[0,:,:].unsqueeze(-1))
        # select_mask_up=select_mask_up.transpose(1,0)
        # select_mask_down=select_mask_down.transpose(1,0)
        # select_mask_up=select_mask_up.unsqueeze(0)
        # select_mask_down=select_mask_down.unsqueeze(0)
        
        '''
        und_mask.shape: (1, 1, 256, 256)
        select_mask_up.shape: (1, 256, 256)
        select_mask_down.shape: (1, 256, 256)
        '''
        

        # und_mask=und_mask.permute(0,2,3,1)#1,256,256,1
        select_mask_up=select_mask_up.unsqueeze(3)#1,256,256,1
        select_mask_down=select_mask_down.unsqueeze(3)

        # und_mask=und_mask.repeat(1,1,1,2)
        select_mask_up=select_mask_up.repeat(1,1,1,2)
        select_mask_down=select_mask_down.repeat(1,1,1,2) #1,256,256,2

        select_mask_up=select_mask_up.to(rank)
        select_mask_down=select_mask_down.to(rank)
        # print('und_mask.dtype:',und_mask.dtype)
        # und_mask = und_mask.float()
        select_mask_up = select_mask_up.float()
        select_mask_down = select_mask_down.float()
        
        und_mask=und_mask.unsqueeze(1).repeat(1,2,1,1)
        im_gt=im_gt.contiguous()
        select_mask_down=select_mask_down.permute(0,3,1,2)
        select_mask_up=select_mask_up.permute(0,3,1,2)
       
        # plt.imshow(select_mask_down[:,1,:,:].squeeze().detach().cpu(),cmap='gray')
        # plt.show()

        # plt.imshow(select_mask_up[:,1,:,:].squeeze().detach().cpu(),cmap='gray')
        # plt.show()

        #complex
        # plt.imshow(select_mask_down[0][0].detach().cpu().squeeze(),cmap='gray')
        # plt.title('loss_mask')
        # plt.show()
        # #kspace2image( pseudo2complex(up_kspace))
        # plt.imshow(pseudo2real(complex2pseudo(kspace2image(pseudo2complex(k_und).detach().cpu().squeeze()))).squeeze(),cmap='gray')
        # plt.title('loss_image')
        # plt.show()

        net_img_down,down_kspace=np_undersample(pseudo2complex(k_und).squeeze(),select_mask_down[0][0])
        net_img_up,up_kspace=np_undersample(pseudo2complex(k_und).squeeze(),select_mask_up[0][0])
        
        dc_mask=select_mask_up[:,1,:,:]
        loss_mask=select_mask_down[:,1,:,:]
        under_img = kspace2image(pseudo2complex(k_und))
        
         
        #此部分可能需要改变
        down_kspace=complex2pseudo(down_kspace)
        up_kspace=complex2pseudo(up_kspace)

        net_img_up=complex2pseudo(net_img_up)
        net_img_down=complex2pseudo(net_img_down)
        under_img=complex2pseudo(under_img)
        # net_img_up=torch.from_numpy(net_img_up)
        # net_img_down=torch.from_numpy(net_img_down)
        
        down_kspace=down_kspace.to(rank)
        up_kspace=up_kspace.to(rank)
        dc_mask=dc_mask.to(rank)
        loss_mask=loss_mask.to(rank)

        net_img_up=net_img_up.to(rank)
        net_img_down=net_img_down.to(rank)
        net_img_up=net_img_up.to(torch.float32)
        net_img_down=net_img_down.to(torch.float32)
        
        down_kspace=down_kspace.squeeze(0)
        up_kspace=up_kspace.squeeze(0)

        net_img_up=net_img_up.squeeze(0)
        net_img_down=net_img_down.squeeze(0)

        down_kspace=down_kspace.repeat(args.batch_size,1,1,1)
        up_kspace=up_kspace.repeat(args.batch_size,1,1,1)

        net_img_up=net_img_up.repeat(args.batch_size,1,1,1)
        net_img_down=net_img_down.repeat(args.batch_size,1,1,1)

        if mode == 'test':
            net_img_up = net_img_down = under_img
            select_mask_up = select_mask_down = und_mask
        # output_up, output_down,output_mid = model(net_img_up.contiguous(), net_img_down.contiguous(),under_img.contiguous(),und_mask.contiguous())#output class 设成1
        # plt.imshow()
            
        # input=torch.abs(pseudo2complex(net_img_up)) 
        # mshow=dc_mask
        # kshow=torch.abs(pseudo2complex(up_kspace))
        # im_show=torch.cat((input,mshow,kshow),0)
        # from data.utils import imsshow
        # imsshow(im_show.cpu().detach().numpy(),titles=['input','mshow','kshow'],num_col=3,cmap='gray',is_colorbar=True)

        #net_img_up:1 2 256 256        dc_mask  1, 256,256 
        output_up = model(net_img_up.contiguous(),dc_mask.contiguous(),up_kspace.contiguous())#output class 设成1
        # writer.add_graph(model,input_to_model=[net_img_up.contiguous(),dc_mask.contiguous(),up_kspace.contiguous()])
        
        #检查输入内容
        #输出顺序
        '''
        用于模拟欠采的mask
        dc划分图像(kspace转成 用于测试kspace与欠采的image是否一致)
        dc mask
        欠采样的图片
        
        '''
        # plt.imshow(und_mask[:,1,:,:].squeeze().detach().cpu(),cmap='gray')
        # plt.show()

        # plt.imshow(pseudo2real(under_img).squeeze().detach().cpu(),cmap='gray')
        # plt.show()

        # plt.imshow(pseudo2real(im_gt).squeeze().detach().cpu(),cmap='gray')
        # plt.show()
        # im_recon =  complex2pseudo( kspace2image( pseudo2complex(up_kspace)))
        # plt.imshow(complex2pseudo( kspace2image( pseudo2complex(up_kspace)))[:,1,:,:].squeeze().detach().cpu(),cmap='gray')
        # # plt.colorbar()
        # plt.show()
        # plt.imshow(dc_mask.squeeze().detach().cpu(),cmap='gray')
        # # plt.colorbar()
        # plt.show()
        # plt.imshow(pseudo2real(net_img_up).squeeze().detach().cpu(),cmap='gray')
        # # plt.colorbar()
        # plt.show()

        if(args.strain!=None):
            img_show=torch.cat((pseudo2real(im_gt),pseudo2real(under_img) ,pseudo2real(output_up)),0)
            gg=pseudo2real(im_gt)
            out=pseudo2real(output_up)
            gg=(gg-torch.min(gg))/(torch.max(gg)-torch.min(gg))
            out=(out-torch.min(out))/(torch.max(out)-torch.min(out))
            psnr_show=compute_psnr(gg,out)
            ssim_show=compute_ssim(gg,out)
            filename2save=f'/home/liuchun/Desktop/add_dc/ssdu_results/train_imgs/{args.strain}'
            if not os.path.exists(filename2save):
                os.makedirs(filename2save)
            imsshow(img_show.data.cpu().numpy(),['gt','undersampled','ssim: {:.3f} psnr: {:.3f}'.format(ssim_show, psnr_show)],1,cmap='gray',is_colorbar=True,filename2save=f'{filename2save}/{iter_num}.png')
        
        # if mode == 'test':
        #     if(args.stest!=None):
        #         img_show=torch.cat((pseudo2real(im_gt),pseudo2real(under_img) ,pseudo2real(output_up)),0)
        #         psnr_show=compute_psnr(pseudo2real(im_gt),pseudo2real(output_up))
        #         ssim_show=compute_ssim(pseudo2real(im_gt),pseudo2real(output_up))
 
        #         print(ssim_show, compute_ssim(pseudo2real(im_gt),pseudo2real((output_up - torch.min(output_up))/(torch.max(output_up)-torch.min(output_up)))))
        #         filename2save=f'/home/liuchun/SSDU_work/ssdu_results/test_imgs/{args.stest}'
        #         if not os.path.exists(filename2save):
        #             os.makedirs(filename2save)
        #         imsshow(img_show.data.cpu().numpy(),['gt','undersampled','ssim: {:.3f} psnr: {:.3f}'.format(ssim_show, psnr_show)],1,cmap='gray',is_colorbar=True,filename2save=f'{filename2save}/{iter_num}.png')


        # output_up_kspace = complex2pseudo(image2kspace(pseudo2complex(output_up)))
        loss_undersampled_kspace=complex2pseudo(image2kspace(pseudo2complex(net_img_down)))
        output_kspace=complex2pseudo(image2kspace(pseudo2complex(output_up)))
        # loss_mask=select_mask_down
        # output_down_kspace = complex2pseudo(image2kspace(pseudo2complex(output_down)))
        # output_mid_kspace = complex2pseudo(image2kspace(pseudo2complex(output_mid)))
        gt_kspace = complex2pseudo(image2kspace(pseudo2complex(im_gt)))

        # SSDU loss
        # plt.imshow(loss_mask.squeeze().detach().cpu()+dc_mask.squeeze().detach().cpu(),cmap='gray')
        # # plt.colorbar()
        # plt.show()

        batch_loss=cal_loss(loss_undersampled_kspace,output_kspace,loss_mask)
        # batch_loss=criterion(output_up_kspace, gt_kspace)#+criterion(output_mid_kspace, gt_kspace)+criterion(output_down_kspace, gt_kspace)
        # batch_loss=criterion(output_up_kspace, k_und)+criterion(output_mid_kspace, k_und)+criterion(output_down_kspace, k_und)

        # if batch_loss<0.1:
        #     save_data=[]
        #     print('save_image in this step')
        #     img=torch.cat((im_gt[0,0,:,:],output_up[0,0,:,:],net_img_up[0,0,:,:],net_img_down[0,0,:,:],under_img[0,0,:,:]),0)
        #     save_data.append(compute_ssim(im_gt,output_up))
        #     save_data.append(compute_ssim(under_img,output_up))
        #     save_data.append(compute_psnr(im_gt,output_up).item())
        #     save_data.append(compute_psnr(under_img,output_up).item())
        #     # save_data
        #     # print('compute_ssim(im_gt,output_up):',compute_ssim(im_gt,output_up))
        #     # print('compute_ssim(under_img,output_up):',compute_ssim(under_img,output_up))
        #     # print('compute_psnr(im_gt,output_up):',compute_psnr(im_gt,output_up))
        #     # print('compute_psnr(under_img,output_up):',compute_psnr(under_img,output_up))
        #     save_excel.append(save_data)#总的保存数据中
        #     save_image(img,f'/home/liuchun/dual_domain/train_image_results/final/{iter_num}.png')
         #保存生成的数据
        # if(save_excel!=[]):
        #     print('come into save_excel')
        #     result=pd.DataFrame(save_excel)
        #     result.to_excel("ssim_psnr.xlsx",float_format='%.5f')
        if mode == 'train':
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
        # else:
        #评估指标
        # ssim+=compute_ssim(output_up,im_gt)
        # psnr+=compute_psnr(output_up,im_gt)  对最后的数据进行归一化
        gg=pseudo2real(im_gt)
        out=pseudo2real(output_up)
        gg=(gg-torch.min(gg))/(torch.max(gg)-torch.min(gg))
        out=(out-torch.min(out))/(torch.max(out)-torch.min(out))

        ssim+=compute_ssim(out,gg)
        psnr+=compute_psnr(out,gg)
           
        loss += batch_loss.item()
   
    loss /= len(dataloader)
    log.append(loss)
    testimg=  abs(torch.complex(output_up[0, 0], output_up[0, 1])).unsqueeze(0)
    testimg = (testimg - torch.min(testimg))/(torch.max(testimg) - torch.min(testimg))
    if mode =='val':
        writer.add_image('images/val', testimg, epoch)
    elif mode =='train':
        writer.add_image('images/train', testimg, epoch)
    #保存生成的数据
    # if(save_excel!=[]):
    #     print('come into save_excel')
    #     result=pd.DataFrame(save_excel)
    #     result.to_excel("ssim_psnr_new.xlsx",float_format='%.5f')

    if mode == 'train':
        # curr_lr = optimizer.param_groups[0]['initial_lr']
        curr_lr = 0.0001
        log.append(curr_lr)
    # else:
    psnr /= len(dataloader)
    ssim /= len(dataloader)
    log.append(psnr)
    log.append(ssim)
    return log


def solvers(rank, ngpus_per_node, args):
    if rank == 0:
        logger = create_logger()
        logger.info('Running distributed data parallel on {} gpus.'.format(args.world_size))
    torch.cuda.set_device(rank)
    torch.distributed.init_process_group(backend='nccl', init_method=args.init_method, world_size=args.world_size, rank=rank)
    # set initial value
    start_epoch = 0
    best_ssim_val = 0.0
    best_ssim_train=0.0
    # model
    
    model = Network(num_layers=args.num_layers, rank=rank,bilinear=args.bilinear)#默认值输入
    
    # print(model.keys())
    # whether load checkpoint  模型参数保存 改成共享参数可能会变化
    if args.pretrained or args.mode == 'test':
        model_path = os.path.join(args.model_save_path, 'best_checkpoint.pth.tar')
        # print('model_path:',model_path)
        assert os.path.isfile(model_path)
        checkpoint = torch.load(model_path, map_location='cuda:{}'.format(rank))
        start_epoch = checkpoint['epoch']
        lr = checkpoint['lr']
        args.lr = lr
        best_ssim_val = checkpoint['best_ssim_val']
        best_ssim_train = checkpoint['best_ssim_train']
        model.load_state_dict(checkpoint['model'])
        if rank == 0:
            logger.info('Load checkpoint at epoch {}.'.format(start_epoch))
            logger.info('Current learning rate is {}.'.format(lr))
            logger.info('Current best ssim in val phase is {}.'.format(best_ssim_val))
            logger.info('Current best ssim in train phase is {}.'.format(best_ssim_train))
            logger.info('The model is loaded.')
    elif args.use_init_weights:
        init_weights(model, init_type=args.init_type, gain=args.gain)
        if rank == 0:
            logger.info('Initialize model with {}.'.format(args.init_type))
            # print('-------------------come rank==0')
    model = model.to(rank)
    # print('model to device successfully!')
    # writer.add_graph(model,input_to_model=[net_img_up.contiguous(), net_img_down.contiguous(),under_img.contiguous(),und_mask.contiguous()])
    # writer.add_graph(model, input_to_model = torch.rand(1, 3, 224, 224))
    # model = DDP(model, device_ids=[rank])#为了实现单GPU进行训练

    # criterion, optimizer, learning rate scheduler
    #损失函数部分 根据论文进行改变
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr,weight_decay=1e-5)
    if not args.pretrained:
        warm_up = lambda epoch: epoch / args.warmup_epochs if epoch <= args.warmup_epochs else 1
        scheduler_wu = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=warm_up)
    scheduler_re = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='max', factor=0.3, patience=20)
    early_stopping = EarlyStopping(patience=50, delta=1e-5)

    #数据集部分
    dataset1 = FastmriKnee('/home/liuchun/Desktop/3ssdu_lm/ssdu_lm/dual_domain/data/knee_singlecoil_well_360.npz')
    # print('dataset1:',len(dataset1))
    dataset=DatasetReconMRI(dataset1)
    # print('dataset:',len(dataset))
    train_loader,val_loader,test_loader=build_loader(dataset,args.batch_size)
    # print('train_loader,val_loader,test_loader:',len(train_loader),len(val_loader),len(test_loader))
    # test step  数据集部分更改
    if args.mode == 'test':

        if rank == 0:
            # logger.info('The size of test dataset is {}.'.format(len(test_set)))
            logger.info('Now testing {}.'.format(args.exp_name))
        model.eval()
        with torch.no_grad():
            test_log = []
            start_time = time.time()
            print('test_loader:',len(test_loader))
            test_log = forward('test', rank, model, test_loader, criterion, optimizer, test_log, args)
            test_time = time.time() - start_time
        # test information
        test_loss = test_log[0]
        test_psnr = test_log[1]
        test_ssim = test_log[2]
        if rank == 0:
            logger.info('time:{:.5f}s\ttest_loss:{:.7f}\ttest_psnr:{:.5f}\ttest_ssim:{:.5f}'.format(test_time, test_loss, test_psnr, test_ssim))
        return

    if rank == 0:
        # logger.info('The size of training dataset and validation dataset is {} and {}, respectively.'.format(len(train_set), len(val_set)))
        logger.info('Now training {}.'.format(args.exp_name))
        writer = SummaryWriter(args.loss_curve_path)
    for epoch in range(start_epoch + 1, args.num_epochs + 1):
        # train_sampler.set_epoch(epoch)
        train_log = [epoch]
        epoch_start_time = time.time()
        model.train()
        # print('train_loader:',len(train_loader))
        train_log = forward('train', rank, model, train_loader, criterion, optimizer, train_log, args, epoch)
        model.eval()
        with torch.no_grad():
            train_log = forward('val', rank, model, val_loader, criterion, optimizer, train_log, args, epoch)
        epoch_time = time.time() - epoch_start_time
        # train information
        epoch = train_log[0]
        train_loss = train_log[1]
        lr = train_log[2]

        train_psnr = train_log[3]
        train_ssim = train_log[4]

        val_loss = train_log[5]
        val_psnr = train_log[6]
        val_ssim = train_log[7]

        is_best_val = val_ssim > best_ssim_val
        best_ssim_val = max(val_ssim, best_ssim_val)
        is_best_train = train_ssim > best_ssim_train
        best_ssim_train = max(train_ssim, best_ssim_train)
        if rank == 0:
            logger.info('epoch:{:<8d}time:{:.5f}s\tlr:{:.8f}\ttrain_loss:{:.7f}\ttrain_psnr:{:.7f}\ttrain_ssim:{:.7f}\tval_loss:{:.7f}\tval_psnr:{:.5f}\t'
                        'val_ssim:{:.5f}'.format(epoch, epoch_time, lr, train_loss, train_psnr, train_ssim, val_loss, val_psnr, val_ssim))
            writer.add_scalars('loss', {'train_loss': train_loss, 'val_loss': val_loss}, epoch)
            # save checkpoint
            # print(1)#isidchdsih
            checkpoint = {
                'epoch': epoch,
                'lr': lr,
                'best_ssim_val': best_ssim_val,
                'best_ssim_train':best_ssim_train,
                'model': model.state_dict(keep_vars=True)
                # 'model': model.module.state_dict()
            }
            if not os.path.exists(args.model_save_path):
                # print('come here!')
                os.makedirs(args.model_save_path)
            model_path = os.path.join(args.model_save_path, 'checkpoint.pth.tar')
            best_model_path = os.path.join(args.model_save_path, 'best_checkpoint.pth.tar')
            torch.save(checkpoint, model_path)
            print('save the checkpoints successfully!')
            if is_best_val:
                shutil.copy(model_path, best_model_path)
        # scheduler
        if epoch <= args.warmup_epochs and not args.pretrained:
            scheduler_wu.step()
        scheduler_re.step(val_ssim)
        early_stopping(val_ssim, loss=False)
        if early_stopping.early_stop:
            if rank == 0:
                logger.info('The experiment is early stop!')
               
            break
    if rank == 0:
        writer.close()
    return


def main():
    args = parser.parse_args()
    args.world_size = args.nodes * args.gpus
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.multiprocessing.spawn(solvers, nprocs=args.gpus, args=(args.gpus, args))


if __name__ == '__main__':
    main()