
from .unet.unet_model import UNet#用此方式可以实现包的导入

#从不同文件夹下导入包
import torch
import torch.nn as nn
import torch.nn.functional as F
from .cascade import CascadeMRIReconstructionFramework
#从不同文件夹下导入包


class ParallelNetwork(nn.Module):
   
    def __init__(self, num_layers, rank,bilinear=False):
        super(ParallelNetwork, self).__init__()
        self.num_layers = num_layers
        self.rank = rank
        self.bilinear=bilinear
       
        self.net = CascadeMRIReconstructionFramework(
            n_cascade=5  #the formor is 5
        )

    def forward(self,im_recon,mask,k0):
         
        output_mid=self.net(im_recon ,mask,k0)
        return  output_mid
