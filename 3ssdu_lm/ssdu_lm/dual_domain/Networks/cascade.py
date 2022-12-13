import torch
import torch.nn as nn
from data.utils import *
from .unet.unet_model import UNet#用此方式可以实现包的导入
import sys
sys.path.append("/home/liuchun/Desktop/add_dc/dual_domain/Networks")
from .dudor_gen import get_generator
class DataConsistencyLayer(nn.Module):
    """
    This class support different types k-space data consistency
    """

    def __init__(self, is_data_fidelity=True):
        super().__init__()
        self.is_data_fidelity = is_data_fidelity
        

        if is_data_fidelity:
            self.data_fidelity = nn.Parameter(torch.tensor(1.0,dtype=torch.float32))
            # self.data_fidelity = nn.Parameter(torch.ones((1,1),dtype=torch.float32))

    def forward(self, im_recon, mask, k0):
        """
        set is_data_fidelity=True to complete the formulation
        
        :param k: input k-space (reconstructed kspace, 2D-Fourier transform of im) complex
        :param k0: initially sampled k-space complex
        :param mask: sampling pattern
        """
        
        # if self.is_data_fidelity:
        #     k=image2kspace(pseudo2complex(im_recon))
        #     k0 = pseudo2complex(k0)
        #     v = self.is_data_fidelity
        #     k_dc = (1 - mask) * k + mask * (k + v * k0 / (1 + v))
        #     im_dc = complex2pseudo(kspace2image(k_dc))  # [B, C=2, H, W] 
        #数据一致性的实现
        k=image2kspace(pseudo2complex(im_recon))
        k0 = pseudo2complex(k0)
        # mask_new = torch.zeros_like(mask)
        # mask_new[mask == 0] = 1
        # mask_new[mask== 1] = 1/(1+self.data_fidelity)

        k_dc = (1 - mask) * k + mask * k0   
        # k_dc = mask_new * k + self.data_fidelity / (1+self.data_fidelity) * k0
        im_dc = complex2pseudo(kspace2image(k_dc))  # [B, C=2, H, W]   
        return im_dc


class CascadeMRIReconstructionFramework(nn.Module):
    def __init__(self,  n_cascade: int):
        super().__init__()
        self.cnn = get_generator('DRDN')######使用新的
        # self.cnn=UNet(n_channels=2, n_classes=2, bilinear=True)
        self.n_cascade = n_cascade

        assert n_cascade > 0
        dc_layers = [DataConsistencyLayer() for _ in range(n_cascade)]
        self.dc_layers = nn.ModuleList(dc_layers)

    def forward(self,im_recon, mask,k0):
        B, C, H, W = im_recon.shape
        B, C, H, W = k0.shape
        assert C == 2
        # assert im_recon.dtype==
        assert (B, H, W) == tuple(mask.shape)
        # im_recon =  complex2pseudo( kspace2image( pseudo2complex(k_und)))
      
        for dc_layer in self.dc_layers:
            # before= torch.abs(kspace2image(pseudo2complex(k0)))
            im_recon=self.cnn(im_recon)    
            # after= torch.abs(pseudo2complex(im_recon))#1 256 256  
            # after=torch.abs(pseudo2complex(ifft2(k0))) 
            # im_show=torch.cat((before,after),0)
            # imsshow(im_show.cpu().detach().numpy(),titles=['before','after'],num_col=2,cmap='gray',is_colorbar=True)


            # im_recon=self.cnn(im_recon)
            im_recon = dc_layer(im_recon, mask, k0)
        return im_recon