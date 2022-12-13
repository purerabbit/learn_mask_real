import torch
import torch.nn as nn
from data.utils import *
from .unet.unet_model import UNet#用此方式可以实现包的导入

class DataConsistencyLayer(nn.Module):
    """
    This class support different types k-space data consistency
    """

    def __init__(self, is_data_fidelity=True):
        super().__init__()
        # self.is_data_fidelity = is_data_fidelity
        
        # if is_data_fidelity:
        #     self.data_fidelity = nn.Parameter(torch.tensor(1.0,dtype=torch.float32))
            # self.data_fidelity = nn.Parameter(torch.ones((1,1),dtype=torch.float32))

    def forward(self, im_recon, k0, mask):
        """
        set is_data_fidelity=True to complete the formulation
        
        :param k: input k-space (reconstructed kspace, 2D-Fourier transform of im) complex
        :param k0: initially sampled k-space complex
        :param mask: sampling pattern
        """
        k=complex2pseudo(image2kspace(pseudo2complex(im_recon)))
        k0 = pseudo2complex(k0)
        k_dc = (1 - mask) * k + mask * k0   
        im_dc = complex2pseudo(kspace2image(k_dc))  # [B, C=2, H, W]   
        return im_dc


class CascadeMRIReconstructionFramework(nn.Module):
    def __init__(self,  n_cascade: int):
        super().__init__()
        self.cnn=UNet(n_channels=2, n_classes=2, bilinear=True)
        self.n_cascade = n_cascade

        assert n_cascade > 0
        dc_layers = [DataConsistencyLayer() for _ in range(n_cascade)]
        self.dc_layers = nn.ModuleList(dc_layers)

    def forward(self, k_und, mask):
        B, C, H, W = k_und.shape
        assert C == 2
        assert (B, H, W) == tuple(mask.shape)
        im_recon =  complex2pseudo( kspace2image( pseudo2complex(k_und)))
      
        for dc_layer in self.dc_layers:
            im_recon=self.cnn(im_recon)
            im_recon = dc_layer(im_recon, k_und, mask)
        return im_recon