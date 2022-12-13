"""
BME1301
DO NOT MODIFY anything in this file.
"""
import matplotlib.pyplot as plt
from typing import Sequence, List, Union

import numpy as np
from numpy.lib.stride_tricks import as_strided
import torch
from torch.utils import data as Data
 
# from data.utils import kspace2image, image2kspace, complex2pseudo, pseudo2real, pseudo2complex
from .utils import kspace2image, image2kspace, complex2pseudo, pseudo2real, pseudo2complex
import sys
import os
sys.path.append("..")
from mask.main_mask import uf_mask
from mask.gen_mask import uniform_selection
# =============================================================================
# Utility
# =============================================================================
def arbitrary_dataset_split(dataset: Data.Dataset,
                            indices_list: Sequence[Sequence[int]]
                            ) -> List[torch.utils.data.Subset]:
    return [Data.Subset(dataset, indices) for indices in indices_list]


def datasets2loaders(datasets: Sequence[Data.Dataset],
                     *,
                     batch_size: Sequence[int] = (1, 1, 1),  # train, val, test
                     is_shuffle: Sequence[bool] = (True, False, False),  # train, val, test  should be change (True, False, False)
                     num_workers: int = 0) -> Sequence[Data.DataLoader]:
    """
    a tool for build N-datasets into N-loaders
    """
   
    assert isinstance(datasets[0], Data.Dataset)
    n_loaders = len(datasets)
    assert n_loaders == len(batch_size)
    assert n_loaders == len(is_shuffle)

    loaders = []
    for i in range(n_loaders):
        loaders.append(
            Data.DataLoader(datasets[i], batch_size=batch_size[i], shuffle=is_shuffle[i], num_workers=num_workers)
        )

    return loaders


def build_loader(dataset, batch_size,
                 train_indices=np.arange(0,100),
                 val_indices=np.arange(100,130),
                 test_indices=np.arange(130, 160),
                 num_workers=4):
    """
    :return: train/validation/test loader
    """
    print('dataset.__len__:',dataset.__len__)
    datasets = arbitrary_dataset_split(dataset, [train_indices, val_indices, test_indices])
    loaders = datasets2loaders(datasets, batch_size=(batch_size,) * 3, is_shuffle=(True, False, False), #should be change is_shuffle=(True, False, False)
                               num_workers=num_workers)
    return loaders


def cartesian_mask(shape, acc, sample_n=24, centred=False):
    """
    Sampling density estimated from implementation of kt FOCUSS
    shape: tuple - (Nslice, Nx, Ny, Ntime)
    acc: float - doesn't have to be integer 4, 8, etc..
    """
    shape=(1,256,256,1)#指定生成尺寸
    '''
    def uf_mask :
    shape,
    center_nums,
    accelerations,
    seed=123
    '''

 
    mask=uf_mask(shape,center_nums=sample_n,accelerations=acc)
    mask=mask[0,:,:,0]
    '''
    此部分生成了模拟欠采的mask,因为在生成的实际过程中使用了seed所以每个mask的种类是一样的
    对于划分 mask 不设置随机种子 使得每次 对于每个数据的mask都具有特异性 与ssdu保持一致
    
    '''
    # def normal_pdf(length, sensitivity):
    #     return np.exp(-sensitivity * (np.arange(length) - length / 2) ** 2)

    # N, Nx, Ny = int(np.prod([shape[0], shape[-1]])), shape[1], shape[2]

    # pdf_x = normal_pdf(Nx, 0.5 / (Nx / 10.) ** 2)
    # lmda = Nx / (2. * acc)
    # n_lines = int(Nx / acc)

    # # add uniform distribution
    # pdf_x += lmda * 1. / Nx

    # if sample_n:
    #     pdf_x[Nx // 2 - sample_n // 2:Nx // 2 + sample_n // 2] = 0
    #     pdf_x /= np.sum(pdf_x)
    #     n_lines -= sample_n

    # mask = np.zeros((N, Nx))
    # for i in range(N):
    #     idx = np.random.choice(Nx, n_lines, False, pdf_x)
    #     mask[i, idx] = 1

    # if sample_n:
    #     mask[:, Nx // 2 - sample_n // 2:Nx // 2 + sample_n // 2] = 1

    # size = mask.itemsize
    # mask = as_strided(mask, (N, Nx, Ny), (size * Nx, size, 0))

    # mask = mask.reshape((shape[0], shape[-1], Nx, Ny))
    # mask = np.transpose(mask, [0, 2, 3, 1])

    # if not centred:
    #     mask = np.fft.ifftshift(mask, axes=(1, 2))

    return mask #256 256


def np_undersample(k0, mask_centered):
    """
    input: k0 (H, W), mask_centered (H, W)
    output: x_u, k_u (H, W)  complex
    """
    assert k0.shape == mask_centered.shape
    # assert k0.dtype == torch.Tensor
    # print('k0.dtype:',k0.dtype)
    if isinstance(k0, torch.Tensor):
        k0 = k0.type(torch.complex64)
    else:
        k0 = k0.astype(np.complex64)
    # print('after-k0.dtype:',k0.dtype)

    k_u = k0 * mask_centered
    x_u = kspace2image(k_u)
    if isinstance(x_u,torch.Tensor):
        x_u = x_u.type(torch.complex64)
    else:
        x_u=x_u.astype(np.complex64)

    if isinstance(k_u,torch.Tensor):
        k_u = k_u.type(torch.complex64)
    else:
        k_u=k_u.astype(np.complex64)
    # k_u = k_u.dtype(torch.complex64)
    return x_u, k_u


# =============================================================================
# Dataset
# =============================================================================
class FastmriKnee(Data.Dataset):
    def __init__(self, path: str):
        """
        :param augment_fn: perform augmentation on image data [C=2, H, W] if provided.
        """
        data_dict = np.load(path)
        # loading dataset
        kspace = data_dict['kspace']  # List[ndarray]
        viz_indices = data_dict['vis_indices']  # List[int]

        # preprocessing
        images = kspace2image(kspace).astype(np.complex64)  # [1000, Nxy, Nxy] complex64
        images = complex2pseudo(images)  # convert to pseudo-complex representation
        self.images = images.astype(np.float32)  # [1000, 2, Nxy, Nxy] float32
        self.viz_indices = viz_indices.astype(np.int64)  # [N,] int64

        # inferred parameter
        self.n_slices = self.images.shape[0]

    def __getitem__(self, idx):
        im_gt = self.images[idx]
        return im_gt  # [2, Nxy, Nxy] float32

    def __len__(self):
        # print('self.n_slices-FastmriKnee:',self.n_slices)
        return self.n_slices


class DatasetReconMRI(Data.Dataset):
    def __init__(self, dataset: Data.Dataset, acc=4.0, num_center_lines=24, augment_fn=None):
        """
        :param augment_fn: perform augmentation on image data [C=2, H, W] if provided.
        """
        self.dataset = dataset

        # inferred parameter
        self.n_slices = len(dataset)

        # parameter for undersampling
        self.acc = acc
        self.num_center_lines = num_center_lines
        self.augment_fn = augment_fn

    def __getitem__(self, idx):
        im_gt = self.dataset[idx]  # [2, Nxy, Nxy] float32

        if self.augment_fn:
            im_gt = self.augment_fn(im_gt)  # [2, Nxy, Nxy] float32

        C, H, W = im_gt.shape
        und_mask = cartesian_mask(shape=(1, H, W, 1), acc=self.acc, sample_n=self.num_center_lines, centred=True
                                  ).astype(np.float32)#[0, :, :, 0]  # [H, W]
        k0 = image2kspace(pseudo2complex(im_gt))
        
        ks=np.ones((256,256,1))
        select_mask_up,select_mask_down=uniform_selection(ks,und_mask)
   
        k0=k0.astype(np.complex64) #最后的float32 约束 不会影响网络训练
        x_und, k_und = np_undersample(k0, und_mask)
        # plt.imshow(und_mask.squeeze(),cmap='gray')
        # plt.title('init_under_mask')
        # plt.show()
        #kspace2image( pseudo2complex(up_kspace))
        # plt.imshow(pseudo2real(complex2pseudo(kspace2image(k0))).squeeze(),cmap='gray')
        # plt.title('init_under_image')
        # plt.show()
        EPS = 1e-8
        #欠采样之后又进行了归一化  这一步可以去掉  
        x_und_abs = np.abs(x_und)
        # norm_min = x_und_abs.min()
        norm_max = x_und_abs.max()
        norm_scale = norm_max  # - norm_min + EPS
        x_und = x_und / norm_scale
        im_gt = im_gt / norm_scale

        k_und = image2kspace(x_und)  # [H, W] Complex
        k_und = complex2pseudo(k_und)  # [C=2, H, W]
        return (
            k_und.astype(np.float32),  # [C=2, H, W]
            und_mask.astype(np.float32),  # [H, W]
            im_gt.astype(np.float32),  # [C=2, H, W]
            select_mask_up.astype(np.float32),
            select_mask_down.astype(np.float32)
        )

    def __len__(self):
        # print('self.n_slices-DatasetReconMRI',self.n_slices)
        return self.n_slices


# if __name__ == '__main__':
#     dataset = FastmriKnee('./data/knee_singlecoil_1000_nor.npz')
#     print(len(dataset))
#     k_und, und_mask, im_gt = dataset[123]
#     print(f"{k_und.shape} {k_und.dtype}")
#     print(f"{und_mask.shape} {und_mask.dtype}")
#     print(f"{im_gt.shape} {im_gt.dtype}")
