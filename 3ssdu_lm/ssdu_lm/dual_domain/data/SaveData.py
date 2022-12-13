#实现将数据提取并保存

from torch.utils.data import Dataset
import math
import numpy as np
import torch
from matplotlib import pyplot as plt
from fastmri.data import SliceDataset
from torch.utils.data import DataLoader
import os
from typing import Sequence, Tuple, Union
import random
import pathlib
import scipy.io as sio
import nibabel as nib

from utils import kspace2image,center_crop_with_padding,image2kspace,preprocess
                #   '/data/fastmri/singlecoil_train'
DATASET_PATH_trian= '/fastmridata/fastmri/singlecoil_train'
DATASET_PATH_val= '/fastmridata/fastmri/singlecoil_val'
DATASET_PATH_test= '/fastmridata/fastmri/singlecoil_test_v2'


def save_d(     dataset_path,
                name,                                  
                center_crop_shape: Tuple[int, int] = (256, 256)
                ):
    dataset = SliceDataset(
    root=dataset_path,
    challenge='singlecoil'
    )

    # num_slices_1=1
    num_slices=3#设置采集长度
    # select ALL kspace from dataset.
    #得到k空间数据
    raw_kspace_list = []
    for idx in range(num_slices):
        datapoint = dataset[idx]
        print('datapoint:',datapoint)
        raw_kspace_list.append(datapoint[0])#只把k空间数据存储到了list
    print('raw_kspace_list:',raw_kspace_list)

    processed_kspace_array,processed_image_array=preprocess(raw_kspace_list)
    processed_kspace_array = processed_kspace_array.astype(np.complex64)#减少数据量
    processed_image_array = processed_image_array.astype(np.complex64)#减少数据量

    np.savez(name,a=processed_kspace_array,b=processed_image_array,c=num_slices)
 
def print_info(data):
    if isinstance(data, np.ndarray):
        print(f"shape {data.shape}, type {data.dtype}.")
    else:
        print(data)

if __name__ == "__main__":
    save_d(DATASET_PATH_trian,'fastmri_train_3c.npz')
    save_d(DATASET_PATH_val,'fastmri_val_3c.npz')
    save_d(DATASET_PATH_test,'fastmri_test_3c.npz')
    print('save data successful!')
    #此处是绝对路径 需要用此方式进行引用
    data_train=np.load('/data/lc/3Dunet/fastmri_train.npz')
    data_val=np.load('/data/lc/3Dunet/fastmri_val.npz')
    data_test=np.load('/data/lc/3Dunet/fastmri_test.npz')
    print_info(data_train)
    print_info(data_val)
    print_info(data_test)


