import numpy as np
import numpy as np
import sys
from numpy.lib.stride_tricks import as_strided
import torch


def norm(tensor, axes=(0, 1, 2), keepdims=True):
    """
    Parameters
    ----------
    tensor : It can be in image space or k-space.
    axes :  The default is (0, 1, 2).
    keepdims : The default is True.

    Returns
    -------
    tensor : applies l2-norm . #默认求2范数

    """
    for axis in axes:#分别求出需要的范数 保存到tensor中
        tensor = np.linalg.norm(tensor, axis=axis, keepdims=True)#求范数

    if not keepdims: return tensor.squeeze()#将输入张量形状中的1去除并返回

    return tensor


def find_center_ind(kspace, axes=(1, 2, 3)):
    """
    Parameters
    ----------
    kspace : nrow x ncol x ncoil.
    axes :  The default is (1, 2, 3).

    Returns
    -------
    the center of the k-space

    """

    center_locs = norm(kspace, axes=axes).squeeze()#何种数据类型，返回的是tensor 对谁计算之后的tensor？

    return np.argsort(center_locs)[-1:]#np.argsort 生成把center_locs从小到大排序的下标
def index_flatten2nd(ind, shape):
    """
    Parameters
    ----------
    ind : 1D vector containing chosen locations.
    shape : shape of the matrix/tensor for mapping ind.

    Returns
    -------
    list of >=2D indices containing non-zero locations

    """

    array = np.zeros(np.prod(shape))
    array[ind] = 1
    ind_nd = np.nonzero(np.reshape(array, shape))

    return [list(ind_nd_ii) for ind_nd_ii in ind_nd]

def normal_pdf(length, sensitivity):
    return np.exp(-sensitivity * (np.arange(length) - length / 2) ** 2)



#生成模拟欠采的mask


#生成划分mask
'''input_data: input k-space, nrow x ncol x ncoil
    input_mask: input mask, nrow x ncol '''

def uniform_selection( input_data, input_mask, num_iter=1):#input_data->kspace   input_mask->mask_init
    
    small_acs_block=(4,4)
    rho=0.5     
    nrow, ncol = input_data.shape[0], input_data.shape[1]#行列 根据input的数据来定

    center_kx = int(find_center_ind(input_data, axes=(1, 2)))
    center_ky = int(find_center_ind(input_data, axes=(0, 2)))

    if num_iter == 0:
        print(f'\n Uniformly random selection is processing, rho = {rho:.2f}, center of kspace: center-kx: {center_kx}, center-ky: {center_ky}')
    #input_mask如何确定？
    temp_mask = np.copy(input_mask)
    temp_mask[center_kx - small_acs_block[0] // 2: center_kx + small_acs_block[0] // 2,
    center_ky - small_acs_block[1] // 2: center_ky + small_acs_block[1] // 2] = 0

    pr = np.ndarray.flatten(temp_mask)
    ind = np.random.choice(np.arange(nrow * ncol),
                            size=np.int(np.count_nonzero(pr) * rho), replace=False, p=pr / np.sum(pr))
    
    #实现原来欠采样的mask上进行采样
    [ind_x, ind_y] = index_flatten2nd(ind, (nrow, ncol))
  
    inpumak=input_mask 
    loss_mask = np.zeros_like(inpumak)
 
    
    loss_mask[ind_x, ind_y] = 1

    trn_mask = input_mask- loss_mask

    return trn_mask, loss_mask


