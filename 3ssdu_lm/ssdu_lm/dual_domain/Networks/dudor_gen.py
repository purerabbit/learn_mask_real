import torch.nn as nn
import numpy as np
from .DRDNnet import DRDN
import pdb

def arange(start, stop, step):
    """ Matlab-like arange
    """
    r = list(np.arange(start, stop, step).tolist())
    if r[-1] + step == stop:
        r.append(stop)
    return np.array(r)


def set_gpu(network, gpu_ids):
    network.to(gpu_ids[0])
    network = nn.DataParallel(network, device_ids=gpu_ids)

    return network


def get_generator(name ):

    if name == 'DRDN':
        ic = 2
       
        network = DRDN(n_channels=ic, G0=32, kSize=3, D=3, C=4, G=32, dilateSet=[1,2,3,3])

    else:
        raise NotImplementedError
 
    return  network 