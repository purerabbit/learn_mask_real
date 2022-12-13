from re import L
import torch.nn as nn
from data.utils import *
#3 type of loss
'''
L(u,v)=||u-v||2/||u||2 + ||u-v||1/||u||1
u:input /\ unsampled ksapce data
v:output ksapce data with mask /\ 

Lac=lamda_img*L_img + lamda_grad*L_grad

'''
'''
input:Y(pred_1,pred_2,pred_u),X_p1,X_p2,X_u

'''
#input(B,C,H,W)


def cal_loss(under_kspace,output_ksapce,mask_loss):

    lossl1=nn.L1Loss()
    lossl2=nn.MSELoss()
    #k_space loss
    L_2=lossl2(under_kspace,output_ksapce*mask_loss)/lossl2(under_kspace,torch.zeros_like(under_kspace))
    L_1=lossl1(under_kspace,output_ksapce*mask_loss)/lossl1(under_kspace,torch.zeros_like(under_kspace))
    L_tot=L_1+L_2
    return L_tot