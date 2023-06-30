#Implementation with Mikael's downsampler 
from __future__ import print_function

import matplotlib.pyplot as plt
import os
import numpy as np
import torch
import argparse
import sys
import random
import hdf5storage
import pandas as pd

from models.network_unet import UNetRes as net
from utils import sr_utils2 
from utils import common_utils 
from utils import utils_model
from utils import utils_image as util
from utils import utils_mosaic
from utils.denoising_utils import *

if torch.cuda.is_available():
  torch.cuda.manual_seed_all(0)
  torch.backends.cudnn.deterministic= True
  torch.backends.cudnn.benchmark = False
  dtype = torch.cuda.FloatTensor
else:
  dtype = torch.FloatTensor

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
parser = argparse.ArgumentParser()
parser.add_argument('--num_iter', default=2000, help='number of iterations of ADMM')
parser.add_argument('--lr', type=float, default=0.01, help='lr')
parser.add_argument('--sigma', type=float, default=3.6, help='reg_weight')
parser.add_argument('--testset_name', type=str, default='set5', help='dataset')
parser.add_argument('--opt', type=str, default='grad_torch', help='algo') #'grad_manuel'
parser.add_argument('--factor', type=float, default='0.2', help='completion factor') 
parser.add_argument('--noise_level_img', type=float, default=0.0, help='noise level of the image')
parser.add_argument('--save_im', '-si', action='store_true', help='enables saving images')

args = parser.parse_args()
num_iter=int(args.num_iter)
LR=float(args.lr)
testset_name=str(args.testset_name)
opt=str(args.opt)
sigma=float(args.sigma)
factor=float(args.factor)
noise_level_img=float(args.noise_level_img)
SAVE_IM=args.save_im

method='completion_regularizer'
x8=True
noise_level_img = noise_level_img/255.0 

psnr_all=[]
ssim_all=[]
psnr_all_last=[]
ssim_all_last=[]
psnr_values_all=[]  

L_path= "testsets/%s"%testset_name # L_path, for Low-quality images
L_paths = util.get_image_paths(L_path)

for idx, img in enumerate(L_paths):
    
    torch.cuda.empty_cache()
    psnr_values=[]
    ssim_values=[]
    
    img_name, ext = os.path.splitext(os.path.basename(img))
    
    print("******************",img_name,"*******************")
    print(method, factor, ", weight of reg: ", sigma, ",",num_iter, "iterations, LR ", LR, ", gradient optimization method: ", opt)
    
    ###########################################       create paths to save files       ############################################

    path_all="Results/completion/%s_%.2f_noise%.4f_sig%.2f_LR%.4f_%d_%s"%(testset_name,factor,noise_level_img*255,sigma,LR,num_iter,opt)
    util.mkdir(path_all)
    
    ###########################################              Load images              ############################################
    img_H = util.imread_uint(img, n_channels=3) #np W,H,3
    
    ###########################################         Define and load model         ############################################
    model_path='models/regularizers/ReG.pth'
    model = net(in_nc=3, out_nc=3, nc=[64, 128, 256, 512], nb=4, act_mode='R', downsample_mode="strideconv", upsample_mode="convtranspose")
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')), strict=True)
    model.eval()
    for _, v in model.named_parameters():
        v.requires_grad = False
    model = model.to(device)
    
    
    ###########################################            Generate       LR         ############################################

    img_L = img_H.copy()
    img_L = util.uint2single(img_L)
    sz_y, sz_x = img_L.shape[0:2]
    np.random.seed(seed=0)
    mask = np.random.uniform(0,1,(sz_y, sz_x, 1))
    mask = mask > (1-factor)
    img_L = img_L *mask
    img_L += np.random.normal(0, noise_level_img, img_L.shape) 
    img_L_tensor, mask_tensor = util.single2tensor4(img_L).to(device), util.single2tensor4(mask).to(device)
    

    ###########################################            Initialization         ############################################
    torch.manual_seed(0)
    x = img_L_tensor + (1 - mask_tensor) * .5
    x.requires_grad=True
    x_fit = torch.tensor(x.shape,dtype=torch.float32,device=device)
    psnr_degraded = util.compare_psnr(img_H, np.transpose(torch_to_np(img_L_tensor.clone().detach()),(1,2,0))*255)
    psnr_initialized = util.compare_psnr(img_H, np.transpose(torch_to_np(x.clone().detach()),(1,2,0))*255)
    ssim_degraded = util.calculate_ssim(img_H, np.transpose(torch_to_np(img_L_tensor.clone().detach()),(1,2,0))*255, border=0)
    ssim_init = util.calculate_ssim(img_H, np.transpose(torch_to_np(x.clone().detach()),(1,2,0))*255, border=0) 
    print("psnr/ssim degraded", round(psnr_degraded,3), round(ssim_degraded,4))
    print("psnr/ssim initial", round(psnr_initialized,3) , round(ssim_init,4))
    
    if SAVE_IM==True:
        util.imsave(img_H, "%s/%s_inputHR.png"%(path_all,img_name))
        util.imsave(util.tensor2uint(x), "%s/%s_initialized_inpainted.png"%(path_all,img_name))
        util.imsave(util.tensor2uint(img_L_tensor), "%s/%s_degraded_image.png"%(path_all,img_name))
    
    p=[]
    p += [x]
    optimizer = torch.optim.Adam(p,LR)
    
    for i in range(num_iter):
            optimizer.zero_grad()
        
            if i % 100 == 0:
                sys.stdout.flush()
       
            x_fit = x * mask_tensor
            
            if opt=='grad_torch' :
                total_loss = 0.5*sr_utils2.norm2_loss(x_fit -img_L_tensor) 
                total_loss.backward(retain_graph=False,create_graph=False)
                running_loss = total_loss.item()
                with torch.torch.no_grad():
                    if x8:
                        x_ = util.augment_img_tensor4(x.clone(), i % 8)

                    grad_out = utils_model.test_mode(model, x_.to(device), mode=2, refield=32, min_size=256, modulo=16)

                    if x8:
                        if i % 8 == 3 or i % 8 == 5:
                            grad_out = util.augment_img_tensor4(grad_out, 8 - i % 8)
                        else:
                            grad_out = util.augment_img_tensor4(grad_out, i % 8)
                    x.grad+=((sigma/255)**2)*grad_out
            elif opt == 'pgd': #projected gradient descent
                with torch.torch.no_grad():
                    grad_out = utils_model.test_mode(model, x, mode=2, refield=32, min_size=256, modulo=16)
                    grad_out*=(1-mask_tensor)
                    x.grad=((sigma/255)**2)*grad_out
            elif opt == 'grad_dir'  or opt == 'grad_manuel':  
                mask_transposed= mask_tensor
                x.grad = mask_transposed * (x_fit -img_L_tensor)  
                with torch.torch.no_grad():
                    grad_out = utils_model.test_mode(model, x, mode=2, refield=32, min_size=256, modulo=16)
                    x.grad+=((sigma/255)**2)*grad_out
                    
            psnr_HR = util.compare_psnr(img_H, np.transpose(torch_to_np(x.clone().detach())*255,(1,2,0)))
            ssim_HR = util.calculate_ssim(img_H, np.transpose(torch_to_np(x.clone().detach()),(1,2,0))*255, border=0)
            
            if i%100==0:    
                print ('Iteration %05d   PSNR_HR %.3f   SSIM_HR %.3f' % (i, psnr_HR,ssim_HR), '\r', end='')

            psnr_values.append(psnr_HR)
            ssim_values.append(ssim_HR)
            if psnr_HR >= np.max(psnr_values):
                best_img_np=torch_to_np(x.clone().detach())
                best_iter=i
                best_psrn=psnr_HR
            if ssim_HR >= np.max(ssim_values):
                best_ssim=ssim_HR

            if opt=='grad_torch' or opt=='grad_dir' or opt =='pgd':
                optimizer.step()
            elif opt == 'grad_manuel':
                with torch.torch.no_grad():
                    x -= LR * x.grad
    psnr_values_all.append(psnr_values)
    if SAVE_IM==True:
        common_utils.draw_graph(path_all,'psnr',img_name,LR,sigma,best_psrn,best_iter,psnr_values)
        common_utils.draw_graph(path_all,'ssim',img_name,LR,sigma,best_psrn,best_iter,psnr_values)
    
    print("best psrn value " , np.max(psnr_values))
    print('best_iter=',best_iter)
    print('last psrn value= ', psnr_values[-1])
    print("best ssim value " , np.max(ssim_values))   
    print('last ssim value= ', ssim_values[-1])
    best_img_torch=np_to_torch(best_img_np)
    
    if SAVE_IM==True:
        util.imsave(util.tensor2uint(best_img_torch), "%s/best_%s_sig%.2f_LR%.3f_psrn_%.3f_%d.png"%(path_all,img_name,sigma,LR,best_psrn,best_iter))
        util.imsave(util.tensor2uint(x), "%s/last_%s_sig%.2f_LR%.3f_psrn_%.3f_%d.png"%(path_all,img_name,sigma,LR,psnr_values[-1],num_iter))
    
    psnr_all.append(round(float(best_psrn),2))
    ssim_all.append(round(float(best_ssim),4))
    psnr_all_last.append(round(float(psnr_values[-1]),2))
    ssim_all_last.append(round(float(ssim_values[-1]),4))


common_utils.write_log("%s/psnr_avg_best.txt"%(path_all),sigma,LR,psnr_all,2)  
common_utils.write_log("%s/psnr_avg_last.txt"%(path_all),sigma,LR,psnr_all_last,2)    
common_utils.write_log("%s/ssim_avg_best.txt"%(path_all),sigma,LR,ssim_all,4)    
common_utils.write_log("%s/ssim_avg_last.txt"%(path_all),sigma,LR,ssim_all_last,4)


###########################################             write psnr to a csv file           ############################################
psnr_values_all_avg=[]
for i in range (len(psnr_values_all[0])): #nbr iter
    avg = 0
    for j in range (len(psnr_values_all)): #nbr image 
        avg+=psnr_values_all[j][i]
    avg/=len(psnr_values_all)
    psnr_values_all_avg.append(avg)
path_="%s/avg_psnr.png"%(path_all)
plt.clf()
plt.plot(range(len(psnr_values_all_avg)), psnr_values_all_avg)
plt.savefig(path_)
psnr_values_all = np.array(psnr_values_all)
psnr_values_all_T=psnr_values_all.T
psnr_values_all_T = pd.DataFrame(psnr_values_all_T)
psnr_values_all_avg = pd.DataFrame(psnr_values_all_avg)
psnr_values_all_T.to_csv(os.path.join(path_all, 'iterations_psnr.csv'), sep=';')
psnr_values_all_avg.to_csv(os.path.join(path_all, 'average_psnr.csv'), sep=';')

