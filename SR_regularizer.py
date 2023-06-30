import matplotlib.pyplot as plt
import os
import numpy as np
import torch
import argparse
import sys
import random					
import cv2
import pandas as pd

from models.downsampler import Downsampler
from models.network_unet import UNetRes as net
from utils import sr_utils2 
from utils import common_utils 
from utils import utils_model
from utils import utils_image as util

if torch.cuda.is_available():
  torch.cuda.manual_seed_all(0)
  torch.backends.cudnn.deterministic= True
  torch.backends.cudnn.benchmark = False
  dtype = torch.cuda.FloatTensor
else:
  dtype = torch.FloatTensor

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
parser = argparse.ArgumentParser()
parser.add_argument('--num_iter', default=1500, help='number of iterations of ADMM')
parser.add_argument('--kernel_type', type=str, default='bicubic', help='type of the kernel')
parser.add_argument('--factor', type=int, default=2, help='SR_factor') 
parser.add_argument('--lr', type=float, default=0.008, help='lr')
parser.add_argument('--gauss_sigma', type=float, default=0.5, help='sigma of gaussian filter')
parser.add_argument('--testset_name', type=str, default='set5', help='dataset')
parser.add_argument('--sigma', type=float, default=1.2, help='weight of regularization')
parser.add_argument('--noise_level_img', type=float, default=0.0, help='added noise')
parser.add_argument('--save_im', '-si', action='store_true', help='enables saving images')


args = parser.parse_args()
num_iter=int(args.num_iter)
KERNEL_TYPE = str(args.kernel_type)
gauss_sigma=float(args.gauss_sigma)
factor = int(args.factor)
LR=float(args.lr)
testset_name=str(args.testset_name)
sigma=float(args.sigma)
noise_level_img = float(args.noise_level_img)
SAVE_IM=args.save_im

if noise_level_img!=0:
    sigma = noise_level_img
    
method='SR_regularizer'
x8=True 
noise_level_img = noise_level_img/255.0 

img_names,ext = common_utils.get_image_names(testset_name)

psnr_all=[]
ssim_all=[]
psnr_all_last=[]
ssim_all_last=[]
psnr_values_all=[]

    
for img_p in img_names: #
    torch.cuda.empty_cache()
    psnr_values=[]
    ssim_values=[]
    
    print("******************",img_p,"*******************")
    print(method, ": factor", factor,",", KERNEL_TYPE, ", noise level: " ,noise_level_img*255, ", weight of reg: ", sigma, ",",num_iter, "iterations, LR ", LR)
    
    
    ###########################################       create paths to save files       ############################################
    
    path_all= "Results/SR/%s_factor%d_%s_noise%.4f_s%.3f_LR%.4f_it%d"%(testset_name,factor,KERNEL_TYPE,noise_level_img*255,sigma,LR,num_iter)
    util.mkdir(path_all)
    
    
    ###########################################              Load images              ############################################
    path_to_image = 'testsets/%s/%s%s'%(testset_name,img_p,ext)
    imgs = sr_utils2.load_LR_HR_imgs_sr(path_to_image , imsize = -1, factor = factor, enforse_div32='CROP')
    imgs['HR_np'] = common_utils.adjust_img_channels(imgs['HR_np']) 
    
    
    ###########################################         Define and load model         ############################################
    model_path='models/regularizers/ReG.pth'
    model = net(in_nc=3, out_nc=3, nc=[64, 128, 256, 512], nb=4, act_mode='R', downsample_mode="strideconv", upsample_mode="convtranspose")
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')), strict=True)
    model.eval()
    for _, v in model.named_parameters():
        v.requires_grad = False
    model = model.to(device)
    
  
    ###########################################         Downsample to the LR         ############################################
    downsampler = Downsampler(factor=factor, kernel_type=KERNEL_TYPE, gauss_sigma=gauss_sigma, antialias=True, pad_mode='circular', freeze_params=True).type(dtype)
    with torch.torch.no_grad():
        out_LR = downsampler(common_utils.np_to_torch(imgs['HR_np']).to(device))
    np.random.seed(0)
    out_LR=common_utils.torch_to_np(out_LR) + (np.random.normal(0, noise_level_img, out_LR.shape)) #we add the noise in numpy in order to have exactly the same noise as the methods we compare with (same seed)
    out_LR=common_utils.np_to_torch(out_LR).to(device)
    img_LR_kernel = out_LR.clone()
    
    
    ###########################################        Bicubic upsampling for initialization    ############################################
    out_HR = (cv2.resize(util.tensor2single(img_LR_kernel), (util.tensor2single(img_LR_kernel).shape[1]*factor, util.tensor2single(img_LR_kernel).shape[0]*factor), interpolation=cv2.INTER_CUBIC)) #.transpose(2,0,1)
    out_HR = out_HR.transpose(2,0,1)
    out_HR= common_utils.np_to_torch(out_HR).to(device)
    out_HR.requires_grad=True
    psnr_HR = util.compare_psnr(imgs['HR_np']*255, common_utils.torch_to_np((out_HR*255).clone().detach()))
    print("bicubic : ", psnr_HR)
    
    
    ###########################################             save images            ############################################
    if SAVE_IM==True:
        util.imsave(util.tensor2uint(common_utils.np_to_torch(imgs['HR_np'])), "%s/%s_inputHR.png"%(path_all,img_p))
        util.imsave(util.tensor2uint(out_LR), "%s/%s_HR_downsampled_%s.png"%(path_all,img_p,KERNEL_TYPE)) #image downsampled with our kernel

    p=[]
    p += [out_HR]
    optimizer = torch.optim.Adam(p,LR)
    
    for i in range(num_iter):
            
            optimizer.zero_grad()

            if i % 100 == 0:
                sys.stdout.flush()
            out_LR = downsampler(out_HR.to(device))

            total_loss = 0.5*sr_utils2.norm2_loss(out_LR-img_LR_kernel) 
            total_loss.backward(retain_graph=False,create_graph=False)
           
            with torch.torch.no_grad():
                if x8:
                    out_HR_ = util.augment_img_tensor4(out_HR.clone(), i % 8)

                grad_out = utils_model.test_mode(model, out_HR_.to(device), mode=2, refield=32, min_size=256, modulo=16)

                if x8:
                    if i % 8 == 3 or i % 8 == 5:
                        grad_out = util.augment_img_tensor4(grad_out, 8 - i % 8)
                    else:
                        grad_out = util.augment_img_tensor4(grad_out, i % 8)
                out_HR.grad+=((sigma/255)**2)*grad_out
                
            psnr_LR = util.compare_psnr(common_utils.torch_to_np(img_LR_kernel*255), common_utils.torch_to_np((out_LR*255).clone().detach()))
            psnr_HR = util.compare_psnr(imgs['HR_np']*255, common_utils.torch_to_np((out_HR*255).clone().detach()))
            ssim_HR = util.calculate_ssim((imgs['HR_np']*255).transpose(1,2,0), ((common_utils.torch_to_np(out_HR.clone().detach()))*255).transpose(1,2,0), border=0)
           
            if i%100==0:    
                print ('Iteration %05d    PSNR_LR %.3f   PSNR_HR %.3f   SSIM_HR %.3f' % (i, psnr_LR, psnr_HR,ssim_HR), '\r', end='')

            psnr_values.append(psnr_HR)
            ssim_values.append(ssim_HR)
            if psnr_HR >= np.max(psnr_values):
                best_img_np=common_utils.torch_to_np(out_HR.clone().detach())
                best_iter=i
                best_psrn=psnr_HR
            if ssim_HR >= np.max(ssim_values):
                best_ssim=ssim_HR
            
            optimizer.step()
            
    psnr_values_all.append(psnr_values)
    if SAVE_IM==True:
        common_utils.draw_graph(path_all,'psnr',img_p,LR,sigma,best_psrn,best_iter,psnr_values)
        common_utils.draw_graph(path_all,'ssim',img_p,LR,sigma,best_psrn,best_iter,psnr_values)
        
    print("best psrn value " , np.max(psnr_values))
    print('best_iter=',best_iter)
    print('last psrn value= ', psnr_values[-1])
    print("best ssim value " , np.max(ssim_values))   
    print('last ssim value= ', ssim_values[-1])
    best_img_torch=common_utils.np_to_torch(best_img_np)
        
    if SAVE_IM==True:
        util.imsave(util.tensor2uint(best_img_torch), "%s/best_%s_LR%.3f_%.3f_psrn_%.3f_%d.png"%(path_all,img_p,LR,sigma,best_psrn,best_iter))
        util.imsave(util.tensor2uint(out_HR), "%s/last_%s_LR%.3f_%.3f_%.3f_%d.png"%(path_all,img_p,LR,sigma,psnr_values[-1],num_iter))

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
    
#Draw the graph showing the average psnr over the whole dataset along the iterations 
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
