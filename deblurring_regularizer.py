import matplotlib.pyplot as plt
import os
import numpy as np
import torch
from PIL import Image	
import argparse
import sys
import random
import hdf5storage
import pandas as pd

from models.downsampler_2 import Downsampler_2
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
parser.add_argument('--lr', type=float, default=0.004, help='lr')
parser.add_argument('--sigma', type=float, default=1.4142, help='reg_weight')
parser.add_argument('--testset_name', type=str, default='set5', help='dataset')
parser.add_argument('--noise_level_img', type=float, default=1.4142, help='noise level of the image')
parser.add_argument('--k_index', type=int, default=2, help='index of the kernel')
parser.add_argument('--save_im', '-si', action='store_true', help='enables saving images')


args = parser.parse_args()
num_iter=int(args.num_iter)
LR=float(args.lr)
testset_name=str(args.testset_name)
sigma=float(args.sigma)
noise_level_img=float(args.noise_level_img)
k_index=int(args.k_index)
SAVE_IM=args.save_im


if noise_level_img!=0:
    sigma = noise_level_img
   
method='deblurring_regularizer'
x8=True
noise_level_img = noise_level_img/255.0 


kernels = hdf5storage.loadmat(os.path.join('kernels', 'kernels_12.mat'))['kernels']
k = kernels[0, k_index].astype(np.float64)
k_tensor = util.single2tensor4(np.expand_dims(k, 2))
[k_tensor] = util.todevice([k_tensor], device)

L_path= "testsets/%s"%testset_name # L_path, for Low-quality images
L_paths = util.get_image_paths(L_path)

psnr_all=[]
ssim_all=[]
psnr_all_last=[]
ssim_all_last=[]
psnr_values_all=[]   

for idx, img in enumerate(L_paths):
    torch.cuda.empty_cache()
    psnr_values=[]
    ssim_values=[]
    
    img_name, ext = os.path.splitext(os.path.basename(img))
    
    print("******************",img_name,"*******************")
    print(method, ": k_index ", k_index,", noise level: " ,noise_level_img*255, ", weight of reg: ", sigma, ",",num_iter, "iterations, LR ", LR)
    
    ###########################################       create paths to save files       ############################################
    path_all="Results/Deblurring/%s_kernel%d_noise%.4f_LR%.4f_%d"%(testset_name,k_index,noise_level_img*255,LR,num_iter)
    util.mkdir(path_all)
    
    
    ###########################################              Load images              ############################################
    img_H = util.imread_uint(img, n_channels=3) #np W,H,3
    img_H = util.modcrop(img_H, 8)  
    
    
    ###########################################         Define and load model         ############################################
    model_path='models/regularizers/ReG.pth'
    model = net(in_nc=3, out_nc=3, nc=[64, 128, 256, 512], nb=4, act_mode='R', downsample_mode="strideconv", upsample_mode="convtranspose")
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')), strict=True)
    model.eval()
    for _, v in model.named_parameters():
        v.requires_grad = False
    model = model.to(device)
    
    
    ###########################################            Generate       LR         ############################################

    downsampler = Downsampler_2(factor=1, kernel=k_tensor[0,0], pad_mode='circular', freeze_params=True).type(dtype)
    
    img_L_tensor = downsampler(util.uint2tensor4(img_H).to(device))
    img_L = common_utils.torch_to_np (img_L_tensor)
    np.random.seed(seed=0)  
    img_L += np.random.normal(0, noise_level_img, img_L.shape) 
    img_L_tensor = common_utils.np_to_torch(img_L).to(device)
    
    
    ###########################################            Initialization         ############################################
    x = util.single2tensor4(np.transpose(img_L,(1,2,0))).to(device)
    x.requires_grad=True  
    x_fit = torch.tensor(x.shape,dtype=torch.float32,device=device)
    psnr_initialized = util.compare_psnr(img_H, np.transpose(common_utils.torch_to_np((x*255).clone().detach()),(1,2,0)))
    ssim_init = util.calculate_ssim(img_H, np.transpose(common_utils.torch_to_np(x.clone().detach()),(1,2,0))*255, border=0)
    print("psnr/ssim initial", round(psnr_initialized,3) , round(ssim_init,4))
    

    ###########################################             save images            ############################################
    
    if SAVE_IM==True:
        util.imsave(img_H, "%s/%s_inputHR.png"%(path_all,img_name))
        util.imsave(util.tensor2uint(img_L_tensor), "%s/%s_degraded_image.png"%(path_all,img_name))
    
    p=[]
    p += [x]
    optimizer = torch.optim.Adam(p,LR)
    
    for i in range(num_iter):
        optimizer.zero_grad()
    
        if i % 100 == 0:
            sys.stdout.flush()
        
        x_fit = downsampler(x.to(device))
        
        total_loss = 0.5*sr_utils2.norm2_loss(x_fit -img_L_tensor) 
        total_loss.backward(retain_graph=False,create_graph=False)
        
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
            
        psnr_HR = util.compare_psnr(img_H, np.transpose(common_utils.torch_to_np((x*255).clone().detach()),(1,2,0)))
        ssim_HR = util.calculate_ssim(img_H, np.transpose(common_utils.torch_to_np(x.clone().detach()),(1,2,0))*255, border=0)
        
        if i%200==0:    
            print ('Iteration %05d   PSNR_HR %.3f   SSIM_HR %.3f' % (i, psnr_HR, ssim_HR), '\r', end='')

        psnr_values.append(psnr_HR)
        ssim_values.append(ssim_HR)
        if psnr_HR >= np.max(psnr_values):
            best_img_np=common_utils.torch_to_np(x.clone().detach())
            best_iter=i
            best_psrn=psnr_HR
        if ssim_HR >= np.max(ssim_values):
            best_ssim=ssim_HR

        optimizer.step()
            
    psnr_values_all.append(psnr_values)
    if SAVE_IM==True:
        common_utils.draw_graph(path_all,'psnr',img_name,LR,sigma,best_psrn,best_iter,psnr_values)
        common_utils.draw_graph(path_all,'ssim',img_name,LR,sigma,best_psrn,best_iter,psnr_values)
        
    print("best psrn value " , np.max(psnr_values))
    print('best_iter=',best_iter)
    print('last psrn value= ', psnr_values[-1])
    print("best ssim value " , np.max(ssim_values))   
    print('last ssim value= ', ssim_values[-1])
    best_img_torch=common_utils.np_to_torch(best_img_np)
    
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