# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 20:26:45 2020

@author: Pasquale
"""
from .common_utils import *

def put_in_center(img_np, target_size):
    img_out = np.zeros([3, target_size[0], target_size[1]])
    
    bbox = [
            int((target_size[0] - img_np.shape[1]) / 2),
            int((target_size[1] - img_np.shape[2]) / 2),
            int((target_size[0] + img_np.shape[1]) / 2),
            int((target_size[1] + img_np.shape[2]) / 2),
    ]
    
    img_out[:, bbox[0]:bbox[2], bbox[1]:bbox[3]] = img_np
    
    return img_out



    
    
def load_LR_HR_imgs_sr2(fname, imsize, factor, enforse_div32=None, noise_level=0):
    '''Loads an image, resizes it, center crops and downscales.

    Args: 
        fname: path to the image
        imsize: new size for the image, -1 for no resizing
        factor: downscaling factor
        enforse_div32: if 'CROP' center crops an image, so that its dimensions are divisible by 32.
    '''
    img_orig_pil, img_orig_np = get_image(fname, -1)

    if imsize != -1:
        img_orig_pil, img_orig_np = get_image(fname, imsize)
        
    # For comparison with GT
    if enforse_div32 == 'CROP':
        new_size = (img_orig_pil.size[0] - img_orig_pil.size[0] % 32, 
                    img_orig_pil.size[1] - img_orig_pil.size[1] % 32)

        bbox = [
                (img_orig_pil.size[0] - new_size[0])/2, 
                (img_orig_pil.size[1] - new_size[1])/2,
                (img_orig_pil.size[0] + new_size[0])/2,
                (img_orig_pil.size[1] + new_size[1])/2,
        ]

        img_HR_pil = img_orig_pil.crop(bbox)
        img_HR_np = pil_to_np(img_HR_pil)
        
    else:
        img_HR_pil, img_HR_np = img_orig_pil, img_orig_np
        
    LR_size = [
               img_HR_pil.size[0] // factor, 
               img_HR_pil.size[1] // factor
    ]

    img_LR_pil = img_HR_pil.resize(LR_size, Image.ANTIALIAS)
    img_LR_np = pil_to_np(img_LR_pil)
    
    noise     = np.random.standard_normal(size=img_LR_np.shape)
    img_LR_np = img_LR_np + noise_level*noise
    img_LR_pil=np_to_pil(img_LR_np)

    print('HR and LR resolutions: %s, %s' % (str(img_HR_pil.size), str (img_LR_pil.size)))

    return {
                'orig_pil': img_orig_pil,
                'orig_np':  img_orig_np,
                'LR_pil':  img_LR_pil, 
                'LR_np': img_LR_np,
                'HR_pil':  img_HR_pil, 
                'HR_np': img_HR_np
           }    


def get_baselines(img_LR_pil, img_HR_pil):
    '''Gets `bicubic`, sharpened bicubic and `nearest` baselines.'''
    img_bicubic_pil = img_LR_pil.resize(img_HR_pil.size, Image.BICUBIC)
    img_bicubic_np = pil_to_np(img_bicubic_pil)

    img_nearest_pil = img_LR_pil.resize(img_HR_pil.size, Image.NEAREST)
    img_nearest_np = pil_to_np(img_nearest_pil)

    img_bic_sharp_pil = img_bicubic_pil.filter(PIL.ImageFilter.UnsharpMask())
    img_bic_sharp_np = pil_to_np(img_bic_sharp_pil)

    return img_bicubic_np, img_bic_sharp_np, img_nearest_np



def tv_loss(x, beta = 0.5):
    '''Calculates TV loss for an image `x`.
        
    Args:
        x: image, torch.Variable of torch.Tensor
        beta: See https://arxiv.org/abs/1412.0035 (fig. 2) to see effect of `beta` 
    '''
    dh = torch.pow(x[:,:,:,1:] - x[:,:,:,:-1], 2)
    dw = torch.pow(x[:,:,1:,:] - x[:,:,:-1,:], 2)


    
    return torch.sum(torch.pow(dh[:, :, :-1] + dw[:, :, :, :-1], beta))

def norm1_loss(x):
    '''Calculates norm1 loss for an image `x`.
        
    Args:
        x: image, torch.Variable of torch.Tensor
        beta: See https://arxiv.org/abs/1412.0035 (fig. 2) to see effect of `beta` 
    '''
    #dh = torch.pow(x[:,:,:,1:] - x[:,:,:,:-1], 2)
    #dw = torch.pow(x[:,:,1:,:] - x[:,:,:-1,:], 2)
    
    vec = torch.abs(x[:,:,:,:])
    
    return torch.sum(vec)

def norm2_loss(x):
    '''Calculates TV loss for an image `x`.
        
    Args:
        x: image, torch.Variable of torch.Tensor
        beta: See https://arxiv.org/abs/1412.0035 (fig. 2) to see effect of `beta` 
    '''
    vec = torch.pow(x[:,:,:,:], 2)
    
    return torch.sum(vec)

def norm2_loss_sans(x):
    '''Calculates TV loss for an image `x`.
        
    Args:
        x: image, torch.Variable of torch.Tensor
        beta: See https://arxiv.org/abs/1412.0035 (fig. 2) to see effect of `beta` 
    '''
    vec = torch.pow(x[:,:,:,:], 1)
    
    return torch.sum(vec)
    
def tv1_loss(x):
    '''Calculates norm1 loss for an image `x`.
        
    Args:
        x: image, torch.Variable of torch.Tensor
        beta: See https://arxiv.org/abs/1412.0035 (fig. 2) to see effect of `beta` 
    '''
    dh = torch.abs(x[:,:,:,1:] - x[:,:,:,:-1])
    dw = torch.abs(x[:,:,1:,:] - x[:,:,:-1,:]) 
    
    return torch.sum(torch.pow(dh[:, :, :-1] + dw[:, :, :, :-1], 1))

def tvalfa_loss(x, beta = 0.5, alfa=0.000001):
    '''Calculates TV loss for an image `x`.
        
    Args:
        x: image, torch.Variable of torch.Tensor
        beta: See https://arxiv.org/abs/1412.0035 (fig. 2) to see effect of `beta` 
    '''
    dh = torch.pow(x[:,:,:,1:] - x[:,:,:,:-1], 2)
    dw = torch.pow(x[:,:,1:,:] - x[:,:,:-1,:], 2)
    
    return torch.sum(torch.pow(dh[:, :, :-1] + dw[:, :, :, :-1] + alfa , beta))


def wtvalfa_loss(x, beta = 0.5, alfa=0.000001, omega=0.5):
    '''Calculates TV loss for an image `x`.
        
    Args:
        x: image, torch.Variable of torch.Tensor
        beta: See https://arxiv.org/abs/1412.0035 (fig. 2) to see effect of `beta` 
    '''
    dh = torch.pow(x[:,:,:,1:] - x[:,:,:,:-1], 2)
    dw = torch.pow(x[:,:,1:,:] - x[:,:,:-1,:], 2)
    
    I = dh[:, :, :-1] + dw[:, :, :, :-1] + alfa;
    I = I / torch.max(I).item()
    I = torch.sqrt(I)
    weight = 1 / omega + I 
    
    return torch.sum(weight*torch.pow(dh[:, :, :-1] + dw[:, :, :, :-1] + alfa , beta))

def load_LR_HR_imgs_sr(fname, imsize, factor, enforse_div32=None):
    '''Loads an image, resizes it, center crops and downscales.

    Args: 
        fname: path to the image
        imsize: new size for the image, -1 for no resizing
        factor: downscaling factor
        enforse_div32: if 'CROP' center crops an image, so that its dimensions are divisible by 32.
    '''
    img_orig_pil, img_orig_np = get_image(fname, -1)
    img_orig_np=pil_to_np(img_orig_pil)
    # For comparison with GT
    if enforse_div32 == 'CROP':
        if factor == 3:
            new_size = (img_orig_pil.size[0] - img_orig_pil.size[0] % 3, 
            img_orig_pil.size[1] - img_orig_pil.size[1] % 3)
            bbox = [0,0, new_size[0],new_size[1]]
        else:
            new_size = (img_orig_pil.size[0] - img_orig_pil.size[0] % 32, 
                        img_orig_pil.size[1] - img_orig_pil.size[1] % 32)

            bbox = [
                    (img_orig_pil.size[0] - new_size[0])/2, 
                    (img_orig_pil.size[1] - new_size[1])/2,
                    (img_orig_pil.size[0] + new_size[0])/2,
                    (img_orig_pil.size[1] + new_size[1])/2,
            ]
        
        img_HR_pil = img_orig_pil.crop(bbox)
        img_HR_np = pil_to_np(img_HR_pil)
        
    else:
        img_HR_pil, img_HR_np = img_orig_pil, img_orig_np
        
    LR_size = [
               img_HR_pil.size[0] // factor, 
               img_HR_pil.size[1] // factor
    ]

    img_LR_pil = img_HR_pil.resize(LR_size, Image.ANTIALIAS)
    img_LR_np = pil_to_np(img_LR_pil)

    print('HR and LR resolutions: %s, %s' % (str(img_HR_pil.size), str (img_LR_pil.size)))

    return {
                'orig_pil': img_orig_pil,
                'orig_np':  img_orig_np,
                'LR_pil':  img_LR_pil, 
                'LR_np': img_LR_np,
                'HR_pil':  img_HR_pil, 
                'HR_np': img_HR_np
           }
def load_LR_HR_imgs_sr3(fname, imsize, factor, enforse_div32=None):
    '''Loads an image, resizes it, center crops and downscales.

    Args: 
        fname: path to the image
        imsize: new size for the image, -1 for no resizing
        factor: downscaling factor
        enforse_div32: if 'CROP' center crops an image, so that its dimensions are divisible by 32.
    '''
    img_orig_pil, img_orig_np = get_image(fname, -1)
    #img_orig_pil = img_orig_pil.crop((1, 1, 256, 256))
    img_orig_np=pil_to_np(img_orig_pil)
    if imsize != -1:
        img_orig_pil, img_orig_np = get_image(fname, imsize)
        
    # For comparison with GT
    if enforse_div32 == 'CROP':
        new_size = (img_orig_pil.size[0] - img_orig_pil.size[0] % 3, 
                    img_orig_pil.size[1] - img_orig_pil.size[1] % 3)
        bbox = [0,0, new_size[0],new_size[1]]
        
        """bbox = [
                (img_orig_pil.size[0] - new_size[0])/2, 
                (img_orig_pil.size[1] - new_size[1])/2,
                (img_orig_pil.size[0] + new_size[0])/2,
                (img_orig_pil.size[1] + new_size[1])/2,
        ]"""
        
        img_HR_pil = img_orig_pil.crop(bbox)
        img_HR_np = pil_to_np(img_HR_pil)
        
    else:
        img_HR_pil, img_HR_np = img_orig_pil, img_orig_np
        
    LR_size = [
               img_HR_pil.size[0] // factor, 
               img_HR_pil.size[1] // factor
    ]

    img_LR_pil = img_HR_pil.resize(LR_size, Image.ANTIALIAS)
    img_LR_np = pil_to_np(img_LR_pil)

    print('HR and LR resolutions: %s, %s' % (str(img_HR_pil.size), str (img_LR_pil.size)))

    return {
                'orig_pil': img_orig_pil,
                'orig_np':  img_orig_np,
                'LR_pil':  img_LR_pil, 
                'LR_np': img_LR_np,
                'HR_pil':  img_HR_pil, 
                'HR_np': img_HR_np
           }
    
    
