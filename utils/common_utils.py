import torch
import torch.nn as nn
import torchvision
import sys

import numpy as np
from PIL import Image
import PIL
import numpy as np
from utils import utils_image as util
import os
import matplotlib.pyplot as plt
from utils.format_utils import is_iterable, to_tensor
import math 
from typing import TYPE_CHECKING, Tuple, Union, List, Optional, Sequence
if TYPE_CHECKING:
    from torch import Tensor

def draw_graph(path,graph_type,img_p,LR,sigma,best_psrn,best_iter,values_list):
    path_="%s/%s_%s_LR%.3f_%.3f_%.3f_%d.png"%(path,graph_type,img_p,LR,sigma,best_psrn,best_iter)
    plt.clf()
    plt.plot(range(len(values_list)), values_list)
    plt.savefig(path_)
    return
def write_log(path,sigma,LR,values_list,round_digits):
    v=open (path, "a")
    v.write("ss ")
    v.write(str(sigma))
    v.write("   LR ")
    v.write(str(LR))
    v.write("   ")
    for i in range(0,len(values_list)):
            a=round(values_list[i],round_digits)
            v.write(str(a))
            v.write("  ")
    avg=float(np.mean(values_list))
    avg=round(avg,2)
    v.write(str(avg))
    v.write("\n")
    return

def write_log_checkpoints(path,checkpoint,sigma,LR,values_list,round_digits):
    v=open (path, "a")
    v.write("ckp ")
    v.write(str(checkpoint))
    v.write(" ss ")
    v.write(str(sigma))
    v.write("   LR ")
    v.write(str(LR))
    v.write("   ")
    for i in range(0,len(values_list)):
            a=round(values_list[i],round_digits)
            v.write(str(a))
            v.write("  ")
    avg=float(np.mean(values_list))
    avg=round(avg,2)
    v.write(str(avg))
    v.write("\n")
    return
def crop_image(img, d=32):
    '''Make dimensions divisible by `d`'''

    new_size = (img.size[0] - img.size[0] % d, 
                img.size[1] - img.size[1] % d)

    bbox = [
            int((img.size[0] - new_size[0])/2), 
            int((img.size[1] - new_size[1])/2),
            int((img.size[0] + new_size[0])/2),
            int((img.size[1] + new_size[1])/2),
    ]

    img_cropped = img.crop(bbox)
    return img_cropped

def get_image_names(testset_name):
    L_path= "testsets/%s"%testset_name # L_path, for Low-quality images
    L_paths = util.get_image_paths(L_path)
    img_names=[]
    for idx, img in enumerate(L_paths):
        img_name, ext = os.path.splitext(os.path.basename(img))
        img_names.append(img_name)
    return img_names,ext
        
    
def get_params(opt_over, net, net_input, downsampler=None):
    '''Returns parameters that we want to optimize over.

    Args:
        opt_over: comma separated list, e.g. "net,input" or "net"
        net: network
        net_input: torch.Tensor that stores input `z`
    '''
    opt_over_list = opt_over.split(',')
    params = []
    
    for opt in opt_over_list:
    
        if opt == 'net':
            params += [x for x in net.parameters() ]
        elif  opt=='down':
            assert downsampler is not None
            params = [x for x in downsampler.parameters()]
        elif opt == 'input':
            net_input.requires_grad = True
            params += [net_input]
        else:
            assert False, 'what is it?'
            
    return params
def adjust_img_channels(img):
    if img.shape[0]!=3:
        img_np_temp = np.zeros((3,img.shape[1], img.shape[2]))
        img_np_temp[0,:,:] = img[0,:,:]  
        img_np_temp[1,:,:] = img[0,:,:]
        img_np_temp[2,:,:] = img[0,:,:]
        img = img_np_temp.astype(np.float32)
    return img 
def get_image_grid(images_np, nrow=8):
    '''Creates a grid from a list of images by concatenating them.'''
    images_torch = [torch.from_numpy(x) for x in images_np]
    torch_grid = torchvision.utils.make_grid(images_torch, nrow)
    
    return torch_grid.numpy()

def plot_image_grid(images_np, nrow =8, factor=1, interpolation='lanczos'):
    """Draws images in a grid
    
    Args:
        images_np: list of images, each image is np.array of size 3xHxW of 1xHxW
        nrow: how many images will be in one row
        factor: size if the plt.figure 
        interpolation: interpolation used in plt.imshow
    """
    n_channels = max(x.shape[0] for x in images_np)
    assert (n_channels == 3) or (n_channels == 1), "images should have 1 or 3 channels"
    
    images_np = [x if (x.shape[0] == n_channels) else np.concatenate([x, x, x], axis=0) for x in images_np]

    grid = get_image_grid(images_np, nrow)
    
    plt.figure(figsize=(len(images_np) + factor, 12 + factor))
    
    if images_np[0].shape[0] == 1:
        plt.imshow(grid[0], cmap='gray', interpolation=interpolation)
    else:
        plt.imshow(grid.transpose(1, 2, 0), interpolation=interpolation)
    
    plt.show()
    
    return grid

def load(path):
    """Load PIL image."""
    img = Image.open(path)
    return img

def get_image(path, imsize=-1):
    """Load an image and resize to a cpecific size. 

    Args: 
        path: path to image
        imsize: tuple or scalar with dimensions; -1 for `no resize`
    """
    img = load(path)

    if isinstance(imsize, int):
        imsize = (imsize, imsize)

    if imsize[0]!= -1 and img.size != imsize:
        if imsize[0] > img.size[0]:
            img = img.resize(imsize, Image.BICUBIC)
        else:
            img = img.resize(imsize, Image.ANTIALIAS)

    img_np = pil_to_np(img)

    return img, img_np



def fill_noise(x, noise_type):
    """Fills tensor `x` with noise of type `noise_type`."""
    if noise_type == 'u':
        x.uniform_()
    elif noise_type == 'n':
        x.normal_() 
    else:
        assert False

def get_noise(input_depth, method, spatial_size, noise_type='u', var=1./10):
    """Returns a pytorch.Tensor of size (1 x `input_depth` x `spatial_size[0]` x `spatial_size[1]`) 
    initialized in a specific way.
    Args:
        input_depth: number of channels in the tensor
        method: `noise` for fillting tensor with noise; `meshgrid` for np.meshgrid
        spatial_size: spatial size of the tensor to initialize
        noise_type: 'u' for uniform; 'n' for normal
        var: a factor, a noise will be multiplicated by. Basically it is standard deviation scaler. 
    """
    if isinstance(spatial_size, int):
        spatial_size = (spatial_size, spatial_size)
    if method == 'noise':
        shape = [1, input_depth, spatial_size[0], spatial_size[1]]
        net_input = torch.zeros(shape)
        
        fill_noise(net_input, noise_type)
        net_input *= var            
    elif method == 'meshgrid': 
        assert input_depth == 2
        X, Y = np.meshgrid(np.arange(0, spatial_size[1])/float(spatial_size[1]-1), np.arange(0, spatial_size[0])/float(spatial_size[0]-1))
        meshgrid = np.concatenate([X[None,:], Y[None,:]])
        net_input=  np_to_torch(meshgrid)
    else:
        assert False
        
    return net_input

def pil_to_np(img_PIL):
    '''Converts image in PIL format to np.array.
    
    From W x H x C [0...255] to C x W x H [0..1]
    '''
    ar = np.array(img_PIL)

    if len(ar.shape) == 3:
        ar = ar.transpose(2,0,1)
    else:
        ar = ar[None, ...]

    return ar.astype(np.float32) / 255.

def np_to_pil(img_np): 
    '''Converts image in np.array format to PIL image.
    
    From C x W x H [0..1] to  W x H x C [0...255]
    '''
    ar = np.clip(img_np*255,0,255).astype(np.uint8)
    
    if img_np.shape[0] == 1:
        ar = ar[0]
    else:
        ar = ar.transpose(1, 2, 0)

    return Image.fromarray(ar)

def np_to_torch(img_np):
    '''Converts image in numpy.array to torch.Tensor.

    From C x W x H [0..1] to  C x W x H [0..1]
    '''
    return torch.from_numpy(img_np)[None, :]

def torch_to_np(img_var):
    '''Converts an image in torch.Tensor format to np.array.

    From 1 x C x W x H [0..1] to  C x W x H [0..1]
    '''
    return img_var.detach().cpu().numpy()[0]


def optimize(optimizer_type, parameters, closure, LR, num_iter):
    """Runs optimization loop.

    Args:
        optimizer_type: 'LBFGS' of 'adam'
        parameters: list of Tensors to optimize over
        closure: function, that returns loss variable
        LR: learning rate
        num_iter: number of iterations 
    """
    if optimizer_type == 'LBFGS':
        # Do several steps with adam first
        optimizer = torch.optim.Adam(parameters, lr=0.001)
        for j in range(100):
            optimizer.zero_grad()
            closure()
            optimizer.step()

        print('Starting optimization with LBFGS')        
        def closure2():
            optimizer.zero_grad()
            return closure()
        optimizer = torch.optim.LBFGS(parameters, max_iter=num_iter, lr=LR, tolerance_grad=-1, tolerance_change=-1)
        optimizer.step(closure2)

    elif optimizer_type == 'adam':
        print('Starting optimization with ADAM')
        optimizer = torch.optim.Adam(parameters, lr=LR)
        
        for j in range(num_iter):
            optimizer.zero_grad()
            closure()
            optimizer.step()
    else:
        assert False
        
def pad_lf(lf: torch.Tensor, pad: Union[int, Sequence[int]], mode: str = 'reflect', use_window: bool = True):
    """
    Pads the Light Field tensor in spatial dimensions with number of pixels defined in 'pad'.
    :param lf: Light Field tensor with dimensions (batch x channels x V x U x Y x X).
    :param pad: int or tuple of ints, number of pixels to pad on each side: (left, right, top, bottom).
    :param mode: padding mode (from pytorch padding function)
    :param use_window: set to True to apply Hann windowing to the padded borders.
    """
    if not is_iterable(pad):
        pad = (pad, pad, pad, pad)
    lf_shape = lf.shape[:-2] + (lf.shape[-2] + pad[2] + pad[3], lf.shape[-1] + pad[0] + pad[1])
    lf = torch.nn.functional.pad(lf.view(-1, lf.shape[-3], lf.shape[-2], lf.shape[-1]), pad, mode=mode).view(lf_shape)
    if use_window:
        lf *= hann_border(lf.shape[-1], lf.shape[-2], pad, device=lf.device)
    return lf


def hann_border(res_x, res_y, borders, device=None):
    """
    Computes Hann window only considering border pixels (for windowing of padded borders only).
    :param res_x: full horizontal image resolution (including borders).
    :param res_y: full vertical image resolution (including borders).
    :param borders: tuple of ints (left, right, top, bottom): number of border pixels on the borders.
    """
    def window(x):
        return .5 - .5 * torch.cos(math.pi * x)  # Hann window (for x in [0,2]).

    inside_x = res_x - borders[0] - borders[1]
    inside_y = res_y - borders[2] - borders[3]
    assert inside_x >= 0 and inside_y >= 0, 'Borders are larger than the image size.'

    borders_st = [torch.arange(borders[i], device=device) / max(1, borders[i]) for i in (0, 2)]
    borders_end = [torch.arange(1 + borders[i], 1 + 2 * borders[i], device=device) / max(1, borders[i]) for i in (1, 3)]

    borders_x = torch.cat((window(borders_st[0]), torch.ones(inside_x, device=device), window(borders_end[0])))
    borders_y = torch.cat((window(borders_st[1]), torch.ones(inside_y, device=device), window(borders_end[1])))

    return borders_y.unsqueeze(1) * borders_x.unsqueeze(0)


def compare_psnr(img1, img2):
    mse = np.mean( (img1 - img2) ** 2 )
    if mse == 0:
      return 100
    PIXEL_MAX = 1.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


