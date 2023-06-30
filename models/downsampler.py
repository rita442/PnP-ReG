import numpy as np
import torch
import torch.nn as nn
import utils.downsampling_utils as dwn

class Downsampler(nn.Module):
    #pad_mode can be either: 'constant', 'reflect', 'replicate' or 'circular' (default=reflect)
    def __init__(self, factor, kernel_type, gauss_sigma=None, antialias=True, pad_mode='reflect', freeze_params=True):
        super(Downsampler, self).__init__()

        filter_param = None

        if kernel_type == 'lanczos2':
            filter = dwn.lanczos2_filter

        elif kernel_type == 'lanczos3':
            filter = dwn.lanczos3_filter

        elif kernel_type == 'bicubic':
            filter = dwn.cubic_filter_a75

        elif kernel_type == 'bicubic_matlab':
            filter = dwn.cubic_filter_a5

        elif kernel_type == 'bilinear':
            filter = dwn.linear_filter

        elif kernel_type == 'gauss':
            filter = dwn.gauss4s_filter
            filter_param = gauss_sigma

        else:
            raise Exception('Unknown kernel type: \'' + str(kernel_type) + '\'')

        #Create discrete cross-correlation kernel for the given filter and integer  downsampling factor.
        self.pad_mode = pad_mode
        self.factor = int(factor)
        self.kernel, center = dwn.gen_dwn_kernel2D(self.factor, filter, filter_param=filter_param, antialias=antialias, cross_correlation=True)
        self.downsampler_ = nn.Conv2d(1, 1, kernel_size=self.kernel.shape, stride=self.factor, padding=0)

        if freeze_params:
            self.downsampler_.weight.requires_grad = False
            self.downsampler_.bias.requires_grad = False

        self.downsampler_.weight[0,0] = self.kernel
        self.downsampler_.bias[:] = 0

        #Prepare padding size with respect to kernel size and center position.
        pad_left_top = center
        pad_right_bottom = self.kernel.shape[0] - center - 1
        self.pad = (pad_left_top,pad_right_bottom,pad_left_top,pad_right_bottom)
        self.pad_flip = (pad_right_bottom,pad_left_top,pad_right_bottom,pad_left_top)
        #print(self.pad_flip )
    #Adjoint operator:
    # Warning ->  This function corresponds exactly to the adjoint operator of the forward downsampling only if the initial
    # image size is a multiple of the downsampling factor and if the padding type is 'circular'.
    #   -For other padding types, there may be differences with the true adjoint on image borders.
    #   -If the initial image (before downsampling) size was not a multiple of the downsampling factor,
    #   re-upsampling with this function won't recover exactly the initial image size.
    #Note: an image downsampled with forward, and re-upsampled with adjoint will appear darker than the original image.
    #This is an intended property of the adjoint operator. To correct the brightness, the result must be multiplied by factor**2.
    def adjoint(self, input):
        x = torch.zeros(input.shape[0],input.shape[1],input.shape[2]*self.factor,input.shape[3]*self.factor)
        x[...,0::self.factor,0::self.factor] = input

        x = nn.functional.pad(x, self.pad_flip, mode=self.pad_mode)
        x_shape = x.shape
        k = self.kernel.flip(0, 1)
        k = k.view(1,1,k.shape[0],k.shape[1])
        x = nn.functional.conv2d(x.view(x_shape[0]*x_shape[1], 1, x_shape[2], x_shape[3]), k)
        return x.view(x_shape[0:2] + x.shape[2:])


    def forward(self, input):
        #downsampled imge size is rounded down if the input size is not a multiple of the downsampling factor.
        new_size = [input.shape[-2] // self.factor, input.shape[-1] // self.factor]

        x = nn.functional.pad(input, self.pad, mode=self.pad_mode)
        x_shape = x.shape

        # Apply convolution. Channels are transferred to batch dimension to be processed independently.
        x = self.downsampler_(x.view(x_shape[0]*x_shape[1], 1, x_shape[2], x_shape[3]))

        # Replace channels to 2nd dimension and crops 1 pixel if needed to round the spatial size at the smallest integer.
        return x.view(x_shape[0:2] + x.shape[2:])[:, :, :new_size[0], :new_size[1]]
