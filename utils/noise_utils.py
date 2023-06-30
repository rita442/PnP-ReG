import torch

####################################################################
#Functions for noise generation
####################################################################
def gen_gaussian_noise(*output_size, device=None):
    device = torch.device(device) if device is not None else torch.device('cpu')
    if device.type == 'cuda':
        return torch.cuda.FloatTensor(*output_size).normal_()
    elif device.type == 'cpu':
        return torch.randn(*output_size)
    else:
        raise RuntimeError(
            f'Unsupported device type: \'{device.type}\'. Supported device types are \'cpu\' or \'cuda\'.')


def gen_uniform_noise(*output_size, device=None):
    device = torch.device(device) if device is not None else torch.device('cpu')
    if device.type == 'cuda':
        return torch.cuda.FloatTensor(*output_size).uniform_()
    elif device.type == 'cpu':
        return torch.rand(*output_size)
    else:
        raise RuntimeError(
            f'Unsupported device type: \'{device.type}\'. Supported device types are \'cpu\' or \'cuda\'.')


####################################################################
# Function for signal and noise scaling, (from 1 down to 0 for the given threshold)
####################################################################
def infinite_noise_scale(x, min_threshold=.5, max_threshold=1, power=1):
    return torch.clamp(1 - (torch.clamp((x-min_threshold)/(max_threshold-min_threshold), 0))**power, 0)


####################################################################
# Functions for random noise level generation
####################################################################
def sigmas_uni_global(batch_size, num_channels, res_y, res_x, avg_sigma, device=None):
    sig = gen_uniform_noise(1, device=device) * 2 * avg_sigma
    return sig.repeat(batch_size, num_channels, res_y, res_x)


def sigmas_uni_channel(batch_size, num_channels, res_y, res_x, avg_sigma, device=None):
    sig = gen_uniform_noise(1, num_channels, 1, 1, device=device) * 2 * avg_sigma
    return sig.repeat(batch_size, 1, res_y, res_x)


def sigmas_uni_batch(batch_size, num_channels, res_y, res_x, avg_sigma, device=None):
    sig = gen_uniform_noise(batch_size, 1, 1, 1, device=device) * 2 * avg_sigma
    return sig.repeat(1, num_channels, res_y, res_x)


def sigmas_uni_batch_channel(batch_size, num_channels, res_y, res_x, avg_sigma, device=None):
    sig = gen_uniform_noise(batch_size, num_channels, 1, 1, device=device) * 2 * avg_sigma
    return sig.repeat(1, 1, res_y, res_x)


def sigmas_uni_spatial(batch_size, num_channels, res_y, res_x, avg_sigma, device=None):
    sig = gen_uniform_noise(1, 1, res_y, res_x, device=device) * 2 * avg_sigma
    return sig.repeat(batch_size, num_channels, 1, 1)


def sigmas_uni_all_dims(batch_size, num_channels, res_y, res_x, avg_sigma, device=None):
    return gen_uniform_noise(batch_size, num_channels, res_y, res_x, device=device) * 2 * avg_sigma


def sigmas_uni_batch_spatial(batch_size, num_channels, res_y, res_x, avg_sigma, device=None):
    sig = gen_uniform_noise(batch_size, 1, res_y, res_x, device=device) * 2 * avg_sigma
    return sig.repeat(1, num_channels, 1, 1)


def sigmas_uni_Xbatch_spatial(batch_size, num_channels, res_y, res_x, avg_sigma, device=None):
    sig = gen_uniform_noise(batch_size, 1, res_y, res_x, device=device) * \
          gen_uniform_noise(batch_size, 1, 1, 1, device=device) * 4 * avg_sigma  # varying noise level pattern per image
    #    sig = gen_uniform_noise(1, 1, res_y, res_x, device=device) * \
    #          gen_uniform_noise(batch_size, 1, 1, 1, device=device) * 4 * avg_sigma  #same noise level pattern for all images
    return sig.repeat(1, num_channels, 1, 1)


def sigmas_uni_XObatch_spatial(batch_size, num_channels, res_y, res_x, avg_sigma, device=None):
    weight = gen_uniform_noise(batch_size, 1, 1, 1, device=device)
    sig = 2 * avg_sigma * (gen_uniform_noise(batch_size, 1, res_y, res_x, device=device) * (1 - weight) +
                           gen_uniform_noise(batch_size, 1, 1, 1, device=device) * weight)
    return sig.repeat(1, num_channels, 1, 1)