import torch
import numpy as np
import warnings

from inspect import getmembers, isfunction

from collections import MutableMapping

#Dictionary class for parameters, with possibility to add a default value
class DictParams(MutableMapping):
    __dict__ = {}
    params_def = {}
    args_dict = {}

    def __init__(self, args_dict, dict_names_with_defaults={}):
        self.args_dict = args_dict
        self.params_def = dict_names_with_defaults.copy()
        for p in self.params_def:
            if p in self.args_dict:
                self.__dict__[p] = args_dict[p]
            else:
                raise KeyError(f'Parameter name: \'{p}\' not found in the arguments list.')

    def add_param(self, name, default=None):
        if name in self.args_dict:
            self.__dict__[name] = self.args_dict[name]
            self.params_def[name] = default
        else:
            raise KeyError(f'Parameter name: \'{p}\' not found in the arguments list.')


    def __getitem__(self, key):
        return self.__dict__[key]

    def __setitem__(self, key, val):
        self.__dict__[key] = val
        if not key in self.params_def:
            self.params_def[key] = None

    def __delitem__(self, key):
        del self.__dict__[key]
        del self.params_def[key]

    def __iter__(self):
        return iter(self.__dict__)

    def __len__(self):
        return len(self.__dict__)

    def __repr__(self):
        return 'DictParam{' + ', '.join([f'\'{p}\': {self.__dict__[p]}' +
                                         ('' if self.params_def[p] is None else f' (default={self.params_def[p]})') for p in self.__dict__]) + '}'


def get_dict_params_check(args_dict, dict_names_with_defaults):
    dict_params_check={}
    for p in dict_names_with_defaults.keys():
        if p in args_dict:
            dict_params_check[p] = [dict_names_with_defaults[p], args_dict[p]]
        else:
            raise NameError(f'Parameter name: \'{p}\' not found in the arguments list.')
    return dict_params_check

def save_checkpoint(filename: str, model: torch.nn.Module, optimizer: torch.optim.Optimizer, multi_gpu=False, **kwargs):
    dto = {
        'model': model.module.state_dict() if multi_gpu else model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'torch_rng': torch.get_rng_state(),
        'numpy_rng': np.random.get_state(),
        'gpu_count': torch.cuda.device_count(),
    }

    num_gpus = torch.cuda.device_count()
    if num_gpus:
        dto['torch_rng_cuda'] = torch.cuda.get_rng_state_all()
    if multi_gpu and num_gpus < 2:
        warnings.warn('The option multi_gpu is set but less than 2 GPUs are available.')

    dto.update(**kwargs)
    torch.save(dto, filename)


def load_checkpoint(filename: str, model: torch.nn.Module, optimizer: torch.optim.Optimizer, multi_gpu=False, list_params_return=None, dict_params_check=None):
    if list_params_return is None: list_params_return = []
    if dict_params_check is None: dict_params_check = {}
    if not torch.cuda.is_available():
        checkpoint_data = torch.load(filename, map_location=torch.device('cpu'))
    else:
        checkpoint_data = torch.load(filename)
    assert type(checkpoint_data) is dict, 'Selected checkpoint file "{}" should contain a dictionary.'.format(filename)

    if hasattr(dict_params_check, 'params_def'):
        list_params_check_no_def = [p for p in dict_params_check if dict_params_check.params_def[p] is None]
        list_params_check_with_def = [p for p in dict_params_check if dict_params_check.params_def[p] is not None]
    else:
        list_params_check_no_def = list(dict_params_check.keys())
        list_params_check_with_def = []

    # Check existence of expected parameters in the file
    expected_params = ([
        'model',
        'torch_rng',
        'numpy_rng',
        'optimizer',
        'gpu_count',
    ])
    expected_params += list_params_return
    expected_params += list_params_check_no_def

    for p in expected_params:
        assert p in checkpoint_data, f'Can\'t find parameter "{p}" in checkpoint file "{filename}".'

    if checkpoint_data['gpu_count']:
        assert 'torch_rng_cuda' in checkpoint_data, f'Can\'t find parameter "torch_rng_cuda" in checkpoint file "{filename}".'

    #check conformity of parameters provided in dict_params_check with the parameters in the file.
    checked_no_default = {p:checkpoint_data[p] == dict_params_check[p] for p in list_params_check_no_def}
    checked_with_default = {p: (p in checkpoint_data and checkpoint_data[p] == dict_params_check[p]) or
                               (not p in checkpoint_data and dict_params_check.params_def[p] == dict_params_check[p])
                            for p in list_params_check_with_def}
    assert all([*checked_no_default.values(), *checked_with_default.values()]), '\n'.join(
        [f'The following parameter(s) have a different value in current code and in checkpoint file "{filename}"'] +
        [f'\'{p}\':\n\tCurrent value: \'{dict_params_check[p]}\'\n\tFile value: \'{checkpoint_data[p]}\''
         for p in checked_no_default if not checked_no_default[p]] +
        [f'\'{p}\':\n\tCurrent value: \'{dict_params_check[p]}\'\n\tFile value undefined, using default: \'{dict_params_check.params_def[p]}\''
         for p in checked_with_default if not checked_with_default[p]]
    )
#    for p in dict_params_check:
#        assert checkpoint_data[p] == dict_params_check[p], \
#            f'The parameter "{p}" has a different value in current code and in checkpoint file "{filename}"\nCurrent value: {dict_params_check[p]}\nCheckpoint file value: {checkpoint_data[p]}.'

    #Set states of model, optimizer and random number generators
    if multi_gpu:
        model.module.load_state_dict(checkpoint_data['model'])
    else:
        model.load_state_dict(checkpoint_data['model'])
    optimizer.load_state_dict(checkpoint_data['optimizer'])
    np.random.set_state(checkpoint_data['numpy_rng'])
    torch.set_rng_state(checkpoint_data['torch_rng'])

    num_gpus_current = torch.cuda.device_count()
    num_gpus_checkpoint = checkpoint_data['gpu_count']
    num_common_gpus = min(num_gpus_current, num_gpus_checkpoint)
    for cuda_id in range(num_common_gpus):
        torch.cuda.set_rng_state(checkpoint_data['torch_rng_cuda'][cuda_id], cuda_id)

    if num_gpus_current != num_gpus_checkpoint:
        warnings.warn('Checkpoint was saved from an environment with a different number of GPU devices.\nThis could '
                      'affect reproducibility of the results if the number of GPU devices actually used during '
                      'computations are different.')
    if multi_gpu and num_gpus_current < 2:
        warnings.warn('The option multi_gpu is set but less than 2 GPUs are available.')

    # Return optimizer and other remaining parameters with names specified in list_params_return
    dict_return = {p: checkpoint_data[p] for p in list_params_return}
    return dict_return


def save_data_row(filename, index, **kwargs):
    import pandas as pd
    names_list = list(kwargs.keys())
    values_list = list(kwargs.values())
    data_row = pd.DataFrame([values_list], index=[index], columns=names_list)
    try:
        d = pd.read_csv(filename, sep=';', index_col=0)
        if d.index[-1] >= index:
            raise ValueError(f'Trying to append a row at an index {index} to a file with higher or equal last row index {d.index[-1]}.')
        pd.concat([d, data_row]).to_csv(filename, sep=';')
    except (pd.errors.EmptyDataError, FileNotFoundError):
        data_row.to_csv(filename, sep=';')
#    except (ValueError, TypeError, PermissionError):
    except:
        raise



#def cycle(iterable):
#    epoch=0
#    while True:
#        epoch += 1
#        for it_in_epoch, x in enumerate(iterable):
#            yield epoch, it_in_epoch, x

# Iterator structure for iterating through and within epochs
class CycleEpochIterator:
    def __init__(self, iterable, epoch=1):
        self.epoch = epoch
        self.iterator = iter(self.__cycle(iterable))
        self.it_in_epoch = 0
        self.is_last_of_epoch = False
        #if(it_in_epoch > 1):
        #    for i in range(it_in_epoch-1):
        #        self.iterator.__next__()
        #else:
        #    self.it_in_epoch=1

    def __cycle(self, iterable):
        while True:
            prev_iterator = iter(iterable)
            x_prev = next(prev_iterator)
            self.is_last_of_epoch = False
            for self.it_in_epoch, x in enumerate(prev_iterator, 1):
                yield x_prev
                x_prev = x
            self.it_in_epoch += 1
            self.is_last_of_epoch = True
            yield x_prev
            self.epoch += 1

    def __iter__(self):
        return self

    def __next__(self):
        return self.iterator.__next__()



####################################################################
#Functions for noise generation
####################################################################
def gen_gaussian_noise(*output_size, device=torch.device('cpu')):
    device = torch.device(device)
    if device.type == 'cuda':
        return torch.cuda.FloatTensor(*output_size).normal_()
    elif device.type == 'cpu':
        return torch.randn(*output_size)
    else:
        raise RuntimeError(
            f'Unsupported device type: \'{device.type}\'. Supported  device types are \'cpu\' or \'cuda\'.')

def gen_uniform_noise(*output_size, device=torch.device('cpu')):
    device = torch.device(device)
    if device.type == 'cuda':
        return torch.cuda.FloatTensor(*output_size).uniform_()
    elif device.type == 'cpu':
        return torch.rand(*output_size)
    else:
        raise RuntimeError(
            f'Unsupported device type: \'{device.type}\'. Supported  device types are \'cpu\' or \'cuda\'.')

####################################################################
#Functions for random noise level generation
####################################################################
def sigmas_uni_global(batch_size, num_channels, res_y, res_x, avg_sigma, device=torch.device('cpu')):
    sig = gen_uniform_noise(1, device=device) * 2 * avg_sigma
    return sig.repeat(batch_size, num_channels, res_y, res_x)

def sigmas_uni_channel(batch_size, num_channels, res_y, res_x, avg_sigma, device=torch.device('cpu')):
    sig = gen_uniform_noise(1, num_channels, 1, 1, device=device) * 2 * avg_sigma
    return sig.repeat(batch_size, 1, res_y, res_x)

def sigmas_uni_batch(batch_size, num_channels, res_y, res_x, avg_sigma, device=torch.device('cpu')):
    sig = gen_uniform_noise(batch_size, 1, 1, 1, device=device) * 2 * avg_sigma
    return sig.repeat(1, num_channels, res_y, res_x)

def sigmas_uni_batch_channel(batch_size, num_channels, res_y, res_x, avg_sigma, device=torch.device('cpu')):
    sig = gen_uniform_noise(batch_size, num_channels, 1, 1, device=device) * 2 * avg_sigma
    return sig.repeat(1, 1, res_y, res_x)

def sigmas_uni_spatial(batch_size, num_channels, res_y, res_x, avg_sigma, device=torch.device('cpu')):
    sig = gen_uniform_noise(1, 1, res_y, res_x, device=device) * 2 * avg_sigma
    return sig.repeat(batch_size, num_channels, 1, 1)

def sigmas_uni_all_dims(batch_size, num_channels, res_y, res_x, avg_sigma, device=torch.device('cpu')):
    return gen_uniform_noise(batch_size, num_channels, res_y, res_x, device=device) * 2 * avg_sigma

def sigmas_uni_batch_spatial(batch_size, num_channels, res_y, res_x, avg_sigma, device=torch.device('cpu')):
    sig = gen_uniform_noise(batch_size, 1, res_y, res_x, device=device) * 2 * avg_sigma
    return sig.repeat(1, num_channels, 1, 1)

def sigmas_uni_Xbatch_spatial(batch_size, num_channels, res_y, res_x, avg_sigma, device=torch.device('cpu')):
    sig = gen_uniform_noise(batch_size, 1, res_y, res_x, device=device) * \
          gen_uniform_noise(batch_size, 1, 1, 1, device=device) * 4 * avg_sigma #varying noise level pattern per image
#    sig = gen_uniform_noise(1, 1, res_y, res_x, device=device) * \
#          gen_uniform_noise(batch_size, 1, 1, 1, device=device) * 4 * avg_sigma  #same noise level pattern for all images
    return sig.repeat(1, num_channels, 1, 1)

def sigmas_uni_XObatch_spatial(batch_size, num_channels, res_y, res_x, avg_sigma, device=torch.device('cpu')):
    weight = gen_uniform_noise(batch_size, 1, 1, 1, device=device)
    sig = 2 * avg_sigma * (gen_uniform_noise(batch_size, 1, res_y, res_x, device=device) * (1-weight) +
                           gen_uniform_noise(batch_size, 1, 1, 1, device=device) * weight )
    return sig.repeat(1, num_channels, 1, 1)


'''
def sigmas_uni_global_(batch_size, num_channels, res_y, res_x, avg_sigma, device=torch.device('cpu')):
    sig = (torch.rand(1) * 2 * avg_sigma).to(device)
    return sig.repeat(batch_size, num_channels, res_y, res_x)


def sigmas_uni_channel_(batch_size, num_channels, res_y, res_x, avg_sigma, device=torch.device('cpu')):
    sig = (torch.rand(1, num_channels, 1, 1) * 2 * avg_sigma).to(device)
    return sig.repeat(batch_size, 1, res_y, res_x)


def sigmas_uni_batch_(batch_size, num_channels, res_y, res_x, avg_sigma, device=torch.device('cpu')):
    sig = (torch.rand(batch_size, 1, 1, 1) * 2 * avg_sigma).to(device)
    return sig.repeat(1, num_channels, res_y, res_x)


def sigmas_uni_batch_channel_(batch_size, num_channels, res_y, res_x, avg_sigma, device=torch.device('cpu')):
    device = torch.device(device)
    if device.type=='cuda':
            sig = torch.cuda.FloatTensor(batch_size, num_channels, 1, 1).uniform_() * 2 * avg_sigma
    elif device.type=='cpu':
            sig = torch.rand(batch_size, num_channels, 1, 1) * 2 * avg_sigma
    else:
        raise RuntimeError(f'Unsupported device type: \'{device.type}\'. Supported  device types are \'cpu\' or \'cuda\'.')
    return sig.repeat(1, 1, res_y, res_x)


def sigmas_uni_spatial_(batch_size, num_channels, res_y, res_x, avg_sigma, device=torch.device('cpu')):
    device = torch.device(device)
    if device.type == 'cuda':
        sig = torch.cuda.FloatTensor(1, 1, res_y, res_x).uniform_() * 2 * avg_sigma
    elif device.type == 'cpu':
        sig = torch.rand(1, 1, res_y, res_x) * 2 * avg_sigma
    else:
        raise RuntimeError(f'Unsupported device type: \'{device.type}\'. Supported  device types are \'cpu\' or \'cuda\'.')
    return sig.repeat(batch_size, num_channels, 1, 1)


def sigmas_uni_all_dims_(batch_size, num_channels, res_y, res_x, avg_sigma, device=torch.device('cpu')):
    device = torch.device(device)
    if device.type == 'cuda':
        return torch.cuda.FloatTensor(batch_size, num_channels, res_y, res_x).uniform_() * 2 * avg_sigma
    elif device.type == 'cpu':
        return torch.rand(batch_size, num_channels, res_y, res_x) * 2 * avg_sigma
    else:
        raise RuntimeError(f'Unsupported device type: \'{device.type}\'. Supported  device types are \'cpu\' or \'cuda\'.')


def sigmas_uni_batch_spatial_(batch_size, num_channels, res_y, res_x, avg_sigma, device=torch.device('cpu')):
    device = torch.device(device)
    if device.type == 'cuda':
        sig = torch.cuda.FloatTensor(batch_size, 1, res_y, res_x).uniform_() * 2 * avg_sigma
    elif device.type == 'cpu':
        sig = torch.rand(batch_size, 1, res_y, res_x) * 2 * avg_sigma
    else:
        raise RuntimeError(f'Unsupported device type: \'{device.type}\'. Supported  device types are \'cpu\' or \'cuda\'.')
    return sig.repeat(1, num_channels, 1, 1)


def sigmas_uni_Xbatch_spatial_(batch_size, num_channels, res_y, res_x, avg_sigma, device=torch.device('cpu')):
    device = torch.device(device)
    if device.type == 'cuda':
        sig = torch.cuda.FloatTensor(batch_size, 1, res_y, res_x).uniform_() * \
              torch.cuda.FloatTensor(batch_size, 1, 1, 1).uniform_() * 4 * avg_sigma
    elif device.type == 'cpu':
        sig = torch.rand(batch_size, 1, res_y, res_x) * torch.rand(batch_size, 1, 1, 1) * 4 * avg_sigma
    else:
        raise RuntimeError(f'Unsupported device type: \'{device.type}\'. Supported  device types are \'cpu\' or \'cuda\'.')

    #sig = torch.rand(batch_size, 1, res_y, res_x) * torch.rand(batch_size, 1, 1, 1) * 4 * avg_sigma   #varying noise level pattern per image
    #sig = torch.rand(batch_size, 1, 1, 1) * torch.rand(1, 1, res_y, res_x) * 4 * avg_sigma           #same noise level pattern for all images
    return sig.repeat(1, num_channels, 1, 1)
'''
