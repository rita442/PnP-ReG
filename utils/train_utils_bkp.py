import torch
import numpy as np
import warnings


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

    # Check existence of expected parameters in the file
    expected_params = ([
        'model',
        'torch_rng',
        'numpy_rng',
        'optimizer',
        'gpu_count',
    ])
    expected_params += list_params_return
    expected_params += list(dict_params_check.keys())

    for p in expected_params:
        assert p in checkpoint_data, f'Can\'t find parameter "{p}" in checkpoint file "{filename}".'

    if checkpoint_data['gpu_count']:
        assert 'torch_rng_cuda' in checkpoint_data, f'Can\'t find parameter "torch_rng_cuda" in checkpoint file "{filename}".'

    #check conformity of parameters provided in dict_params_check with the parameters in the file.
    assert all([checkpoint_data[p] == dict_params_check[p] for p in dict_params_check]), '\n'.join(
        [f'The following parameter(s) have a different value in current code and in checkpoint file "{filename}"'] +
        [f'\'{p}\':\n\tCurrent value:\'{dict_params_check[p]}\'\n\tFile value:\'{checkpoint_data[p]}\''
         for p in dict_params_check if checkpoint_data[p] != dict_params_check[p]])
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
            self.is_last_of_epoch = True
            yield x_prev
            self.epoch += 1
            self.it_in_epoch = 0

    def __iter__(self):
        return self

    def __next__(self):
        return self.iterator.__next__()




#Functions for random noise level generation

def sigmas_uni_global(batch_size, num_channels, res_y, res_x, avg_sigma, device=torch.device('cpu')):
    sig = (torch.rand(1) * 2 * avg_sigma).to(device)
    return sig.repeat(batch_size, num_channels, res_y, res_x)


def sigmas_uni_channel(batch_size, num_channels, res_y, res_x, avg_sigma, device=torch.device('cpu')):
    sig = (torch.rand(1, num_channels, 1, 1) * 2 * avg_sigma).to(device)
    return sig.repeat(batch_size, 1, res_y, res_x)


def sigmas_uni_batch(batch_size, num_channels, res_y, res_x, avg_sigma, device=torch.device('cpu')):
    sig = (torch.rand(batch_size, 1, 1, 1) * 2 * avg_sigma).to(device)
    return sig.repeat(1, num_channels, res_y, res_x)


def sigmas_uni_batch_channel(batch_size, num_channels, res_y, res_x, avg_sigma, device=torch.device('cpu')):
    device = torch.device(device)
    if device.type=='cuda':
            sig = torch.cuda.FloatTensor(batch_size, num_channels, 1, 1).uniform_() * 2 * avg_sigma
    elif device.type=='cpu':
            sig = torch.rand(batch_size, num_channels, 1, 1) * 2 * avg_sigma
    else:
        raise RuntimeError(f'Unsupported device type: \'{device.type}\'. Supported  device types are \'cpu\' or \'cuda\'.')
    return sig.repeat(1, 1, res_y, res_x)


def sigmas_uni_spatial(batch_size, num_channels, res_y, res_x, avg_sigma, device=torch.device('cpu')):
    device = torch.device(device)
    if device.type == 'cuda':
        sig = torch.cuda.FloatTensor(1, 1, res_y, res_x).uniform_() * 2 * avg_sigma
    elif device.type == 'cpu':
        sig = torch.rand(1, 1, res_y, res_x) * 2 * avg_sigma
    else:
        raise RuntimeError(f'Unsupported device type: \'{device.type}\'. Supported  device types are \'cpu\' or \'cuda\'.')
    return sig.repeat(batch_size, num_channels, 1, 1)


def sigmas_uni_all_dims(batch_size, num_channels, res_y, res_x, avg_sigma, device=torch.device('cpu')):
    device = torch.device(device)
    if device.type == 'cuda':
        return torch.cuda.FloatTensor(batch_size, num_channels, res_y, res_x).uniform_() * 2 * avg_sigma
    elif device.type == 'cpu':
        return torch.rand(batch_size, num_channels, res_y, res_x) * 2 * avg_sigma
    else:
        raise RuntimeError(f'Unsupported device type: \'{device.type}\'. Supported  device types are \'cpu\' or \'cuda\'.')


def sigmas_uni_batch_spatial(batch_size, num_channels, res_y, res_x, avg_sigma, device=torch.device('cpu')):
    device = torch.device(device)
    if device.type == 'cuda':
        sig = torch.cuda.FloatTensor(batch_size, 1, res_y, res_x).uniform_() * 2 * avg_sigma
    elif device.type == 'cpu':
        sig = torch.rand(batch_size, 1, res_y, res_x) * 2 * avg_sigma
    else:
        raise RuntimeError(f'Unsupported device type: \'{device.type}\'. Supported  device types are \'cpu\' or \'cuda\'.')
    return sig.repeat(1, num_channels, 1, 1)


def sigmas_uni_Xbatch_spatial(batch_size, num_channels, res_y, res_x, avg_sigma, device=torch.device('cpu')):
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

