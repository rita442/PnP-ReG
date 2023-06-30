import os
import argparse
import sys
from utils.argparse_utils import *
from datetime import datetime
import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from models.network_unet import UNetRes as net

from utils import utils_model
from utils.train_utils import *
from utils.noise_utils import *
from utils.utils_bnorm import add_bn
from utils.utils_bnorm import merge_bn
from ImagePathDataset import ImagePathDataset

def main():
    now = datetime.now()
    
    datasetDir = '/nfs/stk-sirocco-clim.irisa.fr/data/teams/sirocco_clim/sirocco_clim_image/Data_Mikael/datasets'
    trainDatasetFile = 'dataset_8694_2D.csv'
    validDatasetFile = 'dataset_100_2D.csv'
    
    outputPath = './trained_results/regularizer' 
    validResultFile = os.path.join(outputPath, f'validation.csv')

    #############################################       Argument Parser       ##############################################

    class CustomFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawTextHelpFormatter):
        pass

    parser = argparse.ArgumentParser(description='Training of image denoising network.',
                                     formatter_class=CustomFormatter)

    ######################################       User configuration parameters       #######################################

    parser.add_argument('-sv', '--skip_valid', action='store_true',
                        help='Skip validation step (a validation file is still generated with only training loss data).')
    parser.add_argument('-st', '--skip_train', action='store_true',
                        help='Skip training step (for debugging purpose only).')
    parser.add_argument('-oc', '--overwrite_checkpoints', action='store_true',
                        help='Keep only latest checkpoint (overwrites previous checkpoint).\nIf False all checkpoints are kept.')
    parser.add_argument('-nec', '--num_epoch_checkpoint', type=check_nneg_int_zinf, default=1,
                        help='Save a checkpoint every \'num_epoch_checkpoint\' epoch.\nUse 0 to avoid checkpoints.')
    parser.add_argument('-rc', '--restart_checkpoint', type=check_isfile,
                        help='Restart training from a specified checkpoint file.')
    parser.add_argument('-vnif', '--valid_num_iter_fact', type=check_nneg_numeric, default=1.3,
                        help='Compute validation whenever the number of iterations is of the form ceil(valid_num_iter_fact**n).\nIn addition validation is performed at every checkpoint.')
    parser.add_argument('-vmni', '--valid_max_num_imgs', type=check_nneg_int, default=16,
                        help='Maximum number of images used from the validation dataset for validation.')

    ###########################################       Training parameters       ############################################

    parser.add_argument('-lri', '--learning_rate_init', type=check_pos_numeric, default=1e-4,
                        help='Initial learning rate.')
    parser.add_argument('-lrs', '--learning_rate_stop', type=check_nneg_numeric, default=5e-7,
                        help='Stop training when the learning rate becomes lower than this value.\nUse 0 to keep training indefinitely.')
    parser.add_argument('-nilrd', '--num_iter_lr_decay', type=check_nneg_int_zinf, default=100000,
                        help='Number of iterations after which the learning rate is divided by 2.\nUse 0 to never change learning rate.')
    parser.add_argument('-bs', '--batch_size', type=check_pos_int, default=16,
                        help='Batch size for training.')
    parser.add_argument('-cs', '--crop_size', type=check_pos_int, default=128,
                        help='Horizontal and vertical resolution of input data (cropped from images of the dataset).')
    parser.add_argument('-as', '--avg_sigma', type=check_nneg_numeric, default=25/255,
                        help='Average st.dev. of the noise for input noisy data (assumes data values in [0,1]).')
    parser.add_argument('-gsm', '--gen_sigma_method', type=check_gen_sigma_method, default='sigmas_uni_batch',
                        help='Method of noise level map generation.\nSee sigmas_XXX methods in train_utils.py.\nDefault method randomly selects a constant sigma for each image in the input batch.')
    parser.add_argument('-nie', '--num_iter_epoch', type=check_pos_int, default=0,
                        help='Number of iterations per epoch.\nUse 0 to let this parameter be determined automatically with the size of the dataset.'
                             '\nNote1: If this parameter is set lower than the number of possible batches in the datatset, all the data should still be sampled over several epochs thanks to shuffling.'
                             '\nNote2: If this parameter is set higher than the number of possible batches in the datatset, each epoch will cover some (randomly selected) images several times.')
    parser.add_argument('-lnw', '--loader_num_workers', type=check_nneg_int, default=0,
                        help='\'num_workers\' parameter of the training dataloader.')
    parser.add_argument('-im', '--init_model', type=check_isfile,
                        help='Model file to start training from (initialisation).'
                             '\r\n(different from from restart_checkpoint parameter which also requires the full processing states (e.g. optimizer, random generators)).'
                             '\nNote1: this parameter is ignored if restart_checkpoint is also specified.'
                             '\nNote2: if a checkpoint file is given as init_model file, the model is read from the checkpoint file.')

    parser.add_argument('-ins', '--infinite_noise_scaling', action='store_true',
                        help='Scale noisy input signal down to zero for a given noise level considered as infinite noise level.')
    parser.add_argument('-inst1', '--infinite_noise_scaling_t1', type=check_pos_numeric, default=.2,
                        help='First noise level threshold before which signal is not scaled.')
    parser.add_argument('-inst2', '--infinite_noise_scaling_t2', type=check_pos_numeric, default=1,
                        help='Second noise level threshold considered as infinite noise level (at this level and above, noisy input signal is set to zero).')
    parser.add_argument('-insp', '--infinite_noise_scaling_p', type=check_pos_numeric, default=1,
                        help='Parameter controling the transition between the two noise level thresholds for infinite noise scaling (e.g. 1 for linear transition).')

    ###########################################       Legacy parameters       ############################################

    parser.add_argument('-gscpu', '--gen_noise_cpu', action='store_true',
                        help='Force generation of noise level maps on cpu (use for reproducibility on older versions).')
    parser.add_argument('-dc', '--disable_clamp', type=check_binary, default=True,
                        help='Disable clamp of noisy input to the range [0,1]. Must be set explicitly to 0 (i.e. False) to apply clamp. (use for reproducibility on older versions).')
    parser.add_argument('-bn', '--batch_norm', action='store_true',
                        help='Use Batch Normalisation for training (batch norm layers are merged when training is done). (use for reproducibility on older versions)')
    parser.add_argument('-lam', '--lambd', help='weight of L2 in the loss term')
    parser.add_argument('-ldn', '--learn_denoiser_network', type=check_binary, help='learn the denoiser if true', default=False)
    args = argparse.Namespace()
    args = parser.parse_args()
    ########################################################################################################################

    ## Set User parameters
    args.multi_gpu = False              # set to True to use multiple gpus (if available).
    
    ########################################################################################################################
    print('Launching script \'' + __file__ + '\' the ' + now.strftime('%d/%m/%Y at %H:%M:%S'))
    print(f'Running with arguments:\n {args.__dict__}')

    #Create dictionary of parameters from input arguments to save in checkpoint files, and to check consistency between arguments and loaded checkpoint file.
    # A default checking value can be specified for loading legacy checkpoint files that do not have this parameter.
    # If no default is given (or None value) the parameters must be present in the checkpoint file for checking.
    train_params_check = DictParams(vars(args))
    train_params_check.add_param('learning_rate_init')
    train_params_check.add_param('num_iter_lr_decay')
    train_params_check.add_param('batch_size')
    train_params_check.add_param('crop_size')
    train_params_check.add_param('avg_sigma')
    train_params_check.add_param('gen_sigma_method')
    train_params_check.add_param('learning_rate_init')
    train_params_check.add_param('num_iter_epoch')
    train_params_check.add_param('loader_num_workers')
    train_params_check.add_param('gen_noise_cpu', default=True) # legacy checkpoint files were always created using cpu for noise generation.
    train_params_check.add_param('disable_clamp', default=True) # legacy checkpoint files were always created without clamping of noisy data.
    train_params_check.add_param('batch_norm', default=False)   # legacy checkpoint files were always created without batch normalisation.  
    train_params_check.add_param('lambd', default=0.001)
    
    # Choose device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gen_noise_device = 'cpu' if args.gen_noise_cpu else device
    print(f'Running device type: {device}')

    # make the code reproducible
    np.random.seed(0)
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if not os.path.exists(outputPath):
        os.makedirs(outputPath)

    ##############################################       Load datasets       ###############################################

    def load_list_paths(datasetDir, dataset_csv_file):
        df = pd.read_csv(os.path.join(datasetDir, dataset_csv_file))
        list_paths = df.image_path.values
        for i in range(len(list_paths)):
            list_paths[i] = os.path.normpath(os.path.join(datasetDir, list_paths[i]))
        return list_paths


    list_paths_train = load_list_paths(datasetDir, trainDatasetFile)
    list_paths_valid = load_list_paths(datasetDir, validDatasetFile)

    img_transforms = transforms.Compose([
        transforms.RandomCrop(args.crop_size),
        transforms.ToTensor(),
    ])


    dataset_train = ImagePathDataset(list_paths_train, transform=img_transforms, virtual_length = args.batch_size * args.num_iter_epoch)
    dataset_valid = ImagePathDataset(list_paths_valid[:args.valid_max_num_imgs], transform=img_transforms)

    dataloader_train = DataLoader(dataset_train, batch_size=args.batch_size, num_workers=args.loader_num_workers, shuffle=True)
    dataloader_valid = DataLoader(dataset_valid, batch_size=args.batch_size, num_workers=0, shuffle=False)

    iterloader = CycleEpochIterator(dataloader_train)

    #########################################       Training variables setup       #########################################

    num_channels = 3

    model = net(in_nc=num_channels + 1, out_nc=num_channels, nc=[64, 128, 256, 512], nb=4, act_mode='R',
                downsample_mode="strideconv", upsample_mode="convtranspose")
    model2 = net(in_nc=num_channels, out_nc=num_channels, nc=[64, 128, 256, 512], nb=4, act_mode='R',
                downsample_mode="strideconv", upsample_mode="convtranspose") ##
    if args.batch_norm:
        add_bn(model,for_init=False)
    if args.batch_norm:
        add_bn(model2,for_init=False)

    model.to(device) #If running on GPU device, model data must be transferred to gpu before initilizing the optimizer.
    model2.to(device)
    criterion = nn.L1Loss()
    mse = torch.nn.MSELoss() ##
    if args.learn_denoiser_network==True:
        params = list(model.parameters()) + list(model2.parameters())  ##
    else:
        params=model2.parameters()
    optimizer = torch.optim.Adam(params , lr=args.learning_rate_init)
    
    # Initialization (from checkpoint if specified)
    if "restart_checkpoint" in args and args.restart_checkpoint is not None:
        checkpoint_params = load_checkpoint(args.restart_checkpoint, model, optimizer, args.multi_gpu,
                                ['it_global', 'epoch', 'learning_rate_current', 'next_valid_it','model2'], train_params_check)
        it = checkpoint_params['it_global'] + 1
        learning_rate_current = checkpoint_params['learning_rate_current']
        next_valid_it = checkpoint_params['next_valid_it']
        iterloader.epoch = checkpoint_params['epoch'] + 1
        model2.load_state_dict(checkpoint_params['model2'])

        print(f'Starting from checkpoint file:{args.restart_checkpoint}')
    else:
        if args.init_model is not None:
            data = torch.load(args.init_model, map_location=torch.device('cpu'))
            if 'model' in data:
                data = data['model']
            if args.multi_gpu:
                model.module.load_state_dict(data)
            else:
                model.load_state_dict(data)
        it = 1
        learning_rate_current = args.learning_rate_init
        next_valid_it = 1
    if args.learn_denoiser_network:
        model.train()
    model2.train() ##
    ##############################################       Training loop       ###############################################
    stop_criterion = False
    rand_sig=0
    
    while not stop_criterion:
        img = next(iterloader).to(device)
        print(f'Current iteration: {it} (epoch: {iterloader.epoch} / internal iteration: {iterloader.it_in_epoch})', end='\r')

        if args.skip_train:
            loss = torch.zeros(1)
        else:
            sig0 = globals()[args.gen_sigma_method](img.shape[0], 1, img.shape[2], img.shape[3], args.avg_sigma, gen_noise_device).to(device) 
            if args.learn_denoiser_network==True:
                if rand_sig==0:
                    sig=sig0
                    rand_sig=1
                    delta=1
                else:
                    sig = globals()[args.gen_sigma_method](img.shape[0], 1, img.shape[2], img.shape[3], args.avg_sigma, gen_noise_device).to(device) 
                    rand_sig=0
                    delta=0
            else:
                sig = globals()[args.gen_sigma_method](img.shape[0], 1, img.shape[2], img.shape[3], args.avg_sigma, gen_noise_device).to(device) 
                rand_sig=0
                delta=0
            #print(torch.min(sig0).item()*255,torch.min(sig).item()*255)
            img_noise = img + gen_gaussian_noise(img.size(), device=gen_noise_device).to(device) * sig0 ##
#            if not args.disable_clamp:
#                img_noise.clamp_(0,1)
            if args.infinite_noise_scaling:
                sig.clamp_(max=args.infinite_noise_scaling_t2)
                img_noise *= infinite_noise_scale(sig, args.infinite_noise_scaling_t1, args.infinite_noise_scaling_t2, args.infinite_noise_scaling_p) ## 

            img_rec = model(torch.cat((img_noise, sig), dim=1)) ##
            if not args.disable_clamp: #?
                img_rec.clamp_(0,1) #?
            img_reg = model2(img_rec) ##

            
            loss = delta*criterion(img_rec, img)+float(args.lambd)*mse((sig**2)*img_reg,(img_noise-img_rec))
            #print(delta*criterion(img_rec, img))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


        # Change learning rate after a given number of iterations
        if it % args.num_iter_lr_decay == 0:
            learning_rate_current /= 2
            for g in optimizer.param_groups:
                g['lr'] = learning_rate_current
                stop_criterion = learning_rate_current < args.learning_rate_stop


        trigger_checkpoint = (iterloader.is_last_of_epoch and iterloader.epoch % args.num_epoch_checkpoint == 0) or stop_criterion
        trigger_valid = it >= next_valid_it or trigger_checkpoint

        # Evaluate validation (also done at checkpoints and last iteration)
        if trigger_valid and not args.skip_valid:
            next_valid_it *= args.valid_num_iter_fact if it >= next_valid_it else 1
            print(f'Validation at iteration #{it} ...\t\t\t\t\t\t', end='\r')

            #Save random number generators' states
            rng_cpu = torch.get_rng_state()
            rng_gpu = None
            if torch.cuda.is_available():
                rng_gpu = torch.cuda.get_rng_state_all()

            ''''''
            valid_loss_global = 0.0
            valid_loss_pixwise = 0.0
            valid_loss_global_reg = 0.0
            model.eval()
            model2.eval()
            with torch.no_grad():
                torch.manual_seed(0)
                for i, img in enumerate(dataloader_valid):
                    img = img.to(device)
                    # Validation with constant noise (but variable per image)
                    sig0 = sigmas_uni_batch(img.shape[0], 1, img.shape[2], img.shape[3], args.avg_sigma).to(device)
                    sig = sigmas_uni_batch(img.shape[0], 1, img.shape[2], img.shape[3], args.avg_sigma).to(device)
                    noise = torch.randn(img.size()).to(device)
                    img_noise = img + noise * sig0
                    sigg=torch.cat((sig0[:8,:,:,:],sig[8:,:,:,:]),dim=0) #batch de sigma ayant la moitie sig et l'autre moitie sig0
                    if args.infinite_noise_scaling:
                        sig.clamp_(max=args.infinite_noise_scaling_t2)
                        img_noise *= infinite_noise_scale(sig, args.infinite_noise_scaling_t1, args.infinite_noise_scaling_t2, args.infinite_noise_scaling_p)
                    img_rec = model(torch.cat((img_noise, sig0), dim=1)) #pour validation du denoiser
                    img_toreg = model(torch.cat((img_noise, sigg), dim=1)) #pour validation du model2
                    
                    if not args.disable_clamp: #?
                        img_toreg.clamp_(0,1) #?
                    img_reg = model2(img_toreg) #pour validation du model2
                    
                    valid_loss_global += criterion(img_rec, img).item() * img.shape[0]
                    valid_loss_global_reg += mse((sigg**2)*img_reg,(img_noise-img_toreg)).item() * img.shape[0]
                    # Validation with pixel-wise noise (with maximum level controlled per image)
                    sig = sigmas_uni_Xbatch_spatial(img.shape[0], 1, img.shape[2], img.shape[3], args.avg_sigma).to(device)
                    img_noise = img + noise * sig
                    if args.infinite_noise_scaling:
                        sig.clamp_(max=args.infinite_noise_scaling_t2)
                        img_noise *= infinite_noise_scale(sig, args.infinite_noise_scaling_t1, args.infinite_noise_scaling_t2, args.infinite_noise_scaling_p)
                    img_rec = model(torch.cat((img_noise, sig), dim=1))
                    valid_loss_pixwise += criterion(img_rec, img).item() * img.shape[0]
                valid_loss_global /= len(dataloader_valid.dataset)
                valid_loss_pixwise /= len(dataloader_valid.dataset)
                valid_loss_global_reg /= len(dataloader_valid.dataset)
            if args.learn_denoiser_network:
                model.train()
            model2.train()
            # Save validation and training loss.
            save_data_row(validResultFile, it, train_loss=loss.item(), valid_loss_global=valid_loss_global, valid_loss_pixwise=valid_loss_pixwise,valid_loss_global_reg=valid_loss_global_reg)
            ''''''

            #Reset random number generators' states (validation step should not affect the results of the training)
            torch.set_rng_state(rng_cpu)
            if torch.cuda.is_available():
                torch.cuda.set_rng_state_all(rng_gpu)

            print(f'Validation at iteration #{it} -> Done')

        elif trigger_valid:
            next_valid_it *= args.valid_num_iter_fact if it >= next_valid_it else 1
            save_data_row(validResultFile, it, train_loss=loss.item())
            print(f'Saved training loss at iteration #{it}\t\t\t\t\t\t')

        # Save checkpoints regularly
        if trigger_checkpoint:
            print(f'Saving checkpoint at iteration #{it} ...\t\t\t\t\t\t', end='\r')
            if args.overwrite_checkpoints:
                model_filename = os.path.join(outputPath, f'checkpoint_dpir.pth')
            else:
                model_filename = os.path.join(outputPath, f'checkpoint_dpir_it-{it}.pth')
            #torch.save(model.state_dict(), model_filename)
            save_checkpoint(model_filename, model, optimizer, args.multi_gpu, it_global=it, epoch=iterloader.epoch,
                            learning_rate_current=learning_rate_current, next_valid_it=next_valid_it, model2=model2.state_dict(), **train_params_check)
            print(f'Saved checkpoint at iteration #{it} in: {os.path.realpath(model_filename)}')

        #force output flush to print regularly to output file (if stdout is a file)
        if it % 100 == 0:
            sys.stdout.flush()

        it += 1

    #Save model (merge batch norm before if batch norm is used).
    model_filename = os.path.join(outputPath, f'dpir.pth')
    model2_filename = os.path.join(outputPath, f'dpir_2.pth') #####
    
    if args.batch_norm:
        merge_bn(model)
        merge_bn(model2)
    model.to('cpu')
    model2.to('cpu')
    torch.save(model.state_dict(), model_filename)
    torch.save(model2.state_dict(), model2_filename)
    print(f'Saved final model in: {os.path.realpath(model_filename)}')
    print(f'Saved final model2 in: {os.path.realpath(model2_filename)}')


if __name__ == '__main__':
    main()

