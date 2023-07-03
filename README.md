
# PnP-ReG: Learned Regularizing Gradient for Plug-and-Play Gradient Descent

This is the pytorch implementation of our paper "PnP-ReG: Learned Regularizing Gradient for Plug-and-Play Gradient Descent" published at SIAM Journal on Imaging Sciences 2023. In case of using this code, please cite our paper as mentioned in section Citation. 

The repository includes the codes for both training and testing the regularizer. 

[[Paper]](https://epubs.siam.org/doi/10.1137/22M1490843)


# Prerequisites

The code is based on Pytorch. First, you should install PyTorch in your virtual environment depending on your cuda version. We use Python 3.7.3, Pytorch 1.7.1 with Cuda 10.2. The other dependencies can be installed using the following command.

```bash
pip install -r requirements.txt
```




# ReG training 

- The pre-trained denoiser (DRUNet) should be set in models/DRUNet/drunet_color_fixed_LR.pth . 
- The training dataset (same as DRUNet training: DIV2K, Flickr2K, Waterloo_Exploration_Database, BSD500) should be downloaded and stored in ./data/(name of dataset). 

In order to train the network modeling the gradient of a regularizer, by using a pre-trained denoiser DRUNet and updating it jointly with the regularizer, we run the following: 

```bash
python train_ReG.py -nie 43470 -nilrd 100000 -lnw 8 -im "./models/DRUNet/drunet_color_fixed_LR.pth" -lam 0.004 -ldn 1 
```
If you need to use a checkpoint for the regularizer, add -rc as:  

```bash
python train_ReG.py -nie 43470 -nilrd 100000 -lnw 8 -im "./models/DRUNet/drunet_color_fixed_LR.pth" -lam 0.004 -ldn 1 -rc "./trained_results/regularizer/checkpoint_dpir_it-652050.pth"
```

# ReG for testing

The trained regularizer network can be downloaded [here](https://drive.google.com/drive/folders/1nuQbNrqYAn96zOPxrF2dNB0m0WGvtA_D?usp=sharing). You should save it in models/regularizers/ReG.pth in order to use it for testing. 

For performing Super-resolution of factor 2 with an added noise of 0.01 (2.55/255): 

```bash
python SR_regularizer.py --kernel_type bicubic --factor 2 --lr 0.002 --noise_level_img 2.55 --sigma 2.55 --testset_name set5 --num_iter 1500 
```

**Arguments description:**

- kernel_type specifies the blurring kernel (bicubic or gaussian)
- factor specifies the downsampling factor of the SR
- lr is the learning rate of the Gradient-descent algorithm
- noise_level_img is the added AWGN (/255) during the degradation (after blurring and downsampling) 
- sigma is the weight of the regularization (/255)
- testset_name specifies the testing dataset (can be set5, CBSD68)
- num_iter is the number of iterations of the GD (can be set to 1500)
  
- gauss_sigma should be specified when using a gaussian kernel. we use 0.5
- save_im can be added to save the results

For deblurring with the set5 dataset blurred with a Gaussian isotropic kernel of standard deviation 1.6 and added noise of 0.01 (2.55/255):

```bash
python deblurring_regularizer.py --k_index 2 --lr 0.004 --noise_level_img 2.55 --sigma 2.55 --testset_name set5 --num_iter 1500
```

**Arguments description:**

- k_index is the index of the kernel number used in kernels_12.mat. For the results of the paper, we use k_index 2 and 3 for the isotropic gaussian kernels of standard deviation 1.6 and 2.0 respectively, and 5 and 7 for the anisotropic kernels.
- lr is the learning rate of the Gradient-descent algorithm
- noise_level_img is the added AWGN (/255) during the degradation (after blurring) 
- sigma is the weight of the regularization (/255)
- testset_name specifies the testing dataset (can be set5, CBSD68)
- num_iter is the number of iterations of the GD (can be set to 1500)

- save_im can be added to save the results

For pixel-wise inpainting with 20% of the total pixels maintained: 

```bash
python completion_regularizer.py --factor 0.2 --lr 0.01 --sigma 1 --testset_name set5 --num_iter 1500 
```

**Arguments description:**

- factor is the part of pixels to maintain (between 0 and 1). i.e. 0.2 means 20% of the pixels are maintained and 80% are masked
- lr is the learning rate of the Gradient-descent algorithm
- sigma is the weight of the regularization (/255)
- testset_name specifies the testing dataset (can be set5, CBSD68)
- num_iter is the number of iterations of the GD (can be set to 1500)

- save_im can be added to save the results


**For solving SR, deblurring and pixel-wise inpainting with other degradation parameters, check the Table 1 in our paper to properly set the Gradient Descent parameters for each degradation model (LR and weight of the regularization).**


# Citation 
If you use this project, please cite our relevant publication:

```bash
@article{fermanian2023pnp,
  title={PnP-ReG: Learned Regularizing Gradient for Plug-and-Play Gradient Descent},
  author={Fermanian, Rita and Le Pendu, Mikael and Guillemot, Christine},
  journal={SIAM Journal on Imaging Sciences},
  volume={16},
  number={2},
  pages={585--613},
  year={2023},
  publisher={SIAM}
}
```
