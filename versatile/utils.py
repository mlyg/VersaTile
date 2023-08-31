# built-in imports
import os
import pickle

# library imports
import torch
import seaborn as sns
import pandas as pd
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
from skimage.morphology import remove_small_objects, remove_small_holes
from torchvf.numerics import interp_vf, ivp_solver, init_values_semantic
from torchvf.utils.clustering import cluster

# local imports
from numerics.differentiation.finite_differences import sobel_finite_difference
from losses.loss_functions import StainConsistencyLossMSE, StainConsistencyLossMAE, StainConsistencyLossMSEx, StainConsistencyLossMAEx, \
    DicePlusPlus, TopKLoss, FocalLoss, DiceBCELoss, DiceFocalLoss, DiceTopKLoss, MSDExLoss, MADExLoss, MAGELoss, MSGELoss, MAGExLoss, MSGExLoss


def get_loss(name, device):
    if name == 'dice':
        return DicePlusPlus(gamma=1)
    elif name == 'ce':
        return nn.BCEWithLogitsLoss()
    elif name == 'topk':
        return TopKLoss()
    if name == 'dice_ce':
        return DiceBCELoss()
    elif name == 'dice_topk':
        return DiceTopKLoss()
    elif name == 'mae':
        return nn.L1Loss()
    elif name == 'mse':
        return nn.MSELoss()
    elif name == 'maex':
        return MADExLoss()
    elif name == 'msex':
        return MSDExLoss()
    elif name == 'mage':
        return MAGELoss(device)
    elif name == 'msge':
        return MSGELoss(device)
    elif name == 'magex':
        return MAGExLoss(device)
    elif name == 'msgex':
        return MSGExLoss(device)
    elif name == 'diceplusplus':
        return DicePlusPlus(gamma=2)
    elif name == 'focal':
        return FocalLoss()
    elif name == 'dice_focal':
        return DiceFocalLoss()
    elif name == 'sc_mae':
        return StainConsistencyLossMAE(sigmoid=True)
    elif name == 'sc_mse':
        return StainConsistencyLossMSE(sigmoid=True)
    elif name == 'sc_maex':
        return StainConsistencyLossMAEx()
    elif name == 'sc_msex':
        return StainConsistencyLossMSEx()


def get_model(decoder, device, encoder=None, encoder_weights='imagenet'):
    if decoder == 'unet':
        model = smp.Unet(
            encoder_name=encoder,        
            encoder_weights=encoder_weights,   
            in_channels=3,           
            classes=1).to(device)
    elif decoder == 'fpn':
        model = smp.FPN(
            encoder_name=encoder,        
            encoder_weights=encoder_weights,   
            in_channels=3,           
            classes=1,).to(device)
    elif decoder == 'pspnet':
        model = smp.PSPNet(
            encoder_name=encoder,        
            encoder_weights=encoder_weights,   
            in_channels=3,           
            classes=1).to(device)
    elif decoder == 'pan':
        model = smp.PAN(
            encoder_name=encoder,        
            encoder_weights=encoder_weights,   
            in_channels=3,           
            classes=1).to(device)
    elif decoder == 'deeplabv3plus':
        model = smp.DeepLabV3Plus(
            encoder_name=encoder,        
            encoder_weights=encoder_weights,   
            in_channels=3,           
            classes=1).to(device)
    elif decoder == 'manet':
        model = smp.MAnet(
            encoder_name=encoder,        
            encoder_weights=encoder_weights,   
            in_channels=3,           
            classes=1).to(device)
    elif decoder == 'unetplusplus':
        model = smp.UnetPlusPlus(
            encoder_name=encoder,        
            encoder_weights=encoder_weights,   
            in_channels=3,           
            classes=1).to(device)
    elif decoder == 'linknet':
        model = smp.Linknet(
            encoder_name=encoder,        
            encoder_weights=encoder_weights,   
            in_channels=3,           
            classes=1).to(device)

    return model


def get_conic_indices(train_test_split):
    with open(train_test_split, 'rb') as f:
        data_split = pickle.load(f)
    dev_idxs = data_split['train']
    test_idxs = data_split['test']
    return dev_idxs, test_idxs


def plt_loss_curves(data, model, output_path):
    for loss in data:
        train_loss, val_loss = data[loss]
        epochs = list(range(len(train_loss)))
        df = pd.DataFrame({"Epoch":epochs, 'Train':train_loss, 'Val':val_loss})
        sns.lineplot(x='Epoch', y='Train', data=df, label='Train')
        sns.lineplot(x='Epoch', y='Val', data=df, label='Val')
        plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss during training')
        plt.savefig(os.path.join(output_path, f'{model}_{loss}_loss_curves.png'), dpi=400)
        plt.close()


def plt_metric_curves(dice_scores, model, output_path):
    epochs = list(range(len(dice_scores)))
    data = pd.DataFrame({"Epoch":epochs, "Dice":dice_scores})
    sns.lineplot(x='Epoch', y='Dice', data=data, label='Dice')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.title('Score during training')
    plt.savefig(os.path.join(output_path, f'{model}_metric_curves.png'), dpi=400)
    plt.cla()
    plt.clf()


def freeze_encoder(model):
    for child in model.encoder.children():
        for param in child.parameters():
            param.requires_grad = False
    return


def unfreeze(model):
    for child in model.children():
        for param in child.parameters():
            param.requires_grad = True
    return 


def tta(model, img):
    img_1 = img
    img_2 = torch.flip(img,[2])
    img_3 = torch.flip(img,[3])
    img_4 = torch.flip(img,[2,3])

    sem_pred_1, dtm_pred_1 = model(img_1)
    sem_pred_2, _ = model(img_2)
    sem_pred_3, _ = model(img_3)
    sem_pred_4, _ = model(img_4)

    sem_pred_2 = torch.flip(sem_pred_2, [2])
    sem_pred_3 = torch.flip(sem_pred_3, [3])
    sem_pred_4 = torch.flip(sem_pred_4, [2,3])

    sem_pred_1 = torch.sigmoid(sem_pred_1)
    sem_pred_2 = torch.sigmoid(sem_pred_2)
    sem_pred_3 = torch.sigmoid(sem_pred_3)
    sem_pred_4 = torch.sigmoid(sem_pred_4)

    avg_sem_pred = (sem_pred_1 + sem_pred_2 + sem_pred_3 + sem_pred_4) /4

    return avg_sem_pred, dtm_pred_1


def generate_instance_seg(dtm, sem, kernel_size=5, min_samples=15, dx=1, n_steps=25, snap_noise=True, eps=2.25, device='cpu'):
    
    dtm = torch.from_numpy(dtm.astype(float))
    dtm = dtm.float()
    
    grad = sobel_finite_difference(dtm, kernel_size=kernel_size, device=device)

    mag = torch.hypot(grad[:1,:1], grad[:1,1:])
    mag = torch.concat([mag, mag],axis=1)

    # normalise gradients
    grad /= (mag + 1e-7)

    # scale gradients based on distance to centre of object
    grad *= (1 - torch.concat([dtm, dtm], axis=1))

    semantic = (sem > 0.5)
    semantic = remove_small_objects(semantic, min_size=15)
    semantic = remove_small_holes(semantic, area_threshold=15)
    semantic = torch.from_numpy(semantic)

    vf = interp_vf(grad, mode = "nearest")

    init_values = init_values_semantic(semantic, device = "cpu")

    solutions = ivp_solver(
        vf, 
        init_values, 
        dx = dx,
        n_steps = n_steps,
        solver = "euler"
    )[-1] 

    solutions = solutions.cpu()
    semantic = semantic.cpu()

    instance_segmentation = cluster(
        solutions, 
        semantic[0],
        eps = 2.25,
        min_samples = min_samples,
        snap_noise = snap_noise
        )
    
    return instance_segmentation


def pad_image(image, patch_size):
    pad_h = 0
    pad_v = 0

    if image.shape[0]%patch_size:
        pad_h = patch_size - image.shape[0]%patch_size
    if image.shape[1]%patch_size:
        pad_v = patch_size - image.shape[1]%patch_size
    
    return np.pad(image, [(0, pad_h), (0, pad_v), (0,0)], 'constant')        


def gaussian_importance_map(size=224, sigma=1):
    # Create coordinate grids using meshgrid
    x, y = np.meshgrid(np.linspace(-1, 1, size), np.linspace(-1, 1, size))
    # Calculate the distance from the center
    dist = np.sqrt(x**2 + y**2)
    # create Gaussian distribution
    gaussian = np.exp(-(dist**2) / (2 * sigma**2))
    # normalise the distribution
    gaussian /= np.max(gaussian)
    return gaussian


def square_importance_map(size, stride):
    # Create a 2D array filled with zeros
    array = np.zeros((size, size))

    # Calculate the starting and ending indices for the square
    start_index = int(stride / 2)
    end_index = size - int(stride / 2)

    # Fill the square region with ones
    array[start_index:end_index, start_index:end_index] = 1

    return array