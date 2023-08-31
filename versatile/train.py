# built-in imports
import os
import sys
import logging
import argparse
import random
import pickle
import yaml

# library imports
import torch
from torch.utils.data import DataLoader
from hyperopt import hp, fmin, tpe
from stardist.matching import matching_dataset
import numpy as np
import matplotlib.pyplot as plt

# local imports
from datasets.conic import CoNICDataset
from utils import generate_instance_seg, get_conic_indices, plt_loss_curves, plt_metric_curves, get_model, get_loss
from transforms.transforms import train_transforms, val_transforms
from dtm.create_dtm import compute_hybrid_dtm
from metrics.metrics import f1
from stain.randstainna import RandStainNA
from stain.stain_consistency_aug import StainConsistencyAug
from stain.stain_jitter import StainJitter

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


logging.basicConfig(
                    level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[
                        logging.StreamHandler(sys.stdout)
                    ]
                )

logger = logging.getLogger(__name__)

def main(config):

    # load image, label and dtm
    img_data = np.load(config['image_path'], mmap_mode='r')
    label_data = np.load(config['label_path'], mmap_mode='r')

    if config['dtm_path'] is not None:
        dtm_data = np.load(config['dtm_path'], mmap_mode='r')
    else:
        logger.info('Creating DTMs...')
        # CoNIC labels have both instance (0) and class labels (1)
        dtm_data = np.zeros_like(label_data[:,:,:,:1])
        for i, label in enumerate(label_data[:,:,:,:1]):
            dtm = compute_hybrid_dtm(label)
            dtm_data[i] = dtm
        logger.info('Finished creating DTMs.')

    # randomly split dataset 80/20
    dev_idxs, _ = get_conic_indices(config['train_test_split'])
    if config['seed']:
        random.seed(42)
    random.shuffle(dev_idxs)
    train_idxs = dev_idxs[:int(0.8*(len(dev_idxs)))]
    val_idxs = dev_idxs[int(0.8*(len(dev_idxs))):]

    # select stain preprocessing method
    if config['stain_aug'] == 'sca':
        stain_aug = StainConsistencyAug(param_file=config['param_file'])
    elif config['stain_aug'] == 'stain_jitter':
        stain_aug = StainJitter(stain=config['stain'])
    elif config['stain_aug'] == 'randstainna':
        stain_aug = RandStainNA(yaml_file=config['param_file'])
    else:
        stain_aug = None

    # get training and validation datasets
    train_set = CoNICDataset(imgs=img_data, labels=label_data, dtm=dtm_data, indices=train_idxs, stain_aug=stain_aug, transforms=train_transforms(size=config['image_size'], sc_loss=config['sc_loss']), sc_loss=config['sc_loss'])
    val_set = CoNICDataset(imgs=img_data, labels=label_data, dtm=dtm_data, indices=val_idxs, stain_aug=None, transforms=val_transforms(size=config['image_size']))
    
    # get dataloaders
    train_loader = DataLoader(train_set, batch_size=config['batch_size'], shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False,  num_workers=0)

    # check loaders
    logger.info('----- Checking Dataloader -----')

    if config['sc_loss']:
        img, _, label, dtm = next(iter(train_loader))
        logger.info(f'Image shape: {img.shape}')
        logger.info(f'Label shape: {label.shape}')
        logger.info(f'Distance transform field shape: {dtm.shape}')
        logger.info(f'Device used: {DEVICE}')
    else:
        img, label, dtm = next(iter(train_loader))
        logger.info(f'Image shape: {img.shape}')
        logger.info(f'Label shape: {label.shape}')
        logger.info(f'Distance transform field shape: {dtm.shape}')
        logger.info(f'Device used: {DEVICE}')

    # initialise model
    model = get_model(config['decoder'], DEVICE, config['encoder'])
    
    # load pretrained model
    if config['pretrained']:
       checkpoint = torch.load(config['pretrained_path'])
       model.load_state_dict(checkpoint, strict=False)

    # loss functions
    sem_loss = get_loss(config['sem_loss'], device=DEVICE)
    dtm_loss = get_loss(config['dtm_loss'], device=DEVICE)
    gtm_loss = get_loss(config['gtm_loss'], device=DEVICE)

    if config['sc_loss']:
        sc_loss = get_loss('sc_maex', device=DEVICE)
            
    # optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    # scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.95, patience=3, threshold=1e-9)

    # training loop
    val_interval = 1
    best_val_loss = 1000
    best_val_epoch = 0

    train_epoch_losses = []
    train_sem_losses = []
    train_dtm_losses = []
    train_gtm_losses = []

    if config['sc_loss']:
        train_sc_sem_losses = []

    val_epoch_losses = []
    val_sem_losses = []
    val_dtm_losses = []
    val_gtm_losses = []

    dice_scores = []

    for epoch in range(config['max_epoch']):
        logger.info(f"Epoch: {epoch + 1}/{config['max_epoch']}")
        model.train()
        epoch_loss = 0
        epoch_sem_loss = 0
        epoch_dtm_loss = 0
        epoch_gtm_loss = 0
        epoch_sc_sem_loss = 0
        epoch_sc_inst_loss = 0
        for i, train_data in enumerate(train_loader):
            # retrieve data batch
            if config['sc_loss']:
                imgs, imgs_2, inst_label, dtm_label = train_data[0].to(DEVICE), train_data[1].to(DEVICE), train_data[2].to(DEVICE), train_data[3].to(DEVICE)
            else:
                imgs, inst_label, dtm_label = train_data[0].to(DEVICE), train_data[1].to(DEVICE), train_data[2].to(DEVICE)
            # reset gradients
            optimizer.zero_grad()
            # get model outputs
            sem_pred, dtm_pred = model(imgs)
            # semantic mask
            sem_label = torch.where(inst_label > 0, 1.0 ,0.0)
            # get losses
            loss_1 = sem_loss(sem_pred, sem_label)
            if config['dtm_loss'] == 'maex' or config['dtm_loss'] == 'msex':
                loss_2 = dtm_loss(dtm_pred, dtm_label, sem_label)
            else:
                loss_2 = dtm_loss(dtm_pred, dtm_label)
            if config['gtm_loss'] == 'magex' or config['gtm_loss'] == 'msgex':
                loss_3 = gtm_loss(dtm_pred, dtm_label, sem_label)
            else:
                loss_3 = gtm_loss(dtm_pred, dtm_label)
            if config['sc_loss']:
                sem_pred_2, _ = model(imgs_2)
                loss_4 = sc_loss(sem_pred, sem_pred_2, sem_label)
            loss = loss_1 + loss_2 + loss_3
            if config['sc_loss']:
                loss += loss_4
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_sem_loss += loss_1.item()
            epoch_dtm_loss += loss_2.item()
            epoch_gtm_loss += loss_3.item()
            if config['sc_loss']:
                epoch_sc_sem_loss += loss_4.item()
            epoch_len = len(train_set) // train_loader.batch_size
            logger.info(f'{i+1}/{epoch_len}, train loss: {loss.item():.4f}')
        epoch_loss /= len(train_loader)
        scheduler.step(epoch_loss)
        epoch_sem_loss /= len(train_loader)
        epoch_dtm_loss /= len(train_loader)
        epoch_gtm_loss /= len(train_loader)
        epoch_sc_sem_loss /= len(train_loader)
        epoch_sc_inst_loss /= len(train_loader)
        train_epoch_losses.append(epoch_loss)
        train_sem_losses.append(epoch_sem_loss)
        train_dtm_losses.append(epoch_dtm_loss)
        train_gtm_losses.append(epoch_gtm_loss)
        if config['sc_loss']:
            train_sc_sem_losses.append(epoch_sc_sem_loss)

        logger.info(f'Epoch {epoch + 1} average loss: {epoch_loss:.4f}')

        if (epoch + 1) % val_interval==0:
            logger.info('Running validation')
            model.eval()
            val_epoch_loss = 0
            val_epoch_sem_loss = 0
            val_epoch_dtm_loss = 0
            val_epoch_gtm_loss = 0

            val_dice = []
            with torch.no_grad():
                for i, val_data in enumerate(val_loader):
                    val_imgs, val_inst_label, val_dtm_label = val_data[0].to(DEVICE), val_data[1].to(DEVICE), val_data[2].to(DEVICE)
                    val_sem_pred, val_dtm_pred = model(val_imgs)
                    # semantic mask
                    val_sem_label = torch.where(val_inst_label > 0, 1.0 ,0.0)
                    # get losses
                    loss_1 = sem_loss(val_sem_pred, val_sem_label)
                    if config['dtm_loss'] == 'maex' or config['dtm_loss'] == 'msex':
                        loss_2 = dtm_loss(val_dtm_pred, val_dtm_label, val_sem_label)
                    else:
                        loss_2 = dtm_loss(val_dtm_pred, val_dtm_label)
                    if config['gtm_loss'] == 'magex' or config['gtm_loss'] == 'msgex':
                        loss_3 = gtm_loss(val_dtm_pred, val_dtm_label, val_sem_label)
                    else:
                        loss_3 = gtm_loss(val_dtm_pred, val_dtm_label)
                    val_loss = loss_1 + loss_2 + loss_3
                    val_epoch_loss += val_loss.item()
                    val_epoch_sem_loss += loss_1.item()
                    val_epoch_dtm_loss += loss_2.item()
                    val_epoch_gtm_loss += loss_3.item()
                    # get dice score
                    val_sem_pred = torch.sigmoid(val_sem_pred) > 0.5
                    val_dice.append(f1(val_sem_pred.float(), val_sem_label).detach().cpu().numpy())
                    epoch_len = len(val_set) // val_loader.batch_size
                    logger.info(f'{i+1}/{epoch_len}, val loss: {val_loss.item():.4f}')
                
                val_dice = np.mean(val_dice)                    
                val_epoch_loss /= len(val_loader)
                val_epoch_sem_loss /= len(val_loader)
                val_epoch_dtm_loss /= len(val_loader)
                val_epoch_gtm_loss /= len(val_loader)

                val_epoch_losses.append(val_epoch_loss)
                val_sem_losses.append(val_epoch_sem_loss)
                val_dtm_losses.append(val_epoch_dtm_loss)
                val_gtm_losses.append(val_epoch_gtm_loss)
                dice_scores.append(val_dice)

                if val_epoch_loss < best_val_loss:
                    best_val_loss = val_epoch_loss
                    best_val_epoch = epoch + 1
                    torch.save(model.state_dict(), os.path.join(config['output_path'], f"model_{config['run_name']}.pt"))
                    logger.info('Saved latest model')
                logger.info(
                    "current epoch: {} current Dice {:.4f} current loss: {:.4f} best loss: {:.4f} at epoch {}".format(
                        epoch + 1, val_dice, val_epoch_loss, best_val_loss, best_val_epoch)
                )

            if epoch % config['save_img_epoch'] == 0:

                _, (ax0, ax1, ax2, ax3, ax4) = plt.subplots(1, 5)
                ax0.imshow(np.moveaxis(val_imgs[0].detach().cpu().numpy(),0,-1))
                ax1.imshow(val_sem_pred[0][0].detach().cpu())
                ax2.imshow(val_inst_label[0][0].detach().cpu())
                ax3.imshow(val_dtm_pred[0][0].detach().cpu())
                ax4.imshow(val_dtm_label[0][0].detach().cpu())

                plt.savefig(os.path.join(config['output_path'], f"images_{config['run_name']}_{epoch}.pt.png"), dpi = 400, bbox_inches = 'tight')
                plt.close()
                
        losses = {'Overall':[train_epoch_losses, val_epoch_losses]
                , 'SEM':[train_sem_losses, val_sem_losses]
                ,'DTM':[train_dtm_losses,val_dtm_losses]
                ,'GTM':[train_gtm_losses, val_gtm_losses]
                }

        plt_loss_curves(losses, f"loss_{config['run_name']}_plot", config['output_path'])
        plt_metric_curves(dice_scores, f"metrics_{config['run_name']}_plot", config['output_path'])


    logger.info('Beginning Bayesian hyperparameter search for best post-processing hyperparameters')
    # Bayesian hyperparameter search for best post-processing parameters
    # load best model
    model.load_state_dict(torch.load(os.path.join(config['output_path'],f"model_{config['run_name']}.pt")))
    # set model to eval mode
    model.eval()

    sem_preds = []
    dtm_preds = []
    inst_labels = []

    # generate semantic segmentations and dtms for validation dataset
    with torch.no_grad():
        for val_data in val_loader:
            val_imgs, val_inst_label, val_dtm_label = val_data[0].to(DEVICE), val_data[1].to(DEVICE), val_data[2].to(DEVICE)
            val_sem_pred, val_dtm_pred = model(val_imgs)
            val_sem_pred = torch.sigmoid(val_sem_pred)
            val_sem_pred = (val_sem_pred > 0.5).float()
            sem_preds.append(val_sem_pred.detach().cpu().numpy())
            dtm_preds.append(val_dtm_pred.detach().cpu().numpy())
            inst_labels.append(val_inst_label.detach().cpu().numpy())
    
    sem_preds = np.stack(sem_preds)
    dtm_preds = np.stack(dtm_preds)
    inst_labels = np.stack(inst_labels)

    params = {
            'dx': hp.uniform('dx',config['dx'][0], config['dx'][1]),
            'n_steps': hp.quniform('n_steps', config['n_steps'][0],config['n_steps'][1],config['n_steps'][2]),
            }
    
    def optimizer_fn(params):
        pred_masks = np.zeros((config['num_eval'],config['image_size'],config['image_size']))
        for i, (sem_pred, dtm_pred) in enumerate(zip(sem_preds[:config['num_eval']], dtm_preds[:config['num_eval']])):
            mask = generate_instance_seg(dtm_pred, sem_pred, dx=params['dx'], n_steps=int(params['n_steps']))
            pred_masks[i] = mask[0]
        pred_masks = pred_masks.astype(int)
        mts = matching_dataset([label for label in inst_labels[:config['num_eval'],0,0,:,:]], [pred for pred in pred_masks]).panoptic_quality 
        return -mts
                
    best_param = fmin(fn=optimizer_fn,
                    space=params,
                    max_evals=config['max_eval'],
                    rstate=np.random.default_rng(0),
                    algo=tpe.suggest)
 
    with open(os.path.join(config['output_path'], f"param_{config['run_name']}.pkl"), 'wb') as f:
        pickle.dump(best_param, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='VersaTile settings')
    # Add arguments
    parser.add_argument('-c','--config', help="path to config file", required=True, type=str)
    args = vars(parser.parse_args())

    with open(args['config']) as file:
        config = yaml.safe_load(file)

    main(config)