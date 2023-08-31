# built-in imports
import os
import pickle
import logging
import sys
import argparse
import yaml

# library imports
import torch
import numpy as np
from torch.utils.data import DataLoader

# local imports
from datasets.conic import CoNICDataset
from transforms.transforms import val_transforms
from utils import tta, generate_instance_seg, get_conic_indices, get_model

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

    # get test indices 
    _, test_idxs = get_conic_indices(config['train_test_split'])

    # load images
    img_data = np.load(config['image_path'], mmap_mode='r')

    # get test datasets
    test_set = CoNICDataset(imgs=img_data, labels=None, dtm=None, indices=test_idxs, stain_aug=None, transforms=val_transforms(size=config['image_size']))

    # get test dataloader
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False,  num_workers=0)

    # check loaders
    logger.info('----- Checking Dataloader -----')
    img = next(iter(test_loader))
    logger.info(f'Image shape: {img.shape}')
    logger.info(f'Device used: {DEVICE}')
    if config['tta']:
        logger.info('TTA enabled')
    else:
        logger.info('TTA disabled')

    # initialise model
    model = get_model(config['decoder'], DEVICE, config['encoder'])

    # load model
    model.load_state_dict(torch.load(config['model_path']))
    # set model to eval mode
    model.eval()

    # load best post-processing params
    with open(config['param_path'], 'rb') as f:
        best_params = pickle.load(f)

    logger.info(f'post-processing parameters: {best_params}')
    
    inst_labels = []
    
    with torch.no_grad():
        for test_data in test_loader:
            test_imgs = test_data.to(DEVICE)
            if config['tta']:
                test_sem_pred, test_dtm_pred = tta(model, test_imgs)
            else:
                test_sem_pred, test_dtm_pred = model(test_imgs)
                test_sem_pred = torch.sigmoid(test_sem_pred)
            test_sem_pred = (test_sem_pred > 0.5).float()

            inst_label = generate_instance_seg(test_dtm_pred.detach().cpu().numpy(), test_sem_pred.detach().cpu().numpy(), dx=best_params['dx'], n_steps=int(best_params['n_steps']))
            inst_labels.append(inst_label.detach().cpu().numpy().astype(int))
    inst_labels = np.stack(inst_labels)

    np.save(os.path.join(config['output_path'], f"{config['run_name']}_preds.npy"), inst_labels.astype(int))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='VersaTile settings')
    # Add arguments
    parser.add_argument('-c','--config', help="path to config file", required=True, type=str)
    args = vars(parser.parse_args())

    with open(args['config']) as file:
        config = yaml.safe_load(file)

    main(config)