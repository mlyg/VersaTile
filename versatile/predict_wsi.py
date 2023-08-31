# built-in imports
import os
import sys
import logging
import argparse
import pickle 
import yaml

# library imports
import torch
import numpy as np
from skimage import io
from skimage import segmentation
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader

# local imports
from datasets.myocyte import MyocyteDataset
from utils import get_model, tta, pad_image, gaussian_importance_map, generate_instance_seg, square_importance_map
from transforms.transforms import val_transforms


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
    # initialise model
    model = get_model(config['decoder'], DEVICE, config['encoder'])
    # load model
    model.load_state_dict(torch.load(config['model_path']))
    # set model to eval mode
    model.eval()
    
    # load image
    orig_img = io.imread(config['wsi_path'])

    logger.info(f'Image shape before padding: {orig_img.shape}')

    # pad image if necessary
    img = pad_image(orig_img, config['window_size'])

    logger.info(f'Image shape after padding: {img.shape}')

    # define importance map
    importance_map = np.expand_dims(gaussian_importance_map(size=config['window_size']),axis=0)
    sum_map = np.ones((1, config['window_size'], config['window_size']))

    # extract patches from image
    h, w = img.shape[0], img.shape[1]
    w_h = w_w = config['window_size']
    s_h = s_w = config['stride']

    # Generate a list of starting points a.k.a top left of window
    starting_points = [(x, y)  for x in set( list(range(0, h - w_h, s_h)) + [h - w_h] ) 
                            for y in set( list(range(0, w - w_w, s_w)) + [w - w_w] )]

    logger.info(f'Number of patches: {len(starting_points)}')


    # Get list of patches
    image_patches = np.zeros((len(starting_points), config['window_size'], config['window_size'], img.shape[-1])).astype(np.uint8)
    for i, (x, y) in enumerate(starting_points):
        image_patches[i] = img[x:x + w_h, y:y + w_w,:]

    logger.info(f'Image patch shape: {image_patches.shape}')

    # initialise dataset with patches
    dataset = MyocyteDataset(image_patches, transforms=val_transforms(size=224))

    # initialise dataloader with dataset
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, drop_last=False, num_workers=0)

    # run model inference and store results
    semantic_preds = []
    dtm_preds = []

    with torch.no_grad():
        for patch in dataloader:
            if config['tta']:
                sem_pred, dtm_pred = tta(model, patch.to(DEVICE))
            else:
                sem_pred, dtm_pred = model(patch)
                sem_pred = torch.sigmoid(sem_pred)

            semantic_preds.append(sem_pred.detach().cpu().numpy())
            dtm_preds.append(dtm_pred.detach().cpu().numpy())

    semantic_preds = np.concatenate(semantic_preds,axis=0)
    dtm_preds = np.concatenate(dtm_preds,axis=0)

    logger.info(f'Semantic prediction shape: {semantic_preds.shape}')

    # merge predictions while handling overlap
    semantic_pred_merged = np.zeros((1, img.shape[0], img.shape[1]))
    dtm_pred_merged = np.zeros((1, img.shape[0], img.shape[1]))
    overlap = np.zeros((1, img.shape[0], img.shape[1]))
    sums = np.zeros((1, img.shape[0], img.shape[1]))
 
    for i in range(len(semantic_preds)):
        x, y = starting_points[i]
        overlap[:, x:x + w_h, y:y + w_w] += importance_map
        sums[:, x:x + w_h, y:y + w_w] += sum_map

    for i in range(len(semantic_preds)):
        x, y = starting_points[i]
        semantic_pred_merged[:,x:x + w_h, y:y + w_w] += semantic_preds[i] * (importance_map / overlap[:,x:x + w_h, y:y + w_w])
        dtm_pred_merged[:,x:x + w_h, y:y + w_w] += dtm_preds[i] * (importance_map / overlap[:, x:x + w_h, y:y + w_w])
    
    semantic_pred_merged = (semantic_pred_merged > 0.5).astype(float)

    # expand out to 4D tensor
    semantic_pred_merged = np.expand_dims(semantic_pred_merged, axis=0)
    dtm_pred_merged = np.expand_dims(dtm_pred_merged, axis=0)

    with open(config['param_path'], 'rb') as f:
        best_params = pickle.load(f)

    inst_labels = []
    # generate instance segmentation for each patch
    j = 0
    for i in range(len(semantic_preds)):
        x, y = starting_points[i]
        inst_label = generate_instance_seg(dtm_pred_merged[:,:,x:x + w_h,y:y + w_w], semantic_pred_merged[:,:,x:x + w_h,y:y + w_w], dx=best_params['dx'], n_steps=int(best_params['n_steps']))
        inst_label += j
        inst_label[inst_label == j] = 0
        inst_labels.append(inst_label.detach().cpu().numpy())
        j = inst_label.max()
    
    inst_labels = np.concatenate(inst_labels,axis=0)

    logger.info(f'Instance label shape: {inst_labels.shape}')

    # change importance map to square
    importance_map = square_importance_map(size=config['window_size'], stride=config['stride'])
    sum_map = np.ones((1, config['window_size'], config['window_size']))
    overlap = np.zeros((1, img.shape[0], img.shape[1]))
    sums = np.zeros((1, img.shape[0], img.shape[1]))
    for i in range(len(semantic_preds)):
        x, y = starting_points[i]
        overlap[:, x:x + w_h, y:y + w_w] += importance_map
        sums[:, x:x + w_h, y:y + w_w] += sum_map     
    sums = np.where(sums==1,1,0)

    inst_pred = np.zeros((1, img.shape[0], img.shape[1]))

    for i in range(len(inst_labels)):
        x, y = starting_points[i]
        # determine regions where this instance map is definitive
        specific_importance_map = sums[:,x:x + w_h, y:y + w_w] + importance_map
        ambiguous_region = (specific_importance_map < 1)
        # determine objects that are contained within definitive region
        ambiguous_objects_region = ambiguous_region * np.expand_dims(inst_labels[i],axis=0)
        ambiguous_objects_idxs = np.unique(ambiguous_objects_region)
        certain_objects_idx = list(set(np.unique(inst_labels[i])) - set(ambiguous_objects_idxs))
        
        # add certain objects to overall instance map
        for obj in certain_objects_idx:
            inst_pred[:,x:x + w_h, y:y + w_w] = np.where(np.expand_dims(inst_labels[i],axis=0)==obj, inst_pred.max()+1, inst_pred[:,x:x + w_h, y:y + w_w])

        # add ambiguous objects with caution
        if len(ambiguous_objects_idxs) > 1:
            # iterate through ambiguous objects
            for amb_obj_idx in ambiguous_objects_idxs[1:]:
                # get ambiguous object mask
                amb_obj_mask = np.where(np.expand_dims(inst_labels[i],axis=0)==amb_obj_idx,1,0)
                overlap_region = amb_obj_mask * inst_pred[:,x:x + w_h, y:y + w_w]
                # get object indices that are in overlap region
                obj_indices = np.unique(overlap_region)[1:]
                # set those indices to be current object's index
                new_value = inst_pred.max()+1
                for obj_idx in obj_indices:
                    inst_pred[inst_pred==obj_idx] = new_value
                inst_pred[:,x:x + w_h, y:y + w_w] = np.where(np.expand_dims(inst_labels[i],axis=0)==amb_obj_idx, new_value, inst_pred[:,x:x + w_h, y:y + w_w])


    inst_pred = inst_pred[:,:orig_img.shape[0],:orig_img.shape[1]]
    logger.info(f'Instance pred shape: {inst_pred.shape}')

    overlay_image = segmentation.mark_boundaries(orig_img.astype(np.uint8), inst_pred[0], mode='thick')
    io.imsave(os.path.join(config['output_path'], 'wsi.png'), overlay_image)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='WSI settings')
    # Add arguments
    parser.add_argument('-c','--config', help="path to config file", required=True, type=str)
    args = vars(parser.parse_args())

    with open(args['config']) as file:
        config = yaml.safe_load(file)

    main(config)