import torch
import numpy as np

from torch.utils.data import Dataset

class MyocyteDataset(Dataset):
    """PyTorch Dataset class for loading images from Cardiomyocyte dataset"""

    def __init__(self, imgs, labels=None, dtm=None, indices=None, stain_aug=None, transforms=None, annotator=0, sc_loss=False):
        """
        Args:
            imgs (array): array containing images
            labels (array): array containing labels
            dtm (array): array containing distance transform map
            indices (list): list of indices corresponding to dataset
            stain_aug: stain augmentation method
            transforms: image transformations
            annotator (int): which annotator labels (0 or 1) to use
            sc_loss (bool): whether using stain consistency loss
        """

        # set indices as all if not specified
        if not indices:
            indices = list(range(len(imgs)))
        
        # load images corresponding to indices
        self.images = imgs[indices]

        # load instance segmentation labels if available
        if labels is not None:
            # load labels corresponding to indices and annotator
            self.labels = labels[indices,:,:,:,annotator]
        else:
            self.labels = None
        
        # load distance transform map if available
        if dtm is not None:
            self.dtm = dtm[indices]
        else:
            self.dtm = None

        self.stain_aug = stain_aug
        if self.stain_aug is not None:
           self.stain_aug.preprocess(self.images)
    
        self.transforms = transforms
        self.sc_loss = sc_loss

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # select image
        img = self.images[idx].astype(np.uint8)
        # select label
        if self.labels is not None:
            label = self.labels[idx].astype(np.uint8)
        # select distance transform map
        if self.dtm is not None:
            dtm = self.dtm[idx].astype(np.float32)

        # apply stain augmentation
        if self.stain_aug is not None:
            if self.sc_loss:
                # apply stain consistency augmentation
                img_1, sample_stain, sample_conc, sm, conc, conc_stats = self.stain_aug(img, return_mat=True)
                # reuse sampled concentration matrix for second stain consistency augmentation application
                img_2 = self.stain_aug(img, stain_matrix=sm, conc=conc, conc_stats=conc_stats, sample_stain=None, sample_conc=sample_conc)
            else:
                img = self.stain_aug(img)

        # apply transformations
        if self.transforms is not None:
            if self.labels is not None and self.dtm is not None and self.sc_loss:
                augmentations = self.transforms(image=img_1, image1=img_2, masks=[label,dtm])
                img_1 = augmentations['image']
                img_2 = augmentations['image1']
                label = augmentations['masks'][0]
                dtm = augmentations['masks'][1]
            elif self.labels is not None and self.dtm is not None and not self.sc_loss:
                augmentations = self.transforms(image=img, masks=[label,dtm])
                img = augmentations['image']
                label = augmentations['masks'][0]
                dtm = augmentations['masks'][1]
            elif self.labels is not None:
                augmentations = self.transforms(image=img, mask=label)
                img = augmentations['image']
                label = augmentations['mask']
            elif self.dtm is not None:
                augmentations = self.transforms(image=img, mask=dtm)
                img = augmentations['image']
                dtm = augmentations['mask']
            else:
                augmentations = self.transforms(image=img)
                img = augmentations['image']

        if self.labels is not None:        
            label = torch.moveaxis(label, -1, 0).long()
        if self.dtm is not None:
            dtm = torch.moveaxis(dtm, -1, 0)

        if self.labels is not None and self.dtm is not None and self.sc_loss:
            return img_1, img_2, label, dtm
        elif self.labels is not None and self.dtm is not None:    
            return img, label, dtm
        elif self.labels is not None:
            return img, label
        elif self.dtm is not None:
            return img, dtm
        else:
            return img