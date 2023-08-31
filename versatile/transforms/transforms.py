import albumentations as A
from albumentations.pytorch import ToTensorV2

def train_transforms(size=256, sc_loss=False):
    if sc_loss:
        transforms = A.Compose([A.Resize(height=size, width=size),
                    A.RandomResizedCrop(height=size,width=size, p=0.25),
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.RandomRotate90(p=0.5),
                    A.OneOf([
                                A.CLAHE(p=0.25),
                                A.RandomGamma(p=0.25),
                                A.RandomBrightnessContrast(p=0.25)
                    ],p=1),
                    A.Blur(p=0.25),
                    A.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225],
                                max_pixel_value=255.0),
                    ToTensorV2()
                    ],
                    additional_targets={'image1':'image'})
    else:
        transforms = A.Compose([A.Resize(height=size, width=size),
                    A.RandomResizedCrop(height=size,width=size, p=0.25),
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.RandomRotate90(p=0.5),
                    A.OneOf([
                                A.CLAHE(p=0.25),
                                A.RandomGamma(p=0.25),
                                A.RandomBrightnessContrast(p=0.25)
                    ],p=1),
                    A.Blur(p=0.25),
                    A.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225],
                                max_pixel_value=255.0),
                    ToTensorV2()
                    ])
    return transforms

def val_transforms(size=256):
    transforms = A.Compose([A.Resize(height=size, width=size),
                                A.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225],
                                            max_pixel_value=255.0),
                                ToTensorV2()])
    return transforms

