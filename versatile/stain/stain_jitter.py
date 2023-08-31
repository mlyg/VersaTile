import numpy as np
from scipy import linalg


class StainJitter():
    def __init__(self, stain, alpha=0.25, beta=0.05):
        if stain == 'he':
            self.rgb_from_stain = np.array([[0.644211, 0.716556, 0.266844],
                                  [0.092789, 0.954111, 0.283111],
                                  [0., 0., 0.]])
            self.rgb_from_stain[2, :] = np.cross(self.rgb_from_stain[0, :], self.rgb_from_stain[1, :])
            
        elif stain == 'masson':
            self.rgb_from_stain = np.array([[0.7995107, 0.5913521, 0.10528667],
                                       [0.09997159, 0.73738605, 0.6680326],
                                       [0., 0., 0.]])
            self.rgb_from_stain[2, :] = np.cross(self.rgb_from_stain[0, :], self.rgb_from_stain[1, :])
            
        self.alpha = alpha
        self.beta = beta
        
        
    def _separate_stains(self, img):
        # convert image to float
        img = img.astype(np.float64)
        # change image to scale 0-1
        img /= 255.
        np.maximum(img, 1e-6, out=img)  # avoiding log artifacts
        log_adjust = np.log(1e-6)  # used to compensate the sum above
        # augment stain_from_rgb
        stains = (np.log(img) / log_adjust) @ self.stain_from_rgb
        np.maximum(stains, 0, out=stains)
        return stains
    
    def _combine_stains(self, stains):
        log_adjust = -np.log(1e-6)
        log_img = -(stains * log_adjust) @ self.rgb_from_stain
        img = np.exp(log_img)
        img = np.clip(img, a_min=0, a_max=1)
        img = (img * 255).astype(np.uint8)
        return img
    
    def _sample(self):
        sample_alpha = np.random.uniform(1-self.alpha,1+self.alpha,3)
        sample_beta = np.random.uniform(-self.beta,self.beta,3)
        return sample_alpha, sample_beta
    
    # to be compatible with Dataset
    def preprocess(self, images):
        pass

    def __call__(self, img):
        
        sample_alpha, sample_beta = self._sample()
        
        # extract stains
        stains = self._separate_stains(img)
        
        # augment stain
        stains *= sample_alpha
        stains += sample_beta
        
        img = self._combine_stains(stains)
         
        return img