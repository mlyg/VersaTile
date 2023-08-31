# library import
import numpy as np
import skfmm
from numpy import ma
from skimage.morphology import skeletonize
from scipy.spatial.distance import cdist


def get_skeleton_centre(skeleton):
    """Get central coordinates of a skeleton"""
    indices = np.argwhere(skeleton == 1)
    distances = cdist(indices, indices)
    sum_distances = np.sum(distances, axis=1)
    center_index = np.argmin(sum_distances)
    skeleton_centre = np.ones_like(skeleton).astype(int)
    skeleton_centre[tuple(indices[center_index])] = -1
    return skeleton_centre


def compute_hybrid_dtm(labels, alpha=0.5):
    """Computes hybrid DTM using both skeleton centroid and EDT from skeleton"""
    labels = np.squeeze(labels)
    
    dtm = np.zeros_like(labels).astype(float)
    
    for i in range(1, labels.max()+1):
        mask = np.where(labels==i,1,0)
        
        skeleton = skeletonize(mask, method='lee')
        skeleton_centre = get_skeleton_centre(mask)
        skeleton_centre_masked = ma.masked_array(skeleton_centre, mask=1-mask)
        
        skeleton_map = np.where(skeleton,-1, 1)
        skeleton_map_masked = ma.masked_array(skeleton_map, mask=1-mask)
        
        try:
            skeleton_edt = skfmm.distance(skeleton_map_masked)
            skeleton_centre_edt = skfmm.distance(skeleton_centre_masked)
        except ValueError:
            continue
        
        skeleton_edt.mask = False
        skeleton_centre_edt.mask = False
        
        skeleton_edt = (skeleton_edt - np.min(skeleton_edt))/(np.max(skeleton_edt)- np.min(skeleton_edt) + 1e-5)
        skeleton_centre_edt = (skeleton_centre_edt - np.min(skeleton_centre_edt))/(np.max(skeleton_centre_edt)- np.min(skeleton_centre_edt) + 1e-5)
        
        combined =  alpha*skeleton_edt + (1-alpha)*(skeleton_centre_edt)
        
        combined = (1 - combined) * mask        
        
        dtm += combined
    
    return dtm
