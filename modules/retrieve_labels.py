# encoding:utf-8
import numpy as np
from scipy.ndimage.morphology import binary_dilation, binary_erosion
from skimage.measure import label
import global_functions


def get_ROI_label(img):
    # gets as input as single ROI image called from global functions.ROI_based and returns the GS corresponding to that
    GS = img[np.nonzero(img)]
    if GS.any():
        GS = GS[0]
    else:
        GS = 0
    GS_dic = {'ortho_ROI_GS':GS}
    GS_dic = global_functions.single_values_to_img(GS_dic, img.shape)
    return GS_dic

def get_labels(ROIs):

    orthogonalized_ROIS = global_functions.ROI_based_calclulations(ROIs, ROIs, get_ROI_label)
    real_ROIS= {'ROI_GS':ROIs}
    to_return = global_functions.merge_dictionairies(orthogonalized_ROIS, real_ROIS)
    return to_return