#encoding:utf-8
from __future__ import division
import numpy as np
import pywt
from scipy.fftpack import dct
import global_functions
from scipy.ndimage.interpolation import zoom

def fix_shape(wavelet_coeffs, original_shape):

    coeffs_shape = wavelet_coeffs.shape
    scale_0 = original_shape[0]/coeffs_shape[0]
    # scale_1 = original_shape[1]/coeffs_shape[1]
    #if not scale_0 == scale_1: raise ValueError
    rescaled = zoom(wavelet_coeffs, scale_0)
    diff = rescaled.shape[0] - original_shape[0]
    if diff<0:
        rescaled = np.pad(rescaled, [[0, abs(diff)], [0, 0]], 'edge')
    elif diff>0:
        rescaled = rescaled[:original_shape[0], :]
    diff = rescaled.shape[1] - original_shape[1]
    if diff<0:
        rescaled = np.pad(rescaled, [[0, 0], [0, abs(diff)]], 'edge')
    elif diff>0:
        rescaled = rescaled[:, :original_shape[1]]

    if not rescaled.shape == original_shape:
        rescaled = np.reshape(rescaled, original_shape) # has a third axis corresponding to real and imaginary party for Daubechies, but is 0 for imaginary allways
    return rescaled

def pad_to_shape(coeffs, original_shape):
    """"pad with zeors to get the original slice shape for each scale. return_arry[0,0] and [0,1] are the original dims"""
    return_array = np.zeros(original_shape)
    img_shape = coeffs.shape
    if img_shape[0]>=original_shape[0] and img_shape[1]>=original_shape[1]:
        return np.reshape(coeffs, img_shape)
    return_array[0,0] = img_shape[0]
    return_array[0,1] = img_shape[1]
    return_array[1:img_shape[0]+1, 1:img_shape[1]+1] = coeffs
    return return_array

def get_original(array, family):

    not_affected = ['Daubechies_HL', 'Daubechies_HH']
    if family in not_affected:
        return array
    dimension1 = int(array[0,0])
    dimension2 = int(array[0,1])
    return_array = array[1:dimension1, 1:dimension2]
    return return_array

def C4(img):
    """Returns Daubechies and Haar wavelet coefficients

    Daubeschies wavelet coefficients are averaged on a 7 x 7 window as mentioned in:

    R. Lopes, A. Ayache, N. Makni, P. Puech, A. Villers, S. Mordon, and
    N. Betrouni, “Prostate cancer characterization on mr images using fractal
    features,” Medical physics, vol. 38, no. 1, pp. 83–95, 2011.

    Depending on the input image, the maximum level of scale decomposition might vary.
    This depends on pywavelet library that is used to extract the coefficients.
    """
    # 8 vanishing moments daubechies wavelets up to level 6 (maximum allowed by pywt)
    ws = 7
    coefficients = pywt.wavedec2(img, 'db4', level=1) # not entering the level will use maxlevel
    daubechies = coefficients
    # questionable_FD_approximation = coefficients
    daubechies =[list(a_tuple) for a_tuple in daubechies[1:]]
    daubechies.insert(0, coefficients[0])
    for i in range(1,len(coefficients)):
        for j in range(len(coefficients[i])):
            # daubechies[i][j] = ndimage.uniform_filter(coefficients[i][j], ws)
            # averaging over 7x7 neighborhood
            daubechies[i][j] = global_functions.pixelwise_features(img, ws, np.mean)

    dct_coeffs = {"DCT":dct(img)}
    original_shape = img.shape
    haar = pywt.wavedec2(img, 'haar', level=1)
    haar_0 = {"Haar_LL":fix_shape(haar[0], original_shape)}
    haar_1 = {"Haar_LH":fix_shape(haar[1][0], original_shape)}
    haar_2 = {"Haar_HL":fix_shape(haar[1][1], original_shape)}
    haar_3 = {"Haar_HH":fix_shape(haar[1][2], original_shape)}
    daub_0 = {"Daubechies_LL":fix_shape(daubechies[0], original_shape)}
    daub_1 = {"Daubechies_LH":fix_shape(daubechies[1][0], original_shape)}
    daub_2 = {"Daubechies_HL":fix_shape(daubechies[1][1], original_shape)}
    daub_3 = {"Daubechies_HH":fix_shape(daubechies[1][2], original_shape)}
    C4 = global_functions.merge_dictionairies(dct_coeffs, haar_0, haar_1, haar_2, haar_3, daub_0, daub_1, daub_2, daub_3)
    return C4

