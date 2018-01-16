#encoding: utf-8
#from __future__ import division
import math
import numpy as np
import skimage.feature
from scipy import ndimage
from skimage.filters import gabor
import global_functions
from itertools import izip




def li(img):
    """Implementation of the Li filter

    Filter is described in detail in:

    Q. Li, S. Sone, and K. Doi, “Selective enhancement filters for nodules,
    vessels, and airway walls in two-and three-dimensional ct scans,” Medical
    physics, vol. 30, no. 8, pp. 2040–2051, 2003.

    Roughly, it is a dot and line sensitive filter.

    :param np.array img: image on which the filter is applied

    :returns: a list containing two numpy arrays. The first is the filter that refers to dots, while the second refers to lines. Each of these has the shape of the image (a value for each pixel)
    """
    li_filter_dot = np.zeros(img.shape)
    li_filter_line = np.zeros(img.shape)
    Hxx, Hxy, Hyy = skimage.feature.hessian_matrix(img)
    l1, l2 = skimage.feature.hessian_matrix_eigvals(Hxx, Hxy, Hyy)
    for i in range(len(l1)):
        for j in range(len(l1[0])):
            if l1[i,j]<0 and l2[i,j]<0:
               li_filter_dot[i, j] = l2[i,j]**2/abs(l1[i,j])
            if l1[i,j]<0:
                li_filter_line[i,j] = abs(l1[i,j]) - abs(l2[i,j])
    return li_filter_dot, li_filter_line



def C3(img, resolution):

    """Returns all gradient features

    All the gradient features include the following:

    - Kirsch gradients

    - Roberts gradients

    - Sobel filter

    - Unoriented gradients

    - local binary patterns

    - local directional derivative patterns

    - Gabor filter coefficients

    - Gaussian blurred images

    Note that for the Gabor and Gaussian filters, scales range between 2 and 8 mm as to capture scales within
    which there could be a lesion. This method is mentioned in:

    G. Litjens, O. Debats, J. Barentsz, N. Karssemeijer, and H. Huisman,
    “Computer-aided detection of prostate cancer in mri,” IEEE transactions
    on medical imaging, vol. 33, no. 5, pp. 1083–1092, 2014.

    :param np.array img: the image about which features are calculated
    :param float resolution: represents the physical dimension to which a pixel corresponds

    :return: dictionary with all gradient features.
    """

    # Kirsch kernels
    g = [[[-3, -3, -3], [-3, 0, -3], [5, 5, 5]],
         [[-3, -3, -3], [5, 0, -3], [5, 5, -3]],
         [[5, -3, -3], [5, 0, -3], [5, -3, -3]],
         [[5, 5, -3], [5, 0, -3], [-3, -3, -3]],
         [[5, 5, 5], [-3, 0, -3], [-3, -3, -3]],
         [[-3, 5, 5], [-3, 0, 5], [-3, -3, -3]],
         [[-3, -3, 5], [-3, 0, 5], [-3, -3, 5]],
         [[-3, -3, -3], [-3, 0, 5], [-3, 5, 5]]]

    kirsch = [ndimage.filters.convolve(img, g[i]) for i in range(8)]
    kirsch = global_functions.pretty(kirsch, "Kirsch_kernel_rotation_", 8)

    # Roberts Kernels
    g = [[[-1, 0], [0, 1]],
         [[0, 1], [-1, 0]]]
    roberts = [ndimage.filters.convolve(img, g[i]) for i in range(2)]
    roberts = global_functions.pretty(roberts, "Roberts_kernel_rotation_", 2)

    # Sobel filter
    sobel = ndimage.filters.sobel(img)
    sobel = {"Sobel filter": sobel}

    # Gradients without orientation
    grd = np.gradient(img)
    grd = global_functions.pretty(grd, "Unoriented_gradients_axis_", 2)

    # Gabor coefficients
    # Frequencies exponentially (power of 2) ranging between 2 and 8 mm
    # in order to change these scales one can change start and stop params
    # 8 scales are now considered
    # sigma is standard 1 and 3 scales per orientation
    start_res = np.log2(2/resolution[0])
    stop_res = np.log2(8/resolution[0])
    scale_exponents = np.linspace(start_res, stop_res, 8)
    resolutions = 2**scale_exponents

    angles = [0, (45*math.pi/180), (90*math.pi/180), (135*math.pi/180)]
    str_angles = ['0', '45', '90', '135']
    gbr = {}
    i = 0
    for res in resolutions:
        for theta, str_angle in izip(angles, str_angles):
            #gabor filter yields only real coefficients, thus we only keep the real part
            real_gabor = gabor(img, res, theta)[0]
            real_gabor = {"Gabor_scale"+str(i)+"_direction_"+str_angle:real_gabor}
            gbr.update(real_gabor)
        i += 1
    # Gaussian filters for the same resolution range as for gabor filters
    # exponentially ranging sigmas between 2 and 8 mm
    gauss = [ndimage.filters.gaussian_filter(img, s) for s in resolutions]

    li_dot = []
    li_line = []
    for blurred in gauss:
        li_1, li_2 = li(blurred)
        li_dot.append(li_1)
        li_line.append(li_2)
    li_dot = global_functions.pretty(li_dot, "Li_filter_dots_scale_", 8)
    li_line = global_functions.pretty(li_line, "Li_filter_lines_scale_", 8)
    gauss = global_functions.pretty(gauss, "Gauss_scale_", 8)

    # rotation invariant local binary patterns as described in ojala paper
    # in kwak they don't mention radius and number of points, so it is a question
    # dimensionality is that of the occuring histogram!


    C3_features = global_functions.merge_dictionairies(kirsch, gauss, roberts, sobel, grd, gbr, li_dot, li_line)

    # C3 = {"kirsch":kirsch, "gauss":gauss, "roberts":roberts, "sobel":sobel, "plane_gradient":grd,\
    #       "gabor":gbr, "lbp_1_8":lbp_1, "lbp_2_16":lbp_2, "lddp_1_2_8":lddp_1, "lddp_2_4_16":lddp_2,\
    #       "li_filter_dot": li_dot, "li_filter_line":li_line}
    return C3_features