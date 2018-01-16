# encoding: utf-8
from __future__ import division

import numpy as np
from scipy.ndimage import interpolation
from scipy.stats import skew, kurtosis
import global_functions


def window_statistics(window, intensity_levels):
    """Calculates first order statistical radiomic features.

    Features are described in detail in literature review. These include first four order moments,
    and some special features, the reference of which can be find in the block comments in the code.
    All features are calclulated for a neighborhood of size *ws x ws* around each pixel.
    Parameter ws is defined in function *C1.C1* source code which in turn calls C1.window_statistics via *global_functions.pixelwise_features*
    giving as input the window, rather than the entire image. Then this function is called for a window
    of size ws x ws for each pixel in the image and for this window (referring to a single pixel) all features
    are returned in form of a dictionary.

    :param np.array window: window on which features are calculated
    :param int intensity_levels: represents the dataset global maximum of intensity levels
    :returns: dictionary with first order statistics, shape of the image containing 1x4 lists
    """
    mean  = np.mean(window)
    N = (window.shape[0]*window.shape[1])
    intensity_pdf = np.array([a/N for a in list(np.histogram(window, intensity_levels)[0])])
    # gray values range between 1-256 is assumed- fixed in current version
    std = np.sum(((i - mean)**2)*p for p, i in zip(intensity_pdf, range(len(intensity_pdf))))
    skew = np.sum(((i - mean)**3)*p for p, i in zip(intensity_pdf, range(len(intensity_pdf))))
    kurtosis = np.sum(((i - mean)**4)*p for p, i in zip(intensity_pdf, range(len(intensity_pdf))))

    # A.Madabhushi, M.D.Feldman, D.N.Metaxas, J.Tomaszeweski, and
    # D.Chute, “Automated detection of prostatic adenocarcinoma from high
    # -resolution ex vivo mri,” IEEE transactions on medical imaging, vol. 24,
    # no.12, pp.1611–1625, 2005.

    f_1 = np.median(window)
    f_2 = np.sqrt(np.sum((window - f_1)**2))/N
    f_3 = abs(f_1 - np.sum(window)/N)
    # F. Khalvati, A. Modhafar, A. Cameron, A. Wong, and M. A. Haider,
    # “A multi-parametric diffusion magnetic resonance imaging texture feature
    # model for prostate cancer analysis,” in Computational Diffusion MRI,
    # pp. 79–88, Springer, 2014
    remove_zeros = intensity_pdf[np.nonzero(intensity_pdf)]
    En = -np.sum((np.log2(a)*a for a in remove_zeros))
    U = np.sum((a**2 for a in intensity_pdf))
    # J. T. Kwak, S. Xu, B. J. Wood, B. Turkbey, P. L. Choyke, P. A. Pinto,
    # S. Wang, and R. M. Summers, “Automated prostate cancer detection
    # using t2-weighted and high-b-value diffusion-weighted magnetic resonance
    # imaging,” Medical physics, vol. 42, no. 5, pp. 2368–2378, 2015.
    radius1 = 1
    points = 8
    # in the above reference, authors use local binary patterns. the radius and points selected are the same as for lbp
    # which is calculated  and documented in C3 and in the literature review.
    angles = np.linspace(0, 2*np.pi, points+1)[:-1]
    data1 = []
    for dy,dx in zip(np.sin(angles), np.cos(angles)):
        # data1.append(
        #     np.ravel(interpolate.shift(window, [radius1*dy,radius1*dx], order=1)))
        data1.append(
            np.ravel(interpolation.shift(window, [radius1*dy,radius1*dx], order=1)))
    data1 = np.array(data1)
    var_lbp = np.var(data1)
    return [mean, std, skew, kurtosis, f_1, f_2, f_3, var_lbp, En, U]

#
# def C1(img):
#     """Returns first order features. See C1.window_statistics for more
#
#     *ws* is an important parameter (window size) and it is defined within the function source code.
#     It refers to a window which is *wsxws* pixels. Preset value is 9x9.
#
#     .. seealso:: documentation of *C1.window_statistics*
#
#
#     :param np.array img: the image about which features are calculated
#     :param int intensity_levels: represents the dataset global maximum of intensity levels
#     :returns: dictionary with all features
#     """
#     ws = 9
#     intensity_levels = np.max(img)
#     features = global_functions.pixelwise_features(img, ws, window_statistics, 10, intensity_levels)
#     mean = global_functions.pixelwise_features(img, ws, np.mean)
#     # return {"first order":features}
#     C1_features = {"mean":features[:,:,0], "standard_deviation":features[:,:,1], "skewness":features[:,:,2], "kurtosis":features[:,:,3],\
#             "median1":features[:,:,4], "median2":features[:,:,5], "median3":features[:,:,6], "lbp_like_variance":features[:,:,7],\
#             "entropy":features[:,:,8], "U":features[:,:,9]}
#     #C1_Ext = global_functions.pixel_values_to_img(C1_features, img.shape)
#     return C1_features

def f2(window):
    f_2 = np.sqrt(np.sum((window - np.median(window)) ** 2)) / window.size
    return f_2

def f3(window):
    f_3 = abs(np.median(window) - np.sum(window) / window.size)
    return f_3

def lbp_var(window):
    #scipy ndimage returns flattened array
    orig_shape = int(np.sqrt(window.size))
    window = np.reshape(window, (orig_shape, orig_shape))
    radius1 = 1
    points = 8
    # in the above reference, authors use local binary patterns. the radius and points selected are the same as for lbp
    # which is calculated  and documented in C3 and in the literature review.
    angles = np.linspace(0, 2*np.pi, points+1)[:-1]
    data1 = []
    for dy,dx in zip(np.sin(angles), np.cos(angles)):
        # data1.append(
        #     np.ravel(interpolate.shift(window, [radius1*dy,radius1*dx], order=1)))
        data1.append(
            np.ravel(interpolation.shift(window, [radius1*dy,radius1*dx], order=1)))
    data1 = np.array(data1)
    var_lbp = np.var(data1)
    return var_lbp

def En(window, intensity_levels):
    #intensity_levels = intensity_levels[0]
    N = window.size
    intensity_pdf = np.array([float(a) / float(N) for a in list(np.histogram(window, intensity_levels)[0])])
    remove_zeros = intensity_pdf[np.nonzero(intensity_pdf)]
    E_n = -np.sum((np.log2(a) * a for a in remove_zeros))
    return E_n

def U(window, intensity_levels):
    N = window.size
    intensity_pdf = np.array([float(a) / float(N) for a in list(np.histogram(window, intensity_levels)[0])])
    U_ = np.sum((a**2 for a in intensity_pdf))
    return U_

def C1(img):
    """Returns first order features. See C1.window_statistics for more

    *ws* is an important parameter (window size) and it is defined within the function source code.
    It refers to a window which is *wsxws* pixels. Preset value is 9x9.

    .. seealso:: documentation of *C1.window_statistics*


    :param np.array img: the image about which features are calculated
    :param int intensity_levels: represents the dataset global maximum of intensity levels

    :returns: dictionary with all features
    """
    ws = 9
    intensity_levels = np.max(img)
    #features = global_functions.pixelwise_features(img, ws, window_statistics, 10, intensity_levels)
    mean_v = global_functions.pixelwise_features(img, ws, np.mean)
    std_dev = global_functions.pixelwise_features(img, ws, np.std)
    skewness = global_functions.pixelwise_features(img, ws, skew)
    kurt = global_functions.pixelwise_features(img, ws, kurtosis)
    med1 = global_functions.pixelwise_features(img, ws, np.median)
    med2 = global_functions.pixelwise_features(img, ws, f2)
    med3 = global_functions.pixelwise_features(img, ws, f3)
    var_lbp = global_functions.pixelwise_features(img, ws, lbp_var)
    E_n = global_functions.pixelwise_features(img, ws, En, intensity_levels)
    U_ = global_functions.pixelwise_features(img, ws, U, intensity_levels)

    # return {"first order":features}
    C1_features = {"mean":mean_v, "standard_deviation":std_dev, "skewness":skewness, "kurtosis":kurt,\
            "median1":med1, "median2":med2, "median3":med3, "lbp_like_variance":var_lbp,\
            "entropy":E_n, "U":U_}
    #C1_Ext = global_functions.pixel_values_to_img(C1_features, img.shape)
    return C1_features
