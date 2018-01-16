#encoding:utf-8
import numpy as np
import global_functions

def value_range(window):
    x = np.max(window) - np.min(window)
    return x


# def C5(img, mode=False):
#     """Returns fractal dimension features.
#
#     The algorithms that have been deployed to extract fractal features is the popular 3D box counting.
#     In the following reference:
#
#     R. Lopes, A. Ayache, N. Makni, P. Puech, A. Villers, S. Mordon, and
#     N. Betrouni, “Prostate cancer characterization on mr images using fractal
#     features,” Medical physics, vol. 38, no. 1, pp. 83–95, 2011.
#
#     It is recommended that 8-tap Daubechie wavelet coefficients could be used to estimate the fractal dimension (FD)
#     but this method has not been implemented. 3D box counting is presented in:
#
#     B. Stark, M. Adams, D. Hathaway, and M. Hagyard, “Evaluation of two
#     fractal methods for magnetogram image analysis,” Solar Physics, vol. 174,
#     no. 1-2, pp. 297–309, 1997.
#
#     Fractal dimmensions are estimated for the image itself given as input *img* and the histogram of the same image,
#     a method recommended in
#
#     D. Lv, X. Guo, X. Wang, J. Zhang, and J. Fang, “Computerized charac-
#     terization of prostate cancer by fractal analysis in mr images,” Journal of
#     magnetic resonance imaging, vol. 30, no. 1, pp. 161–168, 2009.
#
#     Implementation of the algorithm has been properly adapted from an implementation by Francesco Turci,
#     retrieved online from https://francescoturci.wordpress.com/2016/03/31/box-counting-in-numpy/ on 10 Feb 2017.
#
#     Argument *mode*'s default value is false. If *mode* is true, a sliding window box counting estimation is returned for
#     the image FD. Sliding window estimation might be more precise, but it takes way more time, as it invokes
#     *global_functions.pixelwise_features* for each pixel for each scale and then calculates the sum for each one of them.
#     Implementation from Francesco Turci uses a fixed window implementation. Difference between sliding and fixed windows
#     can be seen in the following figures, where on the upper side a sliding window is presented and on the down side a fixed one.
#     These images have been retrieved from:
#     https://imagej.nih.gov/ij/plugins/fraclac/FLHelp/BoxCounting.htm on 15 Feb 2017
#
#     .. figure:: fixed.*
#        :align: center
#        :figclass: align-center
#
#        Fixed window
#
#     .. figure:: sliding.*
#        :align: center
#        :figclass: align-center
#
#        Sliding window
#
#     :param np.array img: Image the fractal dimension of which is estimated. Also, image histogramm's FD is estimated.
#     :param int intensity_levels: represents the dataset global maximum of intensity levels
#     :param bool mode: Sliding window method for True, Fixed window for False
#     :return: dictionary with image and histogram FDs
#     """
#
#     # considering only scales in a logarithmic list
#     intensity_levels = np.max(img)
#     G = np.max(img)
#     M = img.shape[0]
#     scales = np.logspace(1, 8, num=20, endpoint=False, base=2)
#     Ns = []
#     # looping over several scales
#     if mode:
#         # sliding window if mode, else, implementation by Turci.
#         for s in scales:
#             #H, edges = np.histogramdd(img, bins=(np.arange(0, Lx, scale), np.arange(0, Ly, scale)))
#             d_s = global_functions.pixelwise_features(img, s, value_range)
#             N_scale = np.sum(np.ceil((M*d_s)/G))
#             Ns.append(N_scale)
#             # linear fit, polynomial of degree 1
#     else:
#         Lx = img.shape[0]
#         Ly = img.shape[1]
#         for s in scales:
#             step = int(np.ceil(s))
#             N_scale = 0
#             for i in range(0, Lx, step ):
#                 for j in range(0, Ly, step):
#                     support_field = img[i:i+step, j:j+step]
#                     N_scale += np.ceil(M * (np.max(support_field) - np.min(support_field)) / G)
#             Ns.append(N_scale)
#
#     #warnings.simplefilter('error')
#     try:
#         coeffs = np.polyfit(np.log(scales), np.log(Ns), 1)
#     except:
#         pass
#
#     # repeating the above algorithm for 1D signal (histogram)
#     Ns_hist = []
#     hist = np.histogram(img, intensity_levels)[0]
#     N = len(hist)
#     G_hist = max(hist)
#     for s in scales:
#         N_scale = 0
#         step = int(np.ceil(s))
#         for i in range(0, N, step):
#             try:
#                 support_field = hist[i:i+step]
#                 N_scale += np.ceil(N * (max(support_field) - min(support_field)) / G_hist)
#             except IndexError:
#                 break
#         Ns_hist.append(N_scale)
#
#     coeffs_hist = np.polyfit(np.log(scales), np.log(Ns_hist), 1)
#
#     C5_basic = {"image_FD": -coeffs[0], "histogram_FD":-coeffs_hist[0]}
#     if np.isnan(coeffs[0]) or np.isnan(coeffs_hist[0]):
#         pass
#     C5_ext = global_functions.single_values_to_img(C5_basic, img.shape)
#     return C5_ext

def C5(img, return_array = False):

    """Returns fractal dimension features.

    The algorithms that have been deployed to extract fractal features is the popular 3D box counting.
    In the following reference:

    R. Lopes, A. Ayache, N. Makni, P. Puech, A. Villers, S. Mordon, and
    N. Betrouni, “Prostate cancer characterization on mr images using fractal
    features,” Medical physics, vol. 38, no. 1, pp. 83–95, 2011.

    It is recommended that 8-tap Daubechie wavelet coefficients could be used to estimate the fractal dimension (FD)
    but this method has not been implemented. 3D box counting is presented in:

    B. Stark, M. Adams, D. Hathaway, and M. Hagyard, “Evaluation of two
    fractal methods for magnetogram image analysis,” Solar Physics, vol. 174,
    no. 1-2, pp. 297–309, 1997.

    Fractal dimmensions are estimated for the image itself given as input *img* and the histogram of the same image,
    a method recommended in

    D. Lv, X. Guo, X. Wang, J. Zhang, and J. Fang, “Computerized charac-
    terization of prostate cancer by fractal analysis in mr images,” Journal of
    magnetic resonance imaging, vol. 30, no. 1, pp. 161–168, 2009.

    Implementation of the algorithm has been properly adapted from an implementation by Francesco Turci,
    retrieved online from https://francescoturci.wordpress.com/2016/03/31/box-counting-in-numpy/ on 10 Feb 2017.

    :param np.array img: Image the fractal dimension of which is estimated. Also, image histogramm's FD is estimated.
    :param int intensity_levels: represents the dataset global maximum of intensity levels
    :param bool return_array: if True, an array with all features for the specified pixel is returned. This is used to describe the feature vector referring to the specific pixel in a sliding window aproach. Therefore, it should be only set in sliding window aproach, otherwise an exception should be expected. Default is False (in accordance with default values of the main function).

    :returns: dictionary with image and histogram FDs or numpy array in sliding window mode
    """
    intensity_levels = int(np.max(img))
    global_max  = np.max(img)
    scales = np.logspace(1, 8, num=20, endpoint=False, base=2)
    Ns = []
    # looping over several scales
    Lx = img.shape[0]
    Ly = img.shape[1]
    for s in scales:
        step = int(np.ceil(s))
        s_prime = global_max * float(s)/float(Lx)
        N_scale = 0
        for i in range(0, Lx, step ):
            for j in range(0, Ly, step):
                support_field = img[i:i+step, j:j+step]
                N_scale += (np.max(support_field) - np.min(support_field))
        Ns.append(float(N_scale)/float(s_prime))

    #warnings.simplefilter('error')
    coeffs = np.polyfit(np.log(scales), np.log(Ns), 1)

    # repeating the above algorithm for 1D signal (histogram)
    Ns_hist = []
    hist = np.histogram(img, intensity_levels)[0]
    N = len(hist)
    global_max = max(hist)
    for s in scales:
        N_scale = 0
        s_prime = global_max * float(s) / float(N)
        step = int(np.ceil(s))
        for i in range(0, N, step):
            try:
                support_field = hist[i:i+step]
                N_scale += (max(support_field) - min(support_field))
            except IndexError:
                break
        Ns_hist.append(float(N_scale)/float(s_prime))

    coeffs_hist = np.polyfit(np.log(scales), np.log(Ns_hist), 1)

    C5_basic = {"image_FD": -coeffs[0], "histogram_FD":-coeffs_hist[0]}
    if np.isnan(coeffs[0]) or np.isnan(coeffs_hist[0]):
        pass
    C5_ext = global_functions.single_values_to_img(C5_basic, img.shape)
    if return_array:
        to_return = np.array(C5_ext.values())[:, 0, 0]  # for sliding window
    else:
        to_return = C5_ext
    return to_return

