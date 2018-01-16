#encoding: utf-8
import mahotas
import numpy as np
import global_functions

def lddp(image, radius1, radius2, points, ignore_zeros=False, preserve_shape=True):
    """
    Custom implementation of 2nd order local directional derivative pattern
    Originally obtained from mahotas.features.lbp_transform function.

    An inner and an outer radius with respect to a point, which is each image pixel are selected.
    Then, a set of points are obtained by interpolation on these radii, according to the number defined by *points*
    argument. Note that if, for example, 8 points are given, there are 8 points that are considered on the inner radius
    defined by equal angles starting from the central point and each one of them. If these two points (the origin, or the
    centre) and each point define a straight line, also 8 points on the same lines are considered for the outer radius.

    For reference see :

    Guo, Zhenhua, et al. "Local directional derivative pattern for rotation invariant texture classification."
    Neural Computing and Applications 21.8 (2012): 1893-1904.

    :param np.array image: numpy array input image
    :param int radius1: inner radius (in pixels)
    :param int radius2: outer radius (in pixels)
    :param int points: number of points to consider. It should be given regarding the inner radius, as the second set of points will be aligned to the ones lying in the inner circle.

    :return: lddp histogram
    """
    from mahotas import interpolate
    from mahotas.features import _lbp
    from mahotas import histogram

    if ignore_zeros and preserve_shape:
        raise ValueError('mahotas.features.lbp_transform: *ignore_zeros* and *preserve_shape* cannot both be used together')

    image = np.asanyarray(image, dtype=np.float64)
    if image.ndim != 2:
        raise ValueError('mahotas.features.lbp_transform: This function is only defined for two dimensional images')

    if ignore_zeros:
        Y,X = np.nonzero(image)
        def select(im):
            return im[Y,X].ravel()
    else:
        select = np.ravel

    pixels = select(image)
    angles = np.linspace(0, 2*np.pi, points+1)[:-1]
    data1 = []
    for dy,dx in zip(np.sin(angles), np.cos(angles)):
        data1.append(
            select(interpolate.shift(image, [radius1*dy,radius1*dx], order=1)))
    data1 = np.array(data1)

    data2 = []
    for dy,dx in zip(np.sin(angles), np.cos(angles)):
        data2.append(
            select(interpolate.shift(image, [radius2*dy,radius2*dx], order=1)))
    data2 = np.array(data2)
    data = np.array(data2 + pixels -2*data1)
    codes = (data >= 0).astype(np.int32)
    codes *= (2**np.arange(points)[:,np.newaxis])
    codes = codes.sum(0)

    codes = _lbp.map(codes.astype(np.uint32), points)
    if preserve_shape:
        codes = codes.reshape(image.shape)

    final = histogram.fullhistogram(codes.astype(np.uint32))

    codes = np.arange(2**points, dtype=np.uint32)
    iters = codes.copy()
    codes = _lbp.map(codes.astype(np.uint32), points)
    pivots = (codes == iters)
    npivots = np.sum(pivots)
    compressed = final[pivots[:len(final)]]
    compressed = np.append(compressed, np.zeros(npivots - len(compressed)))
    return compressed


def chainGLRLM(img, intensity_levels, width):
    """returns normalized grey level run length matrix for already rotated image.

    An image (img input parameter) is given as input. The image has to already be rotated.
    A rotated image looks like this: if the original image is an np.array with shape (512, 512),
    a rotated image at 45 degrees will have 1,2,..., 1024, 1023,...,2,1 row elements.
    This means that the pixels are given as row elements for each diagonal line.
    Then all subsequent occurences of all gray level are counted for each row and returned as the GLRLMat.
    For more information see:

    M. M. Galloway, “Texture analysis using gray level run lengths,” Com-
    puter graphics and image processing, vol. 4, no. 2, pp. 172–179, 1975.

    :param np.array img: Rotated image for which the consecutive occurences of all gray levels are caluclated
    :param int intensity_levels: the number of intensity levels present in an image
    :param int width: the width of the image in pixels

    :returns list: 2 x 2 list containing countings of consecutive occurences of all gray levels
    """
    GLRLMat = [[0 for i in range(width)] for j in range(intensity_levels)]
    for imrow in img:
        for i in range(len(imrow)):
            counter = 0
            try:
                while imrow[i] == imrow[i+1]:
                    counter += 1
                    i += 1
                GLRLMat[imrow[i]][counter] += counter + 1
            except IndexError:
                # GLRLMat[imrow[i]][counter] = counter + 1
                break
    return GLRLMat


def GLRLM_features(p):
    """Returns 5 Grey Level Run Length Features as described in M. Galloway paper

    M. M. Galloway, “Texture analysis using gray level run lengths,” Com-
    puter graphics and image processing, vol. 4, no. 2, pp. 172–179, 1975.

    :param list p: p is the matrix containing gray level run lengths (2D)

    :returns: a list containing all 5 features
    """
    RF = []
    N = sum(sum(i) for i in zip(*p))
    if N == 0: N = 1
    RF.append(0)
    for i in range(len(p)):
        for j in range(len(p[0])):
            RF[0] += (p[i][j] / pow((j + 1), 2))
    RF[0] /= N

    RF.append(0)
    for i in range(len(p)):
        for j in range(len(p[0])):
            RF[1] += (p[i][j] * pow((j + 1), 2))
    RF[1] /= N

    RF.append(0)
    for i in range(len(p)):
        RF[2] += pow(sum(p[i]), 2)
    RF[2] /= N

    RF.append(0)
    pColwise = np.rot90(p, 1)
    for i in range(len(pColwise)):
        RF[3] += pow(sum(pColwise[i]), 2)
    RF[3] /= N

    RF.append(0)
    P = len(p) * len(p[0])
    RF[4] = N / P
    return RF


def GLRLM(img, intensity_levles):
    """calls *chainGLRLM* for rotated versions of input image and returns dictionary with GLRLM features"""
    width = img.shape[0]
    # intenisty_levels = len(mahotas.fullhistogram(img))
    features = []
    # 0 deg
    chains_0 = img
    GRLM_0 = chainGLRLM(chains_0, intensity_levles, width)
    features.append(GLRLM_features(GRLM_0))

    # 45 degrees Grey level run length matrices
    # rotate image
    chains_45 = []
    lenaRot = np.rot90(img, 3)
    for i in range(len(img) - 1, -(len(img) - 1), -1):
        chains_45.append(np.diag(lenaRot, i))
    GRLM_45 = chainGLRLM(chains_45, intensity_levles, width)
    features.append(GLRLM_features(GRLM_45))

    # 90 degrees
    chains_90 = np.rot90(img, 1)
    GRLM_90 = chainGLRLM(chains_90, intensity_levles, width)
    features.append(GLRLM_features(GRLM_90))

    # 135 degrees
    chains_135 = []
    for i in range(len(img) - 1, -(len(img) - 1), -1):
        chains_135.append(np.diag(img, i))
    GRLM_135 = chainGLRLM(chains_135, intensity_levles, width)
    features.append(GLRLM_features(GRLM_135))

    featuresCol = np.rot90(features, 1)
    features1 = []
    features2 = []
    for row in featuresCol:
        features1.append(np.mean(row))
        features2.append((np.ptp(row)))
    features =[list(a) for a in zip(*[features1, features2])]
    return features

def image_histogram_equalization(image, number_bins=256):
    # from http://www.janeriksolem.net/2009/06/histogram-equalization-with-python-and.html

    # get image histogram
    image_histogram, bins = np.histogram(image.flatten(), number_bins, normed=True)
    cdf = image_histogram.cumsum() # cumulative distribution function
    cdf = number_bins * cdf / cdf[-1] # normalize

    # use linear interpolation of cdf to find new pixel values
    image_equalized = np.interp(image.flatten(), bins[:-1], cdf)

    return image_equalized.reshape(image.shape), cdf


def C2(img, return_array = False):
    """returns 2nd order statistics as described in literature review.
    GLRLM and GLCM are calculated over a 5 grey level image to avoid sparse matrices.
    LBP and LDDP are calculated over a 256 grey level image.
    14th Haralick feature is not calculated as mahotas implementation used here is buggy and
    for a sliding window excecution hangs at arbitrary points.
    However, for ROI based aproach, it seems to be working.
    Still, it is not considered safe to use it.

    For references see:

    M. M. Galloway, “Texture analysis using gray level run lengths,” Com-
    puter graphics and image processing, vol. 4, no. 2, pp. 172–179, 1975.

    R. M. Haralick, K. Shanmugam, et al., “Textural features for image clas-
    sification,” IEEE Transactions on systems, man, and cybernetics, no. 6,
    pp. 610–621, 1973.

    Note that for the last reference, all 28 features are calculated.

    :param np.array img: image for which the second order statistical features presented in the above papers are extracted
    :param bool return_array: if True, an array with all features for the specified pixel is returned. This is used to describe the feature vector referring to the specific pixel in a sliding window aproach. Therefore, it should be only set in sliding window aproach, otherwise an exception should be expected. Default is False (in accordance with default values of the main function).

    :returns: dictionary or numpy array with features
    """
    #intensity_levels = np.max(img)
    #img = global_functions.trim_array_to_ROI(img)

    #MERK! LBP and LDDP are ROI based!
    # in case that a strip like ROI occurs
    img = image_histogram_equalization(img, number_bins=256)[0].astype(int)
    intensity_levels = np.max(img)
    original_shape = img.shape
    lbp_1C = mahotas.features.lbp(img, 1, 8)
    lbp_1 = global_functions.pretty(lbp_1C, 'lbp_R1_P8_', len(lbp_1C))
    for entry in lbp_1:
        lbp_1[entry] = global_functions.single_values_to_img({entry:lbp_1[entry]}, img.shape)[entry]
    # lbp_2C = mahotas.features.lbp(img, 2, 16)
    # lbp_2 = global_functions.pretty(lbp_2C, 'lbp_R1_P8_', len(lbp_2C))
    # for entry in lbp_2:
    #     lbp_2[entry] = global_functions.single_values_to_img({entry:lbp_2[entry]}, img.shape)[entry]
    # lddp_1 = global_functions.single_values_to_img({"lddp_R1_1_R2_2_P8": lddp(img, 1, 2, 8)}, img.shape)
    # lddp_2 = global_functions.single_values_to_img({"lddp_R1_2_R2_4_P16": lddp(img, 2, 4, 16)}, img.shape)
    lddp_1C = lddp(img, 1, 2, 8)
    lddp_1 = global_functions.pretty(lddp_1C, "lddp_R1_1_R2_2_P8_", len(lddp_1C))
    for entry in lddp_1:
        lddp_1[entry] = global_functions.single_values_to_img({entry:lddp_1[entry]}, img.shape)[entry]
    # lddp_2C = lddp(img, 2, 4, 16)
    # lddp_2 = global_functions.pretty(lddp_2C, "lddp_R1_1_R2_2_P8_", len(lddp_2C))
    # for entry in lddp_2:
    #     lddp_2[entry] = global_functions.single_values_to_img({entry: lddp_2[entry]}, img.shape)[entry]
    img = image_histogram_equalization(img, number_bins=4)[0].astype(int)
    #MERK! 14 FEATURE IS BUGGY!might be calculated or throw no error but hangs in some images!
    #try:
    #    #haralick = mahotas.features.haralick(img, compute_14th_feature=True)
    #    pass
    #except:
    #MERK! 14 FEATURE IS BUGGY!might be calculated or throw no error but hangs in some images!
    buggy_14 = mahotas.features.haralick(img)
    haralick = np.append(buggy_14, np.ones((4, 1)), axis=1)
    GLCM_features = [[np.mean(a), np.ptp(a)] for a in zip(*haralick)]
    GLRLM_features = GLRLM(img, intensity_levels)
    # to_return = {}
    r1 = global_functions.pretty(GLCM_features, "Haralick_mean_", 14, 0)
    r1 = global_functions.single_values_to_img(r1, original_shape)
    r2 = global_functions.pretty(GLCM_features, "Haralick_range_", 14, 1)
    r2 = global_functions.single_values_to_img(r2, original_shape)
    # for i in range(13,-1,-1):
        # to_return["haralick" + str(i+1) + "mean"] = GLCM_features[i][0]
        # to_return["haralick" + str(i+1) + "range"] = GLCM_features[i][1]

    r3 = global_functions.pretty(GLRLM_features, "GLRLM_mean_", 5, 0)
    r3 = global_functions.single_values_to_img(r3, original_shape)
    r4 = global_functions.pretty(GLCM_features, "GLRLM_range_", 5, 1)
    r4 = global_functions.single_values_to_img(r4, original_shape)

    # for i in range(4,-1,-1):
    #     to_return["GLRLM" + str(i+1) + "mean"] = GLRLM_features[i][0]
    #     to_return["GLRLM" + str(i+1) + "range"] = GLCM_features[i][1]


    to_return = global_functions.merge_dictionairies(r1, r2, r3, r4, lbp_1, lddp_1)
    if return_array:
        to_return = np.array(to_return.values())[:,0,0] #for sliding window
    # C2_ext = global_functions.single_values_to_img(to_return, img.shape)
    return to_return
