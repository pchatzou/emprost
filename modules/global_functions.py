#encoding: utf-8
import numpy as np
import numpy.ma as ma
import itertools
from scipy import ndimage
from skimage.measure import label
import re

def regex_match(necessity, string_to_match):

    """"Gathering all regex matches used throughout the code to make it applicable to other datasets by modifying.
    Therefore, if we have to train on a new dataset where sequence description is in a new format, only this
    function has to be editted and all regex matches used to make the dataset mapping will be up to date. Generally
    regex are extensivelly used to navigate through map files and identify if something is a directory, etc. In short,
    it is used to find what is what.

    ..note:: Downside is that this script will have to be imported to other scripts under module directory.
             This makes it impossible to import any script from modules directory to *global_functions* script,
             otherwise mutual import will happen and an exception will be raised.

    :param string necessity: can be *patient image*, *patient image*, *axial T2*, *axial DWI*, *windows or unix directory*
    It indicates what we are looking for, which means that we are questioning if a string is a directory or an image etc.
    :param string_to_match:

    :returns: Bool, True if the string is what we are looking for, false otherwise.
    """
    expressions = {'patient image':re.compile(r'^PC-[0-9]{10}.*?.nii$|^aa[a,b]0*[0-9]{2}.*?.nii$|^01\.00[0-9]{2}$'),
                   'patient directory': re.compile(r'^PC-[0-9]{10}$|^01.[0-9]{4}$|^aa[a,b]0*[0-9]{2}$'),
                   'axial T2':re.compile(r'(ax.*?t2|t2.*?ax)(?!.*pelvis.*$)', re.IGNORECASE),
                   'axial DWI': re.compile(r'diff.*?adc|adc.*?diff|^dADC|apparent diff', re.IGNORECASE),
                   'windows or unix directory': re.compile(r'^[A-Z]:.*?|^/.*?')
                   }
    if re.match(expressions[necessity], string_to_match) is not None:
        return True
    else:
        return False

def pretty(data, string, fun_range, *indeces):

    value_dic = {}
    for i in range(fun_range-1,-1,-1):
        the_value = data[i]
        for idx in indeces:
            the_value = the_value[idx]
        value_dic[string + str(i+1)] = the_value
    return value_dic

def merge_dictionairies(*dics):

    x = {}
    for dic in dics:
        x.update(dic)
    return x


def custom_pixelwise_features(img, ws, function, *extra_args):
    """Returns features for the windows within the image calculated by function.

    It is a common issue that many features have to be calculated for each pixel within the image,
    with respect to a neighborhood of pixels, referred to as window. Window has a size *ws x ws*.
    *pixelwise_features* calls iteratively a function that calculates all these features.
    Function *function* can have in this version arbitrarilly many arguments given as
    the input specified by *extra_args*. *pixelwise_features* calls then this function for every
    window within the image. This means, that , for example, two directly subsequent windows only differ in one value.

    The important thing about *pixelwise_features* is that it takes in account boundary conditions.
    For a pixel that lies close to the boundary and a window of size *ws x ws* would yield an index error,
    is truncated to a window just fitting in the image. For example, the window of a pixel at *img[1,0]*
    for *ws = 9*, would be *img[0:5, 0:4]*.

    It is used for sliding window estimation of C2 and C5 features. Otherwise, *pixelwise_features* which calls an implementation
    from *ndimage* library, which is way more efficient is used.

    :param np.array img: image for which the features are calculated
    :param int ws: window size
    :param function.object function: function that calculates the features. It only takes one input, namely window (e.g. img[0:5, 0:4]
    :param any *extra_args: Any further set of parameters that function *function* could take as input

    :returns: dictionary where keys are features names and entries are 2d arrays with feature values over the image.
    """

    dummy_img = np.ones(img.shape)
    feat_characteristics = function(dummy_img)
    features_num = len(feat_characteristics)
    feat_names = feat_characteristics.keys()
    if (ws%2 == 0):
        # raise ValueError("radiomics.C1 : window size has to be odd numbers")
        pass
    shape = list(img.shape)
    shape.append(features_num)
    x_start_shifted = np.array([a - int(ws/2) for a in range(shape[0])])
    x_start = [a[a >= 0] for a in x_start_shifted]
    x_stop_shifted = np.array([a + int(ws / 2) for a in range(shape[0])])
    x_stop = [a[a < len(img)] for a in x_stop_shifted]
    y_start_shifted = np.array([a - int(ws/2) for a in range(shape[1])]) # supposing square support region
    y_start = [a[a >= 0] for a in y_start_shifted]
    y_stop_shifted = np.array([a + int(ws / 2) for a in range(shape[1])])
    y_stop = [a[a < len(img[0])] for a in y_stop_shifted]

    for i in range(len(x_start)):
        if x_start[i].size == 0:
            x_start[i] = 0
        else:
            x_start[i] = int(x_start[i])
    for i in range(len(x_stop)):
        if x_stop[i].size == 0:
            x_stop[i] = int(shape[0])
        else:
            x_stop[i] = int(x_stop[i])
    for i in range(len(y_start)):
        if y_start[i].size == 0:
            y_start[i] = 0
        else:
            y_start[i] = int(y_start[i])
    for i in range(len(y_stop)):
        if y_stop[i].size == 0:
            y_stop[i] = int(shape[1])
        else:
            y_stop[i] = int(y_stop[i])
    features = np.zeros(shape)
    for i in range(shape[0]):
        for j in range(shape[1]):
            features[i][j] = function(img[x_start[i]:x_stop[i], y_start[j]:y_stop[j]], *extra_args)
    features = np.swapaxes(features, 1, 2)
    features = np.swapaxes(features, 0, 1)
    dict_to_return = {a_key:an_array for a_key, an_array in itertools.izip(feat_names, features)}
    return dict_to_return

def pixelwise_features(img, ws, function, *extra_args):

    """Sliding window feature estimation by using *ndimage.filters.generic_filter*, which is much
    faster than the custom implementation.

    :param np.array img: image for which the features are calculated
    :param int ws: window size
    :param function.object function: function that calculates the features. It only takes one input, namely window (e.g. img[0:5, 0:4]
    :param any *extra_args: Any further set of parameters that function *function* could take as input

    :returns: 2d numpy array with feature values over the image.
    """

    feat_shape = img.shape
    features = np.zeros(feat_shape)
    ndimage.filters.generic_filter(img, function, size=ws, output=features, extra_arguments=extra_args)
    return features

def single_values_to_img(dictionary, img_shape):
    """"For ROI based features, replicate img.shape times the single value over the entire ROI"""
    extended_dic = {}
    for item in dictionary:
        if type(dictionary[item]) is list or type(dictionary[item]) is np.ndarray:
            i=0
            for value in dictionary[item]:
                level = np.tile(value, img_shape)
                hist_bin_name = item + '_%d'%i
                extended_dic[hist_bin_name] = level
                i += 1
        else:
            level = np.tile(dictionary[item], img_shape)
            extended_dic[item] = level
    return extended_dic

def set_in_range(img, indices, values):

    """Crop an array to the boundaries specified by indices and fill it with specified values. New image shape
    and values must have the same shape.

    :param 2d numpy array img:
    :param numpy array indices:
    :param 2d numpy array values:

    :returns: 2d numpy array with new shape and values
    """
    rows_0 = np.min(indices[1])
    rows_1 = np.max(indices[1])
    columns_0 = np.min(indices[0])
    columns_1 = np.max(indices[0])
    img[columns_0:columns_1, rows_0:rows_1] = values
    return img

def trim_array_to_ROI(img, return_support_region = False):

    """Crop an image to the boundaries specified by the circumscribed rectangle of the region defined by a mask.

    :param 2d numpy masked array img: mask is the region of interest.
    :param bool return_support_region: If set to true, also the boudaries of the circumscribed rectangle are returned.
    Default is False.

    :returns: 2d numpy array, cropped image

    :returns: If *return_support_region* is set, also a list with coordinates of the ROI circumscribed rectangle is returned.
    """
    #img.mask = np.logical_not(img.mask)
    edges_y = []
    for row in img:
        edges = ma.notmasked_edges(row)
        if edges is not None:
            edges_y.append(edges)
    edges_x = []
    for col in np.swapaxes(img, 0, 1):
        edges = ma.notmasked_edges(col)
        if edges is not None:
            edges_x.append(edges)
    if edges_x == [] and edges_y == []:
        return None
    edges_x = np.swapaxes(edges_x, 0, 1)
    edges_y = np.swapaxes(edges_y, 0, 1)
    min_x = np.min(edges_x[0])
    max_x = np.max(edges_x[1])
    min_y = np.min(edges_y[0])
    max_y = np.max(edges_y[1])
    # do not return stripes
    if max_x - min_x <= 1:
        if min_x == 0:
            max_x += 1
        elif max_x == edges_x[1, -1]:
            min_x -= 1
        else:
            max_x += 1
    if max_y - min_y <= 1:
        if min_y == 0:
            max_x += 1
        elif max_y == edges_y[1, -1]:
            min_y -= 1
        else:
            max_y += 1
    new_data = img.data[min_x : max_x+1, min_y : max_y+1]
    support_region = [(min_x, max_x+1), (min_y, max_y+1)]
    if return_support_region:
        return new_data, support_region
    return new_data

def healthy_rois(img, roi_inds, ROIs, a_function):
    """For ROI based calculation of C2 and C5 features. Apply the function defined as input to regions
    defined by the dual of circumscribed rectangles around regions of interest defined at ROIs.
    Healthy ROIs are defined by the intersections of lines defining the boundaries of circumscribed rectangles
    around regions defined by ROIs, thus they are compact and recangularly shaped.

    :param 2d numpy array img:
    :param list roi_inds: indices of lines defining the boundaries of circumscribed rectangles
    around regions defined by ROIs
    :param 2d numpy array ROIs:
    :param function object a_function: function to apply on healthy regions

    :returns: features dictionary, where keys are features' names and entries are 2d numpy arrays with the feature values over each pixel.

    :returns: A list with coordinates of the healthy ROI circumscribed rectangle is also returned.
    """
    features = []
    indices = []
    roi_inds.append([(0, img.shape[0]), (0, img.shape[1])])
    row_inds = np.sort(np.unique(np.ravel(np.array([x[0] for x in roi_inds]))))
    col_inds = np.sort(np.unique(np.ravel(np.array([x[1] for x in roi_inds]))))
    for i in range(len(row_inds)-1):
        for j in range(len(col_inds)-1):
            if not np.any(ROIs[row_inds[i]:row_inds[i+1], col_inds[j]:col_inds[j+1]]): # is healthy
                min_x = row_inds[i]
                max_x = row_inds[i+1]
                min_y = col_inds[j]
                max_y = col_inds[j+1]
                if max_x - min_x == 1:
                    if min_x <= 0:
                        max_x += 2
                        min_x = 0
                    elif max_x >= img.shape[0]:
                        min_x -= 2
                        max_x = img.shape[0]
                    else:
                        max_x += 1
                if max_y - min_y == 1:
                    if min_y <= 0:
                        max_y += 2
                        min_y = 0
                    elif max_y >= img.shape[-1]:
                        min_y -= 2
                        max_y = img.shape[-1]
                    else:
                        max_y += 1
                indices.append([(min_x, max_x), (min_y, max_y)])
                sub_image = img[min_x:max_x, min_y:max_y]
                features.append(a_function(sub_image))
    roi_inds.pop(-1) # the instance gets affected globaly!!
    return features, indices


def ROI_based_calclulations(img, ROIs, the_function):

    """For ROI based calculation of C2 and C5 features. Apply the function defined as input to regions
    defined by the circumscribed rectangles around regions of interest defined at ROIs. ROIs is a numpy array
    representing the mask where 0 refers to healthy tissue and 1 to lesions. There can multiple lesions of
    arbitrary shape. Also a 4-connected 1-pixel binary errosion
    and dilation is applied on ROIs to compensate for registration errors of the ROIs masks, that usually are
    the outputs of a registation and salt and pepper noise is frequently observed.
    Lesions with a total area less than 10 pixels are disregarded as they are probably a result of registration errors that
    are not compensated during erosion-dilation.

    :param 2d numpy array img:
    :param 2d numpy array ROIs:
    :param function object the_function:

    :returns: features dictionary, where keys are features' names and entries are 2d numpy arrays with the feature values over each pixel.
    """
    roi_features = []
    roi_indices = []
    corrected_ROIs = np.zeros(ROIs.shape)
    # 0 is the healthy, handled later
    #CHANGE
    non_compact_ROIs, components = label(ROIs, neighbors=4, return_num=True)
    if components > 1:  # 0 is healthy, 1 is one compact region. then usually it's noise if it's more
        remove_registration_errors = np.logical_not(ROIs)
        remove_registration_errors = ndimage.binary_erosion(remove_registration_errors, iterations=1)
        remove_registration_errors = ndimage.binary_dilation(remove_registration_errors, iterations=1)
        ROIs = np.logical_not(remove_registration_errors)
        non_compact_ROIs, components = label(ROIs, neighbors=4, return_num=True)
    ROIs = np.logical_or(corrected_ROIs, np.logical_not(ROIs))  # this roi masks out True values.
    for subroi in range(1, components + 1):  # 0 is the rest, the non-roi
        #CHANGE
        this_subroi = np.where(non_compact_ROIs == subroi, False, True)
        if np.sum(np.logical_not(this_subroi)) > 10:
            roi_img = ma.masked_array(img, this_subroi)
            roi_img, their_indices = trim_array_to_ROI(roi_img, return_support_region=True)
            roi_features.append(the_function(roi_img))
            roi_indices.append(their_indices)
        else:
            ROIs = np.where(non_compact_ROIs == subroi, True, ROIs) # change noise to healthy to give features
    hr, hi = healthy_rois(img, roi_indices, np.logical_not(ROIs), the_function) #needs True to consider as sick
    roi_features.extend(hr)
    roi_indices.extend(hi)
    # reconstruct
    features = {}
    for feature_name in list(roi_features[0]):
        features[feature_name] = np.zeros(img.shape)
    for this_roi_feature_dic, this_roi_support in itertools.izip(roi_features, roi_indices):
        # assuming compact rois
        for feature_name in this_roi_feature_dic:
            features[feature_name] = set_in_range(features[feature_name], this_roi_support,
                                                  this_roi_feature_dic[feature_name])

    return features

# from scipy.misc import toimage
# toimage(this_roi).show()