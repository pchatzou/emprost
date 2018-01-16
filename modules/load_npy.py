# encoding:utf-8
import numpy as np
import os
import itertools
import SimpleITK as sitk
import global_functions
from global_functions import regex_match


def get_feature_names(patient_directory):

    """Get all features names in a list with the corresponding order that they are saved in the vectors of the .npy
    feature file.

    :param string patient_directory: Where feature name files are saved.

    :return: list: all feature names in the correct order.
    """
    os.chdir(patient_directory)
    feature_files = []
    for a_file in os.listdir('./'):
        if a_file.endswith('.txt') and a_file.startswith('features_names'):
            feature_files.append(a_file)
    if len(feature_files) == 6:
        feature_file_name_list = ['features_names_C1.txt', 'features_names_C2.txt', 'features_names_C3.txt',
                                  'features_names_C4.txt', 'features_names_C5.txt', 'features_names_labels.txt']
    else:
        feature_file_name_list = sorted(feature_files, key=lambda f: list(f)[-5]) # if not all features are calculated, sort based on family
    all_features_description = []
    for a_file in feature_file_name_list:
        with open(a_file, 'r') as features_names:
            for a_name in features_names:
                all_features_description.append(a_name.rstrip())
    return all_features_description

def get_features_subset(patient_directory, feature_file, feature_names):

    """Get a subset of all features for an image as a 4d numpy array.

    :param string patient_directory:
    :param string feature_file:
    :param list feature_names: Features that should be retrieved

    :returns:  4d numpy array: Requested features of the specific image for the volume of interest.
    """
    os.chdir(patient_directory)
    the_file = np.load(feature_file)
    all_features_description = get_feature_names(patient_directory)
    for a_file in os.listdir('./'):
        if a_file.endswith('mask_cor.nii'):
            mask = sitk.ReadImage(a_file)
            npmask = sitk.GetArrayFromImage(mask)
            npmask = np.ma.masked_array(npmask, np.logical_not(npmask))
    z_idx=0
    for mask_slice in npmask:
        masked = global_functions.trim_array_to_ROI(mask_slice, return_support_region=True)
        temp =[]
        if masked is not None:
            for name_idx in feature_names:
                ft_idx = all_features_description.index(name_idx)
                this_slice_feature = the_file[z_idx][ 5:-5, 5:-5, ft_idx] # excluding edge artifacts for window size 9
                unmasked_elements = np.ma.masked_array(this_slice_feature, np.logical_not(masked[0][5:-5,5:-5]))
                unmasked_elements = unmasked_elements[~unmasked_elements.mask]
                temp.append(np.ravel(unmasked_elements))
            try:
                feature_bag = np.append(feature_bag, np.array(temp), axis=1)
            except:
                feature_bag = np.array(temp)
            z_idx += 1

    return feature_bag


def load_file(patient_directory, feature_file, feature_name, return_array = False):

    """Retrieve a single feature from a feature file and either return it for further processing or save it as itk image
    for visualization. Visualization shows the true feature values for the volume of interest over which the feature has
    been calculated and sets the rest of the image to zero, so there is a common world matrix with T2 image and masks
    and comparison is made possible. However, those zero values do not correspond to actual feature values. Moreover,
    during classification or processing, no feature values outside the mask are taken into account.

    :param patient_directory:
    :param feature_file:
    :param feature_name:
    :param return_array: If set to True, array is returned for further processing, otherwise an itk image file is saved for visualizing the specific feature. Default value is False.

    :returns 3d numpy array: a feature value (only if return_array is set to True)
    """
    if return_array:
        feature_bag = get_features_subset(patient_directory, feature_file, feature_name)
        feature_bag = feature_bag.T
        return feature_bag
    os.chdir(patient_directory)
    the_file = np.load(feature_file)
    all_features_description = get_feature_names(patient_directory)
    for a_file in os.listdir('./'):
        if a_file.endswith('mask_cor.nii'):
            mask = sitk.ReadImage(a_file)
            npmask = sitk.GetArrayFromImage(mask)
            npmask = np.ma.masked_array(npmask, np.logical_not(npmask))
        if regex_match('patient image', a_file): #supposing a single modality, ege axial T2 in this case
            image_itk = sitk.ReadImage(a_file)

    idx = all_features_description.index(feature_name)
    a_feature = np.zeros(npmask.shape)
    i=0
    T2_index = 0
    for mask_slice, feature_slice in itertools.izip(npmask, a_feature):
        masked = global_functions.trim_array_to_ROI(mask_slice, return_support_region=True)
        if masked is not None:
            a_feature[T2_index] = global_functions.set_in_range(feature_slice, masked[1], the_file[i][ :, :, idx])
            i += 1
        T2_index += 1
    #a_feature = np.swapaxes(a_feature, 0, 2)
    if not os.path.exists('./feature_images'):
        os.makedirs('./feature_images')
    os.chdir('./feature_images')
    file_name = feature_name + '.nii'
    a_feature_itk = sitk.GetImageFromArray(a_feature)
    a_feature_itk.SetSpacing(image_itk.GetSpacing())
    a_feature_itk.SetOrigin(image_itk.GetOrigin())
    a_feature_itk.SetDirection(image_itk.GetDirection())
    sitk.WriteImage(a_feature_itk, file_name)



# if __name__ == "__main__":
#
#     load_file(sys.argv[1], sys.argv[2], sys.argv[3])
# load_file('C:/Users/157136/Desktop/pground/wd/PC-2519475016', 'PC-2519475016AXIAL_T2_features.npy', 'ROI_GS')
# load_file('C:/Users/157136/Desktop/pground/wd/PC-2519475016', 'PC-2519475016AXIAL_T2_features.npy', 'ortho_ROI_GS')
#load_file('F:/pground/debug_pgg/game', 'PC-1498300703AXIAL_T2_features.npy', 'ROI_GS')
#features_t_test('C:\\Users\\157136\\Desktop\\test1\\features')
#get_feature_names('F:/PCMM/pground/wd/PC-3120951400')
#load_file('/media/pavlos/Elements/SERVER/working_directory/to_upload/wd_TCIA_all_features_whole_sliding/aaa0072', 'aaa0072T2_AXIAL_SM_FOV_features.npy', 'Daubechies_LH')
