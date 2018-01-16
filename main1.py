#encoding: utf-8
from __future__ import division
import os
import numpy as np
from modules import *
import numpy.ma as ma
import SimpleITK as sitk
import itertools
import sys

def rescale(img, ratio):
    """Modified wavelet pyramid scheme to bring an image down to a specific scale.
    Applies a 2*ratio pixel kernel Gaussian filter to the image to smooth and avoid aliasing.
    Then input image is sampled on a grid defined by ratio. If ratio is not integer and image pixel indices are
    :math: `i \in [0,N_x]` and :math: `j \in [0, N_y]` respectively,then a sampling point is i*ratio[0].
    The index according to which its value is retrieved is floor(i*ratio(0)), since the value is constant in the pixel.
    It is better to use scipy.ndimage.interpolate.zoom function but this implementation is necessary to
    run the program.

    3d images and 3d ratio as input

    :param: 3d numpy array img
    :param: list ratio: ratio[0] corresponds to x sampling ratio, ratio[1] to y sampling ratio and ratio[2] to z. Since this function is made to subsample MRI images, ratio[0]==ratio[1]

    :returns: 3d numpy array: downsampled image
    """
    n= 2 # MERK!! define optimal gaussian low pass filter sigma according to the ratio
    s = np.floor(np.array(ratio)*n).astype(int)
    # filtered_img = filters.gaussian_filter(img, s)
    filtered_img = img
    sampling_grid = []
    for coordinate in range(len(ratio)):
        sampling_grid.append([])
        sampling_grid[coordinate] = []
        for i in range(img.shape[coordinate]):
            #sampling_grid[coordinate].append(i*(1+ratio[coordinate]))
            sampling_grid[coordinate].append(i * (ratio[coordinate]))
        discrete_grid = np.floor(np.array(sampling_grid[coordinate])).astype(int)
        boundary_check = np.nonzero(discrete_grid < img.shape[coordinate])
        discrete_grid = list(discrete_grid[boundary_check])
        sampling_grid[coordinate] = discrete_grid
    downscaled_img = []
    for x in range(len(sampling_grid[0])):
        downscaled_img.append([])
        index_x = sampling_grid[0][x]
        for y in range(len(sampling_grid[1])):
            downscaled_img[x].append([])
            index_y = sampling_grid[1][y]
            for z in range(len(sampling_grid[2])):
                index_z = sampling_grid[2][z]
                # index = np.floor(np.asarray([index_x,index_y, index_z])).astype(int)
                downscaled_img[x][y].append(filtered_img[index_x, index_y, index_z])
    return np.array(downscaled_img)

def dictionaries_to_arrays(features_dictionary, patient_directory, feature_set):
    """Due to change in design, had to convert all features that are reurned in dictionairies as
    multi dimensinal np arrays. for exapmple, a 512x 512 slice with 10 C1 featues is now a 512x512x10 array
    dictionary entries are writen in text files which will indicate feature values sequence.
    According to these files, features can be retireived later. Each patient folder contains the files that correspond
    to the calculated features for the specific patient.

    Feature name files also contain a dummy feature which is zero everywhere. This is for initialization and it does not
    correspond to an actual feature.

    :param: dictionary features_dictionary: keys are features names and entries are 2d numpy arrays that contain the value of the specified feature for each pixel in the image.
    :param: string patient_directory: directory where feature name files are saved.
    :param: string feature_set: the set of features (C1, C2, etc) It defines the name of the feature name files as features_names_C1.txt etc

    :returns: numpy array: 3d numpy array with vectors containing the feature values at each pixel."""
    # intialization
    names = list(features_dictionary)
    first_feature = np.array(features_dictionary[names[0]])
    feature_values = np.zeros(first_feature.shape)
    features_names = ['dummy']
    for feature_name in names:
        feature_values = np.dstack((feature_values, features_dictionary[feature_name]))
        features_names.append(feature_name)
    # assuming same features for each modality
    os.chdir(patient_directory)
    file_name = 'features_names_' + feature_set + '.txt'
    if os.path.isfile(file_name):
        return feature_values
    with open(file_name, 'a') as features_names_file:
        for a_feature in features_names:
            features_names_file.write(a_feature + '\n')
    return feature_values

def calculate_features(img, rois, patient_directory, resolution, features_switch = ['f1', 'f2', 'f3', 'f4', 'f5'], sliding_window = False):

    """Calculates features for an image slice. All functions that implement the selected feature calculation in the selected way
    are called depending on the inputs.

    :param 2d numpy array img: Image slice
    :param 2d numpy array rois: Regions of interest on the image slice labeled 0 for healthy and 1 for lesion
    :param patient_directory: Where to save feature names description files
    :param resolution: Image pixel physical spacing (retrieved from the MRI image header information, not avilable in numpy)
    :param list features_switch: Which of the features to calculate (from families C1, C2, C3, C4, C5). Default value is ['f1', 'f2', 'f3', 'f4', 'f5'], which calculates all features.
    :param bool sliding_window: If true, C2 and C5 features are calculated according to the 9x9 sliding window approach, otherwise by the circumscribed rectangle aproach. Default is False.

    :returns: numpy array: 3d numpy array with vectors containing the feature values at each pixel.
    """
    if sliding_window:
        feature_instructions = {'f1':'r.append(dictionaries_to_arrays(C1.C1(img), patient_directory, "C1"))',
                                'f2':'C2_dic = global_functions.custom_pixelwise_features(img, 9, C2.C2, True)\nr.append(dictionaries_to_arrays(C2_dic, patient_directory, "C2"))',
                                'f3':'r.append(dictionaries_to_arrays(C3.C3(img, resolution), patient_directory, "C3"))',
                                'f4':'r.append(dictionaries_to_arrays(C4.C4(img), patient_directory, "C4"))',
                                'f5':'C5_dic = global_functions.custom_pixelwise_features(img, 9, C5.C5, True)\nr.append(dictionaries_to_arrays(C5_dic, patient_directory, "C5"))'
                                }
    else:
        feature_instructions = {'f1':'r.append(dictionaries_to_arrays(C1.C1(img), patient_directory, "C1"))',
                                'f2':'C2_dic = global_functions.ROI_based_calclulations(img, rois, C2.C2)\nr.append(dictionaries_to_arrays(C2_dic, patient_directory, "C2"))',
                                'f3':'r.append(dictionaries_to_arrays(C3.C3(img, resolution), patient_directory, "C3"))',
                                'f4':'r.append(dictionaries_to_arrays(C4.C4(img), patient_directory, "C4"))',
                                'f5':'C5_dic = global_functions.ROI_based_calclulations(img, rois, C5.C5)\nr.append(dictionaries_to_arrays(C5_dic, patient_directory, "C5"))'
                                }

    r = []
    labels = dictionaries_to_arrays(retrieve_labels.get_labels(rois), patient_directory, 'labels')
    for a_family_of_features in features_switch :
        exec(feature_instructions[a_family_of_features])
    r.append(labels)
    all_features = np.dstack(tuple(r))
    return all_features



def apply_np_mask(image, mask):

    """Apply a mask on an image. Before this function is called, it is adviseable to call change_to_nifti.set_masks_world
     as it is possible that some mask files and MRI image files have different world masks and this will raise an exception.

    :param itk image file image: MRI itk image file
    :param itk image file mask: mask itk image file on the specific MRI image file (which means that if MRI image is a DWI and the mask is extracted on a T2 image it will raise an exception.)

    :returns: circumscribed rectangular volume of interest on which features will be calculated. As the prostate is a small proportion of the entire image and feature calculation is computationally expensive, this speeds up excecution a lot.
    """
    mask_size = mask.GetSize()
    image_size = image.GetSize()
    ratio = [a/b for a, b in itertools.izip(mask_size, image_size)]

    npimage = sitk.GetArrayFromImage(image)
    npimage = np.swapaxes(npimage, 0, 2)
    npmask = sitk.GetArrayFromImage(mask)
    npmask = np.swapaxes(npmask, 0, 2)
    npmask = rescale(npmask, ratio)

    npmask = np.logical_not(npmask)
    masked_array = ma.array(data = npimage, mask = npmask)
    reduced_volume = []
    iterable_array = np.swapaxes(masked_array, 0, 2)  # iterate over z
    for slice in iterable_array:
        reduced_slice = global_functions.trim_array_to_ROI(slice)
        if reduced_slice is not None:
            reduced_volume.append(reduced_slice)
    reduced_volume = np.array(reduced_volume)
    return reduced_volume

def write_features_to_files(patient_directory, image_name, rois, mask, features_switch, sliding_window):# ROIs_GS):

    """Calls calculate_features for each slice trimmed within the volume of interest (thus eliminating pixels both
     in z and in x,y directions) and saves a file with the features calculated within the patient directory. The
     file is in .npy format and it is a 4d numpy array. Features can retrieved by the feature name description files
     that are saved in the same directory.

    :param string patient_directory: where to save the feature files
    :param image_name: MRI image on which features are calculated. This name is used to name after the feature file. Thus, if extended to multiparametric MRI, there can be several feature files in the same directory corresponding to different modality images, as image name is after the sequence description and the patient ID.
    :param itk image file rois: itk image file for ROIs on the specific MRI image file (which means that if MRI image is a DWI and the mask is extracted on a T2 image it will raise an exception.) 1 refers to a lesion and 0 refers to healthy.
    :param itk image file mask: itk image file for masks on the specific MRI image file (which means that if MRI image is a DWI and the mask is extracted on a T2 image it will raise an exception.) 1 refers to the prostate or peripheral zone of the prostate volume and 0 refers to healthy.
    :param list features_switch: Which of the features to calculate (from families C1, C2, C3, C4, C5). Default value is ['f1', 'f2', 'f3', 'f4', 'f5'], which calculates all features.
    :param  bool sliding_window: If true, C2 and C5 features are calculated according to the 9x9 sliding window approach, otherwise by the circumscribed rectangle aproach. Default is False.
    """
    os.chdir(patient_directory)
    print patient_directory
    file_name = image_name.split('.')[0] + '_features'
    if os.path.isfile(file_name+'.npy'):
        return
    image = sitk.ReadImage(os.path.join(patient_directory, image_name))
    mask = sitk.ReadImage(os.path.join(patient_directory, mask))
    rois = sitk.ReadImage(os.path.join(patient_directory, rois))
    mask.SetSpacing(image.GetSpacing())
    mask.SetOrigin(image.GetOrigin())
    mask.SetDirection(image.GetDirection())
    resolution = image.GetSpacing()
    # MERK! mask in z,y,x format
    # swapaxes 1, 2 yields z, x, y to enumerate slices in for
    npimage = apply_np_mask(image, mask)
    #------------
    nprois = apply_np_mask(rois, mask)
    # this is a way non effective effort to estimate the total number of pixels, but it is more robust to future code changes
    npresult = np.empty(npimage.shape, dtype=object)
    i=0
    for slice, rois_slice in itertools.izip(npimage, nprois):
        npresult[i] = calculate_features(slice, rois_slice, patient_directory, resolution, features_switch, sliding_window)
        i += 1
    os.chdir(patient_directory)
    np.save(file_name, npresult)
    os.chdir('../')
    with open('processed_cases.txt', 'a') as progress:
        progress.write(image_name + '\n')

def dump_patient_folders(working_directory, features_switch, sliding_window):

    """Walk all over the working directory, find files for which features should be calculated and calculate
    the selected features for them according to the specified method (sliding window or ROI based.) Moreover,
    a progress file is saved in the working directory. This contains all images about which features have already been calculated.
    Therefore, if an exception is raised or excecution is halted for any reason, it can reset. It is also possible to
    expand in more modalities, as progress file's images names' depend on the modality and the patient ID.

    :param string working_directory: Where all patient directories containing images, masks and ROIs are saved.
    :param list features_switch: Which of the features to calculate (from families C1, C2, C3, C4, C5). Default value is ['f1', 'f2', 'f3', 'f4', 'f5'], which calculates all features.
    :param bool sliding_window: If true, C2 and C5 features are calculated according to the 9x9 sliding window approach,
     otherwise by the circumscribed rectangle aproach. Default is False.
    """
    os.chdir(working_directory)
    patients = next(os.walk(working_directory))[1]
    already_processed = []
    if os.path.isfile('processed_cases.txt'):
        with open('processed_cases.txt','r') as progress:
            for line in progress:
                already_processed.append(line.rstrip())
    for a_directory in patients:
        patient_directory = working_directory + '/' +  a_directory
        for image in os.listdir(patient_directory):
            if image.endswith('mask_cor.nii'):
                mask = image
            if image == 'rois_mask_.nii':
                rois_mask = image
        for image in os.listdir(patient_directory):
            if global_functions.regex_match('patient image', image):
                if image not in already_processed:
                    write_features_to_files(patient_directory, image, rois_mask, mask, features_switch, sliding_window) #ROIs_GS_per_patient[a_directory])


def main2(working_directory, data_root, mask_root, features_switch = ['f1', 'f2', 'f3', 'f4', 'f5'], whole_prostate = True, sliding_window = False):

    """"Call all dataset preprocessing routines, namely separate_modalities.process_files, change_to_nifti.set_masks_world
    and then call dump_patient_folders to calculate all features for all files. Also select whether the whole prostate in feature
    estimation is used or just the peripheral zone.
    :param string working_directory: Where all patient directories containing images, masks and ROIs are saved.
    :param string data_root: root directory of the raw dataset
    :param string mask_root: root directory of the mask and roi files
    :param list features_switch: Which of the features to calculate (from families C1, C2, C3, C4, C5). Default value is
    ['f1', 'f2', 'f3', 'f4', 'f5'], which calculates all features. It is important to feed the features switch in order
     (eg never give f3, f5,f1 but f1, f3, f5)
    :param bool whole_prostate: Whether to use the entire prostate or simply the peripheral zone. Default is True (whole prostate).
    :param bool sliding_window: If true, C2 and C5 features are calculated according to the 9x9 sliding window approach,
     otherwise by the circumscribed rectangle aproach. Default is False.
     """
    working_directory = working_directory
    mask_root_directory = mask_root
    data_root_directory = data_root
    if whole_prostate:
        mask_format_name = 'no_previous_mask_init.nii'
    else:
        mask_format_name = 'pz_mask.nii'

    separate_modalities.process_files(working_directory, data_root_directory, mask_root_directory, mask_format_name)
    change_to_nifti.set_masks_world(working_directory)
    dump_patient_folders(working_directory, features_switch, sliding_window)


def fun1(input_string):

    """"Calculate features"""


    if input_string == 1: #TCIA all features whole prostate
        wd = '/root/work/wd_TCIA_all_features_whole'
        data_root_directory = '/root/work/TCIA/DOI'
        masks_root_directory = '/root/work/msk_tcia/msk_root'
        if not os.path.isdir(wd):
            os.mkdir(wd)
        main2(wd, data_root_directory, masks_root_directory, features_switch=['f1', 'f2', 'f3', 'f4', 'f5'], whole_prostate=True, sliding_window=False)
    elif input_string ==2 : # TCIA all features whole prostate sliding window
        wd = '/root/work/wd_TCIA_all_features_whole_sliding'
        data_root_directory = '/root/work/TCIA/DOI'
        masks_root_directory = '/root/work/msk_tcia/msk_root'
        if not os.path.isdir(wd):
            os.mkdir(wd)
        main2(wd, data_root_directory, masks_root_directory, features_switch=['f1', 'f2', 'f3', 'f4', 'f5'], whole_prostate=True, sliding_window=True)
    elif input_string == 3: #TCIA all features pz
        wd = '/root/work/wd_TCIA_all_features_pz'
        data_root_directory = '/root/work/TCIA/DOI'
        masks_root_directory = '/root/work/msk_tcia/msk_root'
        if not os.path.isdir(wd):
            os.mkdir(wd)
        main2(wd, data_root_directory, masks_root_directory, features_switch=['f1', 'f2', 'f3', 'f4', 'f5'], whole_prostate=False, sliding_window=False)
    elif input_string == 4: # TCIA all features pz sliding window
        wd = '/root/work/wd_TCIA_all_features_pz_sliding'
        data_root_directory = '/root/work/TCIA/DOI'
        masks_root_directory = '/root/work/msk_tcia/msk_root'
        if not os.path.isdir(wd):
            os.mkdir(wd)
        main2(wd, data_root_directory, masks_root_directory, features_switch=['f1', 'f2', 'f3', 'f4', 'f5'], whole_prostate=False, sliding_window=True)
    elif input_string == 5: #PCMM all features whole prostate
        wd = '/root/work/wd_PCMM_all_features_whole'
        data_root_directory = '/root/work/PCMM'
        masks_root_directory = '/root/work/msk_root'
        if not os.path.isdir(wd):
            os.mkdir(wd)
        main2(wd, data_root_directory, masks_root_directory, features_switch=['f1', 'f2', 'f3', 'f4', 'f5'], whole_prostate=True, sliding_window=False)
    elif input_string == 6: #PCMM all features whole prostate sliding window
        wd = '/root/work/wd_PCMM_all_features_whole_sliding'
        data_root_directory = '/root/work/PCMM'
        masks_root_directory = '/root/work/msk_root'
        if not os.path.isdir(wd):
            os.mkdir(wd)
        main2(wd, data_root_directory, masks_root_directory, features_switch=['f1', 'f2', 'f3', 'f4', 'f5'], whole_prostate=True, sliding_window=True)
    elif input_string == 7: #PCMM pz all features
        wd = '/root/work/wd_PCMM_all_features_pz'
        data_root_directory = '/root/work/PCMM'
        masks_root_directory = '/root/work/msk_root'
        if not os.path.isdir(wd):
            os.mkdir(wd)
        main2(wd, data_root_directory, masks_root_directory, features_switch=['f1', 'f2', 'f3', 'f4', 'f5'], whole_prostate=False, sliding_window=False)
    elif input_string == 8: #PCMM pz all features sliding window
        wd = '/root/work/wd_PCMM_all_features_pz_sliding'
        data_root_directory = '/root/work/PCMM'
        masks_root_directory = '/root/work/msk_root'
        if not os.path.isdir(wd):
            os.mkdir(wd)
        main2(wd, data_root_directory, masks_root_directory, features_switch=['f1', 'f2', 'f3', 'f4', 'f5'], whole_prostate=False, sliding_window=True)
    else:
        print('gamw to spiti sou!')



if __name__ == "__main__":

    fun1(7)
    fun1(8)



#main2('F:/wd2','C:/PCMM' ,'C:/Users/157136/Documents/msk_root')
#main2('F:/TCIA/wd2','F:/TCIA/DOI/aaa0044' ,'F:/TCIA/msk_root/aaa044')
#classification.reduction_analysis('F:/TCIA/wd2')
#main2('C:/Users/157136/Desktop/pground/wddd', 'C:/Users/157136/Desktop/pground/PCMM', 'C:/Users/157136/Documents/msk_root/s55', features_switch = ['f5'], whole_prostate = True, sliding_window=True)
#main2('/media/pavlos/Elements/playground/wd', '/media/pavlos/Elements/TCIA/DOI/aaa0063', '/media/pavlos/Elements/TCIA/msk_root/aaa0063', features_switch=['f2'], whole_prostate=False, sliding_window=True)
#main2('/media/pavlos/Elements/TCIA/wdd_2', '/media/pavlos/Elements/TCIA/DOI/aaa0054', '/media/pavlos/Elements/TCIA/msk_root/aaa0054', features_switch=['f5'], whole_prostate=True, sliding_window=True)
#
# wd = '/media/pavlos/Elements/SERVER/working_directory/version_2/wd_PCMM_f5_pz_ROI'
# data_root_directory = '/media/pavlos/Elements/to_upload/PCMM'
# masks_root_directory = '/media/pavlos/Elements/to_upload/msk_root'
# if not os.path.isdir(wd):
#     os.mkdir(wd)
# main2(wd, data_root_directory, masks_root_directory, features_switch=['f5'], whole_prostate=False, sliding_window=False)

#run 7 and 8, then upload all minus PCMM pz (2 directories) and run full classification main