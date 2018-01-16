#encoding:utf-8
import dicom
import os
import registration
import SimpleITK as sitk
from shutil import copyfile
from scipy.ndimage import morphology
import numpy as np
from global_functions import regex_match


# def file_list(working_directory, data_directory, reset = False):
#     """"Return a dictionary with all full paths for all interesting files.
#     Dictionary's key is patient-study name, records are list with modality-path.eg.
#     dict = {patient1:[['T2W', 0.78, 'C:\users\pavlos\patients\patient1\T2W'],['DWI', 1.5, 'C:\users\pavlos\patients\patient1\T2W']] etc}"""
#     all_tags = {}
#     if os.path.isfile(os.path.join(working_directory, 'all_modalities.txt')):
#         if not reset:
#             return
#         else:
#             os.remove(os.path.join(working_directory, 'all_modalities.txt'))
#     with open('all_modalities.txt', 'w') as outstream:
#         for dirpath, dirname, file_names in os.walk(data_directory):
#             dirpath = dirpath.replace('\\','/')
#             for a_file in file_names:
#                 if a_file.endswith('.dcm'):
#                     dcm_tags = dicom.read_file(os.path.join(dirpath, a_file))
#                     if dcm_tags.PatientID not in all_tags:
#                         all_tags[dcm_tags.PatientID] = []
#                         outstream.write('\n' + dcm_tags.PatientID + '\n')
#                     if hasattr(dcm_tags, 'SeriesDescription'):
#                         sequence = dcm_tags.SeriesDescription
#                     elif hasattr(dcm_tags, 'SequenceName'):
#                         sequence = dcm_tags.SequenceName
#                     else:
#                         # only 3 files lack these attributes
#                         print(dirpath + 'has no sequence description')
#                         sequence = 'NA'
#                     if hasattr(dcm_tags, 'PixelSpacing'):
#                         resolution = dcm_tags.PixelSpacing[0]
#                     else:
#                         resolution = 'NA'
#                         print(dirpath + 'has no pixel spacing information')
#                     modality_entry = [sequence, resolution, dirpath]
#                     outstream.write(sequence + '&\t' + str(resolution) + '&\t' + dirpath + '\n')
#                     all_tags[dcm_tags.PatientID].append(modality_entry)
#                     # Break as many dicom files in a single directory are slices of the same 3D image
#                     break
#
#     return

def file_list2(working_directory, data_directory, reset=False):

    """Create and save file *all_modalites.txt* in the working directory, which is the major map to a raw dataset.
    This file contains in lines the patient ID followed by lines that refer to the sequence description, pixel spacing
    and file full path (dicom directory). It is readable by humans. Then the next patient ID follows and so on.

    :param string working_directory:
    :param string data_directory: raw dataset root directory
    :param bool reset: If an *all_modalities.txt* file already exists, delete it and make a new map. Default is False.
    """
    all_tags = {}
    if os.path.isfile(os.path.join(working_directory, 'all_modalities.txt')):
        if not reset:
            return
        else:
            os.remove(os.path.join(working_directory, 'all_modalities.txt'))
    os.chdir(working_directory)
    with open('all_modalities.txt', 'w') as outstream:
        for dirpath, dirname, file_names in os.walk(data_directory):
            dirpath = dirpath.replace('\\', '/')
            for a_file in file_names:
                if a_file.endswith('.dcm'):
                    dcm_tags = dicom.read_file(os.path.join(dirpath, a_file))
                    if dcm_tags.PatientID not in all_tags:
                        all_tags[dcm_tags.PatientID] = []
                        outstream.write('\n' + dcm_tags.PatientID + '\n')
                        #outstream.write(dcm_tags.PatientName +'\n')
                    if hasattr(dcm_tags, 'SeriesDescription'):
                        sequence = dcm_tags.SeriesDescription
                    elif hasattr(dcm_tags, 'SequenceName'):
                        sequence = dcm_tags.SequenceName
                    else:
                        # only 3 files lack these attributes
                        print(dirpath + 'has no sequence description')
                        sequence = 'NA'
                    if hasattr(dcm_tags, 'PixelSpacing'):
                        resolution = dcm_tags.PixelSpacing[0]
                    else:
                        resolution = 'NA'
                        print(dirpath + 'has no pixel spacing information')
                    modality_entry = [sequence, resolution, dirpath]
                    outstream.write(sequence + '&\t' + str(resolution) + '&\t' + dirpath + '\n')
                    all_tags[dcm_tags.PatientID].append(modality_entry)
                    # Break as many dicom files in a single directory are slices of the same 3D image
                    break

def retrieve_modalities(modality_name, working_directory, reset = False):
    """Create and save files in the working directory, that are minor maps to a raw dataset.
    Theses files are generated by map *all_modalities.txt* and like the format of *all_modalities.txt*, contain in lines
    the patient ID followed by lines that refer to the sequence description, pixel spacing
    and file full path (dicom directory).They are readable by humans. Then the next patient ID follows and so on. The difference is that
    they only refer to files that correspond to a specific sequence description. This description is retrieved by *all_modalities.txt*
    by a regular expression matching, so that they match the pattern of a specific sequence (e.g. T2 or DWI). In case a new dataset
    is introduced, where sequence descriptions differ, regex matches can be editted in a single point in the code, which is in
    global_functions.regex_match

    :param string modality_name: can be only T2 or DWI.
    :param string working_directory:
    :param bool reset: If the file already exists, delete it and make a new map. Default is False.
    """
    os.chdir(working_directory)
    output_file = modality_name +'_directories.txt'
    if os.path.isfile(output_file):
        if reset:
            os.remove(os.path.join(working_directory, output_file))
        else:
            return
    search_patterns = {'T2':'axial T2', 'DWI':'axial DWI'}
    looking_for = search_patterns[modality_name]
    modality_directories = {}
    if modality_name not in search_patterns:
        raise KeyError
    with open('all_modalities.txt','r') as dataset:
        for line in dataset:
            line = line.rstrip()
            if regex_match('patient directory', line): # line which is patient ID
                # bad design: line with patient ID is before all modalities, so there will be a dictionary key when needed
                modality_directories[line] = []
                patient_ID = line
            if regex_match(looking_for, line):
                modality_directories[patient_ID].append(line)
    with open(output_file, 'w') as outstream:
        for entry in modality_directories:
            if modality_directories[entry]:
                outstream.write(entry + '\n')
                for details in modality_directories[entry]:
                    outstream.write(str(details))
                outstream.write('\n\n')


def min_res(file_name, working_directory):
    os.chdir(working_directory)
    resolutions = []
    with open(file_name, 'r') as instream:
        for line in instream:
            line_contents = line.split('&\t')
            if len(line_contents) > 1:
                resolutions.append(float(line_contents[1])) # resolution
    min_res = max(resolutions)
    return min_res

def update_pointers(file_stream, pointers):

    current_patient = ''
    for line in file_stream:
        line = line.rstrip()
        # print(line)
        # create a dictionary from scratch
        if regex_match('patient directory', line):
            pointers[line] = ''
            current_patient = line  # dictionary with patient ID as key
        elif line:  # not empty
            directory = line.split('&\t')[-1]
            if regex_match('windows or unix directory', directory):
                pointers[current_patient] = directory  # key already created above, if last entry matches a directory format put it as "pointer"
            else:
                del pointers[current_patient]  # directory not found
        else:  # empty line
            pass
    return pointers

def get_dictionary(modality_directories, reset = False):
    """Return a dictionary with patient IDs as keys and modality directories as entries corresponding to them.
    For parallel dictionairies creation. Input is T2_directories.txt or DWI_direc, or masks_corrected"""
    pointers = {}
    current_modality = modality_directories.split('_')[0]
    existing_status = False
    # we are supposed to be in working directory
    for a_file in os.listdir('./'):
        current_file = a_file.split('_')[0]
        if a_file.endswith('_status.txt') and current_file == current_modality:
            if reset:
                os.remove(a_file)
                break
            existing_status = True
            with open(current_file, 'r') as instream:
                last_patient_processed = ''
                for line in instream:
                    last_patient_processed = line
    with open(modality_directories, 'r') as instream:
        if not existing_status:
            pointers = update_pointers(instream, pointers)
        else:
            # write nothing in dictionary until the last patient is found. Then add next entry.
            patient_found = False
            for line in instream:
                if line != last_patient_processed and not patient_found:
                    pass
                elif line == last_patient_processed:
                    patient_found = True
                if patient_found:
                    pointers = update_pointers(instream, pointers)
    return pointers

def fix_dictionaries(reference, dic2):
    """"check if dicitonaries have the same keys. If not, delete entries to make them.

    :param dictionary reference: true dictionary
    :param dictionary dic2: dictionary to edit
    """
    fixed_dic2 = {}
    fixed_reference = {}
    common_entries = set(reference.keys()).intersection(set(dic2.keys()))
    for a in common_entries:
        fixed_reference[a] = reference[a]
        fixed_dic2[a] = dic2[a]
    return fixed_reference, fixed_dic2

def process_files(working_directory, data_root_directory, mask_root_directory, mask_format_name):

    """Entire dataset mapping and preprocessing, which includes mapping the dataset, retrieving modality
    and mask directories. Then check for which of all candidate patient IDs all necessary data is present.
    Necessary data includes in current version:
    * axial T2 image
    * prostate or peripheral zone segmentation image
    * Regions of interest image
    Finally only keep patients for which all this information is present.
    For these patients, create within the working directory a subdirectory named after the patient ID in which
    all these information are copied and feature files are going to be stored.

    :param string working_directory: At this point working directory can be completely empty
    :param string data_root_directory:
    :param string mask_root_directory: It is the directory under which all patient ID subderectories are present. Those must contain an itk image referring to the prostate or peripheral zone segmentation, an itk image referring to ROIs and a *MeVisLab* dicom/tif image corresponding to the axial T2 image of the patient, after which masks are extracted. This is necessary to retrieve patient ID and spacing information.
    :param string mask_format_name: indicates if whole prostate segmentations or peripheral zone segmentations shall be used. Can be *no_previous_mask_init.nii* for the entire prostate or *pz_mask.nii* for the peripheral zone.
    """
    os.chdir(working_directory)
    file_list2(working_directory, data_root_directory)
    correct_masks(working_directory, mask_root_directory, mask_format_name)
    retrieve_modalities('T2', working_directory)
    mask_list(working_directory, mask_root_directory, 'rois_directories.txt', 'rois_mask_.nii')
    #rois = get_rois(mask_root_directory)
    # retrieve_modalities('DWI', working_directory)
    T2s = get_dictionary('T2_directories.txt')
    #legacy
    masks = get_dictionary('corrected_masks.txt')
    rois = get_dictionary('rois_directories.txt')
    # DWIs = get_dictionary('DWI_directories.txt')

    T2s, masks = fix_dictionaries(T2s, masks)
    rois, T2s = fix_dictionaries(rois, T2s)
    rois, masks = fix_dictionaries(rois, masks)
    #T2s, DWIs = fix_dictionaries(T2s, DWIs)
    #DWIs, masks = fix_dictionaries(DWIs, masks)
    registered_files = []
    if os.path.isfile('registered_files.txt'):
        with open('registered_files.txt','r') as instream:
            for line in instream:
                registered_files.append(line.rstrip())
    for patient in T2s: # same as saying for patient in DWIs
        registration.dcm_series_to_nii(T2s[patient], working_directory)
        dst_file = working_directory + '/' + patient + '/rois_mask_.nii'
        src_file = os.path.join(rois[patient], 'rois_mask_.nii')
        copyfile(src_file, dst_file)


def mask_list(working_directory, mask_root_directory, status_file_name, file_name_ending):
    """"Create text files with patient IDs and the corresponding directories of masks or ROIs.
    It can generally be used to map any file referring to this patient and is stored withn
    the patient subdirectory under the mask root directory.

    :param string working_directory:
    :param string mask_root_directory:
    :param string status_file_name: name of the minor map text file"""
    if os.path.isfile(os.path.join(working_directory, status_file_name)):
        os.remove(os.path.join(working_directory, status_file_name))
    all_masks = {}
    mask_found = {}
    for dirpath, dirname, file_names in os.walk(mask_root_directory):
        dirpath = dirpath.replace('\\','/')
        for a_file in file_names:
            if a_file.endswith('.dcm'):
                patient_ID = dicom.read_file(os.path.join(dirpath, a_file)).PatientID
                all_masks[patient_ID] = dirpath
                mask_found[patient_ID] = False
                break
        for a_file in file_names:
            if a_file.endswith(file_name_ending):
                mask_found[patient_ID] = True
    #if not os.path.isfile('uncorrected_mask_paths.txt'):
    os.chdir(working_directory)
    with open(status_file_name, 'w') as outstream:
        for a_mask in all_masks:
            if mask_found[a_mask]:
                outstream.write(a_mask + '\n' + 'x &\t x &\t' + all_masks[a_mask] + '\n')
            else:
                outstream.write(a_mask + '\n' + 'x &\t x &\t no mask found' + '\n')

def save_mask(cor_mask_for_itk, patient_ID, working_directory):

    os.chdir(working_directory)
    current_patient_folder = working_directory + '/' +patient_ID
    if not os.path.exists(current_patient_folder):
        os.mkdir(current_patient_folder)
    os.chdir(current_patient_folder)
    # assuming corrected mask
    file_name = 'mask_cor.nii'
    itk_mask = sitk.GetImageFromArray(cor_mask_for_itk)
    writer = sitk.ImageFileWriter()
    writer.SetFileName(file_name)
    writer.Execute(itk_mask)
    os.chdir(working_directory)
    i = 0
    with open('corrected_masks.txt', 'a') as outstream:
        outstream.write(patient_ID + '\n' + 'x &\t x &\t' + current_patient_folder + '\n')
        i += 1

def copy_mask(itk_mask, patient_ID, working_directory):

    os.chdir(working_directory)
    current_patient_folder = working_directory + '/' +patient_ID
    if not os.path.exists(current_patient_folder):
        os.mkdir(current_patient_folder)
    os.chdir(current_patient_folder)
    # assuming corrected mask
    file_name = 'mask_cor.nii'
    dist_file1 = os.path.join(current_patient_folder, file_name)
    copyfile(itk_mask, dist_file1)
    os.chdir(working_directory)
    i = 0
    with open('corrected_masks.txt', 'a') as outstream:
        outstream.write(patient_ID + '\n' + 'x &\t x &\t' + current_patient_folder + '\n')
        i += 1

def check_mask_integrity(mask_dir, working_directory, patient_ID, mask_format_name, fix_masks = False):
    """"Masks are extracted with a in-house developped tool (SegmentationTool3) on *MeVisLab*
    some masks are hollow on some planes (only the contours are saved). Check for a mask if it is hollow or not in each plane.
    If it is so, fill holes with ones and save a corrected copy. It performs operations on a single mask file each time.
    This function is by default deactivated, as most masks are fixed manually. However, it is still possible to deploy.

    :param string mask_dir: mask directory
    :param string working_directory:
    :param string patient_ID:
    :param string mask_format_name: which kind of segmentation files to fix (peripheral zone or entire prostate). It has not been tested on ROI segmentations. Either way, it is not necessary for ROI segmentations. Thus, it can be *no_previous_mask_init.nii* for the entire prostate or *pz_mask.nii* for the peripheral zone.
    :param bool fix_masks: If set to True, perform a check, correct problematic masks and save corrected files. Default is False.
    """
    if fix_masks:
        for a_file in os.listdir(mask_dir):
            if a_file.endswith(mask_format_name):
                itkmask = sitk.ReadImage(os.path.join(mask_dir, a_file))
                mask = sitk.GetArrayFromImage(itkmask)
                # MERK! mask in z,y,x format
                mask = np.swapaxes(mask, 0, 2)

        # we already checked if mask exists in correct_masks
        corrected_mask = np.zeros(mask.shape)
        for i in range(mask.shape[-1]):
            slice = mask[:,:,i]
            if 1 in slice:
                corrected_slice = morphology.binary_fill_holes(slice)
                corrected_mask[:,:,i] = corrected_slice
                i += 1
        corrected_mask = corrected_mask.astype(int)
        cor_mask_for_itk = np.swapaxes(corrected_mask, 2, 0)
        save_mask(cor_mask_for_itk, patient_ID, working_directory)
    else:
        for a_file in os.listdir(mask_dir):
            if a_file.endswith(mask_format_name):
                itk_mask = os.path.join(mask_dir, a_file)
                copy_mask(itk_mask, patient_ID, working_directory)


def correct_masks(working_directory, mask_root_directory, mask_format_name):
    """"Read directories file from *uncorrected_mask_paths.txt*, serial feed to check_mask_integrity and update file entry for mask.
    *corrected_masks.txt* map is thereafter created, which contains the directories of the masks actually used during feature
    extraction.

    :param string mask_dir: mask directory
    :param string working_directory:
    :param string mask_format_name: which kind of segmentation files to fix (peripheral zone or entire prostate). It has not been tested on ROI segmentations. Either way, it is not necessary for ROI segmentations. Thus, it can be *no_previous_mask_init.nii* for the entire prostate or *pz_mask.nii* for the peripheral zone.
    """

    os.chdir(working_directory)
    mask_list(working_directory, mask_root_directory, 'uncorrected_mask_paths.txt', mask_format_name)

    if os.path.isfile('corrected_masks.txt'):
        os.remove(working_directory + '/' +'corrected_masks.txt')
    mask_dirs = get_dictionary('uncorrected_mask_paths.txt')
    for pIDs in mask_dirs:
            if 'found' in mask_dirs[pIDs]: # result of split with ' ' for no mask found
                with open('corrected_masks.txt', 'a') as outstream:
                    outstream.write(pIDs + '\n')
                    outstream.write('no mask' + '\n')
            else:
                check_mask_integrity(mask_dirs[pIDs], working_directory, pIDs, mask_format_name)

#
#if __name__ == "__main__":
#
#     file_list('F:/TCIA/wd', 'F:/TCIA/DOI')
#file_list2('F:/TCIA/wd', 'F:/TCIA/DOI', reset=True)
#file_list2('/media/pavlos/Elements/playground/xx', '/media/pavlos/Elements/to_upload/TCIA/DOI')
