#encoding:utf-8
import SimpleITK as sitk
import os
import dicom
from global_functions import regex_match

def dcm_series_to_nii(dump_directory, working_directory, overwrite = False):
    """Reads dicom files from dump directory and writes single nii file
       in working directory for feeding in elastix. Also creates the patient directory within the working directory.

       ..note:: if a file is already there it is overwritten

    :param string dump_directory: directory of the dicom files
    :param string working_directory:
    :param bool overwrite: If set to True, patient folder is removed and a new one is created. Default is False.
    """

    # read image series
    os.chdir(working_directory)
    reader = sitk.ImageSeriesReader()
    dcms_names = sitk.ImageSeriesReader_GetGDCMSeriesFileNames(dump_directory)
    reader.SetFileNames(dcms_names)
    uni_dicom = reader.Execute()
    # create nii file name as PatientID+seriesDescription
    for a_file in os.listdir(dump_directory):
        if a_file.endswith('.dcm'):
            get_tags = dicom.read_file(dump_directory + '/' + a_file)
            break
    if hasattr(get_tags, 'SeriesDescription'):
        series = get_tags.SeriesDescription
    elif hasattr(get_tags, 'SeriesName'):
        series = get_tags.SeriesName
    else:
        series = 'unspecified_series'
    patient_ID = get_tags.PatientID
    output_file_name = patient_ID + series + '.nii'
    output_file_name = output_file_name.replace(' ', '_')
    current_patient_folder = working_directory + '/' + patient_ID
    if not os.path.exists(current_patient_folder):
        os.mkdir(current_patient_folder)
    os.chdir(current_patient_folder)
    if os.path.isfile(output_file_name):
        if overwrite:
            os.remove(os.path.join(current_patient_folder, output_file_name))
        else:
            return
    sitk.WriteImage(uni_dicom, output_file_name)
    return output_file_name, current_patient_folder

def roi_registration(directory):

    """Fine tuned call of *Simple Elastix* to apply a 3D BSpline registration on the ROIs.
    In order to call it, a build of *SimpleITK* that includes *Simple Elastix* must be present.
    Registered file and transform are saved in the patient subdirectory within the mask root directory.
    It takes as input a single argument, which is a directory. This must contain the fixed and moving
    images, that must be named as *no_previous_mask_init.nii* and *macro_mask_init.nii* and both
    come from manual preprocessing. If for any reason the file is not proper for processing, a *note.txt*
    file mentioning the reason for this insufficiency shall also reside within the directory.
    In that case, where a *note.txt* file is also present, no registration is attempted.

    :param string directory:"""
    os.chdir(directory)
    elastix = sitk.SimpleElastix()
    os.chdir('..')
    if 'note.txt' in os.listdir('./'):
        return
    fixedImg = sitk.ReadImage('no_previous_mask_init.nii')
    elastix.SetFixedImage(fixedImg)
    os.chdir(directory)
    movingImg = sitk.ReadImage('macro_mask_init.nii')
    elastix.SetMovingImage(movingImg)
    parameterMap = sitk.GetDefaultParameterMap('bspline')
    parameterMap['FixedImageDimension'] = ['3']
    parameterMap['MovingImageDimension'] = ['3']
    parameterMap['ImagePyramidSchedule'] = ['8 8 8 4 4 4 2 2 2 1 1 1']
    parameterMap['FinalBSplineInterpolationOrder'] = ['0']
    elastix.SetParameterMap(parameterMap)
    elastix.Execute()
    tp = elastix.GetTransformParameterMap()
    transformix = sitk.SimpleTransformix()
    transformix.SetTransformParameterMap(tp)
    movingImg2 = sitk.ReadImage('rois_mask_init.nii')
    transformix.SetMovingImage(movingImg2)
    transformix.Execute()
    result = transformix.GetResultImage()
    os.chdir('..')
    sitk.WriteImage(result, 'rois_mask_.nii')

def patient_identifier(mask_root_directory, all_modalities_file, patient_no):
    """Obsolete, shall not be used"""
    for dirpath, dirname, file_names in os.walk(mask_root_directory):
        dirpath = dirpath.replace('\\', '/')
        for a_file in file_names:
            if a_file.endswith('.dcm'):
                dcm_tags = dicom.read_file(os.path.join(dirpath, a_file))
                patient_name = dcm_tags.PatientName
                if patient_no in patient_name:
                #if patient_no in patient_name:
                    return dirpath, False
    with open(all_modalities_file, 'r') as data_map:
        for line in data_map:
            if patient_no in line:
                while True:
                    line = next(data_map, '')
                    line = line.rstrip()
                    line_contents = line.split('&\t')
                    if regex_match('axial T2',line_contents[0]):
                        return line_contents[-1], True

def recursive_patient_registration(msk_root):

    """Obsolete, shall not be used"""
    for dirpath, dirname, file_names in os.walk(msk_root):
            dirpath = dirpath.replace('\\','/')
            for a_dir in dirname:
                d2 = os.path.join(dirpath, a_dir)
                d2 = os.path.join(d2, 'pre')
                roi_registration(d2)

#recursive_patient_registration('/home/pchatzoudis/msk_root')
#recursive_patient_registration('/home/pchatzoudis/msk_root')
#recursive_patient_registration('C:/Users/157136/Documents/msk_root')



#roi_registration('C:/Users/157136/Documents/msk_root/s55/pre')
# xx = patient_identifier('C:/Users/157136/Documents/allPCMM', 'C:/Users/157136/PycharmProjects/llorona/father/src/preprocessing/all_modalities.txt', '01.0027')
# pass
#dcm_series_to_nii('F:/pground/debug_pgg/00001', 'F:/pground/debug_pgg', overwrite=True)
