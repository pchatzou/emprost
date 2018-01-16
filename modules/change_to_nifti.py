import SimpleITK as sitk
import os
import sys

def change_an_image(img_file):

    """ Change an image from mhd or mha itk format to nii itk format. This is mostly for legacy, as in previous efforts
    mhds were used.

    :param string img_file: full path to the file
    """
    img_file = img_file.replace('\\', '/')
    file_name = img_file.split('/')[-1]
    directory = '/'.join(img_file.split('/')[:-1])
    if 'mask' in file_name or 'roi' in file_name:
        mhd_image = sitk.ReadImage(img_file, sitk.sitkUInt8) #read binary files as uint8, minimal space
    else:
        mhd_image = sitk.ReadImage(img_file, sitk.sitkInt32)
    writer = sitk.ImageFileWriter()
    new_name = os.path.splitext(file_name)[0] + '.nii'
    os.chdir(directory)
    if os.path.isfile(new_name):
        return
    writer.SetFileName(new_name)
    writer.Execute(mhd_image)
    os.remove(file_name)
    raw_file_name = os.path.splitext(file_name)[0] + '.raw'
    try:
        os.remove(raw_file_name)
    except:
        pass # mha has no raw

def resize_nifti(img_file):

    """"It is a common case that image files when transferred from DICOM format use a 64 bit float format in itk images.
    This function sets unsigned 8 bit integer format for mask or roi files and int 32 for other images, resulting in an
    important file size reduction.

    :param string img_file: full path to the file
    """

    img_file = img_file.replace('\\', '/')
    file_name = img_file.split('/')[-1]
    directory = '/'.join(img_file.split('/')[:-1])
    if 'mask' in file_name or 'roi' in file_name:
        nii_image = sitk.ReadImage(img_file, sitk.sitkUInt8) #read binary files as uint8, minimal space
    else:
        nii_image = sitk.ReadImage(img_file, sitk.sitkInt32)
    os.chdir(directory)
    if os.path.isfile(file_name):
        os.remove(file_name)
    writer = sitk.ImageFileWriter()
    writer.SetFileName(file_name)
    writer.Execute(nii_image)


def find_mhds(root_directory, change_to_nifti = True):

    """"change all .mhd or .mha images under a root directory to .nii"""

    for dirpath, dirname, file_names in os.walk(root_directory):
        for a_file_name in file_names:
            if change_to_nifti:
                if a_file_name.endswith('.mhd') or a_file_name.endswith('.mha'):
                    full_path_to_file = os.path.join(dirpath, a_file_name)
                    change_an_image(full_path_to_file)
            else:
                if a_file_name.endswith('.nii'):
                    full_path_to_file = os.path.join(dirpath, a_file_name)
                    resize_nifti(full_path_to_file)

def set_masks_world(root_directory):

    """"Each patient folder has 3 nii files, one for masks, one for ROIs and the axial T2. In the working directory,
    each of these is set to the same values according to the mask, which is set according to the T2 DICOM.

    :param string root_directory:
    """

    flag0 = False
    flag1 = False
    for dirpath, dirname, file_names in os.walk(root_directory):
        for a_directory in dirname:
            a_path = os.path.join(dirpath, a_directory)
            fix_files = []
            for a_file_name in os.listdir(a_path):
                if a_file_name == 'rois_mask_.nii':
                    reference_file = os.path.join(a_path, a_file_name)
                    flag0 = True
                if a_file_name.endswith('.nii') and a_file_name != 'rois_mask_.nii':
                    fix_files.append(os.path.join(a_path, a_file_name))
            if len(fix_files) == 2:
                flag1 = True
            if flag1 and flag0:
                ref_img = sitk.ReadImage(reference_file)
                for a_file in fix_files:
                    print('Setting world coordinates in ' + a_file)
                    img = sitk.ReadImage(a_file)
                    img.SetOrigin(ref_img.GetOrigin())
                    img.SetSpacing(ref_img.GetSpacing())
                    img.SetDirection(ref_img.GetDirection())
                    os.remove(a_file)
                    sitk.WriteImage(img, a_file)


#change_an_image('C:/Users/157136/Desktop/pgg/no_previous_mask_init.mhd', 'no_previous_mask_init.mhd', 'C:/Users/157136/Desktop/pgg')
#find_mhds('C:/Users/157136/Desktop/pgg/')
#find_mhds('C:/Users/157136/Documents/msk_root')
# if __name__ == "__main__":
#
#     find_mhds('scratch/pchatzoudis/msk_root')

#find_mhds('C:/Users/157136/Desktop/pground/msk_root', change_to_nifti=False)
#set_masks_world('F:/pground/wd')
# if __name__ == "__main__":
#
#     set_masks_world(sys.argv[1])
#find_mhds('/media/pavlos/Elements/to_upload/msk_root/')
#set_masks_world('/media/pavlos/Elements/SERVER/working_directory/to_upload/wd_PCMM_all_features_whole_sliding')