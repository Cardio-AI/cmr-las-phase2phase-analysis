import glob
import logging
import os
from time import time

import SimpleITK as sitk
import numpy as np
import pandas as pd
from src.utils.Utils_io import ensure_dir

DEBUG = False



def get_trainings_files(data_path, suffix=None, ftype=None, fold=0, path_to_folds_df=None):
    """
    Load CMR images and masks for a given data path according to different suffixes
    If we path_to_folds (dataframe) is provided, use this dataframe and the parameter fold
    to split our x and y into train, val.
    If no path_to_folds is given use all files for train/validation (usually for inference with another dataset)
    :param data_path:
    :param suffix: Suffix to add before filetype to distinguish images and masks
    :param ftype: file type to load (nii or nrrd)
    :param fold:
    :param path_to_folds_df:
    :return: x_train, y_train, x_val, y_val
    """

    if suffix is None:
        suffix = {"image": "img", "mask": "msk"}
    if ftype is None:
        ftype = ".nii.gz"

    img_suffix = "*{}{}".format(suffix["image_suffix"], ftype)
    # load the cmr files with given pattern from the data path
    x = sorted(glob.glob(os.path.join(data_path, '**', img_suffix), recursive=True))

    if len(x) == 0:
        logging.error('No files found in: {}, with suffix {}, try to list all files in this directory:'.format(data_path,img_suffix))
        files_ = os.listdir(data_path)
        logging.error(files_)
        x = sorted(glob.glob(os.path.join(data_path, '*')))


    if suffix["mask_suffix"] is not None and len(suffix["mask_suffix"])>0:
        mask_suffix = "*{}{}".format(suffix["mask_suffix"], ftype)
        y = sorted(glob.glob(os.path.join(data_path, '**',mask_suffix), recursive=True))
        if len(y) == 0:
            logging.error('No masks found in: {}, with suffix {}, try to list all files in this directory:'.format(data_path,mask_suffix))
    else:
        y=x

    # Split dataset according to df if df given
    if path_to_folds_df and os.path.isfile(path_to_folds_df):
        df = pd.read_csv(path_to_folds_df)
        patients = df[df.fold.isin([fold])]
        # make sure we count each patient only once
        patients_train = patients[patients['modality'] == 'train']['patient'].str.lower().unique()
        patients_test = patients[patients['modality'] == 'test']['patient'].str.lower().unique()
        logging.info('Found {} images/masks in {}'.format(len(x), data_path))
        logging.info('Patients train: {}'.format(len(patients_train)))

        def filter_files_for_fold(list_of_filenames, list_of_patients):
            """Helper to filter one list by a list of substrings"""
            return [temp_str for temp_str in list_of_filenames
                    if get_patient(temp_str).lower() in list_of_patients]

        x_train = sorted(filter_files_for_fold(x, patients_train))
        y_train = sorted(filter_files_for_fold(y, patients_train))
        x_test = sorted(filter_files_for_fold(x, patients_test))
        y_test = sorted(filter_files_for_fold(y, patients_test))
        logging.info('Selected {} of {} files with {} of {} patients for training fold {}'.format(len(x_train), len(x),
                                                                                                  len(patients_train),
                                                                                                  len(df.patient.unique()),
                                                                                                  fold))

    else: # no dataframe given for splitting
        logging.info('No dataframe for splitting provided. Will use all files for train and validation')
        x_train = sorted(x)
        y_train = sorted(y)
        x_test = x_train
        y_test = y_train
        logging.info('Selected {} of {} files'.format(len(x_train), len(x)))
        if len(y) == 0:
            y_train = sorted(x)
            y_test = sorted(x)

    assert (len(x_train) == len(y_train)), 'len(x_train != len(y_train))'


    return x_train, y_train, x_test, y_test

def get_patient(filename_to_2d_nrrd_file):
    """
    Split the nrrd filename and returns the patient id
    split the filename by '_' returns the first two elements of that list
    If the filename contains '__' it returns the part before
    """
    import re
    m = re.search('__', filename_to_2d_nrrd_file)
    if m: # nrrd filename with '__'
        return os.path.basename(filename_to_2d_nrrd_file).split('__')[0]
    if os.path.basename(filename_to_2d_nrrd_file).startswith('patient'): # acdc file
        return os.path.basename(filename_to_2d_nrrd_file).split('_')[0]
    else: # gcn filename
        return '_'.join(os.path.basename(filename_to_2d_nrrd_file).split('_')[:2])

def split_one_4d_sitk_in_list_of_3d_sitk(img_4d_sitk, axis=None):
    """
    Splits a 4D dicom image into a list of 3D sitk images, copy alldicom metadata
    Parameters
    ----------
    img_4d_sitk :
    axis:

    Returns list of 3D-sitk objects
    -------
    """

    img_4d_nda = sitk.GetArrayFromImage(img_4d_sitk)

    if axis: # if we want to split by any other axis than 0, split by this axis and rearange the spacing in the reference sitk
        img_4d_nda = np.split(img_4d_nda,indices_or_sections=img_4d_nda.shape[axis], axis=axis)
        img_4d_nda = [np.squeeze(n) for n in img_4d_nda]
        if axis==1:
            # copy_meta takes the values from spacing, need to swap t and z if we split along the z axis
            old_spacing = img_4d_sitk.GetSpacing()
            img_4d_sitk.SetSpacing((old_spacing[0], old_spacing[1], old_spacing[3], old_spacing[2]))

    # create t 3d volumes
    list_of_3d_sitk = [copy_meta_and_save(new_image=img_3d, reference_sitk_img=img_4d_sitk, full_filename = None, overwrite_spacing = None, copy_direction = True) for img_3d in img_4d_nda]

    return list_of_3d_sitk


def describe_sitk(sitk_img):
    """
    Log some basic informations for a sitk image
    :param sitk_img:
    :return:
    """
    if isinstance(sitk_img, np.ndarray):
        sitk_img = sitk.GetImageFromArray(sitk_img.astype(np.float32))

    logging.info('size: {}'.format(sitk_img.GetSize()))
    logging.info('spacing: {}'.format(sitk_img.GetSpacing()))
    logging.info('origin: {}'.format(sitk_img.GetOrigin()))
    logging.info('direction: {}'.format(sitk_img.GetDirection()))
    logging.info('pixel type: {}'.format(sitk_img.GetPixelIDTypeAsString()))
    logging.info('number of pixel components: {}'.format(sitk_img.GetNumberOfComponentsPerPixel()))


def copy_meta_and_save(new_image, reference_sitk_img, full_filename=None, overwrite_spacing=None, copy_direction=True):
    """
    Copy metadata, UID and structural information from one image to another
    Works also for different dimensions, returns new_image with copied structural info
    Args:
        new_image ():
        reference_sitk_img ():
        full_filename ():
        overwrite_spacing ():
        copy_direction ():

    Returns: new_image with properties of the reference image

    """

    t1 = time()
    try:
        # make sure this method works with nda and sitk images
        if isinstance(new_image, np.ndarray):
            if len(new_image.shape) == 4:
                # 4D needs to be built from a series
                new_image = [sitk.GetImageFromArray(img) for img in new_image]
                new_image = sitk.JoinSeries(new_image)
            else:
                new_image = sitk.GetImageFromArray(new_image)
        if full_filename:
            ensure_dir(os.path.dirname(os.path.abspath(full_filename)))

        if reference_sitk_img is not None:
            assert (isinstance(reference_sitk_img, sitk.Image)), 'no reference image given'
            assert (isinstance(new_image, sitk.Image)), 'only np.ndarrays and sitk images could be stored'

            # copy metadata
            for key in reference_sitk_img.GetMetaDataKeys():
                new_image.SetMetaData(key, get_metadata_maybe(reference_sitk_img, key))
            #logging.debug('Metadata_copied: {:0.3f}s'.format(time() - t1))

            # copy structural informations to image with same dimension and size
            if (reference_sitk_img.GetDimension() == new_image.GetDimension()) and (reference_sitk_img.GetSize() == new_image.GetSize()):
                new_image.CopyInformation(reference_sitk_img)

            # same dimension (e.g. 4) but different size per dimension
            elif (reference_sitk_img.GetDimension() == new_image.GetDimension()):

                # copy spacing, origin and rotation but keep size as it is
                if copy_direction:
                    new_image.SetDirection(reference_sitk_img.GetDirection())
                new_image.SetOrigin(reference_sitk_img.GetOrigin())
                new_image.SetSpacing(reference_sitk_img.GetSpacing())

            # copy structural information to smaller images e.g. 4D to 3D
            elif reference_sitk_img.GetDimension() > new_image.GetDimension():
                shape_ = len(new_image.GetSize())
                reference_shape = len(reference_sitk_img.GetSize())

                # copy direction to smaller images
                # 1. extract the direction, 2. create a matrix, 3. slice by the new shape, 4. flatten
                if copy_direction:
                    direction = np.array(reference_sitk_img.GetDirection())
                    dir_ = direction.reshape(reference_shape, reference_shape)
                    direction = dir_[:shape_, :shape_].flatten()
                    new_image.SetDirection(direction)

                new_image.SetOrigin(reference_sitk_img.GetOrigin()[:shape_])
                new_image.SetSpacing(reference_sitk_img.GetSpacing()[:shape_])

            # copy structural information to bigger images e.g. 3D to 4D, fill with 1.0 spacing
            else:
                ones = [1.0] * (new_image.GetDimension() - reference_sitk_img.GetDimension())
                new_image.SetOrigin((*reference_sitk_img.GetOrigin(), *ones))
                new_image.SetSpacing((*reference_sitk_img.GetSpacing(), *ones))
                # we cant copy the direction from smaller images to bigger ones

            #logging.debug('spatial data_copied: {:0.3f}s'.format(time() - t1))

            if overwrite_spacing:
                new_image.SetSpacing(overwrite_spacing)

        if full_filename:
            # copy uid
            writer = sitk.ImageFileWriter()
            # writer.KeepOriginalImageUIDOn()
            writer.SetFileName(full_filename)
            writer.Execute(new_image)
            logging.debug('image saved: {:0.3f}s'.format(time() - t1))
            return True
        else:
            return new_image
    except Exception as e:
        logging.error('Error with saving file: {} - {}'.format(full_filename, str(e)))
        return False


def get_metadata_maybe(sitk_img, key, default='not_found'):
    # helper for unicode decode errors
    try:
        value = sitk_img.GetMetaData(key)
    except Exception as e:
        logging.debug('key not found: {}, {}'.format(key, e))
        value = default
    # need to encode/decode all values because of unicode errors in the dataset
    if not isinstance(value, int):
        value = value.encode('utf8', 'backslashreplace').decode('utf-8').replace('\\udcfc', 'ue')
    return value


def get_phases_as_onehot(file_path, df, start_id=0, temporal_sampling_factor=1, length=-1, weight=1):
    """
    load the phase info of an acdc data structure
    and converts it into a one-hot vector
    # order of phase classes, learnt by the phase regression model
    # ['ed#', 'ms#', 'es#', 'pf#', 'md#']

    :param file_path:
    :param df:
    :param start_id:
    :param temporal_sampling_factor:
    :param length:
    :param weight:

    :return:
    """
    patient_str = os.path.splitext(os.path.basename(file_path))[0]
    patient_str = extract_id(patient_str)
    assert len(
        patient_str) > 0, 'Empty patient id found, please check the extract_id in Dataset.py (usually there are path problems).'

    # Returns the indices in the following order: 'ED#', 'MS#', 'ES#', 'PF#', 'MD#'
    # Reduce the indices of the Excel sheet by start_id, as the indexes start at 0, the excel-sheet may start at 1
    # Transform them into a one-hot representation
    indices = df[df.patient.str.contains(patient_str)][
        ['ed#', 'ms#', 'es#', 'pf#', 'md#']]
    indices = indices.values[0].astype(int) - start_id

    # scale the idx as we resampled along t (we need to resample the indicies in the same way)
    if temporal_sampling_factor!=1:
        indices = np.round(indices * temporal_sampling_factor).astype(int)
        indices = np.clip(indices, a_min=0, a_max=length)

    if np.any(indices >= length):
        logging.error(
            'Found indices  greater than length of cardiac cycle, please check: {}'.format(indices[indices > length]))

    onehot = np.zeros((indices.size, length))
    onehot[np.arange(indices.size), indices] = weight
    return onehot

def extract_id(filename):
    """
    Extract patient id from given filename. Three patterns pre-defined (GCN, MnM2 and ACD) change here, if different pattern used
    """
    import re
    # Define patterns and their regex
    patterns = [
        r'\d+-([a-zA-Z0-9]+)_\d{4}-\d{2}-\d{2}.*',  # Pattern 1 (GCN): 0000-0ae4r74l_1900-01-01_...
        r'(\d+)_LA_CINE.*',  # Pattern 2 (MnM2): 039_LA_CINE.nii.gz
        r'patient(\d+)_.*'  # Pattern 3 (ACDC): patient001_4d.nii.gz
    ]

    for pattern in patterns:
        match = re.search(pattern, filename)
        if match:
            return match.group(1)  # Extract the first capture group

    return None  # Return None if no pattern matches

def check_input_data(img, mask):
    if isinstance(img, sitk.Image):
        img = sitk.GetArrayFromImage(img).astype(np.float32)

    if isinstance(mask, sitk.Image):
        mask = sitk.GetArrayFromImage(mask).astype(np.float32)

    if not isinstance(img, np.ndarray) and img is not None:
        # try it to unpack a tf.tensor
        img = img.numpy().astype(np.float32)

    if not isinstance(mask, np.ndarray) and mask is not None:
        mask = mask.numpy().astype(np.float32)

    # don't print anything if no images nor masks are given
    if img is None and mask is None:
        logging.error('No image data given')
        raise ('No image data given in augmentation compose')

    given = {"return_image_and_mask": False, "mask_given": True, "img_given": True}

    # replace mask with empty slice if none is given
    if mask is None:
        given["mask_given"] = False
        shape = img.shape
        mask = np.zeros((shape[0], shape[1], 3))
    # replace image with empty slice if none is given
    elif img is None:
        given["img_given"] = False
        shape = mask.shape
        img = np.zeros((shape[0], shape[1], 1))
    # set return_image_and_mask to True
    else:
        given["return_image_and_mask"] = True

    return img, mask, given


def load_phase_reg_exp(exp_root, junk=None):
    """
    Load the predicted numpy files of a 4-fold cross validation experiment

    Parameters
    ----------
    exp_root : (str) path to the experiment root

    Returns (tuple of ndarrays), nda_vects, gt, pred, gt_len, mov, patients
    -------

    """
    from time import time
    import os
    import glob
    t0 = time()

    local_moved = os.path.join(exp_root, 'moved')
    paths_to_vect_npy = sorted(glob.glob(os.path.join(local_moved, '*vects_*.npy')))
    if junk is None:
        junk = len(paths_to_vect_npy)
    paths_to_vect_npy = paths_to_vect_npy[:junk]
    print(paths_to_vect_npy)
    nda_vects = np.concatenate([np.load(path_) for path_ in paths_to_vect_npy], axis=0)
    t1 = time()
    if DEBUG: print('numpy vects loaded: {:0.3f} s'.format(t1 - t0))

    # load the phase gt and pred
    pred_path = os.path.join(exp_root, 'pred')
    paths_to_phase_npy = sorted(glob.glob(os.path.join(pred_path, '*gtpred*.npy')))[:junk]
    nda_phase = np.concatenate([np.load(path_) for path_ in paths_to_phase_npy], axis=1)
    t1 = time()
    if DEBUG: print('numpy phases loaded: {:0.3f} s'.format(t1-t0))
    gt_, pred_ = np.split(nda_phase, axis=0, indices_or_sections=2)

    gt = gt_[0, :, 0]
    pred = pred_[0, :, 0]
    gt_len = gt_[0, :, 1]

    # load the moved examples for easier understanding of the dimensions
    pathtomoved = sorted(glob.glob(os.path.join(local_moved, '*moved*.npy')))[:junk]
    mov = np.concatenate([np.load(path_) for path_ in pathtomoved], axis=0)
    t1 = time()
    if DEBUG: print('numpy moved loaded: {:0.3f} s'.format(t1 - t0))


    # load a mapping to the original patient ids
    patients = []
    if os.path.exists(os.path.join(pred_path, 'patients.txt')):
        with open(os.path.join(pred_path, 'patients.txt'), "r") as f_:
            lines = f_.readlines()
            _ = [patients.append(p) for p in lines]
    if DEBUG: print('created a patient list: {:0.3f} s'.format(time() - t0))
    patients = patients[:junk*10] # one npy file represents a junk of 10 patients
    masks = None

    # We don't need to load the config here, for inference we will not have a config within the path,
    # so we simply check if use_segmentation files are there and load them if available
    pathtomasks = sorted(glob.glob(os.path.join(pred_path, '*segmentation*.npy')))[:junk]
    if len(pathtomasks) > 0:
        masks = np.concatenate([np.load(path_) for path_ in pathtomasks], axis=0)
        t1 = time()
        if DEBUG: print('numpy segmentation loaded: {:0.3f} s'.format(t1 - t0))
    if masks is not None:
        if masks.shape[0] == nda_vects.shape[0]:
            return nda_vects.astype(np.float16), gt, pred, gt_len, mov.astype(np.float16), masks.astype(np.uint8), patients
        elif nda_vects.shape[0] % masks.shape[0] == 0:
            masks = np.tile(masks, (nda_vects.shape[0] // masks.shape[0]))
            return nda_vects.astype(np.float16), gt, pred, gt_len, mov.astype(np.float16), masks.astype(np.uint8), patients
    else:
        return nda_vects.astype(np.float16), gt, pred, gt_len, mov.astype(np.float16), None, patients

