import logging
import sys
import os
import SimpleITK as sitk

from sklearn.preprocessing import RobustScaler
from src.data.Dataset import get_metadata_maybe
import numpy as np

from albumentations import GridDistortion, RandomRotate90, Compose, ReplayCompose, Downscale, ShiftScaleRotate, \
    HorizontalFlip, VerticalFlip, ElasticTransform
import cv2


def load_masked_img(sitk_img_f, mask=False, masking_values=[1, 2, 3], replace=('img', 'msk'), mask_labels=[0, 1, 2, 3],
                    maskAll=True):
    """
    Wrapper for opening a dicom image, this wrapper could also load the corresponding use_segmentation map and mask the loaded image on the fly
     if mask == True use the replace wildcard to open the corresponding use_segmentation mask
     Use the values given in mask_labels to transform the one-hot-encoded mask into channel based binary mask
     Mask/cut the CMR image/volume by the given labels in masking_values

    Parameters
    ----------
    sitk_img_f : full filename for a dicom image/volume, could be any format supported by sitk
    mask : bool, if the sitk image loaded should be cropped by any label of the corresponding mask
    masking_values : list of int, defines the area/labels which should be cropped from the original CMR
    replace : tuple of replacement string to get from the image filename to the mask filename
    mask_labels : list of int
    maskAll: bool,  true: mask all timesteps of the CMR by the mask,
                    false: return the raw CMR for timesteps without a mask
    """

    assert os.path.isfile(sitk_img_f), 'no valid image: {}'.format(sitk_img_f)
    img_original = sitk.ReadImage(sitk_img_f, sitk.sitkFloat32)

    if img_original.GetSize()[-1]==1: # this is a 2D slice saved as 3D nrrd, as expected by MITK
        img_original = img_original[:,:,0]

    if mask:
        sitk_mask_f = sitk_img_f.replace(replace[0], replace[1])
        msk_original = sitk.ReadImage(sitk_mask_f)

        img_nda = sitk.GetArrayFromImage(img_original)
        msk_nda = transform_to_binary_mask(sitk.GetArrayFromImage(msk_original), mask_values=mask_labels)

        # mask by different labels, sum up all masked channels
        temp = np.zeros(img_nda.shape)
        if maskAll:  # mask all timesteps
            for c in masking_values:
                # mask by different labels, sum up all masked channels
                temp += img_nda * msk_nda[..., c].astype(np.bool)
            sitk_img = sitk.GetImageFromArray(temp)
        else:
            for t in range(img_nda.shape[0]):
                if msk_nda[t].sum() > 0:  # mask only timesteps with a given mask
                    for c in masking_values:
                        # mask by different labels, sum up all masked channels
                        temp[t] += img_nda[t] * msk_nda[t][..., c].astype(np.bool)
            sitk_img = sitk.GetImageFromArray(temp)

        # copy metadata
        for tag in img_original.GetMetaDataKeys():
            value = get_metadata_maybe(img_original, tag)
            sitk_img.SetMetaData(tag, value)
        sitk_img.SetSpacing(img_original.GetSpacing())
        sitk_img.SetOrigin(img_original.GetOrigin())

        img_original = sitk_img

    return img_original

def resample_3D(sitk_img, size=(256, 256, 12), spacing=(1.25, 1.25, 8), interpolate=sitk.sitkNearestNeighbor, dtype=None):
    """
    resamples an 3D sitk image or numpy ndarray to a new size with respect to the giving spacing
    This method expects size and spacing in sitk format: x, y, z
    :param sitk_img: sitk.Image
    :param size: (tuple) of int with the following order x,y,z
    :param spacing: (tuple) of float with the following order x,y,z
    :param interpolate:
    :return: the resampled image in the same datatype as submitted, either sitk.image or numpy.ndarray
    """

    return_sitk = True

    if isinstance(sitk_img, np.ndarray):
        return_sitk = False
        sitk_img = sitk.GetImageFromArray(sitk_img)

    assert (isinstance(sitk_img, sitk.Image)), 'wrong image type: {}'.format(type(sitk_img))

    # minor data type corrections
    size = [int(elem) for elem in size]
    spacing = [float(elem) for elem in spacing]

    if sitk_img.GetDimension() == 3:
        sitk_img = sitk_img[:,:,0]
    resampler = sitk.ResampleImageFilter()
    resampler.SetInterpolator(interpolate)
    resampler.SetSize(size)
    resampler.SetOutputDirection(sitk_img.GetDirection())
    resampler.SetOutputSpacing(spacing)
    resampler.SetOutputOrigin(sitk_img.GetOrigin())

    resampled = resampler.Execute(sitk_img)

    # return the same data type as input datatype
    if return_sitk:
        return resampled
    else:
        return sitk.GetArrayFromImage(resampled)


def augmentation_compose_2d_3d_4d(img, mask, probabillity=1, config=None):
    """
    Apply a composition of different augmentation steps,
    either on 2D or 3D image/mask pairs,
    apply
    :param img:
    :param mask:
    :param probabillity:
    :return: augmented image, mask
    """
    from Dataset import check_input_data
    if config is None:
        config = {}

    img, mask, given = check_input_data(img, mask)
    targets, data = {}, {}
    img_placeholder = 'image'
    mask_placeholder = 'mask'

    def update_data_dict(data, img, mask, prefix, z, t=None):
        key_img = f"{prefix}_{z}" if t is None else f"{prefix}_{t}_{z}"
        if given["img_given"]: data[key_img] = img
        if given["mask_given"]: data[key_img.replace(prefix, mask_placeholder)] = mask
        if given["img_given"]: targets[key_img] = "image"
        if given["mask_given"]: targets[key_img.replace(prefix, mask_placeholder)] = "mask"

    def stack_augmented_slices(augmented, prefix, shape, dims):
        stacks = {"image": [], "mask": []}
        for indices in np.ndindex(*shape):
            key_img = f"{prefix}_" + "_".join(map(str, indices))
            if given["img_given"]: stacks["image"].append(augmented[key_img])
            if given["mask_given"]: stacks["mask"].append(augmented[key_img.replace(prefix, mask_placeholder)])
        if given["img_given"]: augmented["image"] = np.stack(stacks["image"], axis=tuple(dims))
        if given["mask_given"]: augmented["mask"] = np.stack(stacks["mask"], axis=tuple(dims))

    # Initialize the data dictionary and process based on ndim
    if img.ndim in (3, 4):
        middle_indices = [s // 2 for s in img.shape[:2]]
        middle_img = img[middle_indices[0]][middle_indices[1]] if img.ndim == 4 else img[middle_indices[0]]
        middle_mask = mask[middle_indices[1]] if given["mask_given"] else mask
        data = {"image": middle_img, "mask": middle_mask}

        # Add all slices to the data dictionary
        for indices in np.ndindex(*img.shape[:2]):
            update_data_dict(data, img[indices], mask[indices], img_placeholder, indices[1],
                             indices[0] if img.ndim == 4 else None)

    # Create and apply augmentation
    aug = _create_aug_compose(p=probabillity, targets=targets, config=config)
    augmented = aug(**data)
    logging.debug(augmented['replay'])

    # Reassemble augmented slices
    if img.ndim == 3:
        stack_augmented_slices(augmented, img_placeholder, img.shape[:1], (0,))
    elif img.ndim == 4:
        stack_augmented_slices(augmented, img_placeholder, img.shape[:2], (0, 1))

    # Return the result
    if given["return_image_and_mask"]:
        return augmented["image"], augmented["mask"]
    return augmented["image"]


def _create_aug_compose(p=1, border_mode=cv2.BORDER_CONSTANT, val=0, targets=None, config=None):
    """
    Create an Albumentations Reply compose augmentation based on the config params
    Parameters
    ----------
    p :
    border_mode :
    val :
    targets :
    config :
    Note for the border mode from openCV:
    BORDER_CONSTANT    = 0,
    BORDER_REPLICATE   = 1,
    BORDER_REFLECT     = 2,
    BORDER_WRAP        = 3,
    BORDER_REFLECT_101 = 4,
    BORDER_TRANSPARENT = 5,
    BORDER_REFLECT101  = BORDER_REFLECT_101,
    BORDER_DEFAULT     = BORDER_REFLECT_101,
    BORDER_ISOLATED    = 16,

    Returns
    -------

    """
    if config is None:
        config = {}
    if targets is None:
        targets = {}
    prob = config.get('AUGMENT_PROB', 0.8)
    border_mode = config.get('BORDER_MODE', border_mode)
    val = config.get('BORDER_VALUE', val)
    augmentations = []
    if config.get('HFLIP', False):augmentations.append(HorizontalFlip(p=prob))
    if config.get('VFLIP', False): augmentations.append(VerticalFlip(p=prob))
    if config.get('RANDOMROTATE', False): augmentations.append(RandomRotate90(p=0.2))
    if config.get('SHIFTSCALEROTATE', False): augmentations.append(
        ShiftScaleRotate(p=prob, rotate_limit=0, shift_limit=0.025, scale_limit=0, value=val, border_mode=border_mode))
    if config.get('GRIDDISTORTION', False): augmentations.append(
        GridDistortion(p=prob, value=val, border_mode=border_mode))
    if config.get('DOWNSCALE', False): augmentations.append(Downscale(scale_min=0.9, scale_max=0.9, p=prob))
    if config.get('ELASTICTRANSFORM', False): augmentations.append(ElasticTransform(p=prob, alpha=20, sigma=20, alpha_affine=20))
    return ReplayCompose(augmentations, p=p,
                         additional_targets=targets)


def transform_to_binary_mask(mask_nda, mask_values=None):
    """
    Transform from a value-based representation to a binary channel based representation
    :param mask_nda:
    :param mask_values:
    :return:
    """
    # transform the labels to binary channel masks

    if mask_values is None:
        mask_values = [0, 1, 2, 3]
    mask = np.zeros((*mask_nda.shape, len(mask_values)), dtype=np.bool)
    for ix, mask_value in enumerate(mask_values):
        mask[..., ix] = mask_nda == mask_value
    return mask


def from_channel_to_flat(binary_mask, start_c=0, threshold=0.5):
    """
    Transform a tensor or numpy nda from a channel-wise (one channel per label) representation
    to a value-based representation
    :param binary_mask:
    :return:
    """
    # convert to bool nda to allow later indexing
    binary_mask = binary_mask >= threshold

    # reduce the shape by the channels
    temp = np.zeros(binary_mask.shape[:-1], dtype=np.uint8)

    for c in range(binary_mask.shape[-1]):
        temp[binary_mask[..., c]] = c + start_c
    return temp


def clip_quantile(img_nda, upper_quantile=.999, lower_boundary=0):
    """
    clip to values between 0 and .999 quantile
    :param img_nda:
    :param upper_quantile:
    :return:
    """

    ninenine_q = np.quantile(img_nda.flatten(), upper_quantile, overwrite_input=False)

    return np.clip(img_nda, lower_boundary, ninenine_q)


def normalise_image(img_nda, normaliser='minmax'):
    """
    Normalise Images to a given range,
    normaliser string repr for scaler, possible values: 'MinMax', 'Standard' and 'Robust'
    if no normalising method is defined use MinMax normalising
    :param img_nda:
    :param normaliser:
    :return:
    """
    # ignore case
    normaliser = normaliser.lower()

    if normaliser == 'standard':
        return (img_nda - np.mean(img_nda)) / (np.std(img_nda) + sys.float_info.epsilon)

        # return StandardScaler(copy=False, with_mean=True, with_std=True).fit_transform(img_nda)
    elif normaliser == 'robust':
        return RobustScaler(copy=False, quantile_range=(0.0, 95.0), with_centering=True,
                            with_scaling=True).fit_transform(img_nda)
    else:
        return (img_nda - img_nda.min()) / (img_nda.max() - img_nda.min() + sys.float_info.epsilon)


def pad_and_crop(ndarray, target_shape=(10, 10, 10)):
    """
    Center pad and crop a np.ndarray in one step
    Accepts any shape (2D,3D, ..nD) to a given target shape
    Expects ndarray.ndim == len(target_shape)
    This method is idempotent, which means the pad operation is the numeric inverse of the crop operation
    Pad and crop must be the complementary,
    In cases of non odd shapes in any dimension this method defines the center as:
    pad_along_dimension_n = floor(border_n/2),floor(border_n/2)+1
    crop_along_dimension_n = floor(margin_n/2)+1, floor(margin_n/2)
    Parameters:
    ----------
    ndarray : numpy.ndarray of any shape
    target_shape : must have the same length as ndarray.ndim

    Returns np.ndarray with each axis either pad or crop
    -------

    """
    cropped = np.zeros(target_shape, dtype=np.float32)
    target_shape = np.array(target_shape)
    logging.debug('input shape, crop_and_pad: {}'.format(ndarray.shape))
    logging.debug('target shape, crop_and_pad: {}'.format(target_shape))

    diff = ndarray.shape - target_shape

    # divide into summands to work with odd numbers
    # take the same numbers for left or right padding/cropping if the difference is dividable by 2
    # else take floor(x),floor(x)+1 for PAD (diff<0)
    # else take floor(x)+1, floor(x) for CROP (diff>0)
    # This behaviour is the same for each axis
    d = list(
        (int(x // 2), int(x // 2)) if x % 2 == 0 else (int(np.floor(x / 2)), int(np.floor(x / 2) + 1)) if x < 0 else (
            int(np.floor(x / 2) + 1), int(np.floor(x / 2))) for x in diff)
    # replace the second slice parameter if it is None, which slice until end of ndarray
    d = list((abs(x), abs(y)) if y != 0 else (abs(x), None) for x, y in d)
    # create a bool list, negative numbers --> pad, else --> crop
    pad_bool = diff < 0
    crop_bool = diff > 0

    # create one slice obj for cropping and one for padding
    pad = list(i if b else (None, None) for i, b in zip(d, pad_bool))
    crop = list(i if b else (None, None) for i, b in zip(d, crop_bool))

    # Create one tuple of slice calls per pad/crop
    # crop or pad from dif:-dif if second param is not None, else replace by None to slice until the end
    # slice params: slice(start,end,steps)
    pad = tuple(slice(i[0], -i[1]) if i[1] != None else slice(i[0], i[1]) for i in pad)
    crop = tuple(slice(i[0], -i[1]) if i[1] != None else slice(i[0], i[1]) for i in crop)

    # crop and pad in one step
    cropped[pad] = ndarray[crop]
    return cropped


def calc_resampled_size(sitk_img, target_spacing):
    """
    Calculate size of resampled image from original sitk image (size and spacing) and given target spacing:
    (orig_size * orig_spacing) / target_spacing
    :param sitk_img:
    :param target_spacing:
    :return: a list with the rounded new image size after resampling.
    """
    if type(target_spacing) in [list, tuple]:
        target_spacing = np.array(target_spacing)

    orig_size = np.array(sitk_img.GetSize())
    orig_spacing = np.array(sitk_img.GetSpacing())

    if len(orig_size) != len(target_spacing):
        orig_size = orig_size[:-1]
        orig_spacing = orig_spacing[:-1]

    logging.debug('old size: {}, old spacing: {}, target spacing: {}'.format(orig_size, orig_spacing,
                                                                             target_spacing))
    new_size = (orig_size * orig_spacing) / target_spacing
    return list(np.around(new_size).astype(np.int64))