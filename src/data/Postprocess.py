import numpy as np
from scipy.interpolate import interp1d
from src.models.KerasLayers import minmax_lambda
from skimage import measure

def align_resample_multi(dirs, norms, gt, gt_len,  target_t=30, normalise_dir=True, normalise_norm=True, rescale=False):
    """
    Alignment wrapper for a full dataset
    align norm and direction by the cardiac phase ED
    resample all 1D feature vectors to the same length and min/max normalise into [0,1] and [-1,1]
    this should help to validate the pre-defined rules, detect outliers, and qualitatively evaluate the motion descriptor
    Args:
        dirs (ndarray):
        gt (ndarray):
        gt_len (int):
        target_t (int):
        rescale (bool)

    Returns:

    """

    dir_ndas, norm_ndas, indices = [], [], []

    number_of_patients = len(dirs)

    for p in range(number_of_patients):
        if p % 20 == 0: print('processing patient : {}'.format(p))
        cardiac_cycle_length = int(gt_len[p, :, 0].sum())
        ind = np.argmax(gt[p][:cardiac_cycle_length], axis=0)
        gt_onehot = gt[p][:cardiac_cycle_length]
        dir_nda = dirs[p][:cardiac_cycle_length]
        norm_nda = norms[p][:cardiac_cycle_length]
        dir_, norm_, ind_ = align_and_resample(cardiac_cycle_length=cardiac_cycle_length,
                                               ind=ind,
                                               gt_onehot=gt_onehot,
                                               dir_1d_mean=dir_nda,
                                               norm_1d_mean=norm_nda,
                                               target_t=target_t,
                                               normalise_dir=normalise_dir,
                                               normalise_norm=normalise_norm,
                                               rescale=rescale)  # align by the gt ED phase or the predicted ED phase
        dir_ndas.append(dir_)
        norm_ndas.append(norm_)
        indices.append(ind_)
    if rescale:
        result = np.stack(dir_ndas, axis=0), np.stack(norm_ndas, axis=0), np.stack(indices, axis=0)
    else:
        # if we do not rescale, each dir, norm will have a different length
        result = dir_ndas, norm_ndas, np.stack(indices, axis=0)
    return result


def align_and_resample(cardiac_cycle_length, ind, gt_onehot, dir_1d_mean, norm_1d_mean, target_t=30, normalise_dir=True,
                       normalise_norm=True, rescale=True):
    """
    Align norm and direction by the cardiac phase ED
    resample all 1D feature vectors to the same length and min/max normalise into [0,1] and [-1,1]
    this should help to validate the pre-defined rules, detect outliers, and if both together explains the cardiac phases
    Args:
        cardiac_cycle_length ():
        ind ():
        gt_onehot ():
        dir_1d_mean():
        norm_1d_mean():
        target_t ():
        normalise_dir ():
        normalise_norm ():
        rescale ():

    Returns:

    """

    lower, mid, upper = -1, 0, 1
    xval = np.linspace(0, 1, target_t)

    gt_ed = ind[0]

    # direction ed aligned
    dir_1d_mean = np.roll(dir_1d_mean, -1 * gt_ed)
    norm_1d_mean = np.roll(norm_1d_mean, -1 * gt_ed)
    gt_onehot_rolled = np.roll(gt_onehot, -1 * gt_ed, axis=0)
    gt_onehot_rolled = np.argmax(gt_onehot_rolled, axis=0)

    if rescale:
        # interpolate to unique length
        f = interp1d(np.linspace(0, 1, norm_1d_mean.shape[0]), norm_1d_mean, kind='linear')
        norm_1d_mean = f(xval)
        # another way to interpolate 1D data
        # norm_nda = np.interp(xval, np.linspace(0, 1, norm_nda.shape[0]), norm_nda)
        if normalise_norm: norm_1d_mean = minmax_lambda([norm_1d_mean, mid, upper])
        f = interp1d(np.linspace(0, 1, dir_1d_mean.shape[0]), dir_1d_mean, kind='linear')
        dir_1d_mean = f(xval)
        if normalise_dir: dir_1d_mean = minmax_lambda([dir_1d_mean, lower, upper])

        # scale, round and clip the gt indices, to get an aligned distribution of the labels
        resize_factor = target_t / cardiac_cycle_length
        gt_onehot_rolled = np.clip(np.rint(gt_onehot_rolled * resize_factor), a_min=0, a_max=target_t - 1)

    return dir_1d_mean, norm_1d_mean, gt_onehot_rolled

def get_predicted_as_segmentation(pred: np.ndarray, return_as: Literal['binary', 'label'] = 'binary',
                                  connected_component=True, start_c=1, threshold=0.5):
    from src.data.Preprocess import from_channel_to_flat, transform_to_binary_mask
    import numpy as np
    assert len(pred.shape) > 2, 'At least 3 dimensions are required'
    mask = from_channel_to_flat(pred, start_c=start_c, threshold=threshold)

    if connected_component:
        if len(mask.shape) == 3:
            mask = clean_3d_prediction_3d_cc(mask)
        elif len(mask.shape) == 4: # CC filter along the temporal axis
            mask = np.stack([clean_3d_prediction_3d_cc(elem3d) for elem3d in mask], axis=0)
        else:
            raise Exception(f'Can not handle dimension {mask.shape} yet')
    if return_as == 'binary':
        mask = transform_to_binary_mask(mask, mask_values=np.unique(mask)[1:])
    return mask


def clean_3d_prediction_3d_cc(pred, premask=False):
    """
    Find the biggest connected component per label
    This is a debugging method, which will plot each step
    returns: a tensor with the same shape as pred, but with only one cc per label
    """

    # avoid labeling images with float values
    assert len(np.unique(pred)) < 10, 'to many labels: {}'.format(len(np.unique(pred)))

    cleaned = np.zeros_like(pred)
    # first remove surrounding predictions
    # combine all labels
    # find a mask for the biggest cc
    # set all other values to zero
    # process as usual

    if premask:
        try:
            # first remove border artefacts that are not connected to the other labels
            # the biggest connected area should be around the heart
            # artefacts might be bigger within one slice or for one label in the whole volume
            # but usually not bigger than all labels in a 3D volume
            pred_combined = (pred>0).astype(np.uint8)
            from scipy import ndimage
            pred_combined = ndimage.binary_erosion(pred_combined, structure=np.ones((3,3,3)))
            biggest_combined = clean_3d_label(1, pred_combined)
            biggest_combined = ndimage.binary_dilation(biggest_combined, structure=np.ones((3,3,3)), iterations=2)
            pred[~biggest_combined] = 0
        except Exception as e:
            print(e)
             # this could fail if we train a model only for one epoch --> predictions are all zero --> test-cases

    for val in np.unique(pred)[1:]:
        biggest = clean_3d_label(val, pred)
        cleaned[biggest] = val
    return cleaned

def clean_3d_label(val, nda):

    """
    has access to pred, no passing required
    """

    # create a placeholder
    biggest = np.zeros_like(nda)
    biggest_size = 0

    # find all cc for this label
    # tensorflow operation is only in 2D
    # all_labels = tfa.image.connected_components(np.uint8(pred==val)).numpy()
    all_labels = measure.label(np.uint8(nda == val), background=0)

    for c in np.unique(all_labels)[1:]:
        mask = all_labels == c
        mask_size = mask.sum() # here we count the number of voxels, if there is one slice with a huge wrong prediction, it will fail
        #mask_size = np.sum(np.any(mask, axis=(1,2))) # here we count the number of slices where this structure is detected
        if mask_size > biggest_size:
            biggest = mask
            biggest_size = mask_size
    return biggest

import cv2