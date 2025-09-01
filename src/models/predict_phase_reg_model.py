# predict cardiac phases for a cv experiment
import logging
import numpy as np
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from typing import List, Union, Literal
from scipy.ndimage import gaussian_filter1d

DEBUG = False


def predict(cfg_file, json_file=None, data_root='', c2l=False, exp=None, number_of_examples=None):
    """
        TODO:
            rethink the current behaviour of processing all files, and than save them to disk, the reason for this is
            that we save a single *.npy file, maybe it would be better to save a single nrrd file per patient

    Predict on the held-out validation split

        :param cfg_file: path to config file
        :type cfg_file: str
        :param data_root: path to data root (cmr data + masks (optional)
        :param c2l: trained on the cluster - inference from local data (path to data from the cluster nodes will be not available locally)
        :param exp: path to trained model/experiment instance (predictions will be saved here)
        :type exp: str
    """

    import json, logging, os
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    import tensorflow as tf
    tf.get_logger().setLevel('FATAL')
    from logging import info
    import numpy as np
    from src.data.Dataset import get_trainings_files
    from src.utils.Utils_io import ConsoleAndFileLogger, ensure_dir, get_post_processing
    from src.data.PhaseGenerators import PhaseRegressionGenerator_v2
    from src.models.PhaseRegModels import PhaseRegressionModel
    from ProjectRoot import change_wd_to_project_root
    change_wd_to_project_root()
    from src.data.Postprocess import get_predicted_as_segmentation
    from src.utils.Tensorflow_helper import choose_gpu_by_id

    # load the experiment config and json
    if type(cfg_file) == type(''):
        with open(cfg_file, encoding='utf-8') as data_file:
            config = json.loads(data_file.read())
    else:
        config = cfg_file

    if json_file is not None and type(json_file) == type(""):
        with open(json_file, encoding='utf-8') as data_file:
            dataset_json = json.loads(data_file.read())
    elif json_file is not None:
        dataset_json = json_file
    else:
        dataset_json = {}

    globals().update(config)

    # ------------------------------------------define GPU id/s to use
    GPU_IDS = config.get('GPU_IDS', '0,1')
    GPUS = choose_gpu_by_id(GPU_IDS)
    print(GPUS)
    print(tf.config.list_physical_devices('GPU'))

    EXPERIMENT = config.get('EXPERIMENT', 'UNDEFINED')
    ConsoleAndFileLogger(EXPERIMENT, logging.INFO)
    info('Loaded config for experiment: {}'.format(EXPERIMENT))
    PRETRAINED_SEG = config.get('PRETRAINED_SEG', False)

    # Load LAX sequences
    # cluster to local data mapping
    if c2l:
        config['DATA_PATH_LAX'] = os.path.join(data_root, 'lax')
        config['MODEL_PATH'] = os.path.join(exp, *config['MODEL_PATH'].split('/')[-2:])
        if not config.get('INFERENCE', False):
            config['DF_FOLDS'] = os.path.join(data_root, 'df_kfold.csv')
            config['DF_META'] = os.path.join(data_root, 'phases.csv')
            config['EXP_PATH'] = exp  # replace the relative path with the SDS path

        if exp is not None and PRETRAINED_SEG:
            config['SEGMENTATION_MODEL'] = os.path.join(exp.replace('phase_regression', ''),
                                                        *config['SEGMENTATION_MODEL'].split('/')[-2:])
            config['SEGMENTATION_WEIGHTS'] = os.path.join(exp.replace('phase_regression', ''),
                                                          *config['SEGMENTATION_WEIGHTS'].split('/')[-4:])


    suffix = dataset_json.get("suffix", None)
    if suffix is None:
        file_ending = "nii.gz"
    else:
        file_ending = suffix["file_ending"]

    x_train_lax, y_train_lax, x_val_lax, y_val_lax = get_trainings_files(data_path=config['DATA_PATH_LAX'],
                                                                         suffix=suffix,
                                                                         ftype=file_ending,
                                                                         path_to_folds_df=config['DF_FOLDS'],
                                                                         fold=config['FOLD'])

    logging.info('LAX train CMR: {}, LAX train masks: {}'.format(len(x_train_lax), len(y_train_lax)))
    logging.info('LAX val CMR: {}, LAX val masks: {}'.format(len(x_val_lax), len(y_val_lax)))

    chunk_size = 10
    x_train_laxs = [x_train_lax[i:i + chunk_size] for i in range(0, len(x_train_lax), chunk_size)]
    y_train_laxs = [y_train_lax[i:i + chunk_size] for i in range(0, len(y_train_lax), chunk_size)]
    x_val_laxs = [x_val_lax[i:i + chunk_size] for i in range(0, len(x_val_lax), chunk_size)]
    y_val_laxs = [y_val_lax[i:i + chunk_size] for i in range(0, len(y_val_lax), chunk_size)]

    logging.info('Split into chunks of:')
    logging.info('LAX train CMR: {}, LAX train masks: {}'.format(len(x_train_laxs), len(y_train_laxs)))
    logging.info('LAX val CMR: {}, LAX val masks: {}'.format(len(x_val_laxs), len(y_val_laxs)))

    # turn off all augmentation operations while inference
    # create another config for the validation data
    # we want the prediction to run with batchsize of 1
    # otherwise we might inference only on the even number of val files
    # the mirrored strategy needs to get a single gpu instance named, otherwise batchsize=1 does not work
    val_config = config.copy()
    val_config['SHUFFLE'] = False
    val_config['AUGMENT'] = False
    val_config['AUGMENT_PHASES'] = False
    val_config['AUGMENT_TEMP'] = False
    val_config['BATCHSIZE'] = 1
    val_config['HIST_MATCHING'] = False
    val_config['GPUS'] = ['/gpu:0']
    model = PhaseRegressionModel(val_config).get_model()
    logging.info('Trying to load the model weights')
    logging.info('work dir: {}'.format(os.getcwd()))
    logging.info('model weights dir: {}'.format(os.path.join(val_config['MODEL_PATH'], 'model.h5')))
    model.load_weights(os.path.join(val_config['MODEL_PATH'], 'model.h5'))
    logging.info('loaded model weights as h5 file')

    # Settings for masking
    post_processing = get_post_processing(data_json)
    use_segmentation = post_processing.get("use_segmentation", False)
    mask_channels = None

    if use_segmentation is None or use_segmentation is False:
        NNUNET_SEG = False
        val_config['NNUNET_SEG'] = False

    else:
        NNUNET_SEG = dataset_json.get("seg_model_path", False)
        val_config['NNUNET_SEG'] = NNUNET_SEG
        mask_channels = post_processing.get("mask_channels", [])

    # predict on the validation generator
    # this should avoid memory leaks for huge inference datasets
    pred_path = os.path.join(val_config['EXP_PATH'], 'pred')
    moved_path = os.path.join(val_config['EXP_PATH'], 'moved')
    example_path = os.path.join(val_config['EXP_PATH'], 'example')

    ensure_dir(pred_path)
    ensure_dir(moved_path)
    ensure_dir(example_path)

    junk = 0
    for x_train_lax_, y_train_lax_, x_val_lax_, y_val_lax_ in zip(x_train_laxs, y_train_laxs, x_val_laxs, y_val_laxs):
        preds_, moved_, vects_, gts_, segmentations_ = [], [], [], [], []
        logging.info('***********  processing junk: {} of {}'.format(junk, len(x_val_laxs)))
        validation_generator = PhaseRegressionGenerator_v2(x_val_lax_, y_val_lax_, config=val_config, in_memory=False, dataset_json=dataset_json)

        for i, (x, y) in enumerate(validation_generator):
            results = model.predict_on_batch(x)
            if PRETRAINED_SEG:
                preds, moved, vects, seg = results
                segmentations_.append(get_predicted_as_segmentation(seg[0], return_as='label', start_c=1, threshold=0.5,
                                                                    connected_component=True).astype(np.uint8))
            else:
                preds, moved, vects = results
                if NNUNET_SEG:
                    seg = y[3]
                    segmentations_.append(seg)

            preds_.append(preds.astype('float16'))
            moved_.append(moved.astype('float16'))
            vects_.append(vects.astype('float16'))
            gts_.append(y[0])

        fold = "{:04d}".format((100 * val_config['FOLD']) + junk) if val_config['FOLD'] > 0 else "{:04d}".format(junk)

        pred_filename = os.path.join(pred_path, 'gtpred_fold{}.npy'.format(fold))
        moved_filename = os.path.join(moved_path, 'moved_f{}.npy'.format(fold))
        vects_filename = os.path.join(moved_path, 'vects_f{}.npy'.format(fold))

        preds = np.concatenate(preds_, axis=0)
        moved = np.concatenate(moved_, axis=0)
        vects = np.concatenate(vects_, axis=0)

        if PRETRAINED_SEG or NNUNET_SEG:
            segmentation_filename = os.path.join(pred_path, 'segmentation_f{}'.format(fold))
            segmentation = np.stack(segmentations_, axis=0)
        gts = np.concatenate(gts_, axis=0)

        np.save(pred_filename, np.stack([gts, preds], axis=0))
        np.save(moved_filename, moved)
        np.save(vects_filename, vects)

        if PRETRAINED_SEG or NNUNET_SEG:
            np.save(segmentation_filename, segmentation)
        else:
            segmentation = None

        junk = junk + 1
        write_random_example_4d_files_to_disk(PRETRAINED_SEG or NNUNET_SEG, config, example_path, moved,
                                              number_of_examples,
                                              segmentation,
                                              vects, x_val_lax_,
                                              norm_thresh=post_processing.get("norm_threshold", 40),
                                              connected_component_filter=post_processing.get("cc_filter", None),
                                              mask_channels=mask_channels)

    del validation_generator
    del model

    # create a list of patients based on the filenames
    patients_filename = os.path.join(pred_path, 'patients.txt')
    with open(patients_filename, "a+") as f:
        _ = [f.writelines(str(val_config['FOLD']) + '_' + os.path.basename(elem) + '\n') for elem in x_val_lax]
    logging.info('saved as: \n{}\n{} \n example patients processed!'.format(pred_filename, patients_filename))


def connected_components_filter(data, pad_size=10):
    """
    This method returns a mask of the largest connected component and all connected components around the largest
    connected component. Including surround components, is used with a rectangular mask around largest connected
    component. Padding around largest component uses the min and max row and column and add/subtract the pad_size
    from it.  Not ideal right now, but fast!
    :param data: the input data
    :param pad_size: the size of the padding around the largest connected component

    :returns: a masked array, including all connected components inside the radius of pad_size around the largest connected component
    """
    largest_component, labeled_ma = get_largest_connected_components(data)

    true_indices = np.argwhere(largest_component)

    row_min = int(np.maximum(true_indices[:, 0].min() - pad_size / 2, 0))
    row_max = int(np.minimum(true_indices[:, 0].max() + pad_size / 2, largest_component.shape[0]))
    col_min = int(np.maximum(true_indices[:, 1].min() - pad_size / 2, 0))
    col_max = int(np.minimum(true_indices[:, 1].max() + pad_size / 2, largest_component.shape[1]))

    pad_mask = np.zeros_like(largest_component, dtype=bool)
    pad_mask[row_min:row_max, col_min:col_max] = True

    label_binary_ma = labeled_ma != 0
    padded_array = np.logical_or(label_binary_ma, pad_mask)

    ret_ma, _ = get_largest_connected_components(padded_array, padded_array)

    ret_ma = ret_ma * data

    return ret_ma


def get_largest_connected_components(data, data_bin_ma=None):
    """
    This method extracts the largest connected component and a labelled mask of all connected components.
    :param data: the input data an array with various connected components
    :return: the largest connected component and a labelled mask of all connected components
    """
    from scipy import ndimage
    if data_bin_ma is None:
        data_bin_ma = np.abs(data) > 0

    labeled_ma, num_labels = ndimage.label(data_bin_ma)
    region_sizes = ndimage.sum(data_bin_ma, labeled_ma, range(num_labels + 1))

    largest_component_label = np.argmax(region_sizes)
    largest_component = (labeled_ma == largest_component_label) * data

    return largest_component, labeled_ma


def write_random_example_4d_files_to_disk(PRETRAINED_SEG, config, example_path, moved, number_of_examples, segmentation,
                                          vects, x_val_lax, norm_thresh=55, connected_component_filter=None, mask_channels=None):

    if number_of_examples == None:
        number_of_examples = vects.shape[0] - 1  # export all patients
    else:
        number_of_examples = 1
    np.random.seed(42)
    examples = np.random.choice(np.array(range(vects.shape[0])), size=number_of_examples, replace=False)
    logging.info('Saving example patients with direction as nrrd')

    # order of moved axis is wrong, so rearrange them:
    moved = np.transpose(moved, (0, 1, 4, 2, 3))
    focus_size = round(moved.shape[-1] / 96)  # Setting size of focus point in depending on the size of the image

    write_4d_files_to_disk(examples, focus_size, PRETRAINED_SEG, config, example_path, moved, segmentation,
                           vects, x_val_lax, norm_thresh=norm_thresh, connected_component_filter=connected_component_filter, mask_channels=mask_channels)


def write_4d_files_to_disk(examples, focus_size, PRETRAINED_SEG, config, example_path, moved, segmentation,
                           vects, x_val_lax, norm_thresh=55, connected_component_filter=None, mask_channels=None):
    import SimpleITK as sitk
    import os
    for example in examples:
        dir_1d_mean, directions, norm_1d_mean, norm_nda, ct, _ = interpret_deformable(vects_nda=vects[example],
                                                                                      masks=segmentation[
                                                                                          example] if PRETRAINED_SEG else None,
                                                                                      mask_channels=mask_channels
                                                                                      if PRETRAINED_SEG else None,
                                                                                      ct_calculation=[1, 2, 3],
                                                                                      norm_percentile=norm_thresh,
                                                                                      component_padding=
                                                                                      connected_component_filter)

        if np.ma.is_masked(directions):
            directions[directions.mask] = -10
            # directions = directions.data * ~directions.mask
        if np.ma.is_masked(norm_nda):
            norm_nda = norm_nda.data * ~norm_nda.mask

        zeros = np.zeros_like(moved[example])
        zeros[:, :,
        int(ct[0] - focus_size):int(ct[0] + focus_size),
        int(ct[1] - focus_size):int(ct[1] + focus_size)] = 1

        # Testen ob es auch direkt mit GetImageFromArray klappt, ohne for schleifen iteration
        sitk_images = [sitk.GetImageFromArray(vol.astype('float32')) for vol in moved[example]]

        sitk_vects = [sitk.GetImageFromArray(vol.astype('float32'), isVector=True) for vol in
                      np.transpose(vects[example], (3, 1, 2, 0))]
        sitk_dir = [sitk.GetImageFromArray(vol.astype('float32')) for vol in rearrange_axis_of_ndarray(directions)]
        sitk_norm = [sitk.GetImageFromArray(vol.astype('float32')) for vol in rearrange_axis_of_ndarray(norm_nda)]
        sitk_foc = [sitk.GetImageFromArray(vol.astype(np.uint8)) for vol in zeros]

        # Define spacing for saving the images
        spacing = config.get('SPACING', (2.5, 2.5))
        spacing = list(reversed(spacing)) + [1.0, 1.0]

        elem = x_val_lax[example]
        file_type = '.nrrd' if '.nrrd' in elem else '.nii.gz'

        # Save image, vector, direction, norm and focus point each as nrrd/NIFTI
        export_img_f_name = os.path.join(example_path, os.path.basename(elem))
        save_sitk(sitk_images, spacing, export_img_f_name)

        export_vec_f_name = os.path.join(example_path,
                                         os.path.basename(elem).replace(file_type, '_vec.nrrd'))
        save_sitk(sitk_vects, spacing, export_vec_f_name)

        export_dir_f_name = os.path.join(example_path,
                                         os.path.basename(elem).replace(file_type, '_dir.nrrd'))
        save_sitk(sitk_dir, spacing, export_dir_f_name)

        export_norm_f_name = os.path.join(example_path,
                                          os.path.basename(elem).replace(file_type, '_norm.nrrd'))
        save_sitk(sitk_norm, spacing, export_norm_f_name)

        export_foc_f_name = os.path.join(example_path,
                                         os.path.basename(elem).replace(file_type, '_foc.nrrd'))
        save_sitk(sitk_foc, spacing, export_foc_f_name)

        if PRETRAINED_SEG:
            sitk_mask = [sitk.GetImageFromArray(np.flipud(vol.astype(np.uint8))) for vol in np.transpose(segmentation[example], (0, 3, 1, 2))]
            new_mask_clean = sitk.JoinSeries(sitk_mask)
            new_mask_clean.SetSpacing(spacing)
            export_mask_f_name = os.path.join(example_path,
                                              os.path.basename(elem).replace(file_type, '_mask.nrrd'))
            sitk.WriteImage(new_mask_clean, export_mask_f_name)
            seg_based_direction(vects[example], moved[example], segmentation[example], x_val_lax[example],
                                focus_size, example_path, config, file_type)

    return

def seg_based_direction(vect, moved, segmentation, x_val_lax, focus_size, example_path, config,  targetfile_type='nii'):
    import SimpleITK as sitk
    import os
    _, directions_seg, _, norm_nda_seg, ct, _ = interpret_deformable(vects_nda=vect, masks=segmentation,
                                                                     mask_channels=[2, 3], ct_calculation=[1],
                                                                     dir_axis=0)
    if np.ma.is_masked(directions_seg):
        # directions_seg = directions_seg.data * ~directions_seg.mask
        # directions_seg[directions_seg.mask] = -10 # this works well with the jet transparent color map
        dir = directions_seg.data  # * ~directions_seg.mask
        dir[directions_seg.mask] = -1.
        directions_seg = dir
    if np.ma.is_masked(norm_nda_seg):
        # norm_nda_seg = norm_nda_seg.data * ~norm_nda_seg.mask
        n = norm_nda_seg.data  # * ~norm_nda_seg.mask
        n[norm_nda_seg.mask] = -1.
        norm_nda_seg = n
    # create a bucket-based mask from direction movement
    dir_seg_bucket = directions_seg.copy()
    dir_seg_bucket[(directions_seg < -0.5) & (directions_seg > -1)] = 1  # strong contraction [-1,-.5]
    dir_seg_bucket[(directions_seg < 0) & (directions_seg >= -0.5)] = 2  # moderate contraction [-.5, 0]
    dir_seg_bucket[(directions_seg >= 0) & (directions_seg < 0.5)] = 3  # moderate relaxation [0, .5]
    dir_seg_bucket[(directions_seg >= 0.5) & (directions_seg <= 1)] = 4  # strong relaxation [.5, 1]
    dir_seg_bucket[directions_seg == -1] = 0

    ############################################
    zeros = np.zeros_like(moved)
    zeros[:,:, int(ct[0] - focus_size):int(ct[0] + focus_size),  int(ct[1] - focus_size):int(ct[1] + focus_size)] = 1

    sitk_dir_seg = [sitk.GetImageFromArray(vol.astype('float32')) for vol in np.transpose(directions_seg[...,None], (0, 3, 1, 2))]
    sitk_norm_seg = [sitk.GetImageFromArray(vol.astype('float32')) for vol in  np.transpose(norm_nda_seg[...,None], (0, 3, 1, 2))]
    sitk_foc_seg = [sitk.GetImageFromArray(vol.astype(np.uint8)) for vol in np.transpose(zeros, (0, 2, 3, 1))]

    new_dir_clean_seg = sitk.JoinSeries(sitk_dir_seg)
    new_norm_clean_seg = sitk.JoinSeries(sitk_norm_seg)
    new_foc_clean_seg = sitk.JoinSeries(sitk_foc_seg)

    spacing = config.get('SPACING', (1.0, 1.0))
    spacing = list(reversed(spacing)) + [1.0, 1.0]
    new_dir_clean_seg.SetSpacing(spacing)
    new_norm_clean_seg.SetSpacing(spacing)
    new_foc_clean_seg.SetSpacing(spacing)

    elem = x_val_lax
    file_type = '.nrrd' if '.nrrd' in elem else '.nii.gz'

    export_dir_seg_f_name = os.path.join(example_path,
                                         os.path.basename(elem).replace(file_type,
                                                                        '_dir_seg{}'.format(targetfile_type)))
    export_norm_f_seg_name = os.path.join(example_path,
                                          os.path.basename(elem).replace(file_type,
                                                                         '_norm_seg{}'.format(targetfile_type)))
    export_foc_seg_f_name = os.path.join(example_path,
                                         os.path.basename(elem).replace(file_type,
                                                                        '_foc_seg{}'.format(targetfile_type)))

    sitk.WriteImage(new_dir_clean_seg, export_dir_seg_f_name)
    sitk.WriteImage(new_norm_clean_seg, export_norm_f_seg_name)
    sitk.WriteImage(new_foc_clean_seg, export_foc_seg_f_name)

def save_sitk(sitk_img, spacing, export_f_name):
    """
    Save a sitk image to disk.
    """
    import SimpleITK as sitk
    new_img_clean = sitk.JoinSeries(sitk_img)
    new_img_clean.SetSpacing(spacing)
    sitk.WriteImage(new_img_clean, export_f_name)
    return new_img_clean


def rearrange_axis_of_ndarray(array, order=(1, 0, 2, 3), additional_row=True):
    """
    This method rearranges the axis of an ndarray.
    :param array: ndarray to be rearranged
    :param order: order of axis to be rearranged to
    :param additional_row: boolean to add a row to the array
    :return: ndarray rearranged
    """
    if additional_row:
        array = array[np.newaxis, :, :, :]
    array = np.transpose(array, order)
    return array


def get_rv_outline_as_mask(masks, include_septum=False):
    """
    This method creates a mask of the right ventricle myocardium,
    if a mask of the right ventricular cavity exclusive the myocardium is given.

    :param masks: array of masks (2D+t)
    :param include_septum: determines if the septum should be included in the rv mask or not

    :return: mask of simulated right ventricle myocardium
    :rtype: ndarray
    """
    import cv2 as cv
    from scipy.spatial.distance import cdist
    if len(masks.shape) == 4:
        masks = masks[:,:,:,0]
    rv_mask = masks == 3
    myo_mask = masks == 2
    faked_rv_myo = np.zeros_like(masks)
    for timestep in range(rv_mask.shape[0]):
        rv_timestep = rv_mask[timestep]
        rv_contours, _ = cv.findContours(rv_timestep.astype(np.uint8), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        if len(rv_contours) == 0:
            continue
        rv_contours = rv_contours[0][:, 0]
        if not include_septum:
            myo_timestep = myo_mask[timestep]
            myo_contours, _ = cv.findContours(myo_timestep.astype(np.uint8), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
            if len(myo_contours) == 0:
                continue
            myo_contours = myo_contours[0][:, 0]

        new_slice = np.zeros(faked_rv_myo.shape[1:])
        rv_contours = rv_contours[np.min(cdist(rv_contours, myo_contours), axis=1) > 1]
        new_slice[rv_contours[:, 1], rv_contours[:, 0]] = 1

        # dilate rv outline to increase thickness (but only to one side)
        # define thickness
        dilate_thickness = new_slice.shape[0]//32
        kernel = np.ones((dilate_thickness, dilate_thickness), np.uint8)

        new_slice = cv.dilate(new_slice, kernel=kernel, iterations=1)
        new_slice = remove_masking_in_atrium_direction(new_slice, myo_contours, rv_timestep)

        inverse_rv_mask = np.logical_not(rv_mask[timestep])
        faked_rv_myo[timestep] = np.logical_and(new_slice, inverse_rv_mask)
    return faked_rv_myo


def remove_masking_in_atrium_direction(rv_contours_slice, myo_contours, rv_mask):
    """
    Removes the surplus of the rv myocardium mask in the aria of the tricuspid valve (in direction of the atrium).
    """
    import cv2 as cv
    from scipy import ndimage
    myo_mask = np.zeros_like(rv_mask)
    myo_mask[myo_contours[:, 1], myo_contours[:, 0]] = 1

    # create a intersection line between the left ventricle myocardium (+ septum) and the right ventricle cavity
    intersection_line = np.logical_and(myo_mask, rv_contours_slice)
    indices = np.where(intersection_line)

    if indices[0].size == 0 or indices[1].size == 0:
        more_dilated_rv = cv.dilate(rv_contours_slice, kernel=(10,10), iterations=1)
        intersection_line = np.logical_and(myo_mask, more_dilated_rv)
        indices = np.where(intersection_line)
        if indices[0].size == 0 or indices[1].size == 0:
            more_dilated_rv = cv.dilate(more_dilated_rv, kernel=(5,5), iterations=1)
            intersection_line = np.logical_and(myo_mask, more_dilated_rv)
            indices = np.where(intersection_line)

    if indices[0].size == 0 or indices[1].size == 0:
        indices = np.where(rv_contours_slice == 1)
        if indices[0].size == 0 or indices[1].size == 0:
            logging.error(f"No intersection line found.")
            return rv_contours_slice

    first_column = indices[1].min()
    first_row_index= indices[0][indices[1]==first_column][0]
    rvip_a =np.array([first_row_index, first_column])

    last_column = indices[1].max()
    last_row_index = indices[0][indices[1] == last_column][0]
    rvip_b = np.array([last_row_index, last_column])

    distance_a_b = np.linalg.norm(rvip_b - rvip_a)
    radius_size = int(distance_a_b * 0.5)

    center_of_rv = ndimage.center_of_mass(rv_mask)

    distance_a = np.linalg.norm(center_of_rv - rvip_a)
    distance_b = np.linalg.norm(center_of_rv - rvip_b)
    if distance_a > distance_b:
        rvip_atrium = rvip_b
    else:
        rvip_atrium = rvip_a

    atrial_surplus_mask = np.zeros_like(rv_mask)
    atrial_surplus_mask[rvip_atrium[0], rvip_atrium[1]] = 1
    atrial_surplus_mask = (atrial_surplus_mask > 0).astype(np.uint8)
    # Draw a filled circle around atrial intersection point
    cv.circle(atrial_surplus_mask, (rvip_atrium[1],rvip_atrium[0]), radius_size, (1), -1)

    atrial_surplus_mask = np.logical_not(atrial_surplus_mask)
    rv_contours_slice = np.logical_and(rv_contours_slice, atrial_surplus_mask)

    return rv_contours_slice


def get_as_single_mask(segmentation, channels, whole_mask=True) -> np.ndarray:
    '''

    Parameters
    ----------
    masks array of probabilities for each mask
    channels that should be used for masking: 3 rv outline, 0 rv, 1 myo, 2 lv
    different_labels used for mitk where labels used for the different masks eg. 1, 2, 3
    threshold

    Returns
    -------

    '''

    # uses connected component
    mask = segmentation
    channels = np.array(channels)
    rv_outline = 3 in channels or 4 in channels
    include_rv_septum = 4 in channels
    if whole_mask and len(channels) >= 3:
        channels = channels
    else:
        channels = channels[(channels > 0) & (channels < 3)]
    _ = np.zeros_like(mask)
    for channel in channels:
        _ = np.logical_or(_, mask == (channel))
    mask = _
    if rv_outline:
        fake_rv_myo = get_rv_outline_as_mask(segmentation, include_septum=include_rv_septum)
        mask = np.logical_or(mask, fake_rv_myo[...,None])
    mask = mask
    return mask


def get_rv_lv_dir(vects_nda, masks=None, length=-1, plot=True, z=None, dir_axis=0, gtind=None, exp_path=None, patient='temp',
                          save=False):
    from src.utils.detect_phases_from_dir import detect_phases

    lv_args = interpret_deformable(dir_axis=dir_axis, masks=masks, length=length, vects_nda=vects_nda,
                                   mask_channels=[2],  ct_calculation=[1,2], as_angle=False)
    lv_dir_1d_mean, lv_directions, lv_norm_1d_mean, lv_norm_nda, lv_ct, lv_mask = lv_args
    lv_ind = detect_phases(dir_1d_mean=lv_dir_1d_mean[:length])



    rv_args = interpret_deformable(dir_axis=dir_axis, masks=masks, length=length, vects_nda=vects_nda,
                                   mask_channels=[3],  ct_calculation=[3], as_angle=False)
    rv_dir_1d_mean, rv_directions, rv_norm_1d_mean, rv_norm_nda, rv_ct, rv_mask = rv_args
    rv_ind = detect_phases(dir_1d_mean=rv_dir_1d_mean[:length])

    if plot:
        fig = plot_two_direction_instance(lv_dir_1d_mean, rv_dir_1d_mean, lv_directions, rv_directions)
        return fig, [lv_ind, rv_ind]


def plot_two_direction_instance(dir_1d_mean_a, dir_1d_mean_b, direction_a, direction_b):
    import matplotlib.pyplot as plt
    from src.visualization.Visualize import show_2D_or_3D
    # X-axis values

    x = range(len(dir_1d_mean_a))
    fig = plt.figure(figsize=(25,4))
    rows = 3
    # DIR 2D+t
    dir_2d_t = direction_a.copy()
    if np.ma.is_masked(direction_a): dir_2d_t = dir_2d_t.data * ~direction_a.mask
    div_cmap = 'bwr'
    fig = show_2D_or_3D(dir_2d_t, allow_slicing=False, cmap=div_cmap, fig=fig, interpolation=None, vmin=-1, vmax=1)
    ax_ = fig.get_axes()[0]
    _ = ax_.set_yticks([])
    _ = ax_.set_xticks([])
    ax = fig.get_axes()[1]
    _ = ax.set_ylabel(r'$\alpha$ ' + '\n2d+t')  # \nmid
    _ = ax.set_yticks([])
    _ = ax.set_xticks([])
    cax = fig.add_axes([0.45, 0.84, 0.1, 0.03])
    cb = fig.colorbar(ax.get_images()[len(ax.get_images()) // 2], cax=cax, orientation='horizontal')
    cb.ax.tick_params(color="black", labelsize=10, labelcolor='black')

    # Plotting the curves
    pos = 2
    ax2 = fig.add_subplot(rows, 1, pos)
    _ = ax2.plot(x, dir_1d_mean_a, label='Direction LV', color = 'orange')
    _ = ax2.plot(x, dir_1d_mean_b, label='Direction RV', color = 'green')

    # Adding labels and title
    ax2.set_xlabel('Time (t)')
    ax2.legend()

    plt.tight_layout()
    plt.show()

    return fig

def get_focus_point(mask2d: np.ndarray, calculation: Union[Literal['septum'], List, int, None] = 'septum', print_ct=False, whole_mask=True):
    from scipy import ndimage
    if type(calculation) == int:
        calculation = [calculation]
    mask_for_focus = get_as_single_mask(mask2d[...,None], channels=calculation, whole_mask=whole_mask)  # masks of first ts
    center_of_mask = ndimage.center_of_mass(mask_for_focus)
    focus = np.array([center_of_mask[0], center_of_mask[1]])
    if print_ct: print(
        f'Using mask(s): {", ".join(np.array(["right ventricle", "myocardium", "left ventricle", "rv outline"])[calculation])} for ct')
    return focus


def get_combined_masking_norm(vects_nda, mask, dir_axis=0, norm_percentile=55):
    """

    """

    vects_nda_ma = vects_nda * np.broadcast_to(mask, shape=vects_nda.shape)
    heart_coverage_percentage = np.sum([vects_nda_ma != 0]) / vects_nda_ma.size
    new_norm_percentile = (100 - heart_coverage_percentage * 100) + heart_coverage_percentage * norm_percentile
    norm_mask, norm_nda = get_norm(dir_axis, new_norm_percentile, vects_nda_ma)

    return vects_nda_ma, norm_mask, norm_nda


def interpret_deformable(vects_nda, masks=None, mask_channels=None, dir_axis=0, length=None, filename=None,
                         norm_percentile=55, diff_thresh=1.0,  component_padding=None, sigma=0.8, as_angle=False,
                         ct_calculation: Union[Literal['septum'], list, int, None] = 'septum'):
    import numpy as np
    from scipy import ndimage
    from src.visualization.save import write_sitk
    if length is None:
        length = vects_nda.shape[0]
    if vects_nda.dtype is not np.float32: vects_nda = vects_nda.astype("float32")

    combinated_masking = False
    if norm_percentile > 0 and masks is not None:
        combinated_masking = True
    # vects_nda: vectors towards registered voxels (M->F), (40, 128, 128, 2)
    # mask: binary mask from use_segmentation, repeated to 3 channels repeated to match vects_nda, (40, 128, 128, 2)
    # norm_nda: vector_length, (40, 128, 128)
    # dim_: dimension of on image (128, 128)
    # ct: center of mass calculated from mask
    # directions: direction relative to vector pointing toward ct
    # dir_1d_mean: mean of direction vectors
    # norm_1d_mean: mean of vector lengths

    dim_ = vects_nda.shape[1:-1]
    # calc the norm (supervised and self-supervised is similar)
    norm_mask, norm_nda = get_norm(dir_axis, norm_percentile, vects_nda)
    norm_nda = norm_nda.astype(np.float16)
    all_masks = []

    if masks is not None and mask_channels is not None and len(mask_channels) > 0:
        # supervised mask and center derivation
        if DEBUG:
            _ = np.array(["right ventricle", "myocardium", "left ventricle"])
            print(f'Using mask(s): {", ".join(_[mask_channels])} for calculation')

        mask = get_as_single_mask(masks, channels=mask_channels).astype(bool)
        if ct_calculation == 'VOL':
            ct = [dim_[0]/2, dim_[1]/2]
        else:
            if type(ct_calculation) == type(''):
                if ct_calculation == 'MSE':
                    ct_calculation = mask_channels
            ct = get_focus_point(masks[0], print_ct=DEBUG, calculation=ct_calculation)

        _, directions = get_directions(ct=ct, dim_=dim_, length=length, vects_nda=vects_nda, as_angle=as_angle, diff_thresh=diff_thresh, masked=True)
    else:
        # 1st center definition,
        # Volume center & norm_msk COM
        ct = get_balanced_center(dim_, norm_mask)
        vects_nda_ma = vects_nda * np.broadcast_to(norm_mask[None, ..., None], shape=vects_nda.shape)
        mask, directions = get_directions(ct=ct, dim_=dim_, length=length, vects_nda=vects_nda_ma, as_angle=as_angle,
                                          sigma=sigma, component_padding=component_padding, diff_thresh=diff_thresh)
        if mask is None:
            mask = norm_mask
        all_masks.append(mask)
        if filename is not None: write_sitk(directions, filename=filename, suffix='dir1')

        # 2nd center definition
        # COM of a direction mask based on a minimal direction change over time
        ct = np.array([*ndimage.center_of_mass(mask)[0:2]])
        # ct = np.array([int(z), *ndimage.center_of_mass(mask)[1:]])
        # ct = np.array([int(z), int(z)])
        mask, directions = get_directions(ct=ct, dim_=dim_, length=length, vects_nda=vects_nda_ma, as_angle=as_angle,
                                          sigma=sigma, component_padding=component_padding, diff_thresh=diff_thresh)
        if mask is None:
            mask = norm_mask
        if not combinated_masking:
            all_masks.append(mask)

    # calc direction, based on the labels mask or the self-supervised mask
    if filename is not None: write_sitk(directions, filename=filename, suffix='dir2')
    # mask direction with a supervised or self-supervised mask
    # TODO: validate if scaling the direction values with the norm provides further value for disease classification

    # mask norm and directions with a supervised or self-supervised mask
    from src.data.Postprocess import minmax_lambda
    directions = directions * minmax_lambda([norm_nda, 1, 2])

    directions_ma, dir_1d_mean = get_masked_array(directions, mask)
    norm_ma, norm_1d_mean = get_masked_array(norm_nda, mask)

    if filename is not None:
        directions_masked = directions_ma.data * ~directions_ma.mask
        write_sitk(directions_masked, filename=filename, suffix='dir_masked')
    return dir_1d_mean, directions_ma.astype(np.float16), norm_1d_mean, norm_ma, ct, all_masks



def get_masked_array(array, mask, axis=(1, 2), aggregation_func=np.ma.mean):
    """
    Returns a masked array and a one dimensional array representing the in-plane aggregation of the array.

    :param array: array to be masked
    :type array: np.ndarray
    :param mask: mask to apply
    :type mask: np.ndarray
    :param axis: axis along which to apply the mask
    :param aggregation_func: aggregation function to apply to the array (np.ma.mean or np.ma.median)

    :returns:   masked array (array_ma np.array) one dimensional array representing the in-plane aggregation of the array (array_1d np.array).
    """
    if array.shape != mask.shape and len(mask.shape) == 4:
        mask = mask[:, :, :, 0]
    array_ma = np.ma.masked_array(array, mask=np.broadcast_to(~mask, shape=array.shape))
    array_1d = aggregation_func(array_ma, axis=axis)
    return array_ma, array_1d


def get_balanced_center(dim_, norm_msk):
    import scipy
    import numpy as np
    ct_norm = scipy.ndimage.center_of_mass(
        norm_msk)  # x,y =  np.mean(np.where(norm_mask)) compareable results, and usable in a differentiable model with tf
    ct = np.array(dim_) // 2
    ct = (ct + ct_norm) // 2
    return ct


def get_norm(dir_axis, norm_percentile, vects_nda):
    import numpy as np
    from src.models.KerasLayers import minmax_lambda
    from src.data.Preprocess import clip_quantile
    # norm of the vector
    norm_nda = np.linalg.norm(vects_nda[..., dir_axis:], axis=-1)
    norm_nda = clip_quantile(norm_nda, 0.99)
    # norm_nda = minmax_lambda([norm_nda, mid, upper])
    norm_msk = norm_nda.copy()
    norm_msk = np.median(norm_msk[:-1], axis=0)  # exclude the cyclic (last) registration step for the mask generation
    threshold = np.percentile(norm_msk, norm_percentile)
    if threshold > 0.3:
        threshold = 0.3
    norm_msk = norm_msk > threshold
    # norm_msk = norm_msk > norm_percentile
    # for norm msk improvements the following did not work well:
    # connected component filtering before COM
    # Gauss smoothing or any other conv operation such as closing etc.
    # usually there are occlusions that stop these methods to work for each patient
    return norm_msk, norm_nda


def cart2pol(x, y):
    '''
    Parameters:
    - x: float, x coord. of vector end
    - y: float, y coord. of vector end
    Returns:
    - r: float, vector amplitude
    - theta: float, vector angle
    '''

    z = x + y * 1j
    r, theta = np.abs(z), np.angle(z)

    return r, np.rad2deg(theta)


def signed_angle_np(p1, p2):
    p2 = np.broadcast_to(p2, shape=p1.shape)
    assert p1.shape == p2.shape
    shape_ = p1.shape

    size = np.prod(shape_[:-1])
    p1 = p1.reshape((size, shape_[-1]))
    p2 = p2.reshape((size, shape_[-1]))
    ang1 = np.arctan2(p1[..., 1], p1[..., 0])
    ang2 = np.arctan2(p2[..., 1], p2[..., 0])
    rel_angle = np.rad2deg(np.abs(ang1) - np.abs(ang2))
    # for anti clockwise degrees 0 - 360
    # np.rad2deg((ang1 - ang2) % (2 * np.pi))
    rel_angle = rel_angle.reshape(shape_[:-1])

    print('angles (-180:180)', rel_angle.min(), rel_angle.max(), rel_angle.mean())
    bord = 90
    neg_mask = rel_angle < 0
    # rel angle is n the range -180:180, by this relaxing angles have a higher impact on the mean
    # here we create a range -90:90, with abs(angle)>90 = 90 - angle mod 90
    rel_angle = np.where(abs(rel_angle) > bord, abs(rel_angle) - bord, abs(rel_angle))
    rel_angle = np.where(neg_mask, rel_angle * -1, rel_angle)
    # assert np.all((rel_angle < -bord) | (rel_angle > bord))
    print('angles (-90:90)', rel_angle.min(), rel_angle.max(), rel_angle.mean())
    return rel_angle


def get_directions(ct, dim_, length, vects_nda, diff_thresh=1.2, masked=False, dir_axis=0, as_angle=False,
                   sigma=0.8, component_padding=False, norm_thresh = 60):
    """
    Create a focus matrix with the shape specified in dim_.
    In this matrix for each voxel we will have a focus vector pointing towards the center ct.
    For each vector v in the deformable vects_nda calculate the cosine angle (direction) between this deformation vector
    and the corresponding focus vector
    that goes
    Args:
        ct ():
        dim_ ():
        length ():
        vects_nda ():
        diff_thresh ():

    Returns:

    Parameters
    ----------
    masked defines if vectors are already filtered by a use_segmentation mask

    """
    import numpy as np
    import tensorflow as tf
    import scipy.ndimage
    from src.models.KerasLayers import get_idxs_tf, get_focus_tf, flow2direction_lambda
    if component_padding is None: component_padding = False

    idx = get_idxs_tf(dim_)
    c = get_focus_tf(ct[0:len(dim_)], dim_)
    centers = c - idx
    centers_tensor = centers[tf.newaxis, ...]

    # direction relative to the focus point C_n
    if as_angle:
        directions = signed_angle_np(vects_nda[..., 1:], centers_tensor[..., 1:].numpy())  # inplane polar angle
    else:
        directions = flow2direction_lambda([vects_nda[..., dir_axis:], centers_tensor[..., dir_axis:]])[
            ..., 0].numpy()  # remove the last extra channel
        if np.min(directions) < -1:
            print("cosine smaller than -1 detected: " + str(np.min(directions)))
        if np.max(directions) > 1:
            print("cosine large than 1 detected: " + str(np.max(directions)))
    directions_cut = directions[:length]
    dir_rest = directions[length:]
    dir_rest = scipy.ndimage.gaussian_filter(dir_rest, sigma=sigma, mode='wrap')

    # smooth the direction field - especially at the cycle end
    directions_cut = scipy.ndimage.gaussian_filter(directions_cut, sigma=sigma, mode='wrap')
    directions[:length] = directions_cut
    directions[length:] = dir_rest
    dir_mask = None

    if not masked and norm_thresh != 0:  # additional filtering required
        # create mask by another constrain: we include only voxels with a direction change (max - min) greater than
        # smooth the direction field - especially at the cycle end
        # directions = scipy.ndimage.gaussian_filter(directions, sigma=0.8, mode='wrap')
        # np.ma.min(directions_cut, axis=0)
        min_dir_ = np.min(directions_cut, axis=0)
        max_dir_ = np.max(directions_cut, axis=0)
        diff = max_dir_ - min_dir_
        # diff_thresh = np.percentile(diff,diff_thresh)
        if DEBUG: print('diff thresh', diff_thresh)
        dir_mask = diff >= diff_thresh

        # binary opening removes single voxels and close crushed vessels
        structure = np.ones((3, 3))
        dir_mask = scipy.ndimage.binary_opening(dir_mask, structure=structure, iterations=1)

    elif norm_thresh == 0:
        dir_mask = np.zeros_like(directions_cut)
        dir_mask = dir_mask == 0

    if component_padding is not None and component_padding != False:
        dir_mask = connected_components_filter(dir_mask, pad_size=component_padding)

    return dir_mask, directions


def predict_phase_from_deformable(exp_path, create_figures=True, norm_thresh=50, dir_axis=0, roll_by_gt=True,
                                  normalise_dir=False, normalise_norm=False, return_files=False, mask_channels=None,
                                  ct_calculation='septum', save_dir_as_nrrd=False, max_junks=None,
                                  connected_component_filter=None):
    """
    Predict the temporal occurence for five cardiac phases from a cmr-phase-regression experiment folder
    Expects to find all files written from a CV-experiment, e.g.> train_regression_model.py
    Args:
        exp_path (str): full path to a phase regression experiment
        norm_thresh (int): 0 < norm_thresh < 100
        dir_axis (int): out of [0,1], 0 = z,y,x motion, 1 = y,x motion, z- is negative during systole, y,x positive
        roll_by_gt (bool): use the gt labels or the pred labels to align the direction cohort plots

    Returns:

    """
    from time import time
    t0 = time()
    if DEBUG: print('start: {:0.3f} s'.format(t0))
    import numpy as np
    import pandas as pd
    import os
    import logging
    from src.data.Dataset import load_phase_reg_exp
    from src.utils.Metrics import meandiff
    from src.data.Postprocess import align_resample_multi
    from src.visualization.Visualize import plot_dir_norm, plot_dir_norm_split_by, plot_pfd_per_phase_as_violin, \
        plot_scatter

    aligned_length = 40
    vols_alignedscaled = None
    vols_rv_alignedscaled = None

    # load all files of this experiment
    nda_vects, gt, pred, gt_len, mov, masks, patients = load_phase_reg_exp(exp_path, junk=max_junks)
    logging.info(f'vects_nda: {nda_vects.shape}')

    if mask_channels is None:
        masks=[]
    if masks is not None and mask_channels == []:
        mask_channels = np.unique(masks)
        logging.info(f"Set mask channels to {mask_channels}")
    if masks is not None and len(mask_channels) > 1:
        logging.info(f"Set mask channels to {mask_channels}")
    else:
        logging.info("Found no masks.")

    using_segmentation = masks is not None and mask_channels is not None and len(mask_channels) > 0

    if using_segmentation:
        logging.info(
            f'Using mask(s): {", ".join(np.array(["None", "left ventricle", "left ventricle myocardium", "rv myocardium"])[mask_channels])} for calculation')
        if ct_calculation is None:
            ct_calculation = 1
    t1 = time()
    print('files loaded {:0.3f} s, continue with deformable2direction2phase'.format(t1 - t0))

    # predict phase per patient and write result as df into experiment folder
    pred_u = np.zeros_like(gt)
    upred_ind = []
    gt_ind = []
    cycle_len = []
    dir_1ds = []
    norms_1ds = []
    directions = []
    norms = []
    msks = []
    cts = []
    executor = ThreadPoolExecutor(max_workers=12)
    instances = nda_vects.shape[0]
    dir_axis_list = [dir_axis] * instances
    norm_threshold_list = [norm_thresh] * instances
    connected_component_filter_list = [connected_component_filter] * instances
    indicies = list(range(instances))
    f_names = [None] * instances
    if not return_files: mov = [None] * instances  # save memory, we only need to return the cmr for jupyter plots
    cts = [ct_calculation] * instances
    assert len(patients) == instances, f'please check your list of patients ({len(patients)} != {instances})'
    if save_dir_as_nrrd:
        f_names = [os.path.join(exp_path, p) for p in patients]

    if DEBUG: print(pred_u.shape)

    params = [nda_vects, gt_len, gt, dir_axis_list, norm_threshold_list, connected_component_filter_list, indicies, f_names, cts]
    # signature of interpret_deformable_async nda_vect, gt_len, gt, dir_axis, norm_thresh, idx, filename=None,
    # ct_calculation='septum', masks=None, mask_channels=None

    if using_segmentation:
        # array has to be repeated for iterator used in multithreading
        mask_channels_multi = [mask_channels] * instances
        params.extend([masks, mask_channels_multi])
    for result in executor.map(interpret_deformable_async, *params):
        cardiac_cycle_length, dir_1d_mean, ind, indices, norm_1d_mean, weight, i, ct, mask = result

        cycle_len.append(cardiac_cycle_length)
        upred_ind.append(indices)
        gt_ind.append(ind)
        dir_1ds.append(dir_1d_mean)
        norms_1ds.append(norm_1d_mean)
        cts[i] = ct
        msks.append(mask)
        # directions.append(direction_nda)
        # norms.append(norm_nda)
        indices = np.array(indices)
        onehot = np.zeros((indices.size, cardiac_cycle_length))
        onehot[np.arange(indices.size), indices] = weight
        pred_u[i][0:cardiac_cycle_length] = onehot.T
        t_temp = time()
        if DEBUG: print('prediction took: {:0.3f} s'.format(t_temp - t1))
    upred_ind = np.stack(upred_ind, axis=0)
    gt_ind = np.stack(gt_ind, axis=0)
    cycle_len = np.stack(cycle_len, axis=0)
    dir_1ds = np.stack(dir_1ds, axis=0)
    norms_1ds = np.stack(norms_1ds, axis=0)

    # re-create a compatible shape for the metric fn
    gt_ = np.stack([gt, gt_len], axis=1)
    pred_ = np.stack([pred_u, np.zeros_like(pred_u)], axis=1)

    # create some dataframes for further processing
    phases = ['ED', 'MS', 'ES', 'PF', 'MD']
    res = meandiff(gt_, pred_, apply_sum=False, apply_average=False)
    pfd_df = pd.DataFrame(res.numpy(), columns=phases)
    pfd_df['patient'] = patients
    pfd_df.to_csv(os.path.join(exp_path, 'cfd.csv'))
    # save the predicted phases as csv
    pred_df = pd.DataFrame(upred_ind, columns=phases)
    pred_df['patient'] = patients
    pred_df.to_csv(os.path.join(exp_path, 'pred_phases.csv'))

    gt_df = pd.DataFrame(gt_ind, columns=phases)
    gt_df['patient'] = patients
    gt_df.to_csv(os.path.join(exp_path, 'gt_phases.csv'), index=False)
    t2 = time()
    if DEBUG: print('prediction complete: {:0.3f} s'.format(t2 - t1))

    # create some plots
    t3 = time()
    if not roll_by_gt: gt = pred_u
    dirs_alignedscaled, norms_alignedscaled, gt_ind_alignedscaled = align_resample_multi(dirs=dir_1ds,
                                                                                         norms=norms_1ds,
                                                                                         gt=gt,
                                                                                         gt_len=gt_len,
                                                                                         target_t=aligned_length,
                                                                                         normalise_dir=normalise_dir,
                                                                                         normalise_norm=normalise_norm,
                                                                                         rescale=True)
    if create_figures:
        _, _ = plot_dir_norm(dirs_alignedscaled, norms_alignedscaled, gt_ind_alignedscaled, exp_path,
                          fname='alpha_per_patient')

    if return_files: dirs_aligned, norms_aligned, gt_ind_aligned = align_resample_multi(dirs=dir_1ds,
                                                                                        norms=norms_1ds,
                                                                                        gt=gt,
                                                                                        gt_len=gt_len,
                                                                                        target_t=aligned_length,
                                                                                        normalise_dir=normalise_dir,
                                                                                        normalise_norm=normalise_norm,
                                                                                        rescale=False)

    if create_figures:
        _ = plot_pfd_per_phase_as_violin(df_pfd=pfd_df, exp_path=exp_path)
        fig = plot_scatter(exp_path=exp_path, gt_df=gt_df, phases=phases, pred_df=pred_df)

    return_args = [pred_df, gt_df, pfd_df, res, cycle_len]

    if return_files and 'dirs_aligned' in locals():
        if type(masks) is not type([]) and type(masks) is not type(None) and len(masks) > 0:
            msks = masks
        return_args += (nda_vects, msks, gt, pred, gt_len, mov, patients, dir_1ds,
                        norms_1ds, gt_ind_alignedscaled, dirs_alignedscaled, norms_alignedscaled, gt_ind_alignedscaled,
                        vols_alignedscaled, vols_rv_alignedscaled, cts)
    if DEBUG: print('load intermediate files took: {}'.format(time() - t3))
    return return_args


def interpret_deformable_async(nda_vect, gt_len, gt, dir_axis, norm_thresh, connected_component, idx, filename=None,
                               ct_calculation: Union[Literal['septum'], list, int, None] = 'septum',
                               masks=None, mask_channels=None, ):
    import numpy as np
    weight = 1
    cardiac_cycle_length = int(gt_len[:, 0].sum())
    if cardiac_cycle_length > 40:
        print(gt_len.shape, gt_len)
    ind = np.argmax(gt[:cardiac_cycle_length], axis=0)
    dir_1d_mean, direction_nda, norm_1d_mean, norm_nda, ct, masks = interpret_deformable(
        vects_nda=nda_vect,
        dir_axis=dir_axis,
        length=cardiac_cycle_length,
        norm_percentile=norm_thresh,
        component_padding=connected_component,
        filename=filename,
        masks=masks,
        mask_channels=mask_channels,
        ct_calculation=ct_calculation)
    
    from src.utils.detect_phases_from_dir import detect_phases
    indices = detect_phases(dir_1d_mean=dir_1d_mean[:cardiac_cycle_length])

    return cardiac_cycle_length, dir_1d_mean, ind, indices, norm_1d_mean, weight, idx, ct, masks


def plot_direction_instance(dir_1d_mean, directions, exp_path, mid, norm_1d_mean, norm_nda, patient, save, upper):
    from matplotlib.ticker import FormatStrFormatter
    import matplotlib.pyplot as plt
    from src.visualization.Visualize import show_2D_or_3D
    from src.models.KerasLayers import minmax_lambda
    # with sb.plotting_context("paper"):
    plt.rcParams['font.size'] = '16'
    timesteps = directions.shape[0]
    size_per_timestep = (25 / timesteps) * 2  # use a relative hight, to scope with the square mid-cavity plots
    figsize = (25, size_per_timestep)
    fig = plt.figure(figsize=figsize)
    # DIR 2D+t
    dir_2d_t = directions.copy()
    if np.ma.is_masked(directions): dir_2d_t = dir_2d_t.data * ~directions.mask
    div_cmap = 'bwr'
    fig = show_2D_or_3D(dir_2d_t, allow_slicing=False, cmap=div_cmap, fig=fig, interpolation=None, vmin=-1, vmax=1)
    ax_ = fig.get_axes()[0]
    _ = ax_.set_yticks([])
    _ = ax_.set_xticks([])
    ax = fig.get_axes()[1]
    _ = ax.set_ylabel(r'$\alpha$ ' + '\n2d+t')  # \nmid
    _ = ax.set_yticks([])
    _ = ax.set_xticks([])
    cax = fig.add_axes([0.45, 0.84, 0.1, 0.03])
    cb = fig.colorbar(ax.get_images()[len(ax.get_images()) // 2], cax=cax, orientation='horizontal')
    cb.ax.tick_params(color="black", labelsize=10, labelcolor='black')
    rows = 2
    pos = 2
    ax = fig.add_subplot(rows, 1, pos)
    # DIR 2D T x Z
    ax2 = ax.twinx()
    # DIR 1D
    _ = ax2.plot(dir_1d_mean, c='black', label=r'$\alpha_{ap_t}$')
    _ = ax.set_yticks([])
    _ = ax.set_ylabel(r'$\alpha$' + '\nz+t')  # \nap:ba
    ax2.label_outer()
    _ = ax2.tick_params(axis="y", pad=-40)
    ax2.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax2.axhline(0, color='red', linestyle='--', label='Zero Line')
    ax2.axhline(0, color='grey', linestyle='--', label='Zero Line')
    norm_cmap = 'hot'
    # NORM 2D + t
    fig = plt.figure(figsize=figsize)
    norm_2d_t = norm_nda
    norm_2d_t = minmax_lambda([norm_2d_t, mid, upper])
    fig = show_2D_or_3D(norm_2d_t, allow_slicing=False, cmap=norm_cmap, interpolation='none',
                        fig=fig)
    ax = fig.get_axes()[0]
    _ = ax.set_yticks([])
    _ = ax.set_xticks([])
    ax = fig.get_axes()[1]
    _ = ax.set_ylabel(r'$|\vec{v}|$' + '\n2d+t')  # \nmid
    _ = ax.set_yticks([])
    _ = ax.set_xticks([])
    cax = fig.add_axes([0.45, 0.84, 0.1, 0.03])
    cb = fig.colorbar(ax.get_images()[len(ax.get_images()) // 2], cax=cax, orientation='horizontal')
    # cb.set_ticks([])
    cb.ax.tick_params(color="white", labelsize=10, labelcolor='white')
    rows = 2
    pos = 2
    ax = fig.add_subplot(rows, 1, pos)
    ax2 = ax.twinx()
    _ = ax2.plot(minmax_lambda([norm_1d_mean, mid, upper]), c='black', label=r'$|\vec{v}_{t}|$')
    _ = ax2.tick_params(axis="y", pad=-20)
    ax2.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    if save:
        try:
            fig.savefig(os.path.join(exp_path, '{}_norm.svg'.format(patient)))
        except Exception as e:
            print(e)
            fig.savefig(os.path.join(exp_path, '{}_norm.png'.format(patient)))
    return fig


if __name__ == "__main__":
    import argparse, os, sys
    from src.utils.Utils_io import get_json, get_post_processing
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    parser = argparse.ArgumentParser(description='predict a phase registration model')

    # usually the exp root parameters should yield to a config, which encapsulate all experiment parameters
    parser.add_argument('-exp_root', action='store', default='/mnt/sds/sd20i001/sven/code/exp/miccai_baseline')
    parser.add_argument('-data', action='store', default='')
    parser.add_argument('-work_dir', action='store', default='/mnt/ssd/git/cmr-phase-detection')
    parser.add_argument('-c2l', action='store_true', default=False)

    results = parser.parse_args()
    os.chdir(results.work_dir)
    sys.path.append(os.getcwd())
    print('given parameters: {}'.format(results))

    # get all cfgs and dataset json (we expect to find 4 as we usually train a 4-fold cv)
    # call the predict_fn for each cfg
    cfg_files = get_json('config/config.json', results.exp_root)
    dataset_files = get_json('config/dataset.json', results.exp_root)

    patients_txt_file = os.path.join(results.exp_root, 'pred', 'patients.txt')

    if os.path.exists(patients_txt_file):
        import os
        # removing previous inference files using the os.remove() method
        os.remove(patients_txt_file)

    for cfg,data_json in zip(cfg_files, dataset_files):
        post_processing = get_post_processing(data_json)

        predict(cfg_file=cfg, data_root=results.data, c2l=results.c2l, exp=results.exp_root, json_file=data_json)
        try:
            predict(cfg_file=cfg, data_root=results.data, c2l=results.c2l, exp=results.exp_root, json_file=data_json)
            pass
        except Exception as e:
            print(e)

        try:
            logging.info('Predict cardiac keyframes and save them as figures')
            predict_phase_from_deformable(results.exp_root,
                                          create_figures=True,
                                          norm_thresh=post_processing.get('norm_threshold', 40),
                                          connected_component_filter=post_processing.get('cc_Filter', 110),
                                          dir_axis=0,
                                          roll_by_gt=False,
                                          mask_channels=post_processing.get("mask_channels", []),
                                          ct_calculation=post_processing.get("focus_point"),
                                          max_junks=None)
        except Exception as e:
            logging.error('{} predict phase failed with: {}'.format(results.exp_root, e))

    exit()
