import concurrent.futures
import logging
import os
import platform
import random
from concurrent.futures import as_completed
from time import time

import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow


from src.data.Dataset import (describe_sitk, split_one_4d_sitk_in_list_of_3d_sitk, get_phases_as_onehot)
from src.data.Preprocess import resample_3D, load_masked_img, clip_quantile, normalise_image, pad_and_crop, calc_resampled_size
from src.visualization.Visualize import show_2D_or_3D


class BaseGenerator(tensorflow.keras.utils.Sequence):
    """
    Base generator class
    """

    def __init__(self, x=None, y=None, config={}):
        """
        Creates a datagenerator for a list of nrrd images and a list of nrrd masks
        :param x: list of nrrd image file names
        :param y: list of nrrd mask file names
        :param config:
        """

        logging.info('Create DataGenerator')

        if y is not None:  # return x, y
            assert (len(x) == len(y)), 'len(X) != len(Y)'

        def normalise_paths(elem):
            """
            recursive helper to clean filepaths, could handle list of lists and list of tuples
            """
            if type(elem) in [list, tuple]:
                return [normalise_paths(el) for el in elem]
            elif isinstance(elem, str):
                return os.path.normpath(elem)
            else:
                return elem

        # linux/windows cleaning
        if platform.system() == 'Linux':
            x = normalise_paths(x)
            y = normalise_paths(y)

        self.INDICES = list(range(len(x)))
        # override if necessary
        self.SINGLE_OUTPUT = config.get('SINGLE_OUTPUT', False)

        self.IMAGES = x
        self.LABELS = y

        # if streamhandler loglevel is set to debug, print each pre-processing step
        # self.DEBUG_MODE = logging.getLogger().handlers[1].level == logging.DEBUG
        self.DEBUG_MODE = False

        # read the config, set default values if param not given
        self.SCALER = config.get('SCALER', 'MinMax')
        self.AUGMENT = config.get('AUGMENT', False)
        self.AUGMENT_PROB = config.get('AUGMENT_PROB', 0.8)
        self.SHUFFLE = config.get('SHUFFLE', True)
        self.RESAMPLE = config.get('RESAMPLE', False)
        self.SPACING = config.get('SPACING', [1.25, 1.25])
        self.SEED = config.get('SEED', 42)
        self.DIM = config.get('DIM', [256, 256])
        self.BATCHSIZE = config.get('BATCHSIZE', 32)
        self.MASK_VALUES = config.get('MASK_VALUES', [1, 2, 3])
        self.N_CLASSES = len(self.MASK_VALUES)
        # create one worker per image & mask (batchsize) for parallel pre-processing if nothing else is defined
        self.MAX_WORKERS = config.get('GENERATOR_WORKER', self.BATCHSIZE)
        self.MAX_WORKERS = min(32, self.MAX_WORKERS)

        if self.DEBUG_MODE:
            self.MAX_WORKERS = 1  # avoid parallelism when debugging, otherwise the plots are shuffled

        if not hasattr(self, 'X_SHAPE'):
            self.X_SHAPE = np.empty((self.BATCHSIZE, *self.DIM), dtype=np.float32)
            self.Y_SHAPE = np.empty((self.BATCHSIZE, *self.DIM, self.N_CLASSES), dtype=np.float32)

        logging.info(
            'Datagenerator created with: \n shape: {}\n spacing: {}\n batchsize: {}\n Scaler: {}\n Images: {} \n Augment: {} \n Thread workers: {}'.format(
                self.DIM,
                self.SPACING,
                self.BATCHSIZE,
                self.SCALER,
                len(
                    self.IMAGES),
                self.AUGMENT,
                self.MAX_WORKERS))

        self.on_epoch_end()

        if self.AUGMENT:
            logging.info('Data will be augmented (shift,scale and rotate) with albumentation')

        else:
            logging.info('No augmentation')

    def __plot_state_if_debug__(self, img, mask=None, start_time=None, step='raw'):

        if self.DEBUG_MODE:

            try:
                logging.debug('{}:'.format(step))
                logging.debug('{:0.3f} s'.format(time() - start_time))
                describe_sitk(img)
                describe_sitk(mask)
                if self.MASKS:
                    show_2D_or_3D(img, mask)
                    plt.show()
                else:
                    show_2D_or_3D(img)
                    plt.show()
                    # maybe this crashes sometimes, but will be caught
                    if mask:
                        show_2D_or_3D(mask)
                        plt.show()

            except Exception as e:
                logging.debug('plot image state failed: {}'.format(str(e)))

    def __len__(self):

        """
        Denotes the number of batches per epoch
        :return: number of batches
        """
        return int(np.floor(len(self.INDICES) / self.BATCHSIZE))

    def __getitem__(self, index):

        """
        Generate indexes for one batch of data
        :param index: int in the range of  {0: len(dataset)/Batchsize}
        :return: pre-processed batch
        """

        t0 = time()
        assert index < self.__len__(), 'invalid idx in batchgenerator: {} and len {}'.format(index, self.__len__())
        # collect n x indexes with n = Batchsize
        # starting from the given index parameter
        # which is in the range of  {0: len(dataset)/Batchsize}
        idxs = self.INDICES[index * self.BATCHSIZE: (index + 1) * self.BATCHSIZE]

        # Collects the value (a list of file names) for each index
        # list_IDs_temp = [self.LIST_IDS[k] for k in idxs]
        logging.debug('index generation: {}'.format(time() - t0))
        # Generate data
        return self.__data_generation__(idxs)

    def on_epoch_end(self):

        """
        Recreates and shuffle the indexes after each epoch
        :return: None
        """

        self.INDICES = np.arange(len(self.INDICES))
        if self.SHUFFLE:
            np.random.shuffle(self.INDICES)

    def __data_generation__(self, idxs):

        """
        Generates data containing batch_size samples

        :param list_IDs_temp:
        :return: X : (batchsize, *dim, n_channels), Y : (batchsize, *dim, number_of_classes)
        """

        # Initialization

        x = np.empty_like(self.X_SHAPE)
        y = np.empty_like(self.Y_SHAPE)

        futures = set()

        # spawn one thread per worker
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.MAX_WORKERS) as executor:

            t0 = time()
            # Generate data
            for i, ID in enumerate(idxs):

                try:
                    # keep ordering of the shuffled indexes
                    futures.add(executor.submit(self.__preprocess_one_image__, i, ID))

                except Exception as e:
                    logging.error(
                        'Exception {} in datagenerator with: image: {} or mask: {}'.format(str(e), self.IMAGES[ID],
                                                                                           self.LABELS[ID]))

        for i, future in enumerate(as_completed(futures)):
            # use the indexes i to place each processed example in the batch
            # otherwise slower images will always be at the end of the batch
            # Use the ID for exception handling as reference to the file name
            try:
                x_, y_, i, ID, needed_time = future.result()
                if self.SINGLE_OUTPUT:
                    x[i,], _ = x_, y_
                else:
                    x[i,], y[i,] = x_, y_
                logging.debug('img finished after {:0.3f} sec.'.format(needed_time))
            except Exception as e:
                logging.error(
                    'Exception {} in datagenerator with: image: {} or mask: {}'.format(str(e), self.IMAGES[ID],
                                                                                       self.LABELS[ID]))

        logging.debug('Batchsize: {} preprocessing took: {:0.3f} sec'.format(self.BATCHSIZE, time() - t0))
        if self.SINGLE_OUTPUT:
            return x.astype(np.float32), None
        else:
            return np.array(x.astype(np.float32)), np.array(y.astype(np.float32))

    def __preprocess_one_image__(self, i, ID):
        logging.error('not implemented error')


class DataGenerator(BaseGenerator):
    """
    Yields (X, Y) / image,mask for 2D and 3D U-net training
    could be used to yield (X, None)
    """

    def __init__(self, x=None, y=None, config=None):
        if config is None:
            config = {}
        self.MASKING_IMAGE = config.get('MASKING_IMAGE', False)
        self.SINGLE_OUTPUT = False
        self.MASKING_VALUES = config.get('MASKING_VALUES', [1, 2, 3])

        # how to get from image path to mask path
        # the wildcard is used to load a mask and cut the images by one or more labels
        self.REPLACE_DICT = {}
        GCN_REPLACE_WILDCARD = ('img', 'msk')
        ACDC_REPLACE_WILDCARD = ('.nii.gz', '_gt.nii.gz')

        if 'ACDC' in x[0]:
            self.REPLACE_WILDCARD = ACDC_REPLACE_WILDCARD
        else:
            self.REPLACE_WILDCARD = GCN_REPLACE_WILDCARD
        # if masks are given
        if y is not None:
            self.MASKS = True
        super().__init__(x=x, y=y, config=config)



class PhaseRegressionGenerator_v2(DataGenerator):
    """
    yields n input volumes and n output volumes
    """

    def __init__(self, x=None, y=None, config=None, dataset_json=None, in_memory=False):

        if config is None:
            config = {}
        if dataset_json is None:
            dataset_json = {}
        super().__init__(x=x, y=y, config=config)
        self.AUGMENT_PHASES = config.get('AUGMENT_PHASES', False)
        self.AUGMENT_PHASES_RANGE = config.get('AUGMENT_PHASES_RANGE', (-3, 3))
        self.T_SHAPE = config.get('T_SHAPE', 10)
        self.T_SPACING = config.get('T_SPACING', 10)
        self.PHASES = config.get('PHASES', 5)
        self.REPEAT = config.get('REPEAT_ONEHOT', True)
        self.TARGET_SMOOTHING = config.get('TARGET_SMOOTHING', False)
        self.SMOOTHING_KERNEL_SIZE = config.get('SMOOTHING_KERNEL_SIZE', 10)
        self.SMOOTHING_LOWER_BORDER = config.get('SMOOTHING_LOWER_BORDER', 0.1)
        self.SMOOTHING_UPPER_BORDER = config.get('SMOOTHING_UPPER_BORDER', 5)
        self.SMOOTHING_WEIGHT_CORRECT = config.get('SMOOTHING_WEIGHT_CORRECT', 20)
        self.SIGMA = config.get('GAUS_SIGMA', 1)
        self.HIST_MATCHING = config.get('HIST_MATCHING', False)
        self.IMG_INTERPOLATION = config.get('IMG_INTERPOLATION', sitk.sitkLinear)
        self.MSK_INTERPOLATION = config.get('MSK_INTERPOLATION', sitk.sitkNearestNeighbor)
        self.AUGMENT_TEMP = config.get('AUGMENT_TEMP', False)
        self.AUGMENT_TEMP_RANGE = config.get('AUGMENT_TEMP_RANGE', (-3, 3))
        self.RESAMPLE_T = config.get('RESAMPLE_T', False)
        self.ROTATE = config.get('ROTATE', False)
        self.TRANSLATE = config.get('TRANSLATE', True)
        self.ADD_SOFTMAX = config.get('ADD_SOFTMAX', False)
        self.SOFTMAX_AXIS = config.get('SOFTMAX_AXIS', 0)
        self.ROLL2SEPTUM = config.get('ROLL2SEPTUM', True)
        self.ROLL2LV = config.get('ROLL2LV', True)  # default, center crop according to the mean mse along t
        workers = config.get('WORKERS', 12)

        self.PRETRAINED_SEG = config.get('PRETRAINED_SEG', False)
        self.NNUNET_SEG = config.get('NNUNET_SEG', False)

        self.IN_MEMORY = in_memory
        self.THREAD_POOL = concurrent.futures.ThreadPoolExecutor(max_workers=workers)
        self.config = config

        if self.REPEAT:
            self.TARGET_SHAPE = (self.T_SHAPE, self.PHASES)
        else:
            self.TARGET_SHAPE = (self.PHASES, self.T_SHAPE)

        # if this is the case we have a sequence of 3D volumes or a sequence of 2D images
        # b, t, z, x, y, 1
        # test --> 2
        self.X_SHAPE = np.empty((self.BATCHSIZE, self.T_SHAPE, *self.DIM, 2), dtype=np.float32)
        self.X_ROLLED_SHAPE = np.empty((self.BATCHSIZE, self.T_SHAPE, *self.DIM, 1), dtype=np.float32)
        self.Y_SHAPE = np.empty((self.BATCHSIZE, 2, *self.TARGET_SHAPE),
                                dtype=np.float32)  # onehot and mask with gt length

        self.ISNIFTI = False

        logging.info('first file: {}'.format(self.IMAGES[0].lower()))
        if 'nii.gz' in dataset_json.get("suffix",{}).get("file_ending", "nii.gz"):
            self.ISNIFTI = True
        self.STARTID = dataset_json.get("start_id", 0)

        # opens a dataframe with cleaned phases per patient
        self.METADATA_FILE = config.get('DF_META', None)
        if self.METADATA_FILE and os.path.exists(self.METADATA_FILE):
            df = pd.read_csv(self.METADATA_FILE,
                             dtype={'patient': str, 'ed#': int, 'ms#': int, 'es#': int, 'pf#': int, 'md#': int})
            df.columns = df.columns.str.lower()
            self.DF_METADATA = df[['patient', 'ed#', 'ms#', 'es#', 'pf#', 'md#']]
            if self.ISNIFTI: self.DF_METADATA['patient'] = self.DF_METADATA['patient'].str.zfill(3)
        else:
            self.DF_METADATA = None

        logging.info('Temporal phase augmentation: \n{}'
                     '\n'
                     'Repeat volume: \n{}'.format(self.AUGMENT_PHASES, self.REPEAT))

        # in memory pre-processing, watch your memory footprint
        if self.IN_MEMORY:
            print('in memory preprocessing')
            zipped = list()
            futures = [self.THREAD_POOL.submit(self.__fix_preprocessing__, i, i) for i in range(len(self.IMAGES))]
            for i, future in enumerate(as_completed(futures)):
                zipped.append(future.result())
            self.model_inputs, self.onehots, self.reps, self.gt_lengths = list(map(list, zip(*zipped)))

        self.MASKS = None  # need to check if this is still necessary!

        random.seed(config.get('SEED', 42))
        logging.info('params of generator:')
        exclude_vars = ['IMAGES', 'LABELS', 'model_inputs', 'onehots', 'reps', 'gt_lengths', 'config']
        valid_var_types = [int, str, list, bool]
        logging.info(list((k, v) for k, v in vars(self).items() if
                          type(v) in valid_var_types and str(k) not in exclude_vars))

    def on_batch_end(self):
        """
        Use this callback for methods that should be executed after each new batch generation
        """
        pass

    def __data_generation__(self, list_IDs_temp):

        """
        Loads and pre-process one batch

        :param list_IDs_temp: list of int - list of ids to load
        :return: x (4d-CMR),y (Phase vector, rolled 4d-CMR, zero-flowfield)
        [(batchsize, *dim, n_channels)],
        [(batchsize, self.T_SHAPE, number_of_classes), (batchsize, dim, n_channels), (batchsize, dim, 3)]
        """

        # Initialization
        x = np.empty_like(self.X_SHAPE)  # model input
        y2 = np.empty_like(self.X_ROLLED_SHAPE)  # rolled volume
        y = np.empty_like(self.Y_SHAPE)  # model output

        futures = set()

        # One thread per worker
        t0 = time()
        ID = ''
        # Generate data
        for i, ID in enumerate(list_IDs_temp):

            try:
                # remember the ordering of the shuffled indexes,
                # otherwise files, that take longer are always at the batch end
                futures.add(self.THREAD_POOL.submit(self.__preprocess_one_image__, i, ID))

            except Exception as e:
                PrintException()
                print(e)
                logging.error(
                    'Exception {} in datagenerator with: image: {} or mask: {}'.format(str(e), self.IMAGES[ID],
                                                                                        self.LABELS[ID]))
                raise e  # testing phase --> make sure all errors are handled

        for i, future in enumerate(as_completed(futures)):
            # use the indexes to order the batch
            # otherwise slower images will always be at the end of the batch

            try:
                x_, y1_, y2_, i, ID, needed_time, y_seg = future.result()

                x[i,], y2[i,], y[i,] = x_, y1_, y2_
                logging.debug('img finished after {:0.3f} sec.'.format(needed_time))
            except Exception as e:
                # write these files into a dedicated error log
                PrintException()
                print(e)
                logging.error(
                    'Exception {} in datagenerator with:\n'
                    'image:\n'
                    '{}\n'
                    'mask:\n'
                    '{}'.format(str(e), self.IMAGES[ID], self.LABELS[ID]))
                #raise e # usually we should raise this Exception as this means we cant process one file
                # for the MNM2 dataset there are files that we cant process e.g.: 039_SA_CINE.nii.gz
                # In order to run any inference on this data we need to skip these files

        logging.debug('Batchsize: {} preprocessing took: {:0.3f} sec'.format(self.BATCHSIZE, time() - t0))

        return_tuple = tuple([[x], [y, y2, np.zeros((*y2.shape[:-1], 3))]])

        if self.PRETRAINED_SEG:
            y_seg = np.zeros([self.BATCHSIZE, self.T_SHAPE, *self.DIM, self.N_CLASSES])

        if self.PRETRAINED_SEG or self.NNUNET_SEG:
            return_tuple = tuple([[x], [y, y2, np.zeros((*y2.shape[:-1], 3)), y_seg]])
        return return_tuple

    def __fix_preprocessing__(self, i, ID):
        t1 = time()

        x = self.IMAGES[ID]
        # use the load_masked_img wrapper to enable masking of the images by any label,
        # currently not needed, but nice to have for later experiments
        model_inputs = load_masked_img(sitk_img_f=x, mask=self.MASKING_IMAGE,
                                       masking_values=self.MASKING_VALUES, replace=self.REPLACE_WILDCARD)
        gt_length = model_inputs.GetSize()[-1]
        assert (gt_length <= self.T_SHAPE), 'CMR sequence is longer ({}) than defined network input shape ({})'.format(
            model_inputs.GetSize()[-1], self.T_SHAPE)
        t_spacing = self.T_SPACING
        logging.debug('t-spacing: {}'.format(t_spacing))
        temporal_sampling_factor = 1  # don't scale the indices if we don't resample T

        # Create a list of 3D volumes for resampling
        model_inputs = split_one_4d_sitk_in_list_of_3d_sitk(model_inputs, axis=0)

        # If nnU-net use_segmentation is used, call masks and preprocess
        if self.NNUNET_SEG:
            y = self.LABELS[ID]
            mask_inputs = load_masked_img(sitk_img_f=y, mask=self.MASKING_IMAGE,
                                          masking_values=self.MASKING_VALUES, replace=self.REPLACE_WILDCARD,
                                          mask_labels=self.MASK_VALUES)
            mask_inputs = split_one_4d_sitk_in_list_of_3d_sitk(mask_inputs, axis=0)

        if self.DF_METADATA is not None:
            # Returns the indices in the following order: 'ED#', 'MS#', 'ES#', 'PF#', 'MD#'
            onehot_orig = get_phases_as_onehot(x, self.DF_METADATA, self.STARTID, temporal_sampling_factor, gt_length)
        else:  # no phase labels given, create a fake target for self-supervised learning
            onehot_orig = np.ones(shape=(self.PHASES, gt_length))
        # 5,t --> t,5
        onehot_orig = onehot_orig.T
        logging.debug('onehot initialised:')
        if self.DEBUG_MODE: plt.imshow(onehot_orig); plt.show()

        if self.RESAMPLE:
            if model_inputs[0].GetDimension() in [2, 3]:
                # calculate the new size (after resample with the given spacing) of each 3D volume
                # sitk.spacing has the opposite order than np.shape and tf.shape
                # In the config we use the numpy order z, y, x which needs to be reversed for sitk
                # transform the spacing from numpy representation towards the sitk representation
                target_spacing = list(reversed(self.SPACING))
                new_size_inputs = list(map(lambda elem: calc_resampled_size(elem, target_spacing), model_inputs))
            else:
                raise NotImplementedError('dimension not supported: {}'.format(model_inputs[0].GetDimension()))

            logging.debug('dimension: {}'.format(model_inputs[0].GetDimension()))
            logging.debug('Size before resample: {}'.format(model_inputs[0].GetSize()))

            model_inputs = list(map(lambda x:
                                    resample_3D(sitk_img=x[0],
                                                size=x[1],
                                                spacing=target_spacing,
                                                interpolate=self.IMG_INTERPOLATION),
                                    zip(model_inputs, new_size_inputs)))
            if self.NNUNET_SEG:
                mask_inputs = list(map(lambda x:
                                       resample_3D(sitk_img=x[0],
                                                   size=x[1],
                                                   spacing=target_spacing,
                                                   interpolate=sitk.sitkNearestNeighbor),
                                       zip(mask_inputs, new_size_inputs)))

        logging.debug('Spacing after resample: {}'.format(model_inputs[0].GetSpacing()))
        logging.debug('Size after resample: {}'.format(model_inputs[0].GetSize()))

        # transform to nda for further processing
        model_inputs = np.stack(list(map(lambda x: sitk.GetArrayFromImage(x), model_inputs)), axis=0)

        # How many times do we need to repeat that cycle along t to cover the desired output size
        reps = 1
        if self.REPEAT: reps = int(np.ceil(self.T_SHAPE / gt_length))

        self.__plot_state_if_debug__(img=model_inputs[len(model_inputs) // 2], start_time=t1, step='resampled')

        # Center-crop z, x, y, keep t
        # normalise
        model_inputs = pad_and_crop(model_inputs, target_shape=(model_inputs.shape[0], *self.DIM))
        model_inputs = clip_quantile(model_inputs, .95)  # lower_boundary=-1*np.quantile(-1*model_inputs, .90)
        model_inputs = normalise_image(model_inputs, normaliser=self.SCALER)  # normalise per 4D

        if self.NNUNET_SEG:
            # transform to nda for further processing
            mask_inputs = np.stack(list(map(lambda x: sitk.GetArrayFromImage(x), mask_inputs)), axis=0)

            mask_inputs = pad_and_crop(mask_inputs, target_shape=(mask_inputs.shape[0], *self.DIM))

            return model_inputs, onehot_orig, reps, gt_length, mask_inputs
        return model_inputs, onehot_orig, reps, gt_length, None

    def __preprocess_one_image__(self, i, ID):

        """

        Parameters
        ----------
        i : the position within the current batch
        ID : the index of entity which should be loaded

        Returns
        -------

        """

        t0 = time()
        if self.IN_MEMORY:  # load pre-loaded images from the current gen object
            model_inputs, onehot, reps, gt_length = (self.model_inputs[ID], self.onehots[ID],
                                                    self.reps[ID], self.gt_lengths[ID])
            mask_inputs = None
        else:
            model_inputs, onehot, reps, gt_length, mask_inputs = self.__fix_preprocessing__(i, ID)
        t1 = time()

        # roll
        model_targets = np.roll(model_inputs, shift=-1, axis=0)  # [1,2,3,0] = np.roll([0,1,2,3],shift=-1,axis=0)
        # Fake a ring behaviour of the onehot vector by
        # first, tile along t
        # second smooth with a gausian Kernel,
        # third split+maximise element-wise on both matrices
        # we normalise before definition of the target, to make sure both are from the same distribution

        model_inputs = np.tile(model_inputs, (reps,  1, 1))[:self.T_SHAPE]
        model_targets = np.tile(model_targets, (reps,  1, 1))[:self.T_SHAPE]
        onehot = np.tile(onehot, (reps, 1))[:self.T_SHAPE]
        logging.debug('onehot repeated {}:'.format(reps))
        if self.DEBUG_MODE: plt.imshow(onehot); plt.show()

        if self.NNUNET_SEG:
            mask_targets = np.roll(mask_inputs, shift=-1, axis=0)
            mask_inputs = np.tile(mask_inputs, (reps, 1, 1))[:self.T_SHAPE]
            mask_targets = np.tile(mask_targets, (reps, 1, 1))[:self.T_SHAPE]

        msk = np.ones_like(onehot)
        logging.debug('onehot pad/crop:')
        if self.DEBUG_MODE: plt.imshow(onehot); plt.show()

        # Finally normalise the 4D volume in one value space
        # Normalise the one-hot along the first or second axis
        # This can be done by:
        # - divide each element by the sum of the elements + epsilon
        # 𝜎(𝐳)𝑖=𝑧𝑖∑𝐾𝑗=1𝑧𝑗+𝜖 for 𝑖=1,…,𝐾 and 𝐳=(𝑧1,…,𝑧𝐾)∈ℝ𝐾
        # - The standard (unit) softmax function 𝜎:ℝ𝐾→ℝ𝐾 is defined by the formula
        # 𝜎(𝐳)𝑖=𝑒𝑧𝑖∑𝐾𝑗=1𝑒𝑧𝑗 for 𝑖=1,…,𝐾 and 𝐳=(𝑧1,…,𝑧𝐾)∈ℝ𝐾
        onehot = normalise_image(onehot, normaliser='minmax')
        # model_inputs = normalise_image(model_inputs, normaliser=self.SCALER)
        # model_targets = normalise_image(model_targets, normaliser=self.SCALER)
        # logging.debug('background: \n{}'.format(onehot))

        # logging.debug('normalised (sum phases per timestep == 1): \n{}'.format(onehot))
        self.__plot_state_if_debug__(img=model_inputs[len(model_inputs) // 2], start_time=t1,
                                     step='clipped cropped and pad')

        # add length of this cardiac cycle as mask to onhot if we repeat,
        # otherwise we created a mask before the padding step
        if self.REPEAT:
            msk = np.pad(
                np.ones((gt_length, self.PHASES)),
                ((0, self.T_SHAPE - gt_length), (0, 0)))

        model_inputs = np.stack([model_inputs, model_targets], axis=-1)

        onehot = np.stack([onehot, msk], axis=0)
        # make sure we do not introduce nans to the model
        assert not np.any(np.isnan(onehot))
        assert not np.any(np.isnan(model_inputs))
        assert not np.any(np.isnan(model_targets))
        if mask_inputs is None:
            return model_inputs, model_targets[..., None], onehot, i, ID, time() - t0, None
        else:
            return model_inputs, model_targets[..., None], onehot, i, ID, time() - t0, mask_inputs[..., None]

import linecache
import sys


def PrintException():
    exc_type, exc_obj, tb = sys.exc_info()
    f = tb.tb_frame
    lineno = tb.tb_lineno
    filename = f.f_code.co_filename
    linecache.checkcache(filename)
    line = linecache.getline(filename, lineno, f.f_globals)
    print('EXCEPTION IN ({}, LINE {} "{}"): {}'.format(filename, lineno, line.strip(), exc_obj))
