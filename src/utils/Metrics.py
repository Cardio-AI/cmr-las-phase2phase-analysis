import tensorflow as tf
import keras.backend as K

def meandiff( y_true, y_pred, apply_sum=True, apply_average=True):

    """
    Average over the batches
    the sum of the absolute difference between two arrays
    y_true and y_pred are one-hot vectors with the following shape
    batchsize * timesteps * phase classes
    e.g.: 4 * 36 * 5
    First for gt and pred:
    - get the timesteps per phase with the highest probability
    - get the absolute difference between gt and pred
    (- later we can slice each patient by the max value in the corresponding gt indices)
    - sum the diff per entity
    - calc the mean over all examples

    Parameters
    ----------
    y_true :
    y_pred :
    apply_sum :
    apply_average :
    Returns tf.float32 scalar
    -------

    """

    y_true, y_len_msk = tf.unstack(y_true,2,axis=1)
    y_pred, _ = tf.unstack(y_pred,2,axis=1)

    y_true = tf.cast(tf.convert_to_tensor(y_true), tf.float32)
    y_pred = tf.cast(tf.convert_to_tensor(y_pred), tf.float32)
    y_len_msk = tf.cast(tf.convert_to_tensor(y_len_msk), tf.float32)

    # b, 36, 5
    temp_pred = y_pred * y_len_msk
    temp_gt = y_true * y_len_msk

    # get the original lengths of each mask in the current batch
    # b, 1
    y_len = tf.cast(tf.reduce_sum(y_len_msk[:,:,0], axis=1),dtype=tf.int32)#

    #print('y_len shape: {}'.format(y_len.shape))
    # returns b, 5,
    gt_idx = tf.cast(tf.math.argmax(temp_gt, axis=1), dtype=tf.int32)
    pred_idx = tf.cast(tf.math.argmax(temp_pred, axis=1), dtype=tf.int32)
    filled_length = tf.repeat(tf.expand_dims(y_len,axis=1),5,axis=1)

    # b, 5, 3
    stacked = tf.stack([gt_idx, pred_idx, filled_length], axis=-1)

    # sum the error per entity, and calc the mean over the batches
    # for each batch ==> 5, 3 in stacked
    diffs = tf.map_fn(lambda x: get_min_dist_for_list(x), stacked, dtype=tf.int32)
    if apply_sum: diffs = tf.cast(tf.reduce_sum(diffs, axis=1),tf.float32)
    if apply_average: diffs = tf.reduce_mean(diffs)
    diffs = tf.cast(diffs, tf.float32)
    return diffs


@tf.function
def get_min_dist_for_list(vals):
    # vals has the shape 5, 3
    # for each phase tuple (gt,pred,length)
    return tf.map_fn(lambda x :get_min_distance(x),vals, dtype=tf.int32)


@tf.function
def get_min_distance(vals):

    smaller = tf.reduce_min(vals[0:2], keepdims=True)
    bigger = tf.reduce_max(vals[0:2], keepdims=True)
    mod = vals[2]

    diff = bigger - smaller # zero if our prediction is correct
    diff_ring = tf.math.abs(mod - bigger + smaller) # maybe abs is no longer necessary
    min_diff = tf.reduce_min(tf.stack([diff, diff_ring]))
    return min_diff


class MSE:

    def __init__(self, masked=False, loss_fn=tf.keras.losses.mse, onehot=False):

        self.__name__ = 'MSE_{}'.format(masked)
        self.masked = masked
        self.loss_fn = loss_fn
        self.onehot = onehot


    def __call__(self, y_true, y_pred, **kwargs):
        if self.masked:
            if y_pred.shape[1] == 2:  # this is a stacked onehot vector
                y_true, y_msk = tf.unstack(y_true, num=2, axis=1)
                y_pred, zeros = tf.unstack(y_pred, num=2, axis=1)
                # masking works only if we have the gt stacked
                y_msk = tf.cast(y_msk, tf.float32)  # weight the true cardiac cycle by zero and one
                y_true = y_msk * y_true
                y_pred = y_msk * y_pred
        elif self.onehot:
            zeros = tf.zeros_like(y_true[:, 0], tf.float32)
            ones = tf.ones_like(y_true[:, 1], tf.float32)
            msk = tf.stack([ones, zeros], axis=1)
            y_true, y_pred =  msk * y_true, msk * y_pred
            # b, 2,

        if self.loss_fn == 'cce':
            # recent tf version does not support cce with another axis than -1
            # updating tf will break the recent model graph plotting
            y_true, y_pred = tf.transpose(y_true,perm=[0,1,3,2]), tf.transpose(y_pred,perm=[0,1,3,2])
            loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.4,reduction=tf.keras.losses.Reduction.NONE)(y_true,y_pred)
            return loss
        return self.loss_fn(y_true, y_pred)

class SSIM:

    def __init__(self):
        self.__name__ = 'SSIM'

    def __call__(self, y_true, y_pred, **kwargs):

            """def get_shape(tensor):
                static_shape = tensor.shape.as_list()
                dynamic_shape = tf.unstack(tf.shape(tensor))
                dims = [s[1] if s[0] is None else s[0]
                        for s in zip(static_shape, dynamic_shape)]
                return dims

            shape_ytrue = get_shape(y_true)
            t_shape = (shape_ytrue[0],shape_ytrue[-3],shape_ytrue[-2],shape_ytrue[1]*shape_ytrue[2])
            img1 = tf.reshape(tensor=y_true, shape=t_shape)
            img2 = tf.reshape(tensor=y_pred, shape=t_shape)"""
            ssim = tf.image.ssim(y_true, y_pred, max_val=1.0, filter_size=11,
                          filter_sigma=1.5, k1=0.01, k2=0.03)
            return 1- ssim


class Grad:
    """
    N-D gradient loss.
    loss_mult can be used to scale the loss value - this is recommended if
    the gradient is computed on a downsampled vector field (where loss_mult
    is equal to the downsample factor).
    """

    def __init__(self, penalty='l1', loss_mult=None, vox_weight=None):
        self.penalty = penalty
        self.loss_mult = loss_mult
        self.vox_weight = vox_weight

    def _diffs(self, y):
        vol_shape = y.get_shape().as_list()[1:-1]
        ndims = len(vol_shape)

        df = [None] * ndims
        for i in range(ndims):
            d = i + 1
            # permute dimensions to put the ith dimension first
            r = [d, *range(d), *range(d + 1, ndims + 2)]
            yp = K.permute_dimensions(y, r)
            dfi = yp[1:, ...] - yp[:-1, ...]

            if self.vox_weight is not None:
                w = K.permute_dimensions(self.vox_weight, r)
                # TODO: Need to add square root, since for non-0/1 weights this is bad.
                dfi = w[1:, ...] * dfi

            # permute back
            # note: this might not be necessary for this loss specifically,
            # since the results are just summed over anyway.
            r = [*range(1, d + 1), 0, *range(d + 1, ndims + 2)]
            df[i] = K.permute_dimensions(dfi, r)

        return df

    def loss(self, _, y_pred):
        """
        returns Tensor of size [bs]
        """

        if self.penalty == 'l1':
            dif = [tf.abs(f) for f in self._diffs(y_pred)]
        else:
            assert self.penalty == 'l2', 'penalty can only be l1 or l2. Got: %s' % self.penalty
            dif = [f * f for f in self._diffs(y_pred)]


        df = [tf.reduce_mean(K.batch_flatten(f), axis=-1) for f in dif]
        grad = tf.add_n(df) / len(df)

        if self.loss_mult is not None:
            grad *= self.loss_mult
        return grad


def cfd(pred, gt, length):
    """
    Calculate the cyclic smallest difference between two points on a given sequence
    Args:
        pred (): predicted key frame
        gt (): ground truth key frame
        length (): length of cmr sequence

    Returns: cfd (int)

    """
    lower = min(pred,gt)
    upper = max(pred,gt)
    diff = upper-lower
    cycle_diff = abs(length-upper+lower)
    return min(diff,cycle_diff)