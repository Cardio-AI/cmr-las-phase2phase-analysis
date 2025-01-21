import logging
import os
import sys

from src.utils.Utils_io import save_plot, ensure_dir
import SimpleITK as sitk
from matplotlib.ticker import PercentFormatter
from collections import Counter

import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import matplotlib.patches as mpatches
import numpy as np


def plot_pfd_per_phase_as_violin(df_pfd, exp_path=None, phases=['ED','MS','ES','PF','MD'],
                                 y_limit=11, file_name='pFD_violin.png'):
    import seaborn as sb
    import pandas as pd

    sb.set_context('paper')
    sb.set(font_scale=2)
    _ = df_pfd.plot(kind='box')
    ax = sb.violinplot(data=df_pfd)
    ax = sb.stripplot(data=df_pfd, ax=ax)
    ax.set_ylim(0, y_limit)
    _ = ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    if exp_path is not None: plt.savefig(os.path.join(exp_path, file_name))
    if y_limit > 5:
        pd.options.display.float_format = "{:,.2f}".format
    else:
        pd.options.display.float_format = "{:,.4f}".format
    df_summarized = pd.concat([df_pfd.mean(axis=0), df_pfd.std(axis=0), df_pfd.median(axis=0)], axis=1)
    df_summarized.columns = ['mean', 'SD', 'median']
    print('mean overall\n', df_summarized.mean())
    print(df_summarized)


def plot_scatter(gt_df, phases, pred_df, exp_path=None):
    fig = plot_scat_(gt_df, phases, pred_df)
    if exp_path is not None: plt.savefig(os.path.join(exp_path, 'pFD_scatter.svg'))
    if exp_path is not None: plt.savefig(os.path.join(exp_path, 'pFD_scatter.png'))
    return fig


def plot_scat_(gt_df, phases, pred_df):
    import seaborn as sb
    sb.set_context('paper')
    sb.set(font_scale=1.8)
    fig, axes = plt.subplots(1, len(phases), figsize=(25, 4))
    axes = axes.flatten()
    for i, ax in enumerate(axes):
        _ = ax.scatter(pred_df[pred_df.columns[i]], gt_df[gt_df.columns[i]])
        _ = ax.axline((1, 1), slope=1)
        _ = ax.set_title(phases[i])
        _ = ax.set_xlabel('Index - pred')
        _ = ax.set_ylabel('Index - gt')
        _ = ax.set_xlim([0, 35])
        _ = ax.set_ylim([0, 35])
    plt.tight_layout()
    return fig


def plot_dir_norm_split_by(dirs, norms, gt_inds, df_merge, split_by='pathology', exp_path=None, ax1_label=r'Angle $\alpha_t$ of $\phi_t$', ax2_label=r'Norm |$\phi_t$|'):

    assert split_by in df_merge.columns, 'split_by no given in df_merge'
    import seaborn as sb
    import pandas as pd
    sb.set_context('paper')
    sb.set(font_scale=2)
    fig, ax = plt.subplots(figsize=(25, 5))
    ax2 = ax.twinx()
    ax2.margins(0, 0)
    ax.margins(0, 0)
    for p in df_merge[split_by].unique():
        msk = df_merge[split_by] == p
        dirs_p = dirs[msk]
        norms_p = norms[msk]
        df = pd.DataFrame(dirs_p).melt()
        _ = sb.lineplot(x='variable', y='value', data=df, label=r'$\alpha_t$ {}'.format(p), ci='sd', err_style=None,
                        zorder=2, legend=False, ax=ax)
        _ = ax.set_xlabel('Time (t) - linear interpolated to {} frames'.format(dirs.shape[1]))
        _ = ax.set_ylabel(ax1_label)
        _ = ax.set_xticks(np.rint(gt_inds.mean(axis=0)), minor=False)
        _ = ax.legend(loc='upper left')
        df = pd.DataFrame(norms_p).melt()
        _ = sb.lineplot(x='variable', y='value', data=df, label=r'|$\phi_t$| {}'.format(p), ci='sd', linestyle='dashed',
                        err_style=None, zorder=2, legend=False, ax=ax2)
        _ = ax2.set_ylabel(ax2_label)
        _ = ax2.legend(loc='upper right')
    plt.tight_layout()
    if exp_path is not None: plt.savefig(os.path.join(exp_path, 'alpha_per_{}.svg'.format(split_by)))
    if exp_path is not None: plt.savefig(os.path.join(exp_path, 'alpha_per_{}.png'.format(split_by)))
    return fig


def plot_dir_norm(dirs,norms,gt_inds,exp_path=None, ax1_label=r'Instances $\alpha_t$ ', ax2_label=r'Avg. |$\phi_t$|', fname='alpha_per_patient'):
    import seaborn as sb
    import pandas as pd
    sb.set_context('paper')
    sb.set(font_scale=3)
    fig, ax = plt.subplots(figsize=(25, 5))
    ax.margins(0, 0)
    ax2 = ax.twinx()
    ax2.margins(0, 0)
    for n in dirs:
        _ = ax.plot(n, alpha=0.5, zorder=0)

    df = pd.DataFrame(dirs).melt()
    _ = sb.lineplot(x='variable', y='value', data=df, color='blue', label=r'$\alpha_t$', ci='sd', err_style='bars',
                    zorder=2, legend=False, ax=ax)
    # _ = ax.set_title('Mean direction +/- SD - aligned at ED')
    _ = ax.set_xticks(np.rint(gt_inds.mean(axis=0)), minor=False)
    _ = ax.set_xlabel('Time (t) - linear interpolated to 40 frames')
    _ = ax.set_ylabel(ax1_label)
    _ = ax.legend(loc='upper left')

    df = pd.DataFrame(norms).melt()
    _ = sb.lineplot(x='variable', y='value', data=df, color='black', label=r'|$\phi_t$|', ci='sd', linestyle='dashed',
                    err_style='bars', zorder=2, legend=False, ax=ax2)
    _ = ax2.set_ylabel(ax2_label)
    _ = ax2.legend(loc='upper right')

    plt.tight_layout()
    if exp_path is not None: plt.savefig(os.path.join(exp_path, '{}.svg'.format(fname)))
    if exp_path is not None: plt.savefig(os.path.join(exp_path, '{}.png'.format(fname)))
    return fig, ax

def show_2D_or_3D(img=None, mask=None, f_size=(25,3),dpi=100, interpol='none', allow_slicing=True, cmap='gray',
                  fig=None, show=True, **kwargs):
    """
    Debug wrapper for 2D or 3D image/mask vizualisation
    wrapper checks the ndim and calls shoow_transparent or plot 3d
    :param img:
    :param mask:
    :param show:
    :param f_size:
    :return:
    """
    from src.data.Dataset import check_input_data
    img, mask, _ = check_input_data(img, mask)

    if img is not None:
        dim = img.ndim
        temp = img
    else:
        dim = mask.ndim
        temp = mask

    if dim == 2:
        f_size=(8,8)
        return show_slice_transparent(img, mask, f_size=f_size, dpi=dpi, interpol=interpol, cmap=cmap, show=show)
    elif dim == 3 and temp.shape[-1] == 1:  # data from the batchgenerator
        f_size = (8, 8)
        return show_slice_transparent(img, mask, f_size=f_size, dpi=dpi, interpol=interpol, cmap=cmap, show=show)
    elif dim == 3:
        return plot_3d_vol(img_3d=img, mask_3d=mask, fig_size=f_size,allow_slicing=allow_slicing,cmap=cmap, fig=fig, **kwargs)
    elif dim == 4 and temp.shape[-1] == 1:  # data from the batchgenerator
        return plot_3d_vol(img_3d=img, mask_3d=mask, fig_size=f_size,allow_slicing=allow_slicing,cmap=cmap, fig=fig, **kwargs)
    elif dim == 4 and temp.shape[-1] in [3,4]: # only mask
        return plot_3d_vol(img_3d=temp, mask_3d=mask, fig_size=f_size,allow_slicing=allow_slicing,cmap=cmap, fig=fig, **kwargs)
    else:
        logging.error('Unsupported dim: {}, shape: {}'.format(img.ndim, img.shape))
        raise NotImplementedError('Wrong shape Exception in: {}'.format('show_2D_or_3D()'))


def plot_3d_vol(img_3d, mask_3d=None, timestep=0, save=False, path='reports/figures/tetra/3D_vol/temp/',
                fig_size=[25, 8], show=True, allow_slicing=True,cmap='gray', fig=None, **kwargs):
    """
    plots a 3D nda, if a mask is given combine mask and image slices
    :param show:
    :param img_3d:
    :param mask_3d:
    :param timestep:
    :param save:
    :param path:
    :param fig_size:
    :return: plot figure
    """
    plot_fn = show_slice_transparent


    if isinstance(img_3d, sitk.Image):
        img_3d = sitk.GetArrayFromImage(img_3d)

    if isinstance(mask_3d, sitk.Image):
        mask_3d = sitk.GetArrayFromImage(mask_3d)

    # use float as dtype for all plots
    if img_3d is not None:
        img_3d = img_3d.astype(np.float32)
    if mask_3d is not None:
        mask_3d = mask_3d.astype(np.float32)

    if img_3d.max() == 0:
        logging.debug('timestep: {} - no values'.format(timestep))
    else:
        logging.debug('timestep: {} - plotting'.format(timestep))

    if img_3d.shape[-1] in [3,4]: # this image is a mask
        if img_3d.shape[-1] == 4:
            img_3d = img_3d[..., 1:]  # ignore background
        mask_3d = img_3d # handle this image as mask
        img_3d = np.zeros((mask_3d.shape[:-1]))
        plot_fn = show_vec_transparent

    elif img_3d.shape[-1] == 1:
        img_3d = img_3d[..., 0]  # matpotlib cant plot this shape

    if mask_3d is not None:
        if mask_3d.shape[-1] == 4:
            mask_3d = mask_3d[..., 1:]  # ignore background if 4 channels are given
        elif mask_3d.shape[-1] > 4:
            mask_3d = transform_to_binary_mask(mask_3d)

    slice_n = 1
    # slice very huge 3D volumes, otherwise they are too small on the plot
    if (img_3d.shape[0] > 20) and (img_3d.ndim == 3) and allow_slicing:
        slice_n = img_3d.shape[0] // 15
        print('{} sliced first axis {} by {}'.format('plot_3d_vol',img_3d.shape[0], slice_n ))

    img_3d = img_3d[::slice_n]
    mask_3d = mask_3d[::slice_n]if mask_3d is not None else mask_3d

    # number of subplots = no of slices in z-direction
    if fig:
        row = fig.gca().get_gridspec().nrows + 1 # add a new row
    else:
        fig = plt.figure(figsize=fig_size)
        row = 1

    for idx, slice in enumerate(img_3d):  # iterate over all slices
        #row = idx//40 +1
        ax = fig.add_subplot(row, img_3d.shape[0], idx+1)

        if mask_3d is not None:# Set the background (lowest value) to white

            ax = plot_fn(img=slice, mask=mask_3d[idx], show=True, ax=ax, cmap=cmap)
        else:
            #fig = plot_fn(img=slice, mask=None, show=False, ax=ax, cmap=cmap)
            #ax = fig.gca()
            #mixed = show_slice(img=slice, mask=[], show=False, normalize=False)
            ax.imshow(slice, cmap=cmap,**kwargs)

        ax.set_xticks([])
        ax.set_yticks([])
        #real_index = idx + (idx * slice_n)
        #ax.set_title('{}'.format(idx), color='r', fontsize=plt.rcParams['font.size'])


    fig.subplots_adjust(wspace=0, hspace=0)
    if save:
        save_plot(fig, path, str(timestep), override=False)

    if show:
        return fig
    else:
        fig.canvas.draw()
        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        return data


def show_vec_transparent(img=None, mask=None, show=True, f_size=(5, 5), ax=None, dpi=300, interpol='none', cmap=None):
    """
    Plot image + masks in one figure
    """
    from src.data.Dataset import check_input_data
    img, mask, _ = check_input_data(img, mask)
    # check image shape
    if len(img.shape) == 2:
        # image already in 2d shape take it as it is
        x_ = (img).astype(np.float32)
    elif len(img.shape) == 3:
        # take only the first channel, grayscale - ignore the others
        x_ = (img[..., 0]).astype(np.float32)
    else:
        logging.error('invalid dimensions for image: {}'.format(img.shape))
        return

    # check masks shape, handle mask without channel per label
    if len(mask.shape) == 2:  # mask with int as label values
        y_ = transform_to_binary_mask(mask, mask_values=[1, 2, 3]).astype(np.float32)
        # print(y_.shape)
    elif len(mask.shape) == 3 and mask.shape[2] == 1:  # handle mask with empty additional channel
        mask = mask[..., 0]
        y_ = transform_to_binary_mask(mask, mask_values=[1, 2, 3]).astype(np.float32)

    elif len(mask.shape) == 3 and mask.shape[2] == 3:  # handle mask with three channels
        y_ = (mask).astype(np.float32)
        thres = 0.0
        lower = np.where(y_ < thres)
        upper = np.where(y_ > -thres)
        y_[lower and upper] = 0
        y_ = np.abs(y_)
        y_ = (y_ - y_.min()) / (
                    y_.max() - y_.min() + sys.float_info.epsilon)  # scale this mask, this could also be a vectorfield
    elif len(mask.shape) == 3 and mask.shape[2] == 4:  # handle mask with 4 channels (backround = first channel)
        # ignore backround channel for plotting
        y_ = (mask[..., 1:] > 0.5).astype(np.float32)
    else:
        logging.error('invalid dimensions for masks: {}'.format(mask.shape))
        return

    if not ax:  # no axis given
        fig = plt.figure(figsize=f_size, dpi=dpi)
        ax = fig.add_subplot(1, 1, 1, frameon=False)
    else:  # axis given get the current fig
        fig = plt.gcf()

    fig.tight_layout(pad=0)
    ax.axis('off')

    # normalise image
    alpha = 1.
    vmax = 0
    if x_.max() > 0:
        x_ = (x_ - x_.min()) / (x_.max() - x_.min() + sys.float_info.epsilon)
        vmax = 0.6
        alpha = .3
    ax.imshow(x_, 'gray', vmin=0, vmax=vmax)
    ax.imshow(y_, interpolation=interpol, alpha=alpha)

    if show:
        return ax
    else:
        return fig


def transform_to_binary_mask(mask_nda, mask_values=[0, 1, 2, 3]):
    # transform the labels to binary channel masks

    mask = np.zeros((*mask_nda.shape, len(mask_values)), dtype="bool")
    for ix, mask_value in enumerate(mask_values):
        mask[..., ix] = mask_nda == mask_value
    return mask


def show_slice_transparent(img=None, mask=None, show=True, f_size=(5, 5), ax=None, dpi=300, interpol='none',
                           cmap='gray'):
    """
    Plot image + masks in one figure
    """
    if mask is not None: mask_values = [i + 1 for i in range(mask.shape[-1])]

    from src.data.Dataset import check_input_data
    img, mask, _ = check_input_data(img, mask)

    # don't print anything if no images nor masks are given
    if img is None and mask is None:
        logging.error('No image data given')
        return

    # check image shape
    if len(img.shape) == 2:
        # image already in 2d shape take it as it is
        x_ = (img).astype(np.float32)
    elif len(img.shape) == 3:
        # take only the first channel, grayscale - ignore the others
        x_ = (img[..., 0]).astype(np.float32)
    else:
        logging.error('invalid dimensions for image: {}'.format(img.shape))
        return

    # check masks shape, handle mask without channel per label
    if len(mask.shape) == 2:  # mask with int as label values
        y_ = transform_to_binary_mask(mask, mask_values=mask_values).astype(np.float32)
        # print(y_.shape)
    elif len(mask.shape) == 3 and mask.shape[2] == 1:  # handle mask with empty additional channel
        mask = mask[..., 0]
        y_ = transform_to_binary_mask(mask, mask_values=mask_values).astype(np.float32)

    elif len(mask.shape) == 3 and mask.shape[2] == 2:  # handle mask with two channels
        y_ = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.float32)
        y_[..., :mask.shape[2]] = mask[..., :mask.shape[2]]  # slice as many channels as given
    elif len(mask.shape) == 3 and mask.shape[2] == 3:  # handle mask with three channels
        y_ = (mask).astype(np.float32)
        # thres = 0.4
        # lower = np.where(y_<thres)
        # upper = np.where(y_>-thres)
        # y_[lower and upper] = 0
        # y_ = np.abs(y_)
        # y_ = (y_ - y_.min()) / (y_.max() - y_.min() + sys.float_info.epsilon) # scale this mask, this could also be a vectorfield
    elif len(mask.shape) == 3 and mask.shape[2] == 4:  # handle mask with 4 channels (backround = first channel)
        # ignore backround channel for plotting
        y_ = (mask[..., 1:] > 0.5).astype(np.float32)
    else:
        logging.error('invalid dimensions for masks: {}'.format(mask.shape))
        return

    if not ax:  # no axis given
        fig = plt.figure(figsize=f_size, dpi=dpi)
        ax = fig.add_subplot(1, 1, 1, frameon=False)
    else:  # axis given get the current fig
        fig = plt.gcf()

    fig.tight_layout(pad=0)
    ax.axis('off')

    # normalise image
    alpha = 1.
    vmax = 0
    if x_.max() > 0:
        x_ = (x_ - x_.min()) / (x_.max() - x_.min() + sys.float_info.epsilon)
        vmax = 0.6
        alpha = .3
    ax.imshow(x_, cmap=cmap, vmin=0, vmax=vmax)
    ax.imshow(y_, interpolation=interpol, alpha=alpha)

    if show:
        return ax
    else:
        return fig


def plot_displacement(col_titles, first_m, first_vol, moved, moved_m, picks, second_m,
                      second_vol, vect, y_label, plot_masks=True):
    from src.data.Preprocess import normalise_image
    cmap = 'inferno'
    # Define the plot grid size
    nrows = 3
    ncols = 7
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 7))
    vmax = 1
    for i, z in enumerate(picks):
        j = 0
        f_vol, s_vol, mov = first_vol[z], second_vol[z], moved[z]
        # t0
        axes[i, j].imshow(normalise_image(f_vol), 'gray',vmin=0, vmax=0.8)
        if plot_masks: axes[i, j].imshow(first_m[z], alpha=0.6)
        axes[i, j].set_ylabel(y_label[i], rotation=90, size='medium')
        axes[i, j].set_xticks([])
        axes[i, j].set_yticks([])
        j = j + 1
        # t1
        axes[i, j].imshow(normalise_image(s_vol), 'gray',vmin=0, vmax=0.8)
        if plot_masks: axes[i, j].imshow(second_m[z], alpha=0.6)
        axes[i, j].set_xticks([])
        axes[i, j].set_yticks([])
        j = j + 1
        # moved t0 and marks for the moved areas
        axes[i, j].imshow(normalise_image(mov), 'gray',vmin=0, vmax=0.8)
        if plot_masks: axes[i, j].imshow(moved_m[z], alpha=0.6)
        axes[i, j].imshow(np.abs(f_vol - mov), cmap=cmap, alpha=0.6)
        axes[i, j].set_xticks([])
        axes[i, j].set_yticks([])
        j = j + 1
        # vect, abs & min/max normalized
        temp = np.absolute(vect[z])
        axes[i, j].imshow(normalise_image(f_vol), 'gray', vmin=0, vmax=0.8)
        axes[i, j].imshow(normalise_image(temp), alpha=0.6)
        axes[i, j].set_xticks([])
        axes[i, j].set_yticks([])
        j = j + 1

        # magnitude
        temp = np.sqrt(
            np.square(vect[z][..., 0]) + np.square(vect[z][..., 1]) + np.square(vect[z][..., 2]))
        axes[i, j].imshow(normalise_image(f_vol), 'gray', vmin=0, vmax=.8)
        axes[i, j].imshow(temp, cmap='cividis', alpha=0.6)

        axes[i, j].set_xticks([])
        axes[i, j].set_yticks([]);
        j = j + 1
        # diff t0 - t1
        axes[i, j].imshow(normalise_image(f_vol), 'gray', vmin=0, vmax=.8)
        axes[i, j].imshow(np.abs(s_vol - f_vol), cmap=cmap,alpha=0.6)
        axes[i, j].set_xticks([])
        axes[i, j].set_yticks([]);
        j = j + 1
        # diff moved - t1
        axes[i, j].imshow(normalise_image(f_vol), 'gray', vmin=0, vmax=.8)
        axes[i, j].imshow(np.abs(s_vol - mov), cmap=cmap, alpha=0.6)
        axes[i, j].set_xticks([])
        axes[i, j].set_yticks([])
    # set column names
    for i in range(ncols):
        axes[0, i].set_title(col_titles[i])
    fig.subplots_adjust(wspace=0.0, hspace=0.0)
    return fig
