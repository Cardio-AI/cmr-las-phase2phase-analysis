import os

def write_sitk(direction_nda, filename, sub_folder='directions', suffix='dir', file_suffix='.nii.gz',
               spacing=(2.5, 2.5, 2.5, 1)):
    import SimpleITK as sitk
    from src.utils.Utils_io import ensure_dir
    assert direction_nda.ndim == 4, 'invalid 4d shape of the direction_nda param'
    sitk_images = [sitk.GetImageFromArray(vol.astype('float32')) for vol in direction_nda]
    sitk_images = sitk.JoinSeries(sitk_images)
    sitk_images.SetSpacing(spacing)
    if '.nrrd' in filename:
        wildcard = '.nrrd'
    else:
        wildcard = '.nii.gz'
    filepath = os.path.join(os.path.dirname(filename), sub_folder)
    ensure_dir(filepath)
    filename = os.path.join(filepath, '{}_{}{}'.format(
        os.path.basename(filename).replace(wildcard, '').replace(' ', '').replace('\n', ''), suffix, file_suffix))
    sitk.WriteImage(sitk_images, filename)