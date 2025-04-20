import numpy as np
from astropy.io import fits

def cut_image_for_potential_correction(image, factor=2):
    cut_size = image.shape[0] % factor
    if cut_size == 0:
        return image
    else:
        left_cut = int(cut_size/2)
        right_cut = cut_size - left_cut
        image_cut = image[left_cut:-right_cut, left_cut:-right_cut]
    return image_cut

def cut_fits_for_potential_correction(fits_file, factor=2):
    image = fits.getdata(fits_file)
    image_cut = cut_image_for_potential_correction(image, factor=factor)
    fits.writeto(fits_file.replace('.fits', '_cut.fits'), image_cut, overwrite=True)

def source_overlay_mesh_shape_from(masked_imaging=None, src_factor=2):
    if (src_factor is not None) and isinstance(src_factor, int):
        #----------calculate source pixel on image plane----------
        xrange_msk = float(masked_imaging.grid.binned.slim[:, 1].max() - masked_imaging.grid.binned.slim[:, 1].min())
        yrange_msk = float(masked_imaging.grid.binned.slim[:, 0].max() - masked_imaging.grid.binned.slim[:, 0].min())
        dpix = masked_imaging.grid.pixel_scale
        n0_src = int(yrange_msk / dpix / src_factor)
        n1_src = int(xrange_msk / dpix / src_factor)
        return (n0_src, n1_src)
    else:
        raise Exception("src_factor should be an integer.")
