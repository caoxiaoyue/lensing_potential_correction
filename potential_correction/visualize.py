import numpy as np
import matplotlib.pyplot as plt
import copy
from potential_correction.dpsi_inv import FitDpsiImaging
import autolens as al
from scipy.interpolate import griddata
import os


def imshow_masked_data(data_1d, mask_2d, dpix=None, ax=None, **kargs):
    data_2d = np.zeros_like(mask_2d, dtype='float')
    data_2d[~mask_2d] = data_1d
    data_2d_masked = np.ma.masked_array(data_2d, mask=mask_2d)

    #show data_2d with colorbar
    im = ax.imshow(data_2d_masked, **kargs)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    if dpix is not None:
        coord_1d = np.arange(len(mask_2d)) * dpix
        coord_1d = coord_1d - np.mean(coord_1d)
        xgrid, ygrid = np.meshgrid(coord_1d, coord_1d)
        rgrid = np.sqrt(xgrid**2 + ygrid**2)
        limit = np.max(rgrid[~mask_2d])
        ax.set_xlim(-1.0*limit, limit)
        ax.set_ylim(-1.0*limit, limit)

    return ax


def show_fit_dpsi(fit: FitDpsiImaging, output='result.png'):
    plt.figure(figsize=(15, 10))
    cmap = copy.copy(plt.get_cmap('jet'))
    cmap.set_bad(color='white')
    myargs_data = {'origin':'upper'}
    myargs_data['cmap'] = cmap
    myargs_data['extent'] = copy.copy(fit.pair_dpsi_data_obj.data_bound)
    myargs_dpsi = copy.deepcopy(myargs_data)
    xlimit = [
        fit.pair_dpsi_data_obj.xgrid_data_1d.min(),
        fit.pair_dpsi_data_obj.xgrid_data_1d.max(),
    ]
    ylimit = [
        fit.pair_dpsi_data_obj.ygrid_data_1d.min(),
        fit.pair_dpsi_data_obj.ygrid_data_1d.max(),
    ]

    plt.subplot(231)
    ax = plt.gca()
    imshow_masked_data(fit.input_image_residual, fit.masked_imaging.mask, ax=ax, **myargs_data)
    ax.set_title('Data')
    ax.plot(fit.anchor_points[:, 1], fit.anchor_points[:, 0], 'rx')
    ax.set_xlim(*xlimit)
    ax.set_ylim(*ylimit)

    plt.subplot(232)
    ax = plt.gca()
    imshow_masked_data(fit.model_image_residual_slim, fit.masked_imaging.mask, ax=ax, **myargs_data)
    ax.set_title('Model')
    ax.set_xlim(*xlimit)
    ax.set_ylim(*ylimit)

    residual_of_image_residual = fit.input_image_residual - fit.model_image_residual_slim
    plt.subplot(233)
    ax = plt.gca()
    imshow_masked_data(residual_of_image_residual, fit.masked_imaging.mask, ax=ax, **myargs_data)
    ax.set_title('Residual')
    ax.set_xlim(*xlimit)
    ax.set_ylim(*ylimit)

    norm_residual_of_image_residual = residual_of_image_residual/fit.masked_imaging.noise_map.slim
    plt.subplot(234)
    ax = plt.gca()
    imshow_masked_data(norm_residual_of_image_residual, fit.masked_imaging.mask, ax=ax, **myargs_data)
    ax.set_title('Normalized Residual')
    ax.set_xlim(*xlimit)
    ax.set_ylim(*ylimit)

    plt.subplot(235)
    ax = plt.gca()
    imshow_masked_data(fit.dpsi_slim, fit.pair_dpsi_data_obj.mask_dpsi, ax=ax, **myargs_dpsi)
    ax.plot(fit.anchor_points[:, 1], fit.anchor_points[:, 0], 'rx')
    ax.set_title('Dpsi Map')
    ax.set_xlim(*xlimit)
    ax.set_ylim(*ylimit)

    # _, dpsi_cov_mat = fit.solve_dpsi(return_error=True)
    # assert np.allclose(fit.dpsi_slim, _)
    # psi_err_slim = np.sqrt(np.diagonal(dpsi_cov_mat))

    dkappa_slim = fit.pair_dpsi_data_obj.hamiltonian_dpsi @ fit.dpsi_slim
    plt.subplot(236)
    ax = plt.gca()
    imshow_masked_data(dkappa_slim, fit.pair_dpsi_data_obj.mask_dpsi, ax=ax, **myargs_dpsi)
    ax.set_title('Dkappa Map')
    ax.set_xlim(*xlimit)
    ax.set_ylim(*ylimit)

    plt.tight_layout()
    plt.savefig(output, bbox_inches='tight')
    plt.close()


def compare_fit_with_true_perturber(fit: FitDpsiImaging, true_perturber: al.Galaxy, output='result.png'):
    plt.figure(figsize=(15, 10))
    cmap = copy.copy(plt.get_cmap('jet'))
    cmap.set_bad(color='white')
    myargs_data = {'origin':'upper'}
    myargs_data['cmap'] = cmap
    myargs_data['extent'] = copy.copy(fit.pair_dpsi_data_obj.data_bound)
    myargs_dpsi = copy.deepcopy(myargs_data)
    xlimit = [
        fit.pair_dpsi_data_obj.xgrid_data_1d.min(),
        fit.pair_dpsi_data_obj.xgrid_data_1d.max(),
    ]
    ylimit = [
        fit.pair_dpsi_data_obj.ygrid_data_1d.min(),
        fit.pair_dpsi_data_obj.ygrid_data_1d.max(),
    ]
    
    true_psi_slim = true_perturber.potential_2d_from(fit.dpsi_points)
    true_psi_slim -= true_psi_slim.min()
    plt.subplot(231)
    ax = plt.gca()
    imshow_masked_data(true_psi_slim, fit.pair_dpsi_data_obj.mask_dpsi, ax=ax, **myargs_dpsi)
    ax.set_title('True Dpsi')
    ax.plot(fit.anchor_points[:, 1], fit.anchor_points[:, 0], 'rx')
    ax.set_xlim(*xlimit)
    ax.set_ylim(*ylimit)

    model_dpsi_slim = fit.dpsi_slim
    model_dpsi_slim -= model_dpsi_slim.min()
    plt.subplot(232)
    ax = plt.gca()
    imshow_masked_data(model_dpsi_slim, fit.pair_dpsi_data_obj.mask_dpsi, ax=ax, **myargs_dpsi)
    ax.set_title('Model Dpsi')
    ax.set_xlim(*xlimit)
    ax.set_ylim(*ylimit)

    plt.subplot(233)
    ax = plt.gca()
    imshow_masked_data(true_psi_slim - model_dpsi_slim, fit.pair_dpsi_data_obj.mask_dpsi, ax=ax, **myargs_dpsi)
    ax.plot(fit.anchor_points[:, 1], fit.anchor_points[:, 0], 'rx')
    ax.set_title('Dpsi Residual')
    ax.set_xlim(*xlimit)
    ax.set_ylim(*ylimit)

    true_dkappa_slim = true_perturber.convergence_2d_from(fit.dpsi_points)
    plt.subplot(234)
    ax = plt.gca()
    imshow_masked_data(true_dkappa_slim, fit.pair_dpsi_data_obj.mask_dpsi, ax=ax, **myargs_dpsi)
    ax.plot(fit.anchor_points[:, 1], fit.anchor_points[:, 0], 'rx')
    ax.set_title('True Dkappa')
    ax.set_xlim(*xlimit)
    ax.set_ylim(*ylimit)

    model_dkappa_slim = fit.pair_dpsi_data_obj.hamiltonian_dpsi @ fit.dpsi_slim
    plt.subplot(235)
    ax = plt.gca()
    imshow_masked_data(model_dkappa_slim, fit.pair_dpsi_data_obj.mask_dpsi, ax=ax, **myargs_dpsi)
    ax.plot(fit.anchor_points[:, 1], fit.anchor_points[:, 0], 'rx')
    ax.set_title('Model Dkappa')
    ax.set_xlim(*xlimit)
    ax.set_ylim(*ylimit)

    plt.subplot(236)
    ax = plt.gca()
    imshow_masked_data(true_dkappa_slim - model_dkappa_slim, fit.pair_dpsi_data_obj.mask_dpsi, ax=ax, **myargs_dpsi)
    ax.plot(fit.anchor_points[:, 1], fit.anchor_points[:, 0], 'rx')
    ax.set_title('Dkappa Residual')
    ax.set_xlim(*xlimit)
    ax.set_ylim(*ylimit)

    plt.tight_layout()
    plt.savefig(output, bbox_inches='tight')
    plt.close()


def show_source_reconstruction(
    points, 
    values, 
    ax=None, 
    enlarge_factor=1.1,
    npixels=100,
    cmap='jet',
    **kwargs,
):
    """
    Points is defined as autolens [(y1,x1), (y2,x2), ...] order
    """
    points = np.asarray(points)
    points  = points[:, ::-1] #change to numpy/scipy api format -- [(x1,y2), (x2,y2),...] order

    half_width = max(np.abs(points.min()), np.abs(points.max()))
    half_width *= enlarge_factor

    coordinate_1d, dpix = np.linspace(-1.0*half_width, half_width, npixels, endpoint=True, retstep=True)
    xgrid, ygrid = np.meshgrid(coordinate_1d, coordinate_1d)
    extent = [-1.0*half_width-0.5*dpix, half_width+0.5*dpix, -1.0*half_width-0.5*dpix, half_width+0.5*dpix]

    source_image = griddata(points, values, (xgrid, ygrid), method='linear', fill_value=0.0)

    im = ax.imshow(source_image, origin='lower', extent=extent, cmap=cmap, **kwargs) 
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def show_fit_source(fit, output='result.png'):
    """
    fit is a FitSrcImaging instance
    see `from potential_correction.src_inv import FitSrcImaging`
    """
    plt.figure(figsize=(10, 10))
    cmap = copy.copy(plt.get_cmap('jet'))
    cmap.set_bad(color='white')
    myargs_data = {'origin':'upper'}
    myargs_data['cmap'] = cmap
    dpix = fit.masked_imaging.pixel_scales[0]
    limit = fit.masked_imaging.data.shape_native[0] * 0.5 * dpix
    extent = [-limit, limit, -limit, limit]
    myargs_data['extent'] = extent

    plt.subplot(221)
    ax = plt.gca()
    imshow_masked_data(fit.masked_imaging.data, fit.masked_imaging.mask, dpix=dpix, ax=ax, **myargs_data)
    ax.set_title('Data')

    plt.subplot(222)
    ax = plt.gca()
    imshow_masked_data(fit.model_image_slim, fit.masked_imaging.mask, dpix=dpix, ax=ax, **myargs_data)
    ax.set_title('Model')

    norm_residual_slim = (fit.masked_imaging.data-fit.model_image_slim)/fit.masked_imaging.noise_map
    plt.subplot(223)
    ax = plt.gca()
    imshow_masked_data(norm_residual_slim, fit.masked_imaging.mask, dpix=dpix, ax=ax, **myargs_data)
    ax.set_title('Norm Residual')

    plt.subplot(224)
    ax = plt.gca()
    show_source_reconstruction(fit.mapper.mapper_grids.source_plane_mesh_grid, fit.src_slim, ax=ax, enlarge_factor=1.1, npixels=100, cmap='jet')
    ax.set_title('Source')

    plt.tight_layout()
    plt.savefig(output, bbox_inches='tight')
    plt.close()


def show_fit_source_al(fit, output='result.png'):
    """
    fit is a FitImaging instance
    """
    plt.figure(figsize=(10, 10))
    cmap = copy.copy(plt.get_cmap('jet'))
    cmap.set_bad(color='white')
    myargs_data = {'origin':'upper'}
    myargs_data['cmap'] = cmap
    dpix = fit.imaging.pixel_scales[0]
    limit = fit.imaging.data.shape_native[0] * 0.5 * dpix
    extent = [-limit, limit, -limit, limit]
    myargs_data['extent'] = extent

    plt.subplot(221)
    ax = plt.gca()
    imshow_masked_data(fit.imaging.data, fit.imaging.mask, dpix=dpix, ax=ax, **myargs_data)
    ax.set_title('Data')

    plt.subplot(222)
    ax = plt.gca()
    imshow_masked_data(fit.inversion.mapped_reconstructed_image, fit.imaging.mask, dpix=dpix, ax=ax, **myargs_data)
    ax.set_title('Model')

    norm_residual_slim = (fit.imaging.data-fit.inversion.mapped_reconstructed_image)/fit.imaging.noise_map
    plt.subplot(223)
    ax = plt.gca()
    imshow_masked_data(norm_residual_slim, fit.imaging.mask, dpix=dpix, ax=ax, **myargs_data)
    ax.set_title('Norm Residual')

    mapper = fit.inversion.linear_obj_list[0]
    plt.subplot(224)
    ax = plt.gca()
    show_source_reconstruction(mapper.mapper_grids.source_plane_mesh_grid, fit.inversion.reconstruction, ax=ax, enlarge_factor=1.1, npixels=100, cmap='jet')
    ax.set_title('Source')

    plt.tight_layout()
    plt.savefig(output, bbox_inches='tight')
    plt.close()


def show_fit_dpsi_src(fit, output='result.png'):
    """
    fit is a FitDpsiSrcImaging instance
    see `from potential_correction.dpsi_src_inv import FitDpsiSrcImaging`
    """
    fig = plt.figure(figsize=(15, 10))
    cmap = copy.copy(plt.get_cmap('jet'))
    cmap.set_bad(color='white')
    myargs_data = {'origin':'upper'}
    myargs_data['cmap'] = cmap
    myargs_data['extent'] = copy.copy(fit.pair_dpsi_data_obj.data_bound)
    xlimit = [
        fit.pair_dpsi_data_obj.xgrid_data_1d.min(),
        fit.pair_dpsi_data_obj.xgrid_data_1d.max(),
    ]
    ylimit = [
        fit.pair_dpsi_data_obj.ygrid_data_1d.min(),
        fit.pair_dpsi_data_obj.ygrid_data_1d.max(),
    ]
    myargs_dpsi = copy.deepcopy(myargs_data)

    #data, noise, snr// model, residual, norm_residual// dpsi_map, dkappa_map, source_reconstruction
    plt.subplot(331)
    ax = plt.gca()
    imshow_masked_data(fit.masked_imaging.data, fit.masked_imaging.mask, ax=ax, **myargs_data)
    ax.set_title('Data')
    ax.plot(fit.anchor_points[:, 1], fit.anchor_points[:, 0], 'rx')
    ax.set_xlim(*xlimit)
    ax.set_ylim(*ylimit)

    plt.subplot(332)
    ax = plt.gca()
    imshow_masked_data(fit.masked_imaging.noise_map, fit.masked_imaging.mask, ax=ax, **myargs_data)
    ax.set_title('Noise')
    ax.set_xlim(*xlimit)
    ax.set_ylim(*ylimit)

    plt.subplot(333)
    ax = plt.gca()
    imshow_masked_data(fit.masked_imaging.data/fit.masked_imaging.noise_map, fit.masked_imaging.mask, ax=ax, **myargs_data)
    ax.set_title('SNR')
    ax.set_xlim(*xlimit)
    ax.set_ylim(*ylimit)

    plt.subplot(334)
    ax = plt.gca()
    imshow_masked_data(fit.model_image_slim, fit.masked_imaging.mask, ax=ax, **myargs_data)
    ax.set_title('Model')
    ax.set_xlim(*xlimit)
    ax.set_ylim(*ylimit)

    residual = fit.masked_imaging.data - fit.model_image_slim
    plt.subplot(335)
    ax = plt.gca()
    imshow_masked_data(residual, fit.masked_imaging.mask, ax=ax, **myargs_data)
    ax.set_title('Residual')
    ax.set_xlim(*xlimit)
    ax.set_ylim(*ylimit)

    norm_residual = residual/fit.masked_imaging.noise_map
    plt.subplot(336)
    ax = plt.gca()    
    imshow_masked_data(norm_residual, fit.masked_imaging.mask, ax=ax, **myargs_data)
    ax.set_title('Norm Residual')
    ax.set_xlim(*xlimit)
    ax.set_ylim(*ylimit)

    n_src_pixels = fit.src_regularization_matrix.shape[0]
    dpsi_slim = fit.src_dpsi_slim[n_src_pixels:]
    plt.subplot(337)
    ax = plt.gca()
    imshow_masked_data(dpsi_slim, fit.pair_dpsi_data_obj.mask_dpsi, ax=ax, **myargs_dpsi)
    ax.plot(fit.anchor_points[:, 1], fit.anchor_points[:, 0], 'rx')
    ax.set_title('Dpsi Map')
    ax.set_xlim(*xlimit)
    ax.set_ylim(*ylimit)

    dkappa_slim = fit.pair_dpsi_data_obj.hamiltonian_dpsi @ dpsi_slim
    plt.subplot(338)
    ax = plt.gca()
    imshow_masked_data(dkappa_slim, fit.pair_dpsi_data_obj.mask_dpsi, ax=ax, **myargs_dpsi)
    ax.set_title('Dkappa Map')
    ax.set_xlim(*xlimit)
    ax.set_ylim(*ylimit)
    
    src_slim = fit.src_dpsi_slim[0:n_src_pixels]
    plt.subplot(339)
    ax = plt.gca()
    show_source_reconstruction(fit.src_mapper.mapper_grids.source_plane_mesh_grid, src_slim, ax=ax, enlarge_factor=1.1, npixels=100, cmap='jet')
    ax.set_title('Source')

    plt.tight_layout()

    if output == "show":
        return fig
    else:
        os.makedirs(os.path.dirname(output), exist_ok=True)
        fig.savefig(output, bbox_inches='tight')
        plt.close(fig)


def check_grf_reconstruction(fit, true_grf, output="show"):
    fig = plt.figure(figsize=(15, 10))
    cmap = copy.copy(plt.get_cmap('jet'))
    cmap.set_bad(color='white')
    myargs_data = {'origin':'upper'}
    myargs_data['cmap'] = cmap
    myargs_data['extent'] = copy.copy(fit.pair_dpsi_data_obj.data_bound)
    xlimit = [
        fit.pair_dpsi_data_obj.xgrid_data_1d.min(),
        fit.pair_dpsi_data_obj.xgrid_data_1d.max(),
    ]
    ylimit = [
        fit.pair_dpsi_data_obj.ygrid_data_1d.min(),
        fit.pair_dpsi_data_obj.ygrid_data_1d.max(),
    ]
    myargs_dpsi = copy.deepcopy(myargs_data)

    grid_irr = al.Grid2DIrregular(
        values=np.vstack([
            fit.pair_dpsi_data_obj.ygrid_dpsi_1d, 
            fit.pair_dpsi_data_obj.xgrid_dpsi_1d,
        ]).T
    )
    dpsi_data = true_grf.potential_2d_from(grid_irr)
    kappa_data = true_grf.convergence_2d_from(grid_irr)

    plt.subplot(231)
    ax = plt.gca()
    imshow_masked_data(dpsi_data, fit.pair_dpsi_data_obj.mask_dpsi, ax=ax, **myargs_dpsi)
    ax.set_title('Dpsi data')
    ax.plot(fit.anchor_points[:, 1], fit.anchor_points[:, 0], 'rx')
    ax.set_xlim(*xlimit)
    ax.set_ylim(*ylimit)

    n_src_pixels = fit.src_regularization_matrix.shape[0]
    dpsi_slim = fit.src_dpsi_slim[n_src_pixels:]
    plt.subplot(232)
    ax = plt.gca()
    imshow_masked_data(dpsi_slim, fit.pair_dpsi_data_obj.mask_dpsi, ax=ax, **myargs_dpsi)
    ax.set_title('Dpsi model')
    ax.plot(fit.anchor_points[:, 1], fit.anchor_points[:, 0], 'rx')
    ax.set_xlim(*xlimit)
    ax.set_ylim(*ylimit)

    plt.subplot(233)
    ax = plt.gca()
    imshow_masked_data(dpsi_data -  dpsi_slim, fit.pair_dpsi_data_obj.mask_dpsi, ax=ax, **myargs_dpsi)
    ax.set_title('Dpsi residual')
    ax.plot(fit.anchor_points[:, 1], fit.anchor_points[:, 0], 'rx')
    ax.set_xlim(*xlimit)
    ax.set_ylim(*ylimit)

    plt.subplot(234)
    ax = plt.gca()
    imshow_masked_data(kappa_data, fit.pair_dpsi_data_obj.mask_dpsi, ax=ax, **myargs_data)
    ax.set_title('Kappa data')
    ax.set_xlim(*xlimit)
    ax.set_ylim(*ylimit)

    dkappa_slim = fit.pair_dpsi_data_obj.hamiltonian_dpsi @ dpsi_slim
    plt.subplot(235)
    ax = plt.gca()
    imshow_masked_data(dkappa_slim, fit.pair_dpsi_data_obj.mask_dpsi, ax=ax, **myargs_data)
    ax.set_title('Kappa model')
    ax.set_xlim(*xlimit)
    ax.set_ylim(*ylimit)

    plt.subplot(236)
    ax = plt.gca()
    imshow_masked_data(kappa_data - dkappa_slim, fit.pair_dpsi_data_obj.mask_dpsi, ax=ax, **myargs_data)
    ax.set_title('Kappa residual')
    ax.set_xlim(*xlimit)
    ax.set_ylim(*ylimit)

    plt.tight_layout()
    if output == "show":
        return fig
    else:
        os.makedirs(os.path.dirname(output), exist_ok=True)
        fig.savefig(output, bbox_inches='tight')
        plt.close(fig)

