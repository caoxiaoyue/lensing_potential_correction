import autofit as af
import autolens as al
import numpy as np
import potential_correction.util as pul
from potential_correction.dpsi_inv import DpsiPixelization
from scipy.spatial import Delaunay
from abc import ABC, abstractmethod
from scipy.sparse import block_diag
import os
from potential_correction.visualize import show_fit_dpsi_src
from scipy.interpolate import RBFInterpolator
import GPy
from potential_correction.covariance_reg import CurvatureRegularizationDpsi, CovarianceRegularization, FourthOrderRegularizationDpsi
from typing import Dict, List, Optional
import autoarray as aa
from autoarray.util.nn import nn_py
from autoarray import exc
import traceback
import pickle


class SrcFactory(ABC):
    def __init__(self):
        pass


    @abstractmethod
    def eval_func(self, xgrid: np.ndarray, ygrid: np.ndarray):
        pass


    def eval_grad(self, xgrid: np.ndarray, ygrid: np.ndarray, cross_size=0.001):
        if xgrid.shape!= ygrid.shape:
            raise ValueError("xgrid and ygrid must have the same shape")
        origin_shape = xgrid.shape
        points = np.vstack((ygrid.flatten(), xgrid.flatten())).T
        grad_points = pul.gradient_points_from(points, cross_size=cross_size)
        values_at_grad_points = self.eval_func(grad_points[:, 1], grad_points[:, 0])
        src_grad_values = pul.source_gradient_from(values_at_grad_points, grad_points)
        return src_grad_values.reshape(*origin_shape, 2)



class PixSrcFactory(SrcFactory):
    def __init__(self, points: np.ndarray, values: np.ndarray, rbf_interp: bool = True):
        self.points = points # (n_points, 2), in autolens [(y1,x1), (y2,x2),...] order
        self.values = values 
        self.rbf_interp = rbf_interp


    def eval_func(self, xgrid: np.ndarray, ygrid: np.ndarray):
        if not self.rbf_interp:
            if not hasattr(self, "tri"):
                self.tri = Delaunay(np.fliplr(self.points))
            if not hasattr(self, "interp_func"):
                self.interp_func = pul.LinearNDInterpolatorExt(self.tri, self.values)
            return self.interp_func(xgrid, ygrid)
        else:
            if not hasattr(self, "interp_func"):
                self.interp_func = RBFInterpolator(np.fliplr(self.points), self.values, kernel="quintic")
            grid_flat = np.vstack([xgrid.flat, ygrid.flat]).T
            itp_vals = self.interp_func(grid_flat)
            return itp_vals.reshape(xgrid.shape)



class PixSrcFactoryITP(SrcFactory):
    def __init__(self, points: np.ndarray, values: np.ndarray):
        self.points = points # (n_points, 2), in autolens [(y1,x1), (y2,x2),...] order
        self.values = values 
        self.vor_mesh = aa.Mesh2DVoronoi(values=self.points)


    def eval_func(self, xgrid: np.ndarray, ygrid: np.ndarray):
        if not hasattr(self, "tri"):
            self.tri = Delaunay(np.fliplr(self.points))
        if not hasattr(self, "interp_func"):
            self.interp_func = pul.LinearNDInterpolatorExt(self.tri, self.values)
        return self.interp_func(xgrid, ygrid)
        

    def eval_grad(self, xgrid: np.ndarray, ygrid: np.ndarray, cross_size=None):
        if cross_size is None:
            grad_points = self.vor_mesh.split_cross
        else:
            grad_points = pul.gradient_points_from(self.points, cross_size=cross_size)
        values_at_grad_points = self.eval_func(grad_points[:, 1], grad_points[:, 0])
        if not hasattr(self, "interp_grad_func"):
            src_grad_values = pul.source_gradient_from(values_at_grad_points, grad_points)
            self.interp_grad_func = pul.LinearNDInterpolatorExt(self.tri, src_grad_values)
        return self.interp_grad_func(xgrid, ygrid)



class PixSrcFactoryNN(SrcFactory):
    def __init__(self, points: np.ndarray, values: np.ndarray):
        self.points = points # (n_points, 2), in autolens [(y1,x1), (y2,x2),...] order
        self.values = values 
        self.vor_mesh = aa.Mesh2DVoronoi(values=self.points)


    def eval_func(self, xgrid: np.ndarray, ygrid: np.ndarray):
        itp_values = nn_py.natural_interpolation(
            self.points[:, 1],
            self.points[:, 0], 
            self.values, 
            xgrid.ravel(), 
            ygrid.ravel(),
        )
        return itp_values.reshape(xgrid.shape)
        

    def eval_grad(self, xgrid: np.ndarray, ygrid: np.ndarray, cross_size=None):
        if cross_size is None:
            grad_points = self.vor_mesh.split_cross
        else:
            grad_points = pul.gradient_points_from(self.points, cross_size=cross_size)
        values_at_grad_points = self.eval_func(grad_points[:, 1], grad_points[:, 0])
        if not hasattr(self, "grad_values"):
            self.grad_values = pul.source_gradient_from(values_at_grad_points, grad_points)
        itp_values_y = nn_py.natural_interpolation(
            self.points[:, 1],
            self.points[:, 0], 
            self.grad_values[:, 0], 
            xgrid.ravel(), 
            ygrid.ravel(),
        )
        itp_values_x = nn_py.natural_interpolation(
            self.points[:, 1],
            self.points[:, 0], 
            self.grad_values[:, 1], 
            xgrid.ravel(), 
            ygrid.ravel(),
        )
        itp_values = np.vstack([itp_values_y, itp_values_x]).T
        return itp_values



class PixSrcFactoryGPR(SrcFactory):
    def __init__(self, points: np.ndarray, values: np.ndarray, scheme: str = "rbf"):
        self.points = points # (n_points, 2), in autolens [(y1,x1), (y2,x2),...] order
        self.values = values 
        self.scheme = scheme


    def eval_func(self, xgrid: np.ndarray, ygrid: np.ndarray):
        if not hasattr(self, "interp_func"):
            if self.scheme == "matern52":
                ker = GPy.kern.Matern52(2, ARD=False)
            elif self.scheme == "rbf":
                ker = GPy.kern.RBF(2, ARD=False)
            else:
                raise ValueError("Unsupported kernel type")
            self.interp_func = GPy.models.GPRegression(np.fliplr(self.points), self.values.reshape(-1, 1), ker)
            self.interp_func.optimize(optimizer="lbfgsb", messages=1, max_f_eval = 5000)
        grid_flat = np.vstack([xgrid.flat, ygrid.flat]).T
        itp_mean, itp_sigma = self.interp_func.predict(grid_flat, full_cov=False, include_likelihood=False)
        return itp_mean.reshape(xgrid.shape)
    


class DpsiSrcPixelization:
    def __init__(
            self, 
            dpsi_pixelization: DpsiPixelization, 
            src_pixelization: al.Pixelization,
        ):
        self.dpsi_pixelization = dpsi_pixelization
        self.src_pixelization = src_pixelization



class FitDpsiSrcImaging:
    def __init__(
        self, 
        masked_imaging: al.Imaging,
        lens_start: al.Galaxy,
        source_start: SrcFactory,
        dpsi_pixelization: DpsiPixelization, #mesh + regularization properties
        src_pixelization: al.Pixelization,
        anchor_points: Optional[np.ndarray] = np.array([[(), ()], [(), ()]]),
        adapt_image: Optional[al.Array2D] = None,
        settings_inversion: Optional[aa.SettingsInversion] = None,
        preloads: dict = None,
    ):
        self.masked_imaging = masked_imaging #autolens masked imaging object
        self.anchor_points = anchor_points
        self.lens_start = lens_start
        self.source_start = source_start
        self.dpsi_pixelization = dpsi_pixelization
        self.src_pixelization = src_pixelization
        self.adapt_image = adapt_image
        if settings_inversion is None:
            self.settings_inversion = al.SettingsInversion(
                use_w_tilde=False, 
                use_positive_only_solver=True,
                relocate_pix_border=True,
            )
        else:
            self.settings_inversion = settings_inversion
        if preloads is not None:
            for key, value in preloads.items():
                setattr(self, key, value)

        self.masked_imaging = self.masked_imaging.apply_settings(
            settings=al.SettingsImaging(sub_size=4, sub_size_pixelization=4)
        )
        # mask = al.Mask2D(mask=self.pair_dpsi_data_obj.mask_data, pixel_scales=self.masked_imaging.pixel_scales)
        # self.masked_imaging.apply_mask(mask=mask)


    def do_source_inversion(self):
        source_galaxy = al.Galaxy(redshift=1.0, pixelization=self.src_pixelization)

        if self.adapt_image is not None:
            self.adapt_images = al.AdaptImages(galaxy_image_dict={source_galaxy: self.adapt_image})
        else:
            self.adapt_images = None

        tracer = al.Tracer.from_galaxies(galaxies=[self.lens_start, source_galaxy])

        self.src_fit = al.FitImaging(
            dataset=self.masked_imaging, 
            tracer=tracer,
            adapt_images=self.adapt_images,
            settings_inversion=self.settings_inversion,
        )

        self.src_mapper = self.src_fit.inversion.linear_obj_list[0] ##Note, not very elegant, but works for now
        self.src_map_mat = self.src_fit.inversion.operated_mapping_matrix
        # self.src_reg_mat = 0.5 * (self.src_fit.inversion.regularization_matrix + self.src_fit.inversion.regularization_matrix.T) #src_reg_mat may be asymmetric due to numerical errors, make sure it is symmetric
        self.src_reg_mat = self.src_fit.inversion.regularization_matrix


    @property
    def inverse_noise_covariance_matrix(self):
        if not hasattr(self, "inv_cov_mat"):
            noise_1d = self.masked_imaging.noise_map
            self.inv_cov_mat = pul.inverse_covariance_matrix_from(noise_1d)
        return self.inv_cov_mat


    @property
    def psf_matrix(self):
        if not hasattr(self, "psf_mat"):
            self.psf_mat = pul.psf_matrix_numba(self.masked_imaging.psf.native, self.masked_imaging.mask)
        return self.psf_mat
    

    @property
    def source_plane_data_grid(self):
        #Note!!!, we implictly assume that the single plane ray-tracing is used here 
        return self.masked_imaging.grid.binned.slim - self.lens_start.deflections_yx_2d_from(self.masked_imaging.grid.binned.slim)
    

    @property
    def source_plane_source_gradient(self):
        return self.source_start.eval_grad(self.source_plane_data_grid[:, 1], self.source_plane_data_grid[:, 0])


    @property
    def source_gradient_matrix(self):
        if not hasattr(self, "src_grad_mat"):
            self.src_grad_mat = pul.source_gradient_matrix_from(self.source_plane_source_gradient)
        return self.src_grad_mat


    @property
    def pair_dpsi_data_obj(self):
        if not hasattr(self, "_pair_dpsi_data_obj"):
            self._pair_dpsi_data_obj = self.dpsi_pixelization.pair_dpsi_data_mesh(
                self.masked_imaging.mask, 
                self.masked_imaging.pixel_scales[0], #assume a square pixel
            )
        return self._pair_dpsi_data_obj


    @property
    def dpsi_gradient_matrix(self):
        if not hasattr(self, "itp_mat"):
            self.itp_mat = self.pair_dpsi_data_obj.itp_mat
        if not hasattr(self, "dpsi_grad_mat"):
            self.dpsi_grad_mat = pul.dpsi_gradient_matrix_from(self.itp_mat, self.pair_dpsi_data_obj.Hx_dpsi, self.pair_dpsi_data_obj.Hy_dpsi)
        return self.dpsi_grad_mat
    

    @property
    def dpsi_regularization_matrix(self):
        if not hasattr(self, "dpsi_points"):
            self.dpsi_points = np.vstack([self.pair_dpsi_data_obj.ygrid_dpsi_1d, self.pair_dpsi_data_obj.xgrid_dpsi_1d]).T
        if not hasattr(self, "dpsi_reg_mat"):
            if isinstance(self.dpsi_pixelization.regularization, CurvatureRegularizationDpsi) or isinstance(self.dpsi_pixelization.regularization, FourthOrderRegularizationDpsi):
                self.dpsi_reg_mat = self.dpsi_pixelization.regularization.regularization_matrix_from(
                    self.pair_dpsi_data_obj.mask_dpsi, 
                )
            elif isinstance(self.dpsi_pixelization.regularization, CovarianceRegularization):
                self.dpsi_reg_mat = self.dpsi_pixelization.regularization.regularization_matrix_from(self.dpsi_points)
        # self.dpsi_reg_mat = 0.5 * (self.dpsi_reg_mat + self.dpsi_reg_mat.T) #dpsi_reg_mat may be asymmetric due to numerical errors, make sure it is symmetric
        return self.dpsi_reg_mat
    
    
    @property
    def dpsi_mapping_matrix(self):
        if not hasattr(self, "dpsi_map_mat"):
            self.dpsi_map_mat = -1.0 * self.psf_matrix @ self.source_gradient_matrix @ self.dpsi_gradient_matrix
        return self.dpsi_map_mat
    

    @property
    def src_regularization_matrix(self):
        #change for every possible source pixelization
        if not hasattr(self, "src_reg_mat"):
            self.do_source_inversion()
        return self.src_reg_mat


    @property
    def src_mapping_matrix(self):
        #change for every possible source pixelization
        if not hasattr(self, "src_map_mat"):
            self.do_source_inversion()
        return self.src_map_mat


    @property
    def mapping_matrix(self):
        if not hasattr(self, "map_mat"):
            self.map_mat = np.hstack((self.src_mapping_matrix, self.dpsi_mapping_matrix))
        return self.map_mat
    

    @property
    def regularization_matrix(self):
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.block_diag.html
        if not hasattr(self, "reg_mat"):
            self.reg_mat = block_diag([self.src_regularization_matrix, self.dpsi_regularization_matrix])
        return self.reg_mat


    @property
    def data_vector(self):
        if not hasattr(self, "d_vec"):
            self.d_vec = self.mapping_matrix.T @ self.inverse_noise_covariance_matrix @ self.masked_imaging.data
        return self.d_vec
    

    @property
    def curvature_matrix(self):
        if not hasattr(self, "curv_mat"):
            self.curv_mat = self.mapping_matrix.T @ self.inverse_noise_covariance_matrix @ self.mapping_matrix
        return self.curv_mat
    

    @property
    def curvature_regularization_matrix(self):
        return self.curvature_matrix + self.regularization_matrix
    

    def solve_src_dpsi(self, return_error=False):
        if return_error:
            return np.linalg.solve(self.curvature_regularization_matrix, self.data_vector), np.linalg.inv(self.curvature_regularization_matrix)
        else:
            return np.linalg.solve(self.curvature_regularization_matrix, self.data_vector)


    @property
    def log_evidence(self):
        if not hasattr(self, "src_dpsi_slim") or not hasattr(self, "model_image_slim"):
            self.src_dpsi_slim = self.solve_src_dpsi()
            self.model_image_slim = self.mapping_matrix @ self.src_dpsi_slim

        #noise normalization term
        self.noise_term= float(np.sum(np.log(2 * np.pi * self.masked_imaging.noise_map ** 2.0))) * (-0.5)

        #log det cuverd reg term
        sign, logval = np.linalg.slogdet(self.curvature_regularization_matrix)
        if sign != 1:
            # with open("err_fit_imaging.pkl", "wb") as f:
            #     pickle.dump(self, f)
            raise Exception(f"The curve reg matrix is not positive definite.")
        self.log_det_curve_reg_term = logval * (-0.5)

        #log det regularization matrix term
        """
        sign, logval = np.linalg.slogdet(self.regularization_matrix)
        if sign != 1:
            raise Exception(f"The regularization matrix is not positive definite.")
        self.log_det_reg_term = logval * 0.5
        """
        try:
            self.log_det_reg_term_src = pul.log_det_mat(self.src_regularization_matrix, sparse=True) * 0.5
        except:
            raise Exception(f"The source regularization matrix is not positive definite.")
        try:
            sign, logval = np.linalg.slogdet(self.dpsi_regularization_matrix)
            if sign != 1:
                raise Exception(f"The dpsi regularization matrix is not positive definite.")
            self.log_det_reg_term_dpsi = logval * 0.5
        except:
            self.log_det_reg_term_dpsi = pul.log_det_mat(self.dpsi_regularization_matrix, sparse=True) * 0.5
        self.log_det_reg_term = self.log_det_reg_term_src + self.log_det_reg_term_dpsi

        #src-dpsi covariance term
        reg_cov_term = self.src_dpsi_slim.T @ self.regularization_matrix @ self.src_dpsi_slim
        self.reg_cov_term = float(reg_cov_term) * (-0.5)

        #chi2 term
        image_residual = self.masked_imaging.data - self.model_image_slim
        norm_residual = image_residual / self.masked_imaging.noise_map
        self.chi2_term = float(np.sum(norm_residual**2)) * (-0.5)

        #evidence
        return self.noise_term + self.log_det_curve_reg_term + self.log_det_reg_term + self.reg_cov_term + self.chi2_term
    

    def draw_random_solutions(self, n_solutions=300):
        """
        draw the dpsi/source solutions that fit can the data within the level permitted by the noise and regularizatoin levels
        """
        mean, cov_mat = self.solve_src_dpsi(return_error=True)
        # cov_mat = 0.5 * (cov_mat + cov_mat.T) #cov_mat may be asymmetric due to numerical errors, make sure it is symmetric
        cov_mat = cov_mat + 1e-8 * np.eye(cov_mat.shape[0])
        L = np.linalg.cholesky(cov_mat)
    
        r_samps = np.random.randn(n_solutions, len(mean))
        # Compute solution samples
        solutions = mean + np.dot(L, r_samps.T).T

        return solutions

    
    @property
    def best_fit_source(self):
        n_s = self.src_regularization_matrix.shape[0]
        return self.src_dpsi_slim[0:n_s]


    @property
    def best_fit_dpsi(self):
        n_s = self.src_regularization_matrix.shape[0]
        return self.src_dpsi_slim[n_s:]


    @property
    def rescaled_dpsi(self):
        try:
            if not hasattr(self, 'dpsi_at_anchors'):
                tri = Delaunay(np.fliplr(self.dpsi_points)) #to [[x1,y1], [x2, y2] ...] order
                self.dpsi_interpl = pul.LinearNDInterpolatorExt(tri, self.best_fit_dpsi)
                self.dpsi_at_anchors = self.dpsi_interpl(self.anchor_points[:, 1], self.anchor_points[:, 0])
                #Suyu's dpsi rescaling scheme --> avoid wandering source position + uncertain constant factor of lensing potential
            ay, ax, c = pul.solve_dpsi_rescale_factor(self.anchor_points, self.dpsi_at_anchors)
            dpsi_new = ay*self.anchor_points[:, 0] + ax*self.anchor_points[:, 1] + c + self.best_fit_dpsi
            return dpsi_new, ay, ax, c
        except:
            return self.best_fit_dpsi, 0.0, 0.0, 0.0
      
      

class DpsiSrcInvAnalysis(af.Analysis):
    def __init__(
            self, 
            masked_imaging: al.Imaging,
            lens_start: al.Galaxy,
            source_start: SrcFactory,
            anchor_points: Optional[np.ndarray] = np.array([[(), ()], [(), ()]]),
            adapt_image: Optional[al.Array2D] = None,
            settings_inversion: Optional[aa.SettingsInversion] = None,
            preloads: dict = None,
        ):
        self.masked_imaging = masked_imaging
        self.anchor_points = anchor_points
        self.lens_start = lens_start
        self.source_start = source_start
        self.adapt_image = adapt_image
        if settings_inversion is None:
            self.settings_inversion = al.SettingsInversion(
                use_w_tilde=False, 
                use_positive_only_solver=True,
                relocate_pix_border=True,
                image_mesh_min_mesh_pixels_per_pixel=3,
                image_mesh_min_mesh_number=5,
                image_mesh_adapt_background_percent_threshold=0.1,
                image_mesh_adapt_background_percent_check=0.8,
            )
        else:
            self.settings_inversion = settings_inversion
        self.preloads = preloads


    def log_likelihood_function(self, instance: DpsiPixelization):
        fit = FitDpsiSrcImaging(
            masked_imaging=self.masked_imaging,
            anchor_points=self.anchor_points,
            lens_start=self.lens_start,
            source_start=self.source_start,
            dpsi_pixelization=instance.dpsi_pixelization,
            src_pixelization=instance.src_pixelization,
            adapt_image=self.adapt_image,
            settings_inversion=self.settings_inversion,
            preloads=self.preloads,
        )
        try:
            log_ev = fit.log_evidence
        except exc.InversionException as e:
            exception_info = traceback.format_exc()
            print(exception_info)
            return -1e8
        return log_ev
    

    def visualize(self, paths: af.DirectoryPaths, instance, during_analysis=True):
        fit = FitDpsiSrcImaging(
            masked_imaging=self.masked_imaging,
            anchor_points=self.anchor_points,
            lens_start=self.lens_start,
            source_start=self.source_start,
            dpsi_pixelization=instance.dpsi_pixelization, #mesh + regularization properties
            src_pixelization=instance.src_pixelization,
            adapt_image=self.adapt_image,
            settings_inversion=self.settings_inversion,
            preloads=self.preloads,
        )
        fit.log_evidence

        os.makedirs(paths.image_path, exist_ok=True)
        show_fit_dpsi_src(fit=fit, output=f"{paths.image_path}/fit_dpsi_src.png")