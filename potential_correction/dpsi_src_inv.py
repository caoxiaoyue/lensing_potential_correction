import autofit as af
import autolens as al
import numpy as np
import potential_correction.util as pul
from potential_correction.dpsi_inv import DpsiPixelization
from potential_correction.src_inv import FitSrcImaging
from scipy.spatial import Delaunay
from abc import ABC, abstractmethod
from scipy.sparse import block_diag
import os
from potential_correction.visualize import show_fit_dpsi_src
from scipy.interpolate import RBFInterpolator
import GPy
from potential_correction.covariance_reg import CurvatureRegularizationDpsi, CovarianceRegularization


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
        src_grad_values = pul.source_gradient_from(values_at_grad_points, cross_size)
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


    def eval_func(self, xgrid: np.ndarray, ygrid: np.ndarray):
        if not hasattr(self, "tri"):
            self.tri = Delaunay(np.fliplr(self.points))
        if not hasattr(self, "interp_func"):
            self.interp_func = pul.LinearNDInterpolatorExt(self.tri, self.values)
        return self.interp_func(xgrid, ygrid)
        

    def eval_grad(self, xgrid: np.ndarray, ygrid: np.ndarray, cross_size=0.001):
        if not hasattr(self, "interp_grad_func"):
            grad_points = pul.gradient_points_from(self.points, cross_size=cross_size)
            values_at_grad_points = self.eval_func(grad_points[:, 1], grad_points[:, 0])
            src_grad_values = pul.source_gradient_from(values_at_grad_points, cross_size)
            self.interp_grad_func = pul.LinearNDInterpolatorExt(self.tri, src_grad_values)
        return self.interp_grad_func(xgrid, ygrid)


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
        anchor_points: np.ndarray,
        lens_start: al.Galaxy,
        source_start: SrcFactory,
        dpsi_pixelization: DpsiPixelization, #mesh + regularization properties
        src_pixelization: al.Pixelization,
        preloads: dict = None,
    ):
        self.masked_imaging = masked_imaging #autolens masked imaging object
        self.anchor_points = anchor_points
        self.lens_start = lens_start
        self.source_start = source_start
        self.dpsi_pixelization = dpsi_pixelization
        self.src_pixelization = src_pixelization

        if preloads is not None:
            for key, value in preloads.items():
                setattr(self, key, value)


    def do_source_inversion(self):
        source_galaxy = al.Galaxy(redshift=1.0, pixelization=self.src_pixelization)
        tracer = al.Tracer.from_galaxies(galaxies=[self.lens_start, source_galaxy])

        self.src_fit = FitSrcImaging(
            masked_imaging=self.masked_imaging, 
            tracer=tracer,
        )

        self.src_mapper = self.src_fit.construct_mapper()
        self.src_map_mat = self.src_fit.mapping_matrix_from(self.src_mapper)
        self.src_reg_mat = self.src_fit.regularization_matrix_from(self.src_mapper)


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
        return self.masked_imaging.grid - self.lens_start.deflections_yx_2d_from(self.masked_imaging.grid)
    

    @property
    def source_plane_source_gradient(self):
        return self.source_start.eval_grad(self.source_plane_data_grid[:, 1], self.source_plane_data_grid[:, 0])


    @property
    def source_gradient_matrix(self):
        if not hasattr(self, "src_grad_mat"):
            self.src_grad_mat = pul.source_gradient_matrix_from(self.source_plane_source_gradient)
        return self.src_grad_mat


    def pair_dpsi_data_mesh(self):
        self.pair_dpsi_data_obj = self.dpsi_pixelization.pair_dpsi_data_mesh(
            self.masked_imaging.mask, 
            self.masked_imaging.pixel_scales[0], #assume a square pixel
        )


    @property
    def dpsi_gradient_matrix(self):
        if not hasattr(self, "pair_dpsi_data_obj"):
            self.pair_dpsi_data_mesh()
        if not hasattr(self, "itp_mat"):
            self.itp_mat = self.pair_dpsi_data_obj.itp_mat
        if not hasattr(self, "dpsi_grad_mat"):
            self.dpsi_grad_mat = pul.dpsi_gradient_matrix_from(self.itp_mat, self.pair_dpsi_data_obj.Hx_dpsi, self.pair_dpsi_data_obj.Hy_dpsi)
        return self.dpsi_grad_mat
    

    @property
    def dpsi_regularization_matrix(self):
        if not hasattr(self, "pair_dpsi_data_obj"):
            self.pair_dpsi_data_mesh()
        if not hasattr(self, "dpsi_points"):
            self.dpsi_points = np.vstack([self.pair_dpsi_data_obj.ygrid_dpsi_1d, self.pair_dpsi_data_obj.xgrid_dpsi_1d]).T
        if not hasattr(self, "dpsi_reg_mat"):
            if isinstance(self.dpsi_pixelization.regularization, CurvatureRegularizationDpsi):
                self.dpsi_reg_mat = self.dpsi_pixelization.regularization.regularization_matrix_from(
                    self.pair_dpsi_data_obj.mask_dpsi, 
                )
            elif isinstance(self.dpsi_pixelization.regularization, CovarianceRegularization):
                self.dpsi_reg_mat = self.dpsi_pixelization.regularization.regularization_matrix_from(self.dpsi_points)
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


class DpsiSrcInvAnalysis(af.Analysis):
    def __init__(
            self, 
            masked_imaging: al.Imaging,
            anchor_points: np.ndarray,
            lens_start: al.Galaxy,
            source_start: SrcFactory,
            preloads: dict = None,
        ):
        self.masked_imaging = masked_imaging
        self.anchor_points = anchor_points
        self.lens_start = lens_start
        self.source_start = source_start
        self.preloads = preloads


    def log_likelihood_function(self, instance: DpsiPixelization):
        fit = FitDpsiSrcImaging(
            masked_imaging=self.masked_imaging,
            anchor_points=self.anchor_points,
            lens_start=self.lens_start,
            source_start=self.source_start,
            dpsi_pixelization=instance.dpsi_pixelization,
            src_pixelization=instance.src_pixelization,
            preloads=self.preloads,
        )
        log_ev = fit.log_evidence
        return log_ev
    

    def visualize(self, paths: af.DirectoryPaths, instance, during_analysis=True):
        fit = FitDpsiSrcImaging(
            masked_imaging=self.masked_imaging,
            anchor_points=self.anchor_points,
            lens_start=self.lens_start,
            source_start=self.source_start,
            dpsi_pixelization=instance.dpsi_pixelization, #mesh + regularization properties
            src_pixelization=instance.src_pixelization,
        )
        fit.log_evidence

        os.makedirs(paths.image_path, exist_ok=True)
        show_fit_dpsi_src(fit=fit, output=f"{paths.image_path}/fit_dpsi_src.png")