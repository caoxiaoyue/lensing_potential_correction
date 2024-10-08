import autofit as af
import autolens as al
import numpy as np
from potential_correction import dpsi_mesh 
from autoarray.inversion.regularization.abstract import AbstractRegularization
import potential_correction.util as pul
from potential_correction.covariance_reg import CurvatureRegularizationDpsi, CovarianceRegularization, FourthOrderRegularizationDpsi
from typing import Dict, List, Optional
from scipy.spatial import Delaunay


class DpsiPixelization:
    def __init__(self, mesh: dpsi_mesh.RegularDpsiMesh, regularization: AbstractRegularization) -> None:
        self.mesh = mesh
        self.regularization = regularization

    def pair_dpsi_data_mesh(self, mask, pixel_scale):
        return dpsi_mesh.PairRegularDpsiMesh(mask, pixel_scale, self.mesh.factor)
        

class FitDpsiImaging:
    def __init__(
        self, 
        masked_imaging: al.Imaging,
        image_residual: np.ndarray,
        source_gradient: np.ndarray,
        dpsi_pixelization: DpsiPixelization,
        anchor_points: Optional[np.ndarray] = np.array([[(), ()], [(), ()]]),
        preloads: dict = None,
    ):
        self.masked_imaging = masked_imaging #autolens masked imaging object
        self.input_image_residual = image_residual #shape: [n_unmasked_data_pixels,]
        self.source_gradient = source_gradient #shape: [n_unmasked_data_pixels, 2]
        self.anchor_points = anchor_points
        self.dpsi_pixelization = dpsi_pixelization

        if preloads is not None:
            for key, value in preloads.items():
                setattr(self, key, value)

        self.masked_imaging = self.masked_imaging.apply_settings(
            settings=al.SettingsImaging(sub_size=4, sub_size_pixelization=4)
        )
        # mask = al.Mask2D(mask=self.pair_dpsi_data_obj.mask_data, pixel_scales=self.masked_imaging.pixel_scales)
        # self.masked_imaging.apply_mask(mask=mask)


    @property
    def inverse_noise_covariance_matrix(self):
        noise_1d = self.masked_imaging.noise_map.binned.slim
        return pul.inverse_covariance_matrix_from(noise_1d)


    @property
    def psf_matrix(self):
        return pul.psf_matrix_numba(self.masked_imaging.psf.native, self.masked_imaging.mask)


    @property
    def source_gradient_matrix(self):
        return pul.source_gradient_matrix_from(self.source_gradient)


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
        return pul.dpsi_gradient_matrix_from(self.itp_mat, self.pair_dpsi_data_obj.Hx_dpsi, self.pair_dpsi_data_obj.Hy_dpsi)
    

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
        return self.dpsi_reg_mat


    @property
    def mapping_matrix(self):
        return -1.0 * self.psf_mat @ self.src_grad_mat @ self.dpsi_grad_mat


    @property
    def data_vector(self):
        return self.map_mat.T @ self.inv_cov_mat @ self.input_image_residual
    

    @property
    def curvature_regularization_matrix(self):
        return self.map_mat.T @ self.inv_cov_mat @ self.map_mat + self.reg_mat
    

    def construct_useful_matrices(self):
        #output the time consumption of each thse matrix evaluelation
        if not hasattr(self, "psf_mat"):
            self.psf_mat = self.psf_matrix
        if not hasattr(self, "inv_cov_mat"):
            self.inv_cov_mat = self.inverse_noise_covariance_matrix
        if not hasattr(self, "src_grad_mat"):
            self.src_grad_mat = self.source_gradient_matrix
        if not hasattr(self, "dpsi_grad_mat"):
            self.dpsi_grad_mat = self.dpsi_gradient_matrix
        if not hasattr(self, "reg_mat"):
            self.reg_mat = self.dpsi_regularization_matrix
        if not hasattr(self, "map_mat"):
            self.map_mat = self.mapping_matrix
        if not hasattr(self, "d_vec"):
            self.d_vec = self.data_vector
        self.curve_reg_mat = self.curvature_regularization_matrix


    def solve_dpsi(self, return_error=False):
        self.construct_useful_matrices()
        if return_error:
            return np.linalg.solve(self.curve_reg_mat, self.d_vec), np.linalg.inv(self.curve_reg_mat)
        else:
            return np.linalg.solve(self.curve_reg_mat, self.d_vec)
        

    @property
    def log_evidence(self):
        if not hasattr(self, "dpsi_slim") or not hasattr(self, "model_image_residual_slim"):
            self.dpsi_slim = self.solve_dpsi()
            self.model_image_residual_slim = self.map_mat @ self.dpsi_slim 

        #noise normalization term
        self.noise_term= float(np.sum(np.log(2 * np.pi * self.masked_imaging.noise_map ** 2.0))) * (-0.5)

        #log det cuverd reg term
        sign, logval = np.linalg.slogdet(self.curve_reg_mat)
        if sign != 1:
            raise Exception(f"The curve reg matrix is not positive definite.")
        self.log_det_curve_reg_term = logval * (-0.5)

        #log det regularization matrix term
        try:
            sign, logval = np.linalg.slogdet(self.reg_mat)
            if sign != 1:
                raise Exception(f"The regularization matrix is not positive definite.")
            self.log_det_reg_term = logval * 0.5
        except:
            self.log_det_reg_term = pul.log_det_mat(self.reg_mat, sparse=True) * 0.5

        #dpsi covariance term
        reg_cov_term = self.dpsi_slim.T @ self.reg_mat @ self.dpsi_slim
        self.reg_cov_term = float(reg_cov_term) * (-0.5)

        #chi2 term
        residual_of_image_residual = self.input_image_residual - self.model_image_residual_slim
        norm_residual_of_image_residual = residual_of_image_residual / self.masked_imaging.noise_map.binned.slim
        self.chi2_term = float(np.sum(norm_residual_of_image_residual**2)) * (-0.5)

        #evidence
        return self.noise_term + self.log_det_curve_reg_term + self.log_det_reg_term + self.reg_cov_term + self.chi2_term


    @property
    def rescaled_dpsi(self):
        try:
            if not hasattr(self, 'dpsi_at_anchors'):
                tri = Delaunay(np.fliplr(self.dpsi_points)) #to [[x1,y1], [x2, y2] ...] order
                self.dpsi_interpl = pul.LinearNDInterpolatorExt(tri, self.dpsi_slim)
                self.dpsi_at_anchors = self.dpsi_interpl(self.anchor_points[:, 1], self.anchor_points[:, 0])
                #Suyu's dpsi rescaling scheme --> avoid wandering source position + uncertain constant factor of lensing potential
            ay, ax, c = pul.solve_dpsi_rescale_factor(self.anchor_points, self.dpsi_at_anchors)
            dpsi_new = ay*self.anchor_points[:, 0] + ax*self.anchor_points[:, 1] + c + self.dpsi_slim
            return dpsi_new, ay, ax, c
        except:
            return self.dpsi_slim, 0.0, 0.0, 0.0
            
        

# import time
class DpsiInvAnalysis(af.Analysis):
    def __init__(
            self, 
            masked_imaging: al.Imaging,
            image_residual: np.ndarray,
            source_gradient: np.ndarray,
            anchor_points: Optional[np.ndarray] = np.array([[(), ()], [(), ()]]),
            preloads: dict = None,
        ):
        self.masked_imaging = masked_imaging
        self.image_residual = image_residual
        self.source_gradient = source_gradient
        self.anchor_points = anchor_points
        self.preloads = preloads


    def log_likelihood_function(self, instance):
        # print('----fitting----')
        # t0 = time.time()
        fit = FitDpsiImaging(
            masked_imaging=self.masked_imaging,
            image_residual=self.image_residual,
            source_gradient=self.source_gradient,
            anchor_points=self.anchor_points,
            dpsi_pixelization=instance,
            preloads=self.preloads,
        )
        log_ev = fit.log_evidence
        # t1 = time.time()
        # print(f"log evidence: {log_ev}, time: {t1-t0}")
        # print(f"mesh factor: {instance.mesh.factor}, regularization coefficient: {instance.regularization.coefficient}, regularization scale: {instance.regularization.scale}")
        return log_ev