import autolens as al
import numpy as np
import autofit as af
from potential_correction import util as pul
from potential_correction.visualize import show_fit_source, show_fit_source_al
import os
import autolens.plot as aplt
from potential_correction.covariance_reg import CovarianceRegularization
from typing import Dict, List, Optional
import autoarray as aa


class FitSrcImaging:
    def __init__(
        self, 
        masked_imaging: al.Imaging,
        tracer: al.Tracer,
        preloads: dict = None,
    ):
        self.masked_imaging = masked_imaging #autolens masked imaging object
        self.tracer = tracer #autolens tracer object

        if preloads is not None:
            for key, value in preloads.items():
                setattr(self, key, value)

    
    def construct_source_pixelization(self):
        src_pix_ctr_im_plane = al.Grid2DSparse.from_grid_and_unmasked_2d_grid_shape(
            grid=self.masked_imaging.grid,
            unmasked_sparse_shape=self.tracer.galaxies[-1].pixelization.mesh.shape,
        )
        tracer_to_inversion = al.TracerToInversion(tracer=self.tracer, dataset=self.masked_imaging)
        src_pix_ctr_src_plane = tracer_to_inversion.traced_sparse_grid_pg_list[0][-1][0]
        return src_pix_ctr_im_plane, src_pix_ctr_src_plane


    def construct_mapper(self, src_pix_ctr_im_plane=None, src_pix_ctr_src_plane=None):
        if (src_pix_ctr_im_plane is None) or (src_pix_ctr_src_plane is None):
            src_pix_ctr_im_plane, src_pix_ctr_src_plane = self.construct_source_pixelization()

        traced_grid_pixelization = self.tracer.traced_grid_2d_list_from(
            grid=self.masked_imaging.grid_pixelization
        )[-1]

        relocated_grid_mesh= traced_grid_pixelization.relocated_grid_from(
            grid=traced_grid_pixelization
        )

        relocated_mesh_grid = traced_grid_pixelization.relocated_mesh_grid_from(
            mesh_grid=src_pix_ctr_src_plane
        )

        grid_src_mesh = al.Mesh2DDelaunay(
            values=relocated_mesh_grid,
            nearest_pixelization_index_for_slim_index=src_pix_ctr_im_plane.sparse_index_for_slim_index,
            uses_interpolation=True,
        )

        mapper_grids = al.MapperGrids(
            source_plane_data_grid=relocated_grid_mesh, #masked_dataset.grid_pixelization,
            source_plane_mesh_grid=grid_src_mesh,
            image_plane_mesh_grid=src_pix_ctr_im_plane,
        )

        mapper = al.Mapper(mapper_grids=mapper_grids, regularization=None)

        return mapper
    

    def regularization_matrix_from(self, mapper):
        regularization = self.tracer.galaxies[-1].pixelization.regularization
        if isinstance(regularization, CovarianceRegularization):
            regularization_matrix = regularization.regularization_matrix_from(
                points=mapper.mapper_grids.source_plane_mesh_grid
            )
        else:
            regularization_matrix = al.util.regularization.constant_regularization_matrix_from(
                coefficient=regularization.coefficient,
                neighbors=mapper.source_plane_mesh_grid.neighbors,
                neighbors_sizes=mapper.source_plane_mesh_grid.neighbors.sizes,
            )
        return regularization_matrix
    

    def mapping_matrix_from(self, mapper, bluring=True):
        mapping_matrix = al.util.mapper.mapping_matrix_from(
            pix_indexes_for_sub_slim_index=mapper.pix_indexes_for_sub_slim_index,
            pix_size_for_sub_slim_index=mapper.pix_sizes_for_sub_slim_index,  # unused for Voronoi
            pix_weights_for_sub_slim_index=mapper.pix_weights_for_sub_slim_index,  # unused for Voronoi
            pixels=mapper.pixels,
            total_mask_pixels=mapper.source_plane_data_grid.mask.pixels_in_mask,
            slim_index_for_sub_slim_index=mapper.slim_index_for_sub_slim_index,
            sub_fraction=mapper.source_plane_data_grid.mask.sub_fraction,
        )

        if bluring:
            blur_mapping_matrix = self.masked_imaging.convolver.convolve_mapping_matrix(
                mapping_matrix=mapping_matrix
            )
            return blur_mapping_matrix
        else:
            return mapping_matrix
    

    def data_vector_from(self, mapping_matrix):
        data_vector = al.util.inversion_imaging.data_vector_via_blurred_mapping_matrix_from(
            blurred_mapping_matrix=mapping_matrix,
            image=self.masked_imaging.data,
            noise_map=self.masked_imaging.noise_map,
        )
        return data_vector
    

    def curvature_matrix_from(self, mapping_matrix):
        curvature_matrix = al.util.inversion.curvature_matrix_via_mapping_matrix_from(
            mapping_matrix=mapping_matrix, noise_map=self.masked_imaging.noise_map
        )
        return curvature_matrix
    

    def construct_useful_matrices(self):
        if not hasattr(self, "mapper"):
            self.mapper = self.construct_mapper()
        if not hasattr(self, "map_mat"):
            self.map_mat = self.mapping_matrix_from(self.mapper)
        if not hasattr(self, "reg_mat"):
            self.reg_mat = self.regularization_matrix_from(self.mapper)
        if not hasattr(self, "d_vec"):
            self.d_vec = self.data_vector_from(self.map_mat)
        self.curve_reg_mat = self.curvature_matrix_from(self.map_mat) + self.reg_mat


    def solve_source(self, return_error=False):
        self.construct_useful_matrices()
        # reconstruction, _ =  nnls(self.curve_reg_mat, self.d_vec) #let us try nnsl
        reconstruction =  np.linalg.solve(self.curve_reg_mat, self.d_vec) #let us try nnsl
        if return_error:
            return reconstruction, np.linalg.inv(self.curve_reg_mat)
        else:
            return reconstruction


    @property
    def log_evidence(self):
        if not hasattr(self, "src_slim ") or not hasattr(self, "model_image_residual_slim"):
            self.src_slim = self.solve_source()
            self.model_image_slim = self.map_mat @ self.src_slim

        #noise normalization term
        self.noise_term= float(np.sum(np.log(2 * np.pi * self.masked_imaging.noise_map ** 2.0))) * (-0.5)

        #log det cuverd reg term
        sign, logval = np.linalg.slogdet(self.curve_reg_mat)
        if sign != 1:
            raise Exception(f"The curve reg matrix is not positive definite.")
        self.log_det_curve_reg_term = logval * (-0.5)
        # self.log_det_curve_reg_term = pul.log_det_mat(self.curve_reg_mat) * (-0.5)

        #log det regularization matrix term
        """
        sign, logval = np.linalg.slogdet(self.reg_mat)
        if sign != 1:
            np.save("reg_mat.npy", self.reg_mat)
            import pickle
            with open("tracer.pkl", "wb") as f:
                pickle.dump(self.tracer, f)    
            raise Exception(f"The regularization matrix is not positive definite.")
        self.log_det_reg_term = logval * 0.5
        """
        try:
            self.log_det_reg_term = pul.log_det_mat(self.reg_mat, sparse=True) * 0.5
        except:
            raise Exception(f"The regularization matrix is not positive definite.")

        #dpsi covariance term
        reg_cov_term = self.src_slim.T @ self.reg_mat @ self.src_slim
        self.reg_cov_term = float(reg_cov_term) * (-0.5)

        #chi2 term
        image_residual = self.masked_imaging.data - self.model_image_slim
        norm_image_residual = image_residual / self.masked_imaging.noise_map
        self.chi2_term = float(np.sum(norm_image_residual**2)) * (-0.5)

        #evidence
        return self.noise_term + self.log_det_curve_reg_term + self.log_det_reg_term + self.reg_cov_term + self.chi2_term
    

class SrcInvAnalysisCustom(af.Analysis):
    def __init__(
            self, 
            masked_imaging: al.Imaging,
            lens_galaxies: al.Galaxy,
            preloads: dict = None,
        ):
        self.masked_imaging = masked_imaging
        self.lens_galaxies = lens_galaxies
        self.preloads = preloads


    def log_likelihood_function(self, instance):
        # instance is a al.Galaxy object with pixelization and regularization parameters
        tracer = al.Tracer.from_galaxies(galaxies=[self.lens_galaxies, instance])
        fit = FitSrcImaging(
            masked_imaging=self.masked_imaging,
            tracer=tracer,
            preloads=self.preloads,
        )
        log_ev = fit.log_evidence
        return log_ev
    

    def visualize(self, paths: af.DirectoryPaths, instance, during_analysis=True):
        # instance is a al.Galaxy object with pixelization and regularization parameters
        tracer = al.Tracer.from_galaxies(galaxies=[self.lens_galaxies, instance])
        fit = FitSrcImaging(
            masked_imaging=self.masked_imaging,
            tracer=tracer,
            preloads=self.preloads,
        )
        print("the best log evidence is: ", fit.log_evidence)

        os.makedirs(paths.image_path, exist_ok=True)
        show_fit_source(fit=fit, output=f"{paths.image_path}/fit_source.png")


class SrcInvAnalysis(af.Analysis):
    def __init__(
            self, 
            masked_imaging: al.Imaging,
            lens_galaxies: al.Galaxy,
            adapt_image: Optional[al.Array2D] = None,
            settings_inversion: Optional[aa.SettingsInversion] = None,
            # preloads: dict = None,
        ):
        self.masked_imaging = masked_imaging
        self.lens_galaxies = lens_galaxies
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
        # self.preloads = preloads


    def log_likelihood_function(self, instance):
        # instance is a al.Galaxy object with pixelization and regularization parameters
        tracer = al.Tracer.from_galaxies(galaxies=[self.lens_galaxies, instance])
        
        if self.adapt_image is not None:
            adapt_images = al.AdaptImages(galaxy_image_dict={instance: self.adapt_image})
        else:
            adapt_images = None

        fit = al.FitImaging(
            dataset=self.masked_imaging, 
            tracer=tracer,
            adapt_images=adapt_images,
            settings_inversion=self.settings_inversion,
            # preloads=self.preloads,
        )
        log_ev = fit.log_evidence
        return log_ev
    

    def visualize(self, paths: af.DirectoryPaths, instance, during_analysis=True):
        # instance is a al.Galaxy object with pixelization and regularization parameters
        tracer = al.Tracer.from_galaxies(galaxies=[self.lens_galaxies, instance])
        
        if self.adapt_image is not None:
            adapt_images = al.AdaptImages(galaxy_image_dict={instance: self.adapt_image})
        else:
            adapt_images = None

        fit = al.FitImaging(
            dataset=self.masked_imaging, 
            tracer=tracer,
            adapt_images=adapt_images,
            settings_inversion=self.settings_inversion,
            # preloads=self.preloads,
        )
        print("the best log evidence is: ", fit.log_evidence)

        # include_2d = aplt.Include2D(
        #     mask=True, mapper_image_plane_mesh_grid=True, mapper_source_plane_mesh_grid=True
        # )
        # mat_plot = aplt.MatPlot2D(
        #     output=aplt.Output(
        #         filename=f"fit_subplot",
        #         path=paths.image_path,
        #         format="png",
        #         bbox_inches="tight",
        #     )
        # )
        # fit_plotter = aplt.FitImagingPlotter(fit=fit, mat_plot_2d=mat_plot, include_2d=include_2d)
        # fit_plotter.subplot_fit()

        os.makedirs(paths.image_path, exist_ok=True)
        show_fit_source_al(fit=fit, output=f"{paths.image_path}/fit_source.png")