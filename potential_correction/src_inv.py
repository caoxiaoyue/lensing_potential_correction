import autolens as al
import numpy as np
import autofit as af
from potential_correction import util as pul
from potential_correction.visualize import show_fit_source_al
import os
import autolens.plot as aplt
from potential_correction.covariance_reg import CovarianceRegularization
from typing import Dict, List, Optional
import autoarray as aa


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