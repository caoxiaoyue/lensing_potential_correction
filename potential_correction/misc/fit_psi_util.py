# %%
import autolens as al
import numpy as np
import autofit as af
import os
from potential_correction.visualize import imshow_masked_data
from matplotlib import pyplot as plt


class PsiCompoundModel(object):
    def __init__(self, dpsi_model, constant):
        self.psi_model = dpsi_model
        self.constant = constant
    
    def potential_2d_from(self, grid):
        return self.psi_model.potential_2d_from(grid) + self.constant


class FitMaskedPsi(object):
    def __init__(
        self, 
        unmasked_values=None, 
        noise_1d=None,
        noise_cov_mat=None,
        mask=None,
        dpix=0.05,
        nsub=4,
        psi_model: PsiCompoundModel=None,
        use_cov_mat=False,
        mask_for_unmasked_values=None,
    ):
        """
        class used to fit the masked map
        unmasked_values: 1d array of values for the unmasked pixels; shape: (n_unmasked_pix,)
        noise_cov_mat: noise covariance matrix for the unmasked pixels; shape: (n_unmasked_pix, n_unmasked_pix)
        mask: 2d boolean array indicating the masked pixels; shape: (ny_image, nx_image)
        dpix: pixel size in arcsec
        """
        self.unmasked_values = unmasked_values
        self.mask = mask
        self.dpix = dpix
        self.nsub = nsub
        self.psi_model = psi_model

        self.mask = al.Mask2D(mask=self.mask, pixel_scales=self.dpix, sub_size=self.nsub)
        self.grid_unmasked = al.Grid2D.uniform(shape_native=(self.mask.shape), pixel_scales=self.dpix, sub_size=self.nsub)
        self.grid_masked = al.Grid2D(values= self.grid_unmasked.native, mask=self.mask)

        self.use_cov_mat = use_cov_mat
        if self.use_cov_mat:
            self.noise_cov_mat = noise_cov_mat
            self.inverse_noise_cov_mat = np.linalg.inv(self.noise_cov_mat)
            self.noise_1d = np.sqrt(np.diag(self.noise_cov_mat))
        else:
            self.noise_1d = noise_1d

        if mask_for_unmasked_values is None:
            self.mask_for_unmasked_values = np.ones_like(self.unmasked_values, dtype=float)
        else:
            self.mask_for_unmasked_values = mask_for_unmasked_values.astype(float)


    @property
    def n_unmasked_pix(self):
        return len(self.unmasked_values)


    def forward_model(self):
        return self.psi_model.potential_2d_from(grid=self.grid_masked).binned.slim


    @property
    def log_likelihood(self):
        # https://en.wikipedia.org/wiki/Multivariate_normal_distribution
        model_1d = self.forward_model()
        residual_1d = (self.unmasked_values - model_1d)*self.mask_for_unmasked_values
        bool_arr = self.mask_for_unmasked_values.astype(bool)

        if self.use_cov_mat:
            chi_squared_1d = residual_1d.T @ self.inverse_noise_cov_mat @ residual_1d
            inverse_noise_cov_mat = self.inverse_noise_cov_mat[bool_arr, :][:, bool_arr]
            n_unmasked_pix = np.sum(bool_arr)
            sign, logval = np.linalg.slogdet(inverse_noise_cov_mat)
            if sign != 1:
                raise Exception(f"The inverse noise covariance matrix is not positive definite.")
            norm_term = -0.5*n_unmasked_pix* np.log(2*np.pi) + 0.5*logval
            # sign, logval = np.linalg.slogdet(self.inverse_noise_cov_mat)
            # if sign != 1:
            #     raise Exception(f"The inverse noise covariance matrix is not positive definite.")
            # norm_term = -0.5*self.n_unmasked_pix* np.log(2*np.pi) + 0.5*logval
        else:
            chi_squared_1d = np.sum(residual_1d ** 2.0 / self.noise_1d ** 2.0)
            noise_1d = self.noise_1d[bool_arr]
            norm_term = float(np.sum(np.log(2 * np.pi * noise_1d ** 2.0))) * (-0.5)

        return -0.5*chi_squared_1d + norm_term


class MaskedPsiAnalysis(af.Analysis):
    def __init__(
            self, 
            unmasked_values=None, 
            noise_1d=None,
            noise_cov_mat=None,
            mask=None,
            dpix=0.05,
            nsub=4,
            use_cov_mat=False,
            mask_for_unmasked_values=None,
        ):
        self.unmasked_values = unmasked_values
        self.noise_1d = noise_1d
        self.noise_cov_mat = noise_cov_mat
        self.mask = mask
        self.dpix = dpix
        self.nsub = nsub
        self.use_cov_mat = use_cov_mat
        self.mask_for_unmasked_values = mask_for_unmasked_values


    def log_likelihood_function(self, instance):
        #instance: an instance of the PsiCompoundModel class
        fit = FitMaskedPsi(
            unmasked_values=self.unmasked_values,
            noise_1d=self.noise_1d,
            noise_cov_mat=self.noise_cov_mat,
            mask=self.mask,
            dpix=self.dpix,
            nsub=self.nsub,
            psi_model=instance,
            use_cov_mat=self.use_cov_mat,
            mask_for_unmasked_values=self.mask_for_unmasked_values,
        )
        return fit.log_likelihood


    def visualize(self, paths: af.DirectoryPaths, instance, during_analysis=True):
        fit = FitMaskedPsi(
            unmasked_values=self.unmasked_values,
            noise_1d=self.noise_1d,
            noise_cov_mat=self.noise_cov_mat,
            mask=self.mask,
            dpix=self.dpix,
            nsub=self.nsub,
            psi_model=instance,
            use_cov_mat=self.use_cov_mat,
            mask_for_unmasked_values=self.mask_for_unmasked_values,
        )
        fit.log_likelihood
        show_psi_fit(fit=fit, output=f"{paths.image_path}/fit_dpsi.png")


def show_psi_fit(fit: FitMaskedPsi, output: str):
    os.makedirs(os.path.dirname(output), exist_ok=True)

    fig = plt.figure(figsize=(12, 4))
    plt.subplot(131)
    ax = plt.gca()
    plt.title("Data")
    imshow_masked_data(fit.unmasked_values, fit.mask, dpix=fit.dpix, cmap="jet", ax=ax)
    plt.subplot(132)
    ax = plt.gca()
    plt.title("Model")
    imshow_masked_data(fit.forward_model(), fit.mask, dpix=fit.dpix, cmap="jet", ax=ax)
    plt.subplot(133)
    ax = plt.gca()
    plt.title("Residual")
    imshow_masked_data(fit.unmasked_values - fit.forward_model(), fit.mask, dpix=fit.dpix, cmap="jet", ax=ax)
    plt.tight_layout()
    fig.savefig(output, bbox_inches='tight')
    plt.close(fig)


#%%
if __name__ == "__main__":
    # generate masked data
    dpix = 0.05
    nsub = 1
    grid = al.Grid2D.uniform(shape_native=(200, 200), pixel_scales=dpix, sub_size=nsub)
    mask = al.Mask2D.circular_annular(shape_native=(200, 200), pixel_scales=dpix, sub_size=nsub, inner_radius=1.0, outer_radius=2.0)
    masked_grid = al.Grid2D(values=grid.native, mask=mask)
    input_psi_model = al.mp.PowerLaw(centre=(0.0, 0.0), ell_comps=(0.05, 0.05), einstein_radius=1.5, slope=2.0)
    input_psi_model = PsiCompoundModel(dpsi_model=input_psi_model, constant=2.0)
    masked_psi_data = input_psi_model.potential_2d_from(masked_grid).binned.slim
    noise_cov_mat = np.diag(np.ones(len(masked_psi_data))) * 0.01

    plt.figure()
    ax = plt.gca()
    imshow_masked_data(masked_psi_data, mask, dpix=dpix, ax=ax)
    plt.show()


    #define the model
    model = af.Model(
        PsiCompoundModel,
        dpsi_model=af.Model(al.mp.PowerLaw),
        constant=af.UniformPrior(lower_limit=-20.0, upper_limit=20.0)
    )
    instance = model.instance_from_prior_medians()

    fit = FitMaskedPsi(
        unmasked_values=masked_psi_data,
        noise_cov_mat=noise_cov_mat,
        mask=mask,
        dpix=dpix,
        nsub=nsub,
        psi_model=instance,
    )
    fit.log_likelihood


    #%%

    #define the analysis
    analysis = MaskedPsiAnalysis(
        unmasked_values=masked_psi_data,
        noise_cov_mat=noise_cov_mat,
        mask=mask,
        dpix=dpix,
        nsub=nsub,
    )

    #search
    search = af.Nautilus(
        path_prefix=None,
        name="test_fit",
        unique_tag=None,
        n_live=300,
        number_of_cores=128,
    )

    result = search.fit(model=model, analysis=analysis)


# %%
