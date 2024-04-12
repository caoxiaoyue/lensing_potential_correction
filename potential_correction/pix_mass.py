import numpy as np
from scipy.spatial import Delaunay
from scipy.sparse import spmatrix
import autoarray as aa
from autogalaxy.profiles.mass.abstract.abstract import MassProfile
from potential_correction import util as pul
from typing import Optional

class InputDeflectionsMask(MassProfile):
    def __init__(
        self,
        deflections_y_slim: aa.type.Grid2DLike,
        deflections_x_slim: aa.type.Grid2DLike,
        image_plane_grid_slim: aa.type.Grid2DLike,
        mask: aa.type.Mask2D,
        Hy: Optional[spmatrix] = None,
        Hx: Optional[spmatrix] = None,
    ):
        """
        A pixelzied mass model is characterized by the potential/deflections on a set of pre-defined grids
        potential/deflections/convergence on arbitrary positions can be evaluated by the interpolation.

        Parameters
        ----------
        deflections_y : aa.Array2D
            The input array of the y components of the deflection angles.
        deflections_x : aa.Array2D
            The input array of the x components of the deflection angles.
        image_plane_grid
            The image-plane grid from which the deflection angles are defined.
        mask : aa.Mask2D
            The mask region for which the InputDeflectionsMask model are defined
        Hy : scipy.sparse.csr_matrix
            The sparse matrix representing the 1st differnetial operator along the y direction.
        Hx : scipy.sparse.csr_matrix
            The sparse matrix representing the 1st differnetial operator along the x direction.
        """
        super().__init__()

        self.deflections_y_slim = deflections_y_slim
        self.deflections_x_slim = deflections_x_slim
        self.image_plane_grid_slim = image_plane_grid_slim
        self.mask = mask
        self.Hy = Hy
        self.Hx = Hx

        self.construct_interpolator()


    def construct_interpolator(self):
        if self.Hy is None or self.Hy is None:
            self.Hy, self.Hx = pul.diff_1st_operator_numba_from(self.mask, dpix=self.image_plane_grid_slim.pixel_scale)

        self.convergence_slim = (self.Hy @ self.deflections_y_slim + self.Hx @ self.deflections_x_slim) * 0.5

        self.tri = Delaunay(np.fliplr(self.image_plane_grid_slim))
        self.interp_defl_y = pul.LinearNDInterpolatorExt(self.tri, self.deflections_y_slim)
        self.interp_defl_x = pul.LinearNDInterpolatorExt(self.tri, self.deflections_x_slim)
        self.interp_kappa = pul.LinearNDInterpolatorExt(self.tri, self.convergence_slim)


    @aa.grid_dec.grid_2d_to_structure
    def convergence_2d_from(self, grid: aa.type.Grid2DLike):
        return self.interp_kappa(grid[:, 1], grid[:, 0])


    @aa.grid_dec.grid_2d_to_structure
    def potential_2d_from(self, grid: aa.type.Grid2DLike):
        return np.zeros(shape=grid.shape[0])


    @aa.grid_dec.grid_2d_to_vector_yx
    @aa.grid_dec.grid_2d_to_structure
    def deflections_yx_2d_from(self, grid: aa.type.Grid2DLike):
        """
        Calculate the deflection angles at a given set of arc-second gridded coordinates.

        Parameters
        ----------
        grid
            The grid of (y,x) arc-second coordinates the deflection angles are computed on.

        """
        deflections_y = self.interp_defl_y(grid[:, 1], grid[:, 0])
        deflections_x = self.interp_defl_x(grid[:, 1], grid[:, 0])
        return np.stack((deflections_y, deflections_x), axis=-1) 
    

class InputPotentialMask(MassProfile):
    def __init__(
        self,
        lensing_potential_slim: aa.type.Grid2DLike,
        image_plane_grid_slim: aa.type.Grid2DLike,
        mask: aa.type.Mask2D,
        Hy: Optional[spmatrix] = None,
        Hx: Optional[spmatrix] = None,
        Hyy: Optional[spmatrix] = None,
        Hxx: Optional[spmatrix] = None,
    ):
        """
        A pixelzied mass model is characterized by the potential/deflections on a set of pre-defined grids
        potential/deflections/convergence on arbitrary positions can be evaluated by the interpolation.

        Parameters
        ----------
        lensing_potential_slim : aa.Array2D
            the 1d array of the lensing potential values for unmasked pixels.
        image_plane_grid
            The image-plane grid from which the deflection angles are defined.
        mask : aa.Mask2D
            The mask region for which the InputDeflectionsMask model are defined
        Hy : scipy.sparse.csr_matrix
            The sparse matrix representing the 1st differnetial operator along the y direction.
        Hx : scipy.sparse.csr_matrix
            The sparse matrix representing the 1st differnetial operator along the x direction.
        Hyy: scipy.sparse.csr_matrix
            The sparse matrix representing the 2nd differnetial operator along the y direction.
        Hxx: scipy.sparse.csr_matrix
            The sparse matrix representing the 2nd differnetial operator along the x direction.
        """
        super().__init__()

        self.lensing_potential_slim = lensing_potential_slim
        self.image_plane_grid_slim = image_plane_grid_slim
        self.mask = mask
        self.Hy = Hy
        self.Hx = Hx
        self.Hyy = Hyy
        self.Hxx = Hxx

        self.construct_interpolator()


    def construct_interpolator(self):
        if self.Hy is None or self.Hy is None:
            self.Hy, self.Hx = pul.diff_1st_operator_numba_from(self.mask, dpix=self.image_plane_grid_slim.pixel_scale)
        if self.Hyy is None or self.Hxx is None:
            self.Hyy, self.Hxx = pul.diff_2nd_operator_numba_from(self.mask, dpix=self.image_plane_grid_slim.pixel_scale)

        self.deflections_y_slim = self.Hy @ self.lensing_potential_slim
        self.deflections_x_slim = self.Hx @ self.lensing_potential_slim
        self.convergence_slim = (self.Hyy @ self.lensing_potential_slim + self.Hxx @ self.lensing_potential_slim) * 0.5

        self.tri = Delaunay(np.fliplr(self.image_plane_grid_slim))
        self.interp_psi = pul.LinearNDInterpolatorExt(self.tri, self.lensing_potential_slim)
        self.interp_defl_y = pul.LinearNDInterpolatorExt(self.tri, self.deflections_y_slim)
        self.interp_defl_x = pul.LinearNDInterpolatorExt(self.tri, self.deflections_x_slim)
        self.interp_kappa = pul.LinearNDInterpolatorExt(self.tri, self.convergence_slim)


    @aa.grid_dec.grid_2d_to_structure
    def convergence_2d_from(self, grid: aa.type.Grid2DLike):
        return self.interp_kappa(grid[:, 1], grid[:, 0])


    @aa.grid_dec.grid_2d_to_structure
    def potential_2d_from(self, grid: aa.type.Grid2DLike):
        return self.interp_psi(grid[:, 1], grid[:, 0])


    @aa.grid_dec.grid_2d_to_vector_yx
    @aa.grid_dec.grid_2d_to_structure
    def deflections_yx_2d_from(self, grid: aa.type.Grid2DLike):
        """
        Calculate the deflection angles at a given set of arc-second gridded coordinates.

        Parameters
        ----------
        grid
            The grid of (y,x) arc-second coordinates the deflection angles are computed on.

        """
        deflections_y = self.interp_defl_y(grid[:, 1], grid[:, 0])
        deflections_x = self.interp_defl_x(grid[:, 1], grid[:, 0])
        return np.stack((deflections_y, deflections_x), axis=-1) 