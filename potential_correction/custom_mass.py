from autogalaxy.profiles.mass.abstract.abstract import MassProfile
from typing import Tuple
import autoarray as aa
import numpy as np
import numba
import copy
import powerbox as pbox
from potential_correction.pix_mass import InputPotentialMask
import autolens as al


def cart2polar(x, y, center_x=0, center_y=0):
    coord_shift_x = x - center_x
    coord_shift_y = y - center_y
    r = np.sqrt(coord_shift_x**2+coord_shift_y**2)
    phi = np.arctan2(coord_shift_y, coord_shift_x)
    return r, phi


@numba.njit(cache=False, parallel=False)
def nan_to_num(x, posinf=1e10, neginf=-1e10, nan=0.):
    if isinstance(x, float):
        return nan_to_num_single(x, posinf, neginf, nan)
    else:
        return nan_to_num_arr(x, posinf, neginf, nan)
        

@numba.njit(cache=False, parallel=False)
def nan_to_num_arr(x, posinf=1e10, neginf=-1e10, nan=0.):
    for i in range(len(x)):
        if np.isnan(x[i]):
            x[i] = nan
        if np.isinf(x[i]):
            if x[i] > 0:
                x[i] = posinf
            else:
                x[i] = neginf
    return x


@numba.njit(cache=False, parallel=False)
def nan_to_num_single(x, posinf=1e10, neginf=-1e10, nan=0.):
    if np.isnan(x):
        return nan
    elif np.isinf(x):
        if x > 0:
            return posinf
        else:
            return neginf
    else:
        return x
    

@numba.njit(cache=False, parallel=False)
def cal_deflections(
    x: np.ndarray,
    y: np.ndarray,
    b: float = 1.0, 
    t: float = 1.0, 
    q: float = 1.0, 
    PA_radian: float = 0.0, 
    xc: float = 0.0, 
    yc: float = 0.0,
):
    x_shift = x - xc
    y_shift = y - yc
    z_origin = x_shift + y_shift*1j
    z_rotated = np.exp(-1j*PA_radian) * z_origin
    alph = alpha_my_epl(z_rotated.real, z_rotated.imag, b, q, t) * np.exp(1j*PA_radian)
    return alph.real, alph.imag


@numba.njit(cache=False, parallel=False)
def alpha_my_epl(x, y, b, q, t):
    zz = x*q + 1j*y
    R = np.abs(zz)
    phi = np.angle(zz)
    Omega = omega(phi, t, q)
    alph = (2*b)/(1+q)*nan_to_num((b/R)**t*R/b)*Omega #see eq.22, 23 of https://arxiv.org/abs/1507.01819
    return alph


@numba.njit(cache=False, parallel=False, fastmath=True)
def omega(phi, t, q, niter_max=200, tol=1e-16):
    f = (1-q)/(1+q)
    omegas = np.zeros_like(phi, dtype=np.complex128)
    niter = min(niter_max, int(np.log(tol)/np.log(f))+2)  
    Omega = 1*np.exp(1j*phi)
    fact = -f*np.exp(2j*phi)
    for n in range(1, niter):
        omegas += Omega
        Omega *= (2*n-(2-t))/(2*n+(2-t)) * fact #see eq.25, 26, 27 of https://arxiv.org/abs/1507.01819
    omegas += Omega
    return omegas



class EPL(MassProfile):
    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        ell_comps: Tuple[float, float] = (0.0, 0.0),
        einstein_radius: float = 1.0,
        slope: float = 2.0,
    ):
        """
        The elliptical power-law under the so-called "intermediate axis convention"
        Under this convention, the einstein_radius is exactly equal to np.sqrt(S_crit/np.pi),
        where S_crit is the area enclosed by the tangential critical curve.

        Parameters
        ----------
        centre
            The (y,x) arc-second coordinates of the profile centre.
        ell_comps
            The first and second ellipticity components of the elliptical coordinate system.
        einstein_radius
            The arc-second Einstein radius.
        slope
            The density slope of the power-law (lower value -> shallower profile, higher value -> steeper profile).
        """
        super().__init__(centre=centre, ell_comps=ell_comps)
        self.einstein_radius = einstein_radius #the einstein radius in the intermediate axis convention
        self.slope = slope #the 3d density slope of the power-law


    @property
    def _b(self) -> float:
        return self.einstein_radius * np.sqrt(self.axis_ratio)


    @property
    def _t(self) -> float:
        return self.slope - 1.0
    

    @property
    def _PA_radian(self) -> float:
        return self.angle * np.pi / 180.0
    

    @aa.grid_dec.grid_2d_to_structure
    @aa.grid_dec.relocate_to_radial_minimum
    def convergence_2d_from(self, grid: aa.type.Grid2DLike):
        """
        Returns the two dimensional projected convergence on a grid of (y,x) arc-second coordinates.

        The `grid_2d_to_structure` decorator reshapes the ndarrays the convergence is outputted on. See
        *aa.grid_2d_to_structure* for a description of the output.

        Parameters
        ----------
        grid
            The grid of (y,x) arc-second coordinates the convergence is computed on.
        """
        x = grid[:, 1] - self.centre[1]
        y = grid[:, 0] - self.centre[0]
        z = x + 1j*y
        z_rot = np.exp(-1j*self._PA_radian) * z
        z_stretch = z_rot.real * self.axis_ratio + 1j*z_rot.imag
        R = np.abs(z_stretch)
        kappa = (2.0 - self._t) / 2.0 * (self._b / R) ** self._t
        
        return kappa 
    

    @aa.grid_dec.grid_2d_to_structure
    def potential_2d_from(self, grid: aa.type.Grid2DLike):
        alpha = self.deflections_yx_2d_from(grid)

        alpha_x = alpha[:, 1]
        alpha_y = alpha[:, 0]

        x = grid[:, 1] - self.centre[1]
        y = grid[:, 0] - self.centre[0]

        return (x * alpha_x + y * alpha_y) / (2 - self._t)


    @aa.grid_dec.grid_2d_to_vector_yx
    @aa.grid_dec.grid_2d_to_structure
    @aa.grid_dec.relocate_to_radial_minimum
    def deflections_yx_2d_from(self, grid: aa.type.Grid2DLike):
        """
        Calculate the deflection angles on a grid of (y,x) arc-second coordinates.

        Parameters
        ----------
        grid
            The grid of (y,x) arc-second coordinates the deflection angles are computed on.
        """
        deflection_x, deflection_y = cal_deflections(
            x=grid[:, 1],
            y=grid[:, 0],
            b=self._b,
            t=self._t,
            q=self.axis_ratio,
            PA_radian=self._PA_radian,
            xc=self.centre[1],
            yc=self.centre[0],
        )

        return np.vstack((deflection_y, deflection_x)).T


    @property
    def unit_mass(self):
        return "angular"


    def with_new_normalization(self, normalization):
        mass_profile = copy.copy(self)
        mass_profile.einstein_radius = normalization
        return mass_profile
    

class Multipole(MassProfile):
    def __init__(
        self,
        m: int = 2,
        a_m: float = 1.0,
        phi_m: float = 0.0,
        centre: Tuple[float, float] = (0.0, 0.0),
    ):
        """
        This class contains a multipole contribution (for 1 component with m>=2)
        This uses the same definitions as Xu et al.(2013) in Appendix B3 https://arxiv.org/pdf/1307.4220.pdf
        m : int, multipole order, m>=2
        a_m : float, multipole strength
        phi_m : float, multipole orientation in radian
        """
        self.m = m #order of multipole
        self.a_m = a_m #multipole strength
        self.phi_m = phi_m #multipole orientation in radian
        super().__init__(centre=centre)


    @aa.grid_dec.grid_2d_to_structure
    @aa.grid_dec.relocate_to_radial_minimum
    def convergence_2d_from(self, grid: aa.type.Grid2DLike):
        """
        Returns the two dimensional projected convergence on a grid of (y,x) arc-second coordinates.
        """
        x = grid[:, 1] 
        y = grid[:, 0] 
        r, phi = cart2polar(x, y, center_x=self.centre[1], center_y=self.centre[0])
        f_xx = 1./r * np.sin(phi)**2 * self.a_m *np.cos(self.m*(phi-self.phi_m))
        f_yy = 1./r * np.cos(phi)**2 * self.a_m *np.cos(self.m*(phi-self.phi_m))
        f_xy = -1./r * self.a_m * np.cos(phi) * np.sin(phi) * np.cos(self.m*(phi-self.phi_m))

        return 0.5*(f_xx+f_yy)
    

    @aa.grid_dec.grid_2d_to_structure
    def potential_2d_from(self, grid: aa.type.Grid2DLike):
        x = grid[:, 1] 
        y = grid[:, 0] 
        r, phi = cart2polar(x, y, center_x=self.centre[1], center_y=self.centre[0])
        f_ = r*self.a_m /(1-self.m**2) * np.cos(self.m*(phi-self.phi_m))

        return f_


    @aa.grid_dec.grid_2d_to_vector_yx
    @aa.grid_dec.grid_2d_to_structure
    @aa.grid_dec.relocate_to_radial_minimum
    def deflections_yx_2d_from(self, grid: aa.type.Grid2DLike):
        x = grid[:, 1] 
        y = grid[:, 0] 
        r, phi = cart2polar(x, y, center_x=self.centre[1], center_y=self.centre[0])
        m = self.m
        f_x = np.cos(phi)*self.a_m/(1-m**2) * np.cos(m*(phi-self.phi_m)) + np.sin(phi)*m*self.a_m/(1-m**2)*np.sin(m*(phi-self.phi_m))
        f_y = np.sin(phi)*self.a_m/(1-m**2) * np.cos(m*(phi-self.phi_m)) - np.cos(phi)*m*self.a_m/(1-m**2)*np.sin(m*(phi-self.phi_m))

        return np.vstack((f_y, f_x)).T


    @property
    def unit_mass(self):
        return "angular"


    def with_new_normalization(self, normalization):
        mass_profile = copy.copy(self)
        mass_profile.einstein_radius = normalization
        return mass_profile
    

class EPLBoxyDisky(MassProfile):
    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        ell_comps: Tuple[float, float] = (0.0, 0.0),
        einstein_radius: float = 1.0,
        slope: float = 2.0,
        a_m: float = 1.0,
    ):
        """
        EPL (Elliptical Power Law) mass profile combined with Multipole with m=4, so that it's either purely boxy or
        disky with EPL's axis and Multipole's axis aligned.

        Reference to the implementation: https://ui.adsabs.harvard.edu/abs/2022A%26A...659A.127V/abstract

        Parameters
        ----------
        centre: The (y,x) arc-second coordinates of the profile centre.
        ell_comps: The first and second ellipticity components of the elliptical coordinate system.
        einstein_radius: The arc-second Einstein radius.
        slope: The density slope of the power-law (lower value -> shallower profile, higher value -> steeper profile).
        a_m : float, multipole strength
        """
        self.centre = centre
        self.ell_comps = ell_comps
        self.einstein_radius = einstein_radius
        self.slope = slope
        self.a_m = a_m

    @property
    def epl(self):
        if not hasattr(self, '_epl'):
            self._epl = EPL(centre=self.centre, ell_comps=self.ell_comps, einstein_radius=self.einstein_radius, slope=self.slope)
        return self._epl


    @property
    def m4(self):
        if not hasattr(self, '_m4'):
            rescale_am = self.einstein_radius / np.sqrt(self.axis_ratio)
            self._m4 = Multipole(centre=self.centre, m=4, a_m=self.a_m*rescale_am, phi_m=self.angle*np.pi/180.0)
        return self._m4


    @aa.grid_dec.grid_2d_to_structure
    @aa.grid_dec.relocate_to_radial_minimum
    def convergence_2d_from(self, grid: aa.type.Grid2DLike):
        kappa_epl = self.epl.convergence_2d_from(grid)
        kappa_multipole = self.m4.convergence_2d_from(grid)
        return kappa_epl + kappa_multipole
    

    @aa.grid_dec.grid_2d_to_structure
    def potential_2d_from(self, grid: aa.type.Grid2DLike):
        psi_epl = self.epl.potential_2d_from(grid)
        psi_multipole = self.m4.potential_2d_from(grid)
        return psi_epl + psi_multipole


    @aa.grid_dec.grid_2d_to_vector_yx
    @aa.grid_dec.grid_2d_to_structure
    @aa.grid_dec.relocate_to_radial_minimum
    def deflections_yx_2d_from(self, grid: aa.type.Grid2DLike):
        alpha_epl = self.epl.deflections_yx_2d_from(grid)
        alpha_multipole = self.m4.deflections_yx_2d_from(grid)
        return alpha_epl + alpha_multipole


class GaussianRandomField(MassProfile):
    def __init__(
        self,
        n_pixels: int=100,
        pixel_scale: float=0.1,
        Ak: float=1.0,
        beta: float=1.0,
        seed: int = 1,
        mask: aa.type.Mask2D = None,
    ):
        """
        This class generate a mass profile defined by a lensing potential that is a Gaussian random field,
        whose power spectrum is given by the P(k) = Ak * k**(-beta) 
        """
        self.n_pixels = n_pixels
        self.pixel_scale = pixel_scale
        self.Ak = Ak
        self.beta = beta
        self.seed = seed
        self.mask = mask
        super().__init__(centre=(0.0, 0.0))


    def Pk(self, k):
        return self.Ak * k**(-self.beta)


    @property
    def box_length(self):
        return 2 * np.pi * self.n_pixels * self.pixel_scale
    

    @property
    def pb_box(self):
        if not hasattr(self, '_pb_box'):
            self._pb_box = pbox.PowerBox(N=self.n_pixels, dim=2, pk=self.Pk, boxlength=self.box_length, seed=self.seed)
        return self._pb_box
    

    @property
    def lens_potential_unmasked(self):
        if not hasattr(self, '_lens_potential_unmasked'):
            self._lens_potential_unmasked = self.pb_box.delta_x()
        return self._lens_potential_unmasked
    

    @property
    def mask_al(self):
        if not hasattr(self, '_mask_al'):
            self._mask_al = al.Mask2D(self.mask, pixel_scales=self.pixel_scale)
        return self._mask_al


    @property
    def grid(self):
        if not hasattr(self, '_grid'):
            self._grid = al.Grid2D.from_mask(mask=self.mask_al)
        return self._grid
    

    @property
    def lens_potential(self):
        if not hasattr(self, '_lens_potential'):
            self._lens_potential = self.lens_potential_unmasked[~self.mask_al]
        return self._lens_potential
    

    @property
    def pix_mass_profile(self):
        if not hasattr(self, '_pix_mass_profile'):
            self._pix_mass_profile = InputPotentialMask(
                lensing_potential_slim=self.lens_potential,
                image_plane_grid_slim=self.grid,
                mask=self.mask_al,
            )
        return self._pix_mass_profile


    def convergence_2d_from(self, grid: aa.type.Grid2DLike):
        """
        Returns the two dimensional projected convergence on a grid of (y,x) arc-second coordinates.
        """
        return self.pix_mass_profile.convergence_2d_from(grid)
    

    def potential_2d_from(self, grid: aa.type.Grid2DLike):
        return self.pix_mass_profile.potential_2d_from(grid)


    def deflections_yx_2d_from(self, grid: aa.type.Grid2DLike):
        return self.pix_mass_profile.deflections_yx_2d_from(grid)


    @property
    def unit_mass(self):
        return "angular"


    def with_new_normalization(self, normalization):
        mass_profile = copy.copy(self) #NOTE, not works now
        mass_profile.einstein_radius = normalization
        return mass_profile
    