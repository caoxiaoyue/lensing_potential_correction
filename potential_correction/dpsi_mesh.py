import numpy as np
import autolens as al 
from potential_correction import util as pul
from scipy.sparse import csr_matrix
from matplotlib import pyplot as plt


class RegularDpsiMesh:
    def __init__(self, factor: int = 1):
        self.factor = int(factor)

    def __eq__(self, other):
        return self.__dict__ == other.__dict__ and self.__class__ is other.__class__
    
    def __str__(self):
        return "\n".join(["{}: {}".format(k, v) for k, v in self.__dict__.items()])

    def __repr__(self):
        return "{}\n{}".format(self.__class__.__name__, str(self))
    


class PairRegularDpsiMesh:
    def __init__(self, mask, dpix_data, dpsi_factor=2):
        """
        This class represent the potential correction (Dpsi) grid,
        usually sparser than the native ccd image grid (or data grid).

        Parameters
        ----------
        mask: a bool array represents the data mask, which typically marks an annular-like region.
        dpix_data: the pixel size in arcsec for the native ccd image data.
        dpsi_factor: an integer number. for the dpsi grid, its resolution is dpsi_factor times
        coarser than the data grid
        """
        self.mask_data = mask
        self.mask_data, self.diff_types_data = pul.iterative_clean_mask(self.mask_data)
        self.dpix_data = dpix_data
        grid_data = al.Grid2D.uniform(shape_native=self.mask_data.shape, pixel_scales=self.dpix_data)
        self.xgrid_data = np.array(grid_data.native[:,:,1])
        self.ygrid_data = np.array(grid_data.native[:,:,0])
        limit = (self.mask_data.shape[0] * self.dpix_data)*0.5
        self.data_bound = [-limit, limit, -limit, limit]

        self.dpsi_factor = dpsi_factor
        if self.mask_data.shape[0] % self.dpsi_factor != 0:
            raise Exception("the mask.shape[0] must be divisible by dpsi_factor")
        self.shape_2d_dpsi = (int(self.mask_data.shape[0]/self.dpsi_factor), int(self.mask_data.shape[1]/self.dpsi_factor))
        self.dpix_dpsi = float(2.0*limit/self.shape_2d_dpsi[0])
        grid_dpsi = al.Grid2D.uniform(shape_native=self.shape_2d_dpsi, pixel_scales=self.dpix_dpsi)
        self.xgrid_dpsi = np.array(grid_dpsi.native[:,:,1])
        self.ygrid_dpsi = np.array(grid_dpsi.native[:,:,0])
        self.mask_dpsi = pul.dpsi_mask_from(self.mask_data, self.dpsi_factor)
        self.mask_dpsi, self.diff_types_dpsi = pul.iterative_clean_mask(self.mask_dpsi)

        self.grid_1d_from_mask()

        self.get_itp_box_ctr()
        self.get_dpsi2data_mapping() #TODO, further increase the speed of this func

        self.get_gradient_operator_data()
        self.get_gradient_operator_dpsi()
        self.get_hamiltonian_operator_data()
        self.get_hamiltonian_operator_dpsi()


    def grid_1d_from_mask(self):
        self.idx_1d_data = np.where((~self.mask_data).flatten())[0]
        self.xgrid_data_1d = self.xgrid_data.flatten()[self.idx_1d_data]
        self.ygrid_data_1d = self.ygrid_data.flatten()[self.idx_1d_data]

        self.idx_1d_dpsi = np.nonzero((~self.mask_dpsi).flatten())[0]
        self.xgrid_dpsi_1d = self.xgrid_dpsi[~self.mask_dpsi]
        self.ygrid_dpsi_1d = self.ygrid_dpsi[~self.mask_dpsi]           


    def get_itp_box_ctr(self):
        ctr_itp_box = al.Grid2D.uniform(
            shape_native=(self.shape_2d_dpsi[0]-1,self.shape_2d_dpsi[1]-1), 
            pixel_scales=self.dpix_dpsi, 
        )
        self.xc_itp_box = np.array(ctr_itp_box.native[:,:,1]) #2d sparse box center x-grid
        self.yc_itp_box = np.array(ctr_itp_box.native[:,:,0]) #2d sparse box center y-grid

        self.mask_itp_box = pul.itp_box_mask_from(self.mask_dpsi) 
        self.xc_itp_box_1d = self.xc_itp_box[~self.mask_itp_box]
        self.yc_itp_box_1d = self.yc_itp_box[~self.mask_itp_box]

        if len(self.xc_itp_box_1d) == 0:
            raise Exception("The dpsi grid is too sparse, Try decreasing the dpsi_factor to smaller values")


    def get_dpsi2data_mapping(self):
        """
        This function mapping a unmasked vector defined on coarser dpsi grid (shape: [n_unmasked_dpsi_pixels,]), 
        to a new unmasked vector defined on finner data grid (shape: [n_unmasked_data_pixels,]).

        return a sparse matrix, with a shape of [n_unmasked_data_pixels, n_unmasked_dpsi_pixels]
        """
        rows_itp_mat, cols_itp_mat, data_itp_mat = pul.dpsi2data_itp_mat_from(
            self.mask_itp_box,
            self.xc_itp_box,
            self.yc_itp_box,
            self.xgrid_data_1d,
            self.ygrid_data_1d,
            self.xgrid_dpsi,
            self.ygrid_dpsi,
            self.mask_dpsi
        )
        self.itp_mat = csr_matrix((data_itp_mat, (rows_itp_mat, cols_itp_mat)), shape=(len(self.xgrid_data_1d), len(self.xgrid_dpsi_1d)))


    def get_gradient_operator_data(self):
        self.Hy_data, self.Hx_data = pul.diff_1st_operator_numba_from(self.mask_data, self.dpix_data)


    def get_gradient_operator_dpsi(self):
        self.Hy_dpsi, self.Hx_dpsi = pul.diff_1st_operator_numba_from(self.mask_dpsi, self.dpix_dpsi)


    def get_hamiltonian_operator_data(self):
        self.Hyy_data, self.Hxx_data = pul.diff_2nd_operator_numba_from(self.mask_data, self.dpix_data)
        self.hamiltonian_data = self.Hxx_data + self.Hyy_data


    def get_hamiltonian_operator_dpsi(self):
        self.Hyy_dpsi, self.Hxx_dpsi = pul.diff_2nd_operator_numba_from(self.mask_dpsi, self.dpix_dpsi)
        self.hamiltonian_dpsi = self.Hxx_dpsi + self.Hyy_dpsi


    def show_grid(self, output_file='grid.png'):
        plt.figure(figsize=(20,20))
        plt.plot(self.xgrid_data.flatten(), self.ygrid_data.flatten(), '*', color='black', label="data")
        plt.plot(self.xgrid_dpsi.flatten(), self.ygrid_dpsi.flatten(), '*', color='red', label="dpsi")
        plt.plot(self.xgrid_data_1d, self.ygrid_data_1d, 'o', color='black', label="data-unmask")
        plt.plot(self.xgrid_dpsi_1d, self.ygrid_dpsi_1d, 'p', color='red', label="dpsi-unmask")
        plt.plot(self.xc_itp_box.flatten(), self.yc_itp_box.flatten(), '+', color='blue', label="dpsi-box")
        plt.plot(self.xc_itp_box_1d, self.yc_itp_box_1d, '+', color='red', label="dpsi-box-unmask")
        plt.legend()
        plt.savefig(output_file, bbox_inches='tight')
        plt.close()