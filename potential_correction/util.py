import numpy as np
from scipy.interpolate import LinearNDInterpolator as linterp
from scipy.interpolate import NearestNDInterpolator as nearest
import numba
from scipy.sparse import lil_matrix, csr_matrix, csc_matrix
import scipy.special as sc
import math
from scipy.sparse.linalg import splu


@numba.njit(cache=False, parallel=False)
def clean_mask(mask):
    in_mask = np.copy(mask)
    in_mask = in_mask.astype('bool')
    out_mask = np.ones_like(in_mask, dtype='bool')
    n1, n2 = in_mask.shape
    diff_types = np.full((n1, n2, 2), -1, dtype='int')

    for i in range(n1): 
        for j in range(n2):
            if not in_mask[i,j]:
                if_remove_y = True
                if_remove_x = True

                if i == 0:
                    if ~in_mask[i+1,j] and ~in_mask[i+2,j]:
                        if_remove_y = False
                        diff_types[i, j, 0] = 2
                elif i == 1:
                    if ~in_mask[i-1,j] and ~in_mask[i+1,j]:
                        if_remove_y = False
                        diff_types[i, j, 0] = 1
                    elif ~in_mask[i+1,j] and ~in_mask[i+2,j]:
                        if_remove_y = False
                        diff_types[i, j, 0] = 2
                elif i == n1-1:
                    if ~in_mask[i-1,j] and ~in_mask[i-2,j]:
                        if_remove_y = False
                        diff_types[i, j, 0] = 0
                elif i == n1-2:
                    if ~in_mask[i-1,j] and ~in_mask[i+1,j]:
                        if_remove_y = False
                        diff_types[i, j, 0] = 1
                    elif ~in_mask[i-1,j] and ~in_mask[i-2,j]:
                        if_remove_y = False
                        diff_types[i, j, 0] = 0
                else: 
                    if ~in_mask[i-1,j] and ~in_mask[i+1,j]:
                        if_remove_y = False
                        diff_types[i, j, 0] = 1
                    elif ~in_mask[i-1,j] and ~in_mask[i-2,j]:
                        if_remove_y = False
                        diff_types[i, j, 0] = 0
                    elif ~in_mask[i+1,j] and ~in_mask[i+2,j]:
                        if_remove_y = False
                        diff_types[i, j, 0] = 2

                if j == 0:
                    if ~in_mask[i,j+1] and ~in_mask[i,j+2]:
                        if_remove_x = False
                        diff_types[i, j, 1] = 2
                elif j == 1:
                    if ~in_mask[i,j-1] and ~in_mask[i,j+1]:
                        if_remove_x = False
                        diff_types[i, j, 1] = 1
                    elif ~in_mask[i,j+1] and ~in_mask[i,j+2]:
                        if_remove_x = False
                        diff_types[i, j, 1] = 2
                elif j == n2-1:
                    if ~in_mask[i,j-1] and ~in_mask[i,j-2]:
                        if_remove_x = False
                        diff_types[i, j, 1] = 0
                elif j == n2-2:
                    if ~in_mask[i,j-1] and ~in_mask[i,j+1]:
                        if_remove_x = False
                        diff_types[i, j, 1] = 1
                    elif ~in_mask[i,j-1] and ~in_mask[i,j-2]:
                        if_remove_x = False
                        diff_types[i, j, 1] = 0
                else:   
                    if ~in_mask[i,j-1] and ~in_mask[i,j+1]:
                        if_remove_x = False
                        diff_types[i, j, 1] = 1
                    elif ~in_mask[i,j-1] and ~in_mask[i,j-2]:
                        if_remove_x = False
                        diff_types[i, j, 1] = 0
                    elif ~in_mask[i,j+1] and ~in_mask[i,j+2]:
                        if_remove_x = False
                        diff_types[i, j, 1] = 2

                if not (if_remove_y or if_remove_x):
                    out_mask[i,j] = False

    return out_mask, diff_types


def iterative_clean_mask(mask, max_iter=50):
    niter = 0
    old_mask = np.copy(mask)
    clean_success = False

    for i in range(max_iter):
        new_mask, diff_types = clean_mask(old_mask)
        if (new_mask == old_mask).all():
            clean_success = True
            break
        old_mask = new_mask 

    if not clean_success:
        raise Exception(f"The mask are not fully cleaned after {max_iter} iterations")
    
    return new_mask, diff_types


class LinearNDInterpolatorExt(object):
    # https://stackoverflow.com/questions/20516762/extrapolate-with-linearndinterpolator
    # use nearest neighbour interpolation to replace Linear interpolation, to avoid NaN
    def __init__(self, points, values):
        self.funcinterp = linterp(points, values)
        self.funcnearest = nearest(points, values)
    
    def __call__(self, *args):
        z = self.funcinterp(*args)
        chk = np.isnan(z)
        if chk.any():
            return np.where(chk, self.funcnearest(*args), z)
        else:
            return z


def diff_1st_operator_from(mask, dpix=1.0):
    """
    Receive a mask, use it to generate the 1st differential operator matrix Hx and Hy.
    Hx (Hy) has a shape of [n_unmasked_pixels, n_unmasked_pixels],
    when it act on the unmasked data, generating the 1st x/y-derivative of the unmasked data.

    dpix: pixel size in unit of arcsec.

    This func is easy to read, but slower. Use func diff_1st_operator_numba_from for better speed
    """
    #check is mask is cleaned, get differential pixel type 
    new_mask, diff_types = iterative_clean_mask(mask)
    if not (new_mask == mask).all():
        raise Exception("the mask has not been fully cleaned!")

    unmask = ~mask
    i_indices_unmasked, j_indices_unmasked = np.where(unmask)
    n_unmasked_pixels = len(i_indices_unmasked) 
    Hy = lil_matrix((n_unmasked_pixels, n_unmasked_pixels)) #y-direction gradient operator matrix
    Hx = lil_matrix((n_unmasked_pixels, n_unmasked_pixels)) #x-direction gradient operator matrix
    step_y = -1.0*dpix #the minus sign is due to the y-coordinate decrease the pixel_size as index i along axis-0 increase 1.
    step_x = 1.0*dpix #no minus, becasue the x-coordinate increase as index j along axis-1 increase.

    index_dict = {}
    for count in range(n_unmasked_pixels):
        i, j = i_indices_unmasked[count], j_indices_unmasked[count]
        index_dict[(i,j)] = count

    for count in range(n_unmasked_pixels):
        i, j = i_indices_unmasked[count], j_indices_unmasked[count]
        
        #------check y-direction
        if diff_types[i,j,0] == 0:
            Hy[count,index_dict[(i-1,j)]] = -1.0/step_y
            Hy[count,index_dict[(i,j)]] = 1/step_y
        elif diff_types[i,j,0] == 1:
            Hy[count,index_dict[(i-1,j)]] = -1.0/(2*step_y)
            Hy[count,index_dict[(i+1,j)]] = 1.0/(2*step_y)
        elif diff_types[i,j,0] == 2:
            Hy[count,index_dict[(i,j)]] = -1.0/step_y
            Hy[count,index_dict[(i+1,j)]] = 1.0/step_y

        #------check x-direction
        if diff_types[i,j,1] == 0:
            Hx[count,index_dict[(i,j-1)]] = -1.0/step_x
            Hx[count,index_dict[(i,j)]] = 1/step_x
        elif diff_types[i,j,1] == 1:
            Hx[count,index_dict[(i,j-1)]] = -1.0/(2*step_x)
            Hx[count,index_dict[(i,j+1)]] = 1.0/(2*step_x)
        elif diff_types[i,j,1] == 2:
            Hx[count,index_dict[(i,j)]] = -1.0/step_x
            Hx[count,index_dict[(i,j+1)]] = 1.0/step_x

    Hy = Hy.tocsr()
    Hx = Hx.tocsr()

    return Hy, Hx


def diff_1st_operator_numba_from(mask, dpix=1.0):
    new_mask, diff_types = iterative_clean_mask(mask)
    if not (new_mask == mask).all():
        raise Exception("the mask has not been fully cleaned!")
    rows_hx, cols_hx, data_hx, rows_hy, cols_hy, data_hy = diff_1st_operator_numba_func(mask, diff_types, dpix=dpix)

    n_unmasked = np.count_nonzero(~mask)
    Hx = csr_matrix((data_hx, (rows_hx, cols_hx)), shape=(n_unmasked, n_unmasked))
    Hy = csr_matrix((data_hy, (rows_hy, cols_hy)), shape=(n_unmasked, n_unmasked))
    return Hy, Hx
    

@numba.njit(cache=False, parallel=False)
def diff_1st_operator_numba_func(mask, diff_types, dpix=1.0):
    """
    Receive a mask, use it to generate the 1st differential operator matrix Hx and Hy.
    Hx (Hy) has a shape of [n_unmasked_pixels, n_unmasked_pixels],
    when it act on the unmasked data, generating the 1st x/y-derivative of the unmasked data.

    dpix: pixel size in unit of arcsec.
    """
    unmask = ~mask
    i_indices_unmasked, j_indices_unmasked = np.where(unmask)
    n_unmasked_pixels = len(i_indices_unmasked) 

    rows_hx = np.full(n_unmasked_pixels*2, -1, dtype=np.int64)
    cols_hx = np.full(n_unmasked_pixels*2, -1, dtype=np.int64)
    data_hx = np.full(n_unmasked_pixels*2, 0.0, dtype='float')
    rows_hy = np.full(n_unmasked_pixels*2, -1, dtype=np.int64)
    cols_hy = np.full(n_unmasked_pixels*2, -1, dtype=np.int64)
    data_hy = np.full(n_unmasked_pixels*2, 0.0, dtype='float')

    step_y = -1.0*dpix #the minus sign is due to the y-coordinate decrease the pixel_size as index i along axis-0 increase 1.
    step_x = 1.0*dpix #no minus, becasue the x-coordinate increase as index j along axis-1 increase.

    index_dict = {}
    for count in range(n_unmasked_pixels):
        i, j = i_indices_unmasked[count], j_indices_unmasked[count]
        index_dict[(i,j)] = count

    count_sparse_hy = 0
    count_sparse_hx = 0
    for count in range(n_unmasked_pixels):
        i, j = i_indices_unmasked[count], j_indices_unmasked[count]
        
        #------check y-direction
        if diff_types[i,j,0] == 0:
            rows_hy[count_sparse_hy] = count
            cols_hy[count_sparse_hy] = index_dict[(i-1,j)]
            data_hy[count_sparse_hy] = -1.0/step_y
            count_sparse_hy += 1
            rows_hy[count_sparse_hy] = count
            cols_hy[count_sparse_hy] = index_dict[(i,j)]
            data_hy[count_sparse_hy] = 1/step_y
            count_sparse_hy += 1
        elif diff_types[i,j,0] == 1:
            rows_hy[count_sparse_hy] = count
            cols_hy[count_sparse_hy] = index_dict[(i-1,j)]
            data_hy[count_sparse_hy] = -1.0/(2*step_y)
            count_sparse_hy += 1
            rows_hy[count_sparse_hy] = count
            cols_hy[count_sparse_hy] = index_dict[(i+1,j)]
            data_hy[count_sparse_hy] = 1.0/(2*step_y)
            count_sparse_hy += 1
        elif diff_types[i,j,0] == 2:
            rows_hy[count_sparse_hy] = count
            cols_hy[count_sparse_hy] = index_dict[(i,j)]
            data_hy[count_sparse_hy] = -1.0/step_y
            count_sparse_hy += 1
            rows_hy[count_sparse_hy] = count
            cols_hy[count_sparse_hy] = index_dict[(i+1,j)]
            data_hy[count_sparse_hy] = 1.0/step_y
            count_sparse_hy += 1

        #------check x-direction
        if diff_types[i,j,1] == 0:
            rows_hx[count_sparse_hx] = count
            cols_hx[count_sparse_hx] = index_dict[(i,j-1)]
            data_hx[count_sparse_hx] = -1.0/step_x
            count_sparse_hx += 1
            rows_hx[count_sparse_hx] = count
            cols_hx[count_sparse_hx] = index_dict[(i,j)]
            data_hx[count_sparse_hx] = 1.0/step_x
            count_sparse_hx += 1
        elif diff_types[i,j,1] == 1:
            rows_hx[count_sparse_hx] = count
            cols_hx[count_sparse_hx] = index_dict[(i,j-1)]
            data_hx[count_sparse_hx] = -1.0/(2*step_x)
            count_sparse_hx += 1
            rows_hx[count_sparse_hx] = count
            cols_hx[count_sparse_hx] = index_dict[(i,j+1)]
            data_hx[count_sparse_hx] = 1.0/(2*step_x)
            count_sparse_hx += 1
        elif diff_types[i,j,1] == 2:
            rows_hx[count_sparse_hx] = count
            cols_hx[count_sparse_hx] = index_dict[(i,j)]
            data_hx[count_sparse_hx] = -1.0/step_x
            count_sparse_hx += 1
            rows_hx[count_sparse_hx] = count
            cols_hx[count_sparse_hx] = index_dict[(i,j+1)]
            data_hx[count_sparse_hx] = 1.0/step_x
            count_sparse_hx += 1

    return rows_hx, cols_hx, data_hx, rows_hy, cols_hy, data_hy


def diff_2nd_operator_numba_from(mask, dpix=1.0):
    new_mask, diff_types = iterative_clean_mask(mask)
    if not (new_mask == mask).all():
        raise Exception("the mask has not been fully cleaned!")
    rows_hxx, cols_hxx, data_hxx, rows_hyy, cols_hyy, data_hyy = diff_2nd_operator_numba_func(mask, diff_types, dpix=dpix)

    n_unmasked = np.count_nonzero(~mask)
    Hxx = csr_matrix((data_hxx, (rows_hxx, cols_hxx)), shape=(n_unmasked, n_unmasked))
    Hyy = csr_matrix((data_hyy, (rows_hyy, cols_hyy)), shape=(n_unmasked, n_unmasked))
    return Hyy, Hxx


@numba.njit(cache=False, parallel=False)
def diff_2nd_operator_numba_func(mask, diff_types, dpix=1.0):
    """
    Receive a mask, use it to generate the 1st differential operator matrix Hx and Hy.
    Hx (Hy) has a shape of [n_unmasked_pixels, n_unmasked_pixels],
    when it act on the unmasked data, generating the 1st x/y-derivative of the unmasked data.

    dpix: pixel size in unit of arcsec.
    """
    unmask = ~mask
    i_indices_unmasked, j_indices_unmasked = np.where(unmask)
    n_unmasked_pixels = len(i_indices_unmasked) 

    rows_hxx = np.full(n_unmasked_pixels*3, -1, dtype=np.int64)
    cols_hxx = np.full(n_unmasked_pixels*3, -1, dtype=np.int64)
    data_hxx = np.full(n_unmasked_pixels*3, 0.0, dtype='float')
    rows_hyy = np.full(n_unmasked_pixels*3, -1, dtype=np.int64)
    cols_hyy = np.full(n_unmasked_pixels*3, -1, dtype=np.int64)
    data_hyy = np.full(n_unmasked_pixels*3, 0.0, dtype='float')

    step_y = -1.0*dpix #the minus sign is due to the y-coordinate decrease the pixel_size as index i along axis-0 increase 1.
    step_x = 1.0*dpix #no minus, becasue the x-coordinate increase as index j along axis-1 increase.

    index_dict = {}
    for count in range(n_unmasked_pixels):
        i, j = i_indices_unmasked[count], j_indices_unmasked[count]
        index_dict[(i,j)] = count

    count_sparse_hyy = 0
    count_sparse_hxx = 0
    for count in range(n_unmasked_pixels):
        i, j = i_indices_unmasked[count], j_indices_unmasked[count]
        
        #------check y-direction
        if diff_types[i,j,0] == 0:
            rows_hyy[count_sparse_hyy] = count
            cols_hyy[count_sparse_hyy] = index_dict[(i-2,j)]
            data_hyy[count_sparse_hyy] = 1.0/step_y**2
            count_sparse_hyy += 1
            rows_hyy[count_sparse_hyy] = count
            cols_hyy[count_sparse_hyy] = index_dict[(i-1,j)]
            data_hyy[count_sparse_hyy] = -2.0/step_y**2
            count_sparse_hyy += 1
            rows_hyy[count_sparse_hyy] = count
            cols_hyy[count_sparse_hyy] = index_dict[(i,j)]
            data_hyy[count_sparse_hyy] = 1.0/step_y**2
            count_sparse_hyy += 1
        elif diff_types[i,j,0] == 1:
            rows_hyy[count_sparse_hyy] = count
            cols_hyy[count_sparse_hyy] = index_dict[(i-1,j)]
            data_hyy[count_sparse_hyy] = 1.0/step_y**2
            count_sparse_hyy += 1
            rows_hyy[count_sparse_hyy] = count
            cols_hyy[count_sparse_hyy] = index_dict[(i,j)]
            data_hyy[count_sparse_hyy] = -2.0/step_y**2
            count_sparse_hyy += 1
            rows_hyy[count_sparse_hyy] = count
            cols_hyy[count_sparse_hyy] = index_dict[(i+1,j)]
            data_hyy[count_sparse_hyy] = 1.0/step_y**2
            count_sparse_hyy += 1
        elif diff_types[i,j,0] == 2:
            rows_hyy[count_sparse_hyy] = count
            cols_hyy[count_sparse_hyy] = index_dict[(i,j)]
            data_hyy[count_sparse_hyy] = 1.0/step_y**2
            count_sparse_hyy += 1
            rows_hyy[count_sparse_hyy] = count
            cols_hyy[count_sparse_hyy] = index_dict[(i+1,j)]
            data_hyy[count_sparse_hyy] = -2.0/step_y**2
            count_sparse_hyy += 1
            rows_hyy[count_sparse_hyy] = count
            cols_hyy[count_sparse_hyy] = index_dict[(i+2,j)]
            data_hyy[count_sparse_hyy] = 1.0/step_y**2
            count_sparse_hyy += 1

        #------check x-direction
        if diff_types[i,j,1] == 0:
            rows_hxx[count_sparse_hxx] = count
            cols_hxx[count_sparse_hxx] = index_dict[(i,j-2)]
            data_hxx[count_sparse_hxx] = 1.0/step_x**2
            count_sparse_hxx += 1
            rows_hxx[count_sparse_hxx] = count
            cols_hxx[count_sparse_hxx] = index_dict[(i,j-1)]
            data_hxx[count_sparse_hxx] = -2.0/step_x**2
            count_sparse_hxx += 1
            rows_hxx[count_sparse_hxx] = count
            cols_hxx[count_sparse_hxx] = index_dict[(i,j)]
            data_hxx[count_sparse_hxx] = 1.0/step_x**2
            count_sparse_hxx += 1
        elif diff_types[i,j,1] == 1:
            rows_hxx[count_sparse_hxx] = count
            cols_hxx[count_sparse_hxx] = index_dict[(i,j-1)]
            data_hxx[count_sparse_hxx] = 1.0/step_x**2
            count_sparse_hxx += 1
            rows_hxx[count_sparse_hxx] = count
            cols_hxx[count_sparse_hxx] = index_dict[(i,j)]
            data_hxx[count_sparse_hxx] = -2.0/step_x**2
            count_sparse_hxx += 1
            rows_hxx[count_sparse_hxx] = count
            cols_hxx[count_sparse_hxx] = index_dict[(i,j+1)]
            data_hxx[count_sparse_hxx] = 1.0/step_x**2
            count_sparse_hxx += 1
        elif diff_types[i,j,1] == 2:
            rows_hxx[count_sparse_hxx] = count
            cols_hxx[count_sparse_hxx] = index_dict[(i,j)]
            data_hxx[count_sparse_hxx] = 1.0/step_x**2
            count_sparse_hxx += 1
            rows_hxx[count_sparse_hxx] = count
            cols_hxx[count_sparse_hxx] = index_dict[(i,j+1)]
            data_hxx[count_sparse_hxx] = -2.0/step_x**2
            count_sparse_hxx += 1
            rows_hxx[count_sparse_hxx] = count
            cols_hxx[count_sparse_hxx] = index_dict[(i,j+2)]
            data_hxx[count_sparse_hxx] = 1.0/step_x**2
            count_sparse_hxx += 1

    return rows_hxx, cols_hxx, data_hxx, rows_hyy, cols_hyy, data_hyy


@numba.njit(cache=False, parallel=False)
def linear_weight_from_box(box_x, box_y, position=(0.0, 0.0)):
    """
    The function find the linear interpolation (extrapolation) at `position`,
    given the box with corrdinates box_x and box_y
    box_x: An 4 elements tuple/array; save the x-coordinate of box, in the order of [top-left,top-right, bottom-left, bottom-right]
    box_y: An 4 elements tuple/array; similar to box_x, save the y-coordinates
    position: the location of which we estimate the linear interpolation weight; a tuple with (y,x) coordinaes, such as (1.0, 0.0),
    the location at x=0,y=1

    return an array with shape [4,], which save the linear interpolation weight in
    [top-left,top-right, bottom-left, bottom-right] order.
    """
    y, x = position
    box_size = box_x[1] - box_x[0]
    wx = (x - box_x[0])/box_size  #x direction weight 
    wy = (y - box_y[2])/box_size   #y direction weight 

    weight_top_left = (1-wx)*wy
    weight_top_right = wx*wy
    weight_bottom_left = (1-wx)*(1-wy)
    weight_bottom_right = wx*(1-wy)

    return np.array([weight_top_left, weight_top_right, weight_bottom_left, weight_bottom_right])


@numba.njit(cache=False, parallel=False)
def bin_image_numba(arr, bin_factor=1):
    n0 = arr.shape[0] // bin_factor
    n1 = arr.shape[1] // bin_factor
    binned_arr = np.zeros((n0, n1), dtype='float')
    n_per_bin = bin_factor**2
    for i in numba.prange(n0):
        for j in numba.prange(n1):
            for m in numba.prange(i*bin_factor, (i+1)*bin_factor):
                for n in numba.prange(j*bin_factor, (j+1)*bin_factor):
                    binned_arr[i,j] += arr[m,n]
            binned_arr[i, j] = binned_arr[i, j]/n_per_bin
    return binned_arr


@numba.njit(cache=False, parallel=False)
def dpsi_mask_from(mask, dpsi_factor):
    unmask = (~mask).astype('float')
    dpsi_unmask = bin_image_numba(unmask, dpsi_factor)
    return ~np.isclose(dpsi_unmask, 1.0)


@numba.njit(cache=False, parallel=False)
def itp_box_mask_from(mask):
    itp_box_mask = np.ones((mask.shape[0]-1, mask.shape[1]-1), dtype='bool')
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if (~mask[i,j]) and (~mask[i+1,j]) and (~mask[i,j+1]) and ((~mask[i+1,j+1])):
                itp_box_mask[i,j] = False
    return itp_box_mask


@numba.njit(cache=False, parallel=False)
def dpsi2data_itp_mat_from(
        mask_itp_box,
        xc_itp_box,
        yc_itp_box,
        xgrid_data_1d,
        ygrid_data_1d,
        xgrid_dpsi,
        ygrid_dpsi,
        mask_dpsi,
):
    # idx_1d_itp_box = np.nonzero((~mask_itp_box).flatten())[0]
    unmask_dpsi = ~mask_dpsi
    i_indices_unmasked, j_indices_unmasked = np.where(unmask_dpsi)
    n_unmasked_dpsi_pixels = len(i_indices_unmasked) 
    index_dict_dpsi = {}
    for count in range(n_unmasked_dpsi_pixels):
        i, j = i_indices_unmasked[count], j_indices_unmasked[count]
        index_dict_dpsi[(i,j)] = count

    n_unmasked_data_pixels = len(xgrid_data_1d)
    rows_itp_mat = np.full(n_unmasked_data_pixels*4, -1, dtype=np.int64)
    cols_itp_mat = np.full(n_unmasked_data_pixels*4, -1, dtype=np.int64)
    data_itp_mat = np.full(n_unmasked_data_pixels*4, 0.0, dtype='float')
    count_itp_mat = 0
    for count in range(n_unmasked_data_pixels):
        this_x_data = xgrid_data_1d[count]
        this_y_data = ygrid_data_1d[count]
        # dist2 = (xc_itp_box-this_x_data)**2 + (yc_itp_box-this_y_data)**2
        # id_nearest_itp_box = idx_1d_itp_box[np.argmin(dist2)]
        # i = id_nearest_itp_box // mask_itp_box.shape[1]
        # j = id_nearest_itp_box % mask_itp_box.shape[1]

        j_min = np.argmin(np.abs(xc_itp_box[0,:]-this_x_data))
        i_min = np.argmin(np.abs(yc_itp_box[:,0]-this_y_data))

        if ~mask_itp_box[i_min,j_min]:
            i = i_min
            j = j_min
        else:
            search_width = 2
            i = -1
            j = -1
            dist_tmp = 1e8
            while (i==-1):
                for m in range(i_min-search_width, i_min+search_width+1):
                    for n in range(j_min-search_width, j_min+search_width+1):
                        if ~mask_itp_box[m,n]:
                            this_dist = np.sqrt((xc_itp_box[m,n]-this_x_data)**2 + (yc_itp_box[m,n]-this_y_data)**2)
                            if this_dist < dist_tmp:
                                dist_tmp = this_dist
                                i = m
                                j = n
                search_width += 1

        #itp_box_corners_x: [top-left,top-right, bottom-left, bottom-right] corner x-positions 
        itp_box_corners_x = (xgrid_dpsi[i,j], xgrid_dpsi[i,j+1], xgrid_dpsi[i+1,j], xgrid_dpsi[i+1,j+1])
        itp_box_corners_y = (ygrid_dpsi[i,j], ygrid_dpsi[i,j+1], ygrid_dpsi[i+1,j], ygrid_dpsi[i+1,j+1])
        itp_weights = linear_weight_from_box(itp_box_corners_x, itp_box_corners_y, position=(this_y_data, this_x_data))
        itp_idx = (index_dict_dpsi[i, j], index_dict_dpsi[i, j+1], index_dict_dpsi[i+1, j], index_dict_dpsi[i+1, j+1])

        for k in range(4):
            rows_itp_mat[count_itp_mat] = count
            cols_itp_mat[count_itp_mat] = itp_idx[k]
            data_itp_mat[count_itp_mat] = itp_weights[k]
            count_itp_mat += 1

    return rows_itp_mat, cols_itp_mat, data_itp_mat


@numba.njit(cache=False, parallel=False)
def exp_cov_matrix_from(
    scale_coefficient: float,  ##regularization scale
    pixel_points: np.ndarray, #shape: [npixels, 2], save the source pixelization positions on source-plane. [[y1,x1], [y2,x2], ...]
) -> np.ndarray:
    """
    Consutruct the source brightness covariance matrix, which is used to determined the regularization pattern (i.e, how the different 
    source pixels are smoothed).

    the covariance matrix includes one non-linear parameters, the scale coefficient, which is used to determine the typical scale of 
    the regularization pattern.

    Parameters
    ----------
    scale_coefficient
        the typical scale of the regularization pattern .
    pixel_points 
        An 2d array with shape [N_source_pixels, 2], which save the source pixelization coordinates (on source plane). Something like
         [[y1,x1], [y2,x2], ...]

    Returns
    -------
    np.ndarray
        The source covariance matrix (2d array), shape [N_source_pixels, N_source_pixels].
    """

    pixels = len(pixel_points)
    covariance_matrix = np.zeros(shape=(pixels, pixels))

    for i in range(pixels):
        covariance_matrix[i,i] += 1e-8
        for j in range(pixels):
            xi = pixel_points[i, 1]
            yi = pixel_points[i, 0]
            xj = pixel_points[j, 1]
            yj = pixel_points[j, 0]
            d_ij = np.sqrt((xi-xj)**2 + (yi-yj)**2) #distance between the pixel i and j

            covariance_matrix[i, j] += np.exp(-1.0*d_ij/scale_coefficient)

    return covariance_matrix


@numba.njit(cache=False, parallel=False)
def gauss_cov_matrix_from(
    scale_coefficient: float,  ##regularization scale
    pixel_points: np.ndarray, #shape: [npixels, 2], save the source pixelization positions on source-plane. [[y1,x1], [y2,x2], ...]
) -> np.ndarray:
    """
    Consutruct the source brightness covariance matrix, which is used to determined the regularization pattern (i.e, how the different 
    source pixels are smoothed).

    the covariance matrix includes one non-linear parameters, the scale coefficient, which is used to determine the typical scale of 
    the regularization pattern.

    Parameters
    ----------
    scale_coefficient
        the typical scale of the regularization pattern .
    pixel_points 
        An 2d array with shape [N_source_pixels, 2], which save the source pixelization coordinates (on source plane). Something like
         [[y1,x1], [y2,x2], ...]

    Returns
    -------
    np.ndarray
        The source covariance matrix (2d array), shape [N_source_pixels, N_source_pixels].
    """

    pixels = len(pixel_points)
    covariance_matrix = np.zeros(shape=(pixels, pixels))

    for i in range(pixels):
        covariance_matrix[i,i] += 1e-8
        for j in range(pixels):
            xi = pixel_points[i, 1]
            yi = pixel_points[i, 0]
            xj = pixel_points[j, 1]
            yj = pixel_points[j, 0]
            d_ij = np.sqrt((xi-xj)**2 + (yi-yj)**2) #distance between the pixel i and j

            covariance_matrix[i, j] += np.exp(-1.0*d_ij**2/(2*scale_coefficient**2))

    return covariance_matrix


@numba.njit(cache=False, parallel=False)
def matern_kernel(r: float, l: float=1.0, v: float=0.5):
    """
    need to `pip install numba-scipy `
    see https://gaussianprocess.org/gpml/chapters/RW4.pdf for more info

    the distance r need to be scalar
    l is the scale
    v is the order, better < 30, otherwise may have numerical NaN issue.

    v control the smoothness level. the larger the v, the stronger smoothing condition (i.e., the solution is v-th differentiable) imposed by the kernel.
    """
    r = abs(r)
    if r == 0:
        r=0.00000001
    part1 = 2 ** (1 - v) / math.gamma(v)
    part2 = (math.sqrt(2 * v) * r / l) ** v
    part3 = sc.kv(v, math.sqrt(2 * v) * r / l)
    return part1 * part2 * part3


@numba.njit(cache=False, parallel=False)
def matern_cov_matrix_from(
    scale_coefficient: float,  ##regularization scale
    nu: float,
    pixel_points: np.ndarray, #shape: [npixels, 2], save the source pixelization positions on source-plane. [[y1,x1], [y2,x2], ...]
) -> np.ndarray:
    """
    Consutruct the regularization covariance matrix, which is used to determined the regularization pattern (i.e, how the different 
    pixels are correlated).

    the covariance matrix includes two non-linear parameters, the scale coefficient, which is used to determine the typical scale of 
    the regularization pattern. The smoothness order parameters mu, whose value determie the inversion solution is mu-th differentiable.
    Parameters
    ----------
    scale_coefficient
        the typical scale of the regularization pattern .
    pixel_points 
        An 2d array with shape [N_source_pixels, 2], which save the source pixelization coordinates (on source plane). Something like
         [[y1,x1], [y2,x2], ...]

    Returns
    -------
    np.ndarray
        The source covariance matrix (2d array), shape [N_source_pixels, N_source_pixels].
    """

    pixels = len(pixel_points)
    covariance_matrix = np.zeros(shape=(pixels, pixels))

    for i in range(pixels):
        covariance_matrix[i,i] += 1e-8
        for j in range(pixels):
            xi = pixel_points[i, 1]
            yi = pixel_points[i, 0]
            xj = pixel_points[j, 1]
            yj = pixel_points[j, 0]
            d_ij = np.sqrt((xi-xj)**2 + (yi-yj)**2) #distance between the pixel i and j

            covariance_matrix[i, j] += matern_kernel(d_ij, l=scale_coefficient, v=nu) 

    return covariance_matrix

    
def regularization_matrix_gp_from(
    scale: float,  ##regularization scale
    coefficient: float,
    nu: float,
    points: np.ndarray, #shape: [npixels, 2], save the source pixelization positions on source-plane. [[y1,x1], [y2,x2], ...]
    reg_type: str,
) -> np.ndarray:
    if reg_type == 'exp':
        covariance_matrix = exp_cov_matrix_from(scale, points)
    elif reg_type == 'gauss':
        covariance_matrix = gauss_cov_matrix_from(scale, points)
    elif reg_type == 'matern':
        covariance_matrix = matern_cov_matrix_from(scale, nu, points)
    else:
        raise Exception("Unknown reg_type")

    inverse_covariance_matrix = np.linalg.inv(covariance_matrix)
    
    regulariaztion_matrix = coefficient * inverse_covariance_matrix

    return regulariaztion_matrix


@numba.njit(cache=False, parallel=False)
def gradient_points_from(points: np.ndarray, cross_size=0.001):
    """
    calculate the grid used for graident calculation
    Parameters
    ----------
    points 
        An 2d array with shape [N_points, 2], which save the points coordinates. Something like [[y1,x1], [y2,x2], ...]

    Returns
    -------
    np.ndarray
        The grid used for graident calculation, shape [N_points*4, 2].
    """
    n_points = len(points)
    grad_points = np.zeros(shape=(n_points*4, 2))
    for i in range(n_points):
        #left endpoint
        grad_points[4*i,0] = points[i,0] #y
        grad_points[4*i,1] = points[i,1] - cross_size #x
        #right endpoint
        grad_points[4*i+1,0] = points[i,0] #y
        grad_points[4*i+1,1] = points[i,1] + cross_size #x
        #top endpoint
        grad_points[4*i+3,0] = points[i,0] + cross_size #y
        grad_points[4*i+3,1] = points[i,1] #x
        #bottom endpoint
        grad_points[4*i+2,0] = points[i,0] - cross_size #y
        grad_points[4*i+2,1] = points[i,1] #x
    
    return grad_points


@numba.njit(cache=False, parallel=False)
def source_gradient_from(values_on_gradient_points: np.ndarray, cross_size=0.001):
    values_on_gradient_points = values_on_gradient_points.reshape((-1, 4))
    x_diff = (values_on_gradient_points[:, 1] - values_on_gradient_points[:, 0])/(2.0*cross_size) 
    y_diff = (values_on_gradient_points[:, 3] - values_on_gradient_points[:, 2])/(2.0*cross_size)
    return np.vstack((y_diff, x_diff)).T


@numba.njit(cache=False, parallel=False)
def source_gradient_matrix_numba_func(source_gradient):
    N_unmasked_data_points = source_gradient.shape[0]
    rows_idx = np.full(N_unmasked_data_points*2, -1, dtype=np.int64)
    cols_idx = np.full(N_unmasked_data_points*2, -1, dtype=np.int64)
    values = np.full(N_unmasked_data_points*2, 0.0, dtype='float')

    count = 0
    for i in range(N_unmasked_data_points):
        rows_idx[count] = i
        cols_idx[count] = i*2
        values[count] = source_gradient[i,1] #x-derivative
        count +=1
        rows_idx[count] = i
        cols_idx[count] = i*2+1
        values[count] = source_gradient[i,0] #y-derivative
        count += 1

    return rows_idx, cols_idx, values


def source_gradient_matrix_from(source_gradient: np.ndarray):
    """
    Generate the source gradient matrix from the source gradient array (got from the function `source_gradient_from`)

    Input:
    source_gradient: an [N_unmasked_data_points, 2] array. The y/x derivative values at the ray-traced dpsi-grid on the source-plane

    Output:
    source_gradient_matrix: an [N_unmasked_data_points, 2*N_unmasked_data_points] arrary. See equation-9 in our team-document.
    """
    rows_idx, cols_idx, values = source_gradient_matrix_numba_func(source_gradient)
    return csr_matrix((values, (rows_idx, cols_idx)), shape=(source_gradient.shape[0], 2*source_gradient.shape[0]))


def dpsi_gradient_matrix_dense_from(itp_mat, Hx, Hy):
    """
    itp_mat is the interpolation matrix, shape [n_unmasked_data_points, n_unmasked_dpsi_points]
    Accept the x/y differential operator Hx and Hy; both shapes are [n_unmasked_dpsi_points, n_unmasked_dpsi_points]
    Construct the dpsi_gradient_operator with shape [2*n_unmasked_data_points, n_unmasked_dpsi_points]. 
    see eq-8 in our potential correction document
    """
    n_unmasked_data_points = itp_mat.shape[0]
    n_unmasked_dpsi_points = Hx.shape[0]
    dpsi_grad_mat = np.zeros((2*n_unmasked_data_points, n_unmasked_dpsi_points), dtype='float')
    Hx_itp = (itp_mat @ Hx).toarray() #shape:[n_unmasked_data_points, n_unmasked_dpsi_points]
    Hy_itp = (itp_mat @ Hy).toarray() #shape:[n_unmasked_data_points, n_unmasked_dpsi_points]

    for count in range(n_unmasked_data_points):
        dpsi_grad_mat[count*2, :] += Hx_itp[count, :]
        dpsi_grad_mat[count*2+1, :] += Hy_itp[count, :]

    return csr_matrix(dpsi_grad_mat)


def dpsi_gradient_matrix_from(itp_mat, Hx, Hy):
    """
    itp_mat is the interpolation matrix, shape [n_unmasked_data_points, n_unmasked_dpsi_points]
    Accept the x/y differential operator Hx and Hy; both shapes are [n_unmasked_dpsi_points, n_unmasked_dpsi_points]
    Construct the dpsi_gradient_operator with shape [2*n_unmasked_data_points, n_unmasked_dpsi_points]. 
    see eq-8 in our potential correction document
    """
    n_unmasked_data_points = itp_mat.shape[0]
    n_unmasked_dpsi_points = Hx.shape[0]
    dpsi_grad_mat = lil_matrix((2*n_unmasked_data_points, n_unmasked_dpsi_points), dtype='float')
    Hx_itp = itp_mat @ Hx  #shape:[n_unmasked_data_points, n_unmasked_dpsi_points]
    Hy_itp = itp_mat @ Hy  #shape:[n_unmasked_data_points, n_unmasked_dpsi_points]

    for row in range(n_unmasked_data_points):
        col_idx = Hx_itp[row].nonzero()[1]
        values = Hx_itp[row, col_idx].toarray()[0]
        dpsi_grad_mat[row*2, col_idx] = values

        col_idx = Hy_itp[row].nonzero()[1]
        values = Hy_itp[row, col_idx].toarray()[0]
        dpsi_grad_mat[row*2+1, col_idx] = values

    return dpsi_grad_mat.tocsr()


@numba.njit(cache=False, parallel=False)
def psf_matrix_numba(psf_kernel, mask):
    """
    psf_kernel: two array represent the psf kernel
    image_shape: the shape of image
    mask: the mask that defines the image-fitting region
    """
    psf_hw = int(psf_kernel.shape[0]/2)
    if psf_hw*2+1 != psf_kernel.shape[0]:
        raise Exception(f"The psf kernel size is: {psf_kernel.shape[0]}, not an odd number!")
    
    if not np.isclose(np.sum(psf_kernel), 1.0):
        print("The psf has not been normalized")
        print("The summed value of psf kernel is:", np.sum(psf_kernel))
        print("Normalize the psf kernel now...")
        psf_kernel = psf_kernel / np.sum(psf_kernel)

    mask_ext = np.ones((mask.shape[0]+psf_hw*2, mask.shape[1]+psf_hw*2), dtype='bool')
    mask_ext[psf_hw:-psf_hw, psf_hw:-psf_hw] = mask
    image_ext_shape = mask_ext.shape

    indice_0, indice_1 = np.nonzero(~mask_ext)
    n_unmasked_pix = len(indice_0)
    psf_mat = np.zeros((n_unmasked_pix, n_unmasked_pix), dtype='float')

    for ii in range(n_unmasked_pix):
        image_unit = np.zeros(image_ext_shape, dtype='float')
        image_unit[indice_0[ii]-psf_hw:indice_0[ii]+psf_hw+1, indice_1[ii]-psf_hw:indice_1[ii]+psf_hw+1] = psf_kernel[:, :]
        # psf_mat[:, ii] = image_unit[indice_0, indice_1] ##numba give a error with this index scheme
        for jj in range(n_unmasked_pix):
            psf_mat[jj, ii] = image_unit[indice_0[jj], indice_1[jj]]

    return psf_mat


def inverse_covariance_matrix_from(noise_slim):
    """
    Generate the inverse covariance matrix from the slimmed 1d noise-map.
    """
    npix = len(noise_slim)
    inv_cov_mat = lil_matrix((npix, npix), dtype='float')
    for ii in range(npix):
        inv_cov_mat[ii, ii] = 1.0/(noise_slim[ii])**2
    return inv_cov_mat.tocsr()


def log_det_mat(square_matrix, sparse=False):
    if sparse:
        try:
            lu = splu(csc_matrix(square_matrix))
            diagL = lu.L.diagonal()
            diagU = lu.U.diagonal()
            diagL = diagL.astype(np.complex128)
            diagU = diagU.astype(np.complex128)

            return np.real(np.log(diagL).sum() + np.log(diagU).sum())

        except RuntimeError:
            pass

    try:
        if sparse: square_matrix = square_matrix.toarray()
        return 2.0 * np.sum(
            np.log(
                np.diag(np.linalg.cholesky(square_matrix))
            )
        )
    except np.linalg.LinAlgError as e:
        raise Exception(f"The matrix is not positive definite: {e}")
    

from skimage import measure
from skimage.morphology import dilation
def arc_mask_from(snr_map, threshold=3.0, ignor_size=25, ext_size=5):
    bool_map = (snr_map > threshold)

    # Label connected regions in the binary mask
    labels = measure.label(bool_map)

    # Find the properties of the connected regions
    properties = measure.regionprops(labels)

    # Identify the negative pixel islands
    small_islands = [prop.label for prop in properties if prop.area < ignor_size]  # Adjust the area threshold as per your requirement

    # Create a new image with negative pixel islands removed
    mask = np.copy(bool_map)
    for island_label in small_islands:
        mask[labels == island_label] = 0
    mask = dilation(mask, footprint=np.ones((ext_size, ext_size)))

    mask, _ = iterative_clean_mask(~mask, max_iter=50)
    return mask



@numba.njit(cache=False, parallel=False)
def diff_2nd_operator_dpsi_reg_numba_func(mask, dpix=1.0):
    """
    Receive a mask, use it to generate the 2nd differential operator matrix Hxx and Hyy.
    Hxx (Hyy) has a shape of [n_unmasked_pixels, n_unmasked_pixels],
    when it act on the unmasked data, generating the 2nd x/y-derivative of the unmasked data.
    compare with the diff_2nd_operator_numba_func, only the forward scheme is used to calculate the 2nd x/y-derivative.
    
    dpix: pixel size in unit of arcsec.
    """
    unmask = ~mask
    i_indices_unmasked, j_indices_unmasked = np.where(unmask)
    n_unmasked_pixels = len(i_indices_unmasked) 

    rows_hxx = np.full(n_unmasked_pixels*3, -1, dtype=np.int64)
    cols_hxx = np.full(n_unmasked_pixels*3, -1, dtype=np.int64)
    data_hxx = np.full(n_unmasked_pixels*3, 0.0, dtype='float')
    rows_hyy = np.full(n_unmasked_pixels*3, -1, dtype=np.int64)
    cols_hyy = np.full(n_unmasked_pixels*3, -1, dtype=np.int64)
    data_hyy = np.full(n_unmasked_pixels*3, 0.0, dtype='float')

    step_y = -1.0*dpix #the minus sign is due to the y-coordinate decrease the pixel_size as index i along axis-0 increase 1.
    step_x = 1.0*dpix #no minus, becasue the x-coordinate increase as index j along axis-1 increase.

    index_dict = {}
    for count in range(n_unmasked_pixels):
        i, j = i_indices_unmasked[count], j_indices_unmasked[count]
        index_dict[(i,j)] = count

    count_sparse_hyy = 0
    count_sparse_hxx = 0
    for count in range(n_unmasked_pixels):
        i, j = i_indices_unmasked[count], j_indices_unmasked[count]

        #check Hxx
        if (j < unmask.shape[1]-3):
            if unmask[i, j+1]:
                if unmask[i, j+2]:
                    #use 2nd diff forward reg
                    rows_hxx[count_sparse_hxx] = count
                    cols_hxx[count_sparse_hxx] = index_dict[(i,j)]
                    data_hxx[count_sparse_hxx] = 1.0/step_x**2
                    count_sparse_hxx += 1
                    rows_hxx[count_sparse_hxx] = count
                    cols_hxx[count_sparse_hxx] = index_dict[(i,j+1)]
                    data_hxx[count_sparse_hxx] = -2.0/step_x**2
                    count_sparse_hxx += 1
                    rows_hxx[count_sparse_hxx] = count
                    cols_hxx[count_sparse_hxx] = index_dict[(i,j+2)]
                    data_hxx[count_sparse_hxx] = 1.0/step_x**2
                    count_sparse_hxx += 1
                else:
                    #use 1st diff forward reg
                    rows_hxx[count_sparse_hxx] = count
                    cols_hxx[count_sparse_hxx] = index_dict[(i,j)]
                    data_hxx[count_sparse_hxx] = -1.0/step_x
                    count_sparse_hxx += 1
                    rows_hxx[count_sparse_hxx] = count
                    cols_hxx[count_sparse_hxx] = index_dict[(i,j+1)]
                    data_hxx[count_sparse_hxx] = 1.0/step_x
                    count_sparse_hxx += 1
            else:
                #use zero order reg
                rows_hxx[count_sparse_hxx] = count
                cols_hxx[count_sparse_hxx] = index_dict[(i,j)]
                data_hxx[count_sparse_hxx] = 1.0
                count_sparse_hxx += 1
        elif (j < unmask.shape[1]-2):
            if unmask[i, j+1]:
                #use 1st diff forward reg
                rows_hxx[count_sparse_hxx] = count
                cols_hxx[count_sparse_hxx] = index_dict[(i,j)]
                data_hxx[count_sparse_hxx] = -1.0/step_x
                count_sparse_hxx += 1
                rows_hxx[count_sparse_hxx] = count
                cols_hxx[count_sparse_hxx] = index_dict[(i,j+1)]
                data_hxx[count_sparse_hxx] = 1.0/step_x
                count_sparse_hxx += 1
            else:
                #use zero order reg
                rows_hxx[count_sparse_hxx] = count
                cols_hxx[count_sparse_hxx] = index_dict[(i,j)]
                data_hxx[count_sparse_hxx] = 1.0
                count_sparse_hxx += 1
        else:
            #use zero order reg
            rows_hxx[count_sparse_hxx] = count
            cols_hxx[count_sparse_hxx] = index_dict[(i,j)]
            data_hxx[count_sparse_hxx] = 1.0
            count_sparse_hxx += 1

        #check Hyy
        if (i < unmask.shape[0]-3):
            if unmask[i+1, j]:
                if unmask[i+2, j]:
                    #use 2nd diff forward reg
                    rows_hyy[count_sparse_hyy] = count
                    cols_hyy[count_sparse_hyy] = index_dict[(i,j)]
                    data_hyy[count_sparse_hyy] = 1.0/step_y**2
                    count_sparse_hyy += 1
                    rows_hyy[count_sparse_hyy] = count
                    cols_hyy[count_sparse_hyy] = index_dict[(i+1,j)]
                    data_hyy[count_sparse_hyy] = -2.0/step_y**2
                    count_sparse_hyy += 1
                    rows_hyy[count_sparse_hyy] = count
                    cols_hyy[count_sparse_hyy] = index_dict[(i+2,j)]
                    data_hyy[count_sparse_hyy] = 1.0/step_y**2
                    count_sparse_hyy += 1
                else:
                    #use 1st diff forward reg
                    rows_hyy[count_sparse_hyy] = count
                    cols_hyy[count_sparse_hyy] = index_dict[(i,j)]
                    data_hyy[count_sparse_hyy] = -1.0/step_y
                    count_sparse_hyy += 1
                    rows_hyy[count_sparse_hyy] = count
                    cols_hyy[count_sparse_hyy] = index_dict[(i+1,j)]
                    data_hyy[count_sparse_hyy] = 1.0/step_y
                    count_sparse_hyy += 1
            else:
                #use zero order reg
                rows_hyy[count_sparse_hyy] = count
                cols_hyy[count_sparse_hyy] = index_dict[(i,j)]
                data_hyy[count_sparse_hyy] = 1.0
                count_sparse_hyy += 1
        elif (i < unmask.shape[0]-2):
            if unmask[i+1, j]:
                #use 1st diff forward reg
                rows_hyy[count_sparse_hyy] = count
                cols_hyy[count_sparse_hyy] = index_dict[(i,j)]
                data_hyy[count_sparse_hyy] = -1.0/step_y
                count_sparse_hyy += 1
                rows_hyy[count_sparse_hyy] = count
                cols_hyy[count_sparse_hyy] = index_dict[(i+1,j)]
                data_hyy[count_sparse_hyy] = 1.0/step_y
                count_sparse_hyy += 1
            else:
                #use zero order reg
                rows_hyy[count_sparse_hyy] = count
                cols_hyy[count_sparse_hyy] = index_dict[(i,j)]
                data_hyy[count_sparse_hyy] = 1.0
                count_sparse_hyy += 1
        else:
            #use zero order reg
            rows_hyy[count_sparse_hyy] = count
            cols_hyy[count_sparse_hyy] = index_dict[(i,j)]
            data_hyy[count_sparse_hyy] = 1.0
            count_sparse_hyy += 1

    return rows_hxx[:count_sparse_hxx], cols_hxx[:count_sparse_hxx], data_hxx[:count_sparse_hxx], \
        rows_hyy[:count_sparse_hyy], cols_hyy[:count_sparse_hyy], data_hyy[:count_sparse_hyy]

        
            
def dpsi_curvature_reg_matrix_from(mask, return_H=False):
    new_mask, diff_types = iterative_clean_mask(mask)
    if not (new_mask == mask).all():
        raise Exception("the mask has not been fully cleaned!")
    rows_hxx, cols_hxx, data_hxx, rows_hyy, cols_hyy, data_hyy = diff_2nd_operator_dpsi_reg_numba_func(mask)

    n_unmasked = np.count_nonzero(~mask)
    Hxx = csr_matrix((data_hxx, (rows_hxx, cols_hxx)), shape=(n_unmasked, n_unmasked))
    Hyy = csr_matrix((data_hyy, (rows_hyy, cols_hyy)), shape=(n_unmasked, n_unmasked))

    if return_H:
        return Hxx.T @ Hxx + Hyy.T @ Hyy, Hxx, Hyy
    else:
        return Hxx.T @ Hxx + Hyy.T @ Hyy