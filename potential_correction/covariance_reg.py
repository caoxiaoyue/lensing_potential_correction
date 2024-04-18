from potential_correction import util as pul
from autoarray.inversion.linear_obj.linear_obj import LinearObj
from autoarray.inversion.regularization.abstract import AbstractRegularization
import numpy as np
from abc import abstractmethod


class CovarianceRegularization(AbstractRegularization):
    def __init__(self, coefficient: float = 1.0, scale: float = 1.0):
        self.coefficient = coefficient
        self.scale = scale


    def regularization_parameters_from(self, linear_obj: LinearObj) -> np.ndarray:
        return self.coefficient * np.ones(linear_obj.params), self.scale * np.ones(linear_obj.params)
    

    @abstractmethod
    def regularization_matrix_from(self, points) -> np.ndarray:
        """
        points: the position of mesh that is regularized
        """
        pass



class GaussianRegularization(CovarianceRegularization):
    def regularization_matrix_from(self, points) -> np.ndarray:
        """
        points: the position of mesh that is regularized
        """
        return pul.regularization_matrix_gp_from(
            coefficient=self.coefficient,
            scale=self.scale,
            nu=None,
            points=points,
            reg_type='gauss'
        )
    


class ExponentialRegularization(CovarianceRegularization):
    def regularization_matrix_from(self, points) -> np.ndarray:
        """
        points: the position of mesh that is regularized
        """
        return pul.regularization_matrix_gp_from(
            coefficient=self.coefficient,
            scale=self.scale,
            nu=None,
            points=points,
            reg_type='exp'
        )
    


class MaternRegularization(CovarianceRegularization):
    def __init__(self, coefficient: float = 1.0, scale: float = 1.0, nu: float = 0.5):
        self.coefficient = coefficient
        self.scale = float(scale)
        self.nu = float(nu)


    def regularization_parameters_from(self, linear_obj: LinearObj) -> np.ndarray:
        return self.coefficient * np.ones(linear_obj.params), self.scale * np.ones(linear_obj.params), self.nu * np.ones(linear_obj.params)


    def regularization_matrix_from(self, points) -> np.ndarray:
        """
        points: the position of mesh that is regularized
        """
        return pul.regularization_matrix_gp_from(
            coefficient=self.coefficient,
            scale=self.scale,
            nu=self.nu,
            points=points,
            reg_type='matern',
        )
    


class CurvatureRegularizationDpsi(AbstractRegularization):
    def __init__(self, coefficient: float = 1.0):
        self.coefficient = coefficient


    def regularization_parameters_from(self, linear_obj: LinearObj) -> np.ndarray:
        return self.coefficient * np.ones(linear_obj.params)
    

    @abstractmethod
    def regularization_matrix_from(self, mask) -> np.ndarray:
        """
        mask: the mask that defines the pixels that are modeled
        """
        return self.coefficient * pul.dpsi_curvature_reg_matrix_from(mask)