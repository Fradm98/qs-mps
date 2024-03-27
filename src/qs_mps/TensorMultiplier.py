from scipy.sparse.linalg import LinearOperator
import numpy as np

class TensorMultiplierOperator(LinearOperator):
    r"""
    TensorMultiplierOperator

    This is a subclass of LinearOperator from PyLops. It serves
    as a new operator that implements the matrix vector product
    between the effective hamiltonian, given in tensor form by the environment
    left, right and the mpo in the middle - and the guess vector
    given by the flattened tensor in the site to optimize.
    The function implementing this is in the class MPS of qs-mps.

    """
    def __init__(self, shape, matvec, dtype=None):
        super(TensorMultiplierOperator, self).__init__(dtype=np.dtype(dtype), shape=shape)
        self.matvec = matvec

    def _matvec(self, x):
        return self.matvec(x)