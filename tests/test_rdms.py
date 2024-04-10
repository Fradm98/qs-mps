from qs_mps.mps_class import MPS
from qs_mps.utils import tensor_shapes
import numpy as np

L = 12
chain = MPS(L=L, d=2, model="ANNNI", chi=64, h=0, k=0, J=1, eps=1e-5)

up_state = [1]
down_state = [0]
ferro_state = np.array([[up_state,down_state]])
ferro_tensor = [ferro_state for _ in range(L)]
chain.sites = ferro_tensor.copy()
chain.enlarge_chi()
tensor_shapes(chain.sites)
# chain.canonical_form()
# tensor_shapes(chain.sites)
rdm = chain.reduced_density_matrix(sites=[L//2])