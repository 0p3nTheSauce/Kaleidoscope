import numpy as np

def mir_p_o(A):
  C = A.T
  D = np.flip(C, 0)
  E = np.flip(D, 1)
  return E

def mir_n_o(A):
  C = A.T
  return C