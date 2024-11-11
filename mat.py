import numpy as np
import sys
from numba import njit


# def mir(A):
#   C = A.T
#   D = np.flip(C, 0)
#   E = np.flip(D, 1)
#   return E

@njit
def mir(A):
  C = A.T
  D = C[::-1, :]  # Flip vertically (equivalent to np.flip(C, 0))
  E = D[:, ::-1] # Flip horizontally (equivalent to np.flip(D, 1))


def mir2(A):
  C = A.T
  # D = np.flip(A, 0)
  # E = np.flip(D, 1)
  # C = E.T
  return C

def mir_diag(m, d):
  if d == 1:
    return mir(m)
  elif d == -1:
    return mir2(m)
  else:
    #error
    print("invalid diagonal")
    sys.exit(1)



if __name__ == '__main__':
  # Define two matrices
  A = np.array([[1, 2, 0],
                [4, 0, 0],
                [0, 0, 0]])

  B = np.array([[0, 0, 0],
                [0, 0, 2],
                [0, 4, 1]])

  # print("Matrix A\n", A)

  # C = A.T
  # print("transpose of A\n", C)

  # D = np.flip(A, 0)
  # print("0(vertical) flip of A\n", D)

  # E = np.flip(A, 1)
  # print("1(horizontal) flip of A\n", E)

  # print("####################")

  print("Matrix A\n", A)
  C = A.T
  print("transpose of A\n", C)
  D = np.flip(C, 0)
  print("0(vertical) flip\n", D)
  E = np.flip(D, 1)
  print("1(horizontal) flip\n", E)




  print("B\n", B)
    
  print("mir(B)\n", mir(B))

  Z = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])

  print("Z\n", Z)
  
  # W = Z.T
  # print("transpose of Z\n", W)
  
  # Y = np.flip(W, 0)
  # print("0(vertical) flip\n", Y)
  
  # X = np.flip(Y, 1)
  # print("1(horizontal) flip\n", X)
  
  # print("mir(Z)\n", mir(Z))



    
  print("mir2(Z)\n", mir2(Z))