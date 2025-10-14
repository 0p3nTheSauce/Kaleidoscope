import numpy as np
from numba import njit


# njit provides little benefit
@njit(cache=True)
def mir_p_1chan(A):  # positive diagonal
    C = A.T
    D = C[::-1, :]  # Flip vertically (equivalent to np.flip(C, 0))
    E = D[:, ::-1]  # Flip horizontally (equivalent to np.flip(D, 1))
    return E

@njit(cache=True)
def mir_n_1chan(A):  # negative diagonal
    C = A.T
    return C

@njit(cache=True)
def mir_p2(A):
    B = A.transpose(1, 0, 2)
    C = B[::-1, :, :]  # Flip vertically (equivalent to np.flip(C, 0))
    D = C[:, ::-1, :]  # Flip horizontally (equivalent to np.flip(D, 1)
    return D

@njit(cache=True)
def mir_n2(A):
    return A.transpose(1, 0, 2)





if __name__ == "__main__":
    # Define two matrices
    A = np.array([[1, 2, 0], [4, 0, 0], [0, 0, 0]])

    B = np.array([[0, 0, 0], [0, 0, 2], [0, 4, 1]])

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

    print("mir(B)\n", mir_p(B))

    Z = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

    print("Z\n", Z)

    # W = Z.T
    # print("transpose of Z\n", W)

    # Y = np.flip(W, 0)
    # print("0(vertical) flip\n", Y)

    # X = np.flip(Y, 1)
    # print("1(horizontal) flip\n", X)

    # print("mir(Z)\n", mir(Z))

    print("mir2(Z)\n", mir_n(Z))
