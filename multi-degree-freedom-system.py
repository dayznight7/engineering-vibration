import numpy as np
import math
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt


# only works in 2degree system
def eig2d(M, K):

    M_inv = np.linalg.inv(M)
    V, U = np.linalg.eig(M_inv @ K)
    vec1, vec2 = np.split(U, 2, axis=1)
    u1 = vec1.T @ M @ vec1
    u2 = vec2.T @ M @ vec2
    S = np.concatenate((vec1*u1**-0.5, vec2*u2**-0.5), axis=1)

    return S, V


# M = np.array([[3, 0], [0, 1]])
# K = np.array([[4, -2], [-2, 2]])
# S, V = eig2d(M, K)
# print("S = [u1 u2]")
# print(S)
# print("V = [w1^2 w2^2]")
# print(V)
# print("S.T @ M @ S = I")
# print(S.T @ M @ S)
# print("S.T @ K @ S = V, diag(w1^2, w2^2)")
# print(S.T @ K @ S)


def eig(M, K):
    M_inv = np.linalg.inv(M)
    eigenvalues, eigenvectors = np.linalg.eig(M_inv @ K)

    S = eigenvectors
    V = eigenvalues

    column_vectors = np.hsplit(S, S.shape[1])

    for i in range(len(column_vectors)):
        vec = column_vectors[i]
        normalizer = vec.T @ M @ vec
        column_vectors[i] = vec * normalizer**-0.5

    S = np.hstack([np.array(vec).reshape(-1, 1) for vec in column_vectors])

    return S, V


# M = np.diag([1, 2, 3])
# K = np.array([[2, 1, 0], [1, 4, 2], [0, 2, 6]])
# S, V = eig(M, K)
# print("S:")
# print(S)
# print("V:")
# print(V)
# print("S.T M S")
# print(S.T @ M @ S)
# print("S.T @ K @ S")
# print(S.T @ K @ S)


# M = np.diag([1, 2, 3])
# K = np.array([[2, 1, 0], [1, 4, 2], [0, 2, 6]])
# S, V = eig(M, K)
# S2 = np.array([[0.4629, -0.7559, 0.4629],[-0.5,0,0.5],[0.3086,0.3780,0.3086]])
# print("\npython ver")
# print("\nS=")
# print(S)
# print("\nw^2")
# print(S.T @ K @ S)
# print("\nmatlab ver")
# print("\nS=")
# print(S2)
# print("\nw^2")
# print(S2.T @ K @ S2)


# M = np.diag([1, 4])
# K = np.array([[12, -2], [-2, 12]])
# S, V = eig(M, K)
# print("S:")
# print(S)
# print("V:")
# print(V)
# print("S.T M S")
# print(S.T @ M @ S)
# print("S.T @ K @ S")
# print(S.T @ K @ S)
# print(S.T @ M)
