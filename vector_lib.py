import numpy as np
import math

def scale_matrix(A):
    T = np.eye(4)
    A_diag = np.diag(A)
    T[:3, :3] = A_diag
    return T

def translation_matrix(A):
    T = np.eye(4)
    T[:3, -1] = A
    return T


def rotation_matrix(angle, direction, point=None):
    sina = math.sin(angle)
    cosa = math.cos(angle)
    direction = vnorm(np.array(direction[:3])).ravel()
    R = np.diag([cosa, cosa, cosa])
    R += np.outer(direction, direction) * (1.0 - cosa)
    direction *= sina
    R += np.array([[ 0.0,         -direction[2],  direction[1]],
                      [ direction[2], 0.0,          -direction[0]],
                      [-direction[1], direction[0],  0.0]])
    M = np.identity(4)
    M[:3, :3] = R
    if point is not None:
        point = np.array(point[:3], dtype=np.float64, copy=False).ravel()
        M[:3, 3] = point - np.dot(R, point)
    return M


def vabs(A):
    return vdot(A, A)


def vnorm(A):
    mag = np.sqrt(vabs(A))
    return A * (1.0 / np.where(mag == 0, 1, mag))


def vdot(A, B):
    return (A * B).sum(axis=0, keepdims=True)


def vplace(cond, A):
    r = np.concatenate([np.zeros(cond.shape), np.zeros(cond.shape), np.zeros(cond.shape)], axis=0)
    np.place(r[0], cond.ravel(), A[0].copy())
    np.place(r[1], cond.ravel(), A[1].copy())
    np.place(r[2], cond.ravel(), A[2].copy())
    return r

def mix(a, b, c):
    return b * c + a * (1 - c)


def apply_4d_to_3d(S, V):
    return np.dot(S[:3, :3], V) + S[:3, 3:4]


def transform(T, V):
    return apply_4d_to_3d(T, V)


def inverse_transform(T, V):
    return apply_4d_to_3d(np.linalg.inv(T), V)


def triangle_normal(V):
    A = V[:, 1:2] - V[:, 0:1]
    B = V[:, 2:3] - V[:, 0:1]
    return np.cross(A, B, axis=0)

