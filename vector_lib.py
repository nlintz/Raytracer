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
    # Note - rotations are in the clockwise direction
    sina = math.sin(angle)
    cosa = math.cos(angle)
    direction = unit_vector(direction[:3])
    # rotation matrix around unit vector
    R = np.diag([cosa, cosa, cosa])
    R += np.outer(direction, direction) * (1.0 - cosa)
    direction *= sina
    R += np.array([[ 0.0,         -direction[2],  direction[1]],
                      [ direction[2], 0.0,          -direction[0]],
                      [-direction[1], direction[0],  0.0]])
    M = np.identity(4)
    M[:3, :3] = R
    if point is not None:
        # rotation not around origin
        point = np.array(point[:3], dtype=np.float64, copy=False)
        M[:3, 3] = point - np.dot(R, point)
    return M


def unit_vector(data, axis=None, out=None):
    if out is None:
        data = np.array(data, dtype=np.float64, copy=True)
        if data.ndim == 1:
            data /= math.sqrt(np.dot(data, data))
            return data
    else:
        if out is not data:
            out[:] = np.array(data, copy=False)
        data = out
    length = np.atleast_1d(np.sum(data*data, axis))
    np.sqrt(length, length)
    if axis is not None:
        length = np.expand_dims(length, axis)
    data /= length
    if out is None:
        return data


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


if __name__ == "__main__":
    V = np.array([0., 5., 5.]).reshape(3, 1)
    S = scale_matrix([2., 2., 2.])
    # T = rotation_matrix(np.pi/2, np.array([1, 0, 0]), np.array([0., 5., 5.]))
    print transform(S, V)
