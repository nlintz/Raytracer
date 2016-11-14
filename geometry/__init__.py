import global_config
import vector_lib as vl
import numpy as np


class Geometry(object):
    def intersection(self, O, D):
        raise NotImplementedError()

    def intersection_point(self, M, N):
        return M + N * global_config.NUDGE_EPSILON

    def intersect_light(self, M, N, to_l):
        return self.intersection(self.intersection_point(M, N), to_l)


class Primitive(object):
    def __init__(self):
        self.T = np.eye(4)
        self.R = np.eye(4)
        self.S = np.eye(4)

    def translate(self, T):
        self.T = np.dot(self.T, vl.translation_matrix(T))

    def rotate(self, theta, direction, point=None):
        self.R = np.dot(self.R, vl.rotation_matrix(theta, direction, point))

    def scale(self, S):
        self.S = np.dot(self.S, vl.scale_matrix(S))


class Mesh(object):
    def translate(self, T):
        TM = vl.translation_matrix(T)
        for P in self.P:
            new_vertices = [vl.transform(TM, P.vertices[:, i:i+1]) for i in range(P.vertices.shape[1])]
            P.vertices = np.concatenate(new_vertices, axis=1)

    def rotate(self, theta, direction, point=None):
        # TODO - SCALE AROUND CENTER
        RM = vl.rotation_matrix(theta, direction, point)
        for P in self.P:
            new_vertices = [vl.transform(RM, P.vertices[:, i:i+1]) for i in range(P.vertices.shape[1])]
            P.vertices = np.concatenate(new_vertices, axis=1)
            P.normal = P.get_normal()


    def scale(self, S):
        # TODO - SCALE AROUND CENTER
        SM = vl.scale_matrix(S)
        for P in self.P:
            new_vertices = [vl.transform(SM, P.vertices[:, i:i+1]) for i in range(P.vertices.shape[1])]
            P.vertices = np.concatenate(new_vertices, axis=1)
            P.normal = P.get_normal()
