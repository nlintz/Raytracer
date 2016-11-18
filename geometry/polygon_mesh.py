import numpy as np
from geometry import Geometry, Mesh
from geometry.polygon import Polygon
import vector_lib as vl
import global_config


class PolyMesh(Geometry, Mesh):
    def __init__(self, n_faces, face_index, verts_index, vertices):
        self.P = []

        k = 0
        for i in range(n_faces):
            self.P.append(Polygon(vertices[:, verts_index[k:k+face_index[i]]]))
            k += face_index[i]

    def intersection(self, O, D):
        intersections = np.concatenate([polygon.intersection(O, D)[0] for polygon in self.P])
        intersection = np.min(intersections, axis=0, keepdims=True)
        polygon_indexes = np.argmin(intersections, axis=0)

        t_out = intersection
        N_out = np.concatenate([self.P[i].normal for i in polygon_indexes], axis=1)
        M_out = O + t_out * D
        return t_out, M_out, N_out


class Cube(PolyMesh):
    def __init__(self, center, length=1.):
        vertices = np.array([[-0.5, -0.5, -0.5], [0.5, -0.5, -0.5], [0.5, -0.5, 0.5], [-0.5, -0.5, 0.5],
                             [-0.5, 0.5, -0.5], [0.5, 0.5, -0.5], [0.5, 0.5, 0.5], [-0.5, 0.5, 0.5]]).T
        T = vl.translation_matrix(center)
        S = vl.scale_matrix([length, length, length])
        vertices = [vl.transform(T, vl.transform(S, vertices[:, i:i+1])) for i in range(vertices.shape[1])]
        vertices = np.concatenate(vertices, axis=1)
        super(Cube, self).__init__(6, [4, 4, 4, 4, 4, 4], [0, 3, 2, 1, # bottom
                                                           4, 5, 6, 7, # top
                                                           0, 4, 7, 4, # left
                                                           1, 2, 6, 5, # right
                                                           0, 1, 5, 4, # front
                                                           3, 7, 6, 2  # back
                                                           ], vertices)