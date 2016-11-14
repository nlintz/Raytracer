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



class DiffuseCube(Cube):  # Test Class
    def __init__(self, center, length, material):
        self.material = material
        super(DiffuseCube, self).__init__(center, length=length)

class ShinyCube(Cube):  # Test Class
    def __init__(self, center, length, material):
        self.material = material
        super(ShinyCube, self).__init__(center, length=length)
        # self.reflectivity = 0.5
        # self.transparency = 1.


if __name__ == "__main__":
    from camera import Camera
    import matplotlib.pyplot as plt

    camera = Camera(500, 500)
    camera.translate(vl.translation_matrix([0., 0., -5.]))
    # camera.rotate(vl.rotation_matrix(np.pi, [0, 1, 0], [0., 0., 5.]))
    p = PolyMesh(2, [4, 4], [0, 1, 2, 3, 0, 3, 4, 5], np.array([[-5, -5, 5], [5, -5, 5], [5, -5, -5],
                                                                [-5, -5, -5], [-5, 5, -5], [-5, 5, 5]]).T)
    # c = Cube([5, -5, 10])
    # c = Cube([0., 0., 0.])
    ps = PolySphere([0., 0., 0.], 5)
    # print c.intersection(np.array([0., 0., 0.]).reshape(-1, 1), np.array([0., 0., 1.]).reshape(-1, 1))
    # print c.intersection(np.array([0., 0., -1.]).reshape(-1, 1), np.array([0., 1., 0.]).reshape(-1, 1))
    # print c.intersection(np.array([0., 0., -1.]).reshape(-1, 1), np.array([[0., 1., 0.], [0., 0., 1.]]).T)
    # O = np.array([[0., 0., -1.], [0., 0., -1.], [0., 0., 1.]]).T
    # D = np.array([[0., 1., 0.], [0., 0., 1.], [0., 0., -1.]]).T
    t, m, n = ps.intersection(camera.origin, camera.D)
    # plt.imshow((d * (d<global_config.MAX_DISTANCE)).reshape(500, 500)); plt.show()
    plt.imshow(((m * (t<global_config.MAX_DISTANCE)).T).reshape(camera.height, camera.width, 3)); plt.show()
    # plt.imshow(((m * (t<global_config.MAX_DISTANCE)).T).reshape(camera.height, camera.width, 3)); plt.show()

    # print ps.normal_to(O + D*d)
    import ipdb; ipdb.set_trace()