import numpy as np
from geometry import Geometry
from geometry.triangle import Triangle
import vector_lib as vl
import global_config


class TriangleMesh(Geometry):
    def __init__(self, n_faces, face_index, verts_index, vertices, single_sided=1):
        self.P = []
        k = 0
        max_vert_index = 0
        num_tris = 0
        for i in range(n_faces):
            num_tris += face_index[i] - 2
            for j in range(face_index[i]):
                if verts_index[k + j] > max_vert_index:
                    max_vert_index = verts_index[k + j]
            k += face_index[i]
        max_vert_index += 1

        P = []

        for i in range(max_vert_index):
            P.append(vertices[:, i:i+1])

        l = 0
        k = 0
        tris_index = []
        for i in range(n_faces):
            for j in range(face_index[i] - 2):
                tris_index.append(verts_index[k])
                tris_index.append(verts_index[k + j + 1])
                tris_index.append(verts_index[k + j + 2])
            k += face_index[i]

        j = 0
        for i in range(num_tris):
            V = []
            V.append(P[tris_index[j]])
            V.append(P[tris_index[j + 1]])
            V.append(P[tris_index[j + 2]])
            self.P.append(Triangle(np.concatenate(V, axis=1), single_sided=0.))
            j += 3

    def intersection(self, O, D):
        TMNS = [P.intersection(O, D) for P in self.P]
        ts = [t[0] for t in TMNS]
        ms = [m[1] for m in TMNS]
        ns = np.array([n[2] for n in TMNS])
        intersections = np.concatenate(ts)
        intersection = np.min(intersections, axis=0, keepdims=True)

        triangle_indexes = np.argmin(intersections, axis=0)
        t_out = intersection
        N_out = np.array([ns[i, :, j] for j, i in enumerate(triangle_indexes)]).T  # TODO: YOU CAN WRITE THIS BETTER
        M_out = O + t_out * D
        return t_out, M_out, N_out

class TriSphere(TriangleMesh):
    def __init__(self, center, divs, radius=1.):
        num_vertices = (divs - 1) * divs + 2
        u = -np.pi / 2
        v = -np.pi
        du = np.pi / divs
        dv = 2 * np.pi / divs
        st = []
        P = [[0., -radius, 0.]]
        N = [[0., -radius, 0.]]
        for i in range(divs - 1):
            u += du
            v = -np.pi
            for j in range(divs):
                x = radius * np.cos(u) * np.cos(v)
                y = radius * np.sin(u)
                z = radius * np.cos(u) * np.sin(v)
                P.append([x, y, z])
                N.append([x, y, z])
                st.append([u / np.pi + 0.5, v * 0.5 / np.pi + 0.5])
                v += dv
        P.append([0., radius, 0.])
        N.append([0., radius, 0.])
        npolys = divs * divs

        face_index = []
        verts_index = [0 for _ in range((6 + (divs - 1) * 4) * divs)]
        vid = 1
        l = 0
        num_v = 0
        for i in range(divs):
            for j in range(divs):
                if i == 0:
                    face_index.append(3)
                    verts_index[l] = (0)
                    verts_index[l + 1] = (j + vid)
                    if j == (divs - 1):
                        verts_index[l + 2] = vid
                    else:
                        verts_index[l + 2] = (j + vid + 1)
                    l += 3
                elif i == (divs - 1):
                    face_index.append(3)
                    verts_index[l] = (j + vid + 1 - divs)
                    verts_index[l + 1] = (vid + 1)
                    if j == (divs - 1):
                        verts_index[l + 2] = (vid + 1 - divs)
                    else:
                        verts_index[l + 2] = (j + vid + 2 - divs)
                    l += 3
                else:
                    face_index.append(4)
                    verts_index[l] = (j + vid + 1 - divs)
                    verts_index[l + 1] = (j + vid + 1)
                    if j == (divs - 1):
                        verts_index[l + 2] = (vid + 1)
                    else:
                        verts_index[l + 2] = (j + vid + 2)
                    if j == (divs - 1):
                        verts_index[l + 3] = (vid + 1 - divs)
                    else:
                        verts_index[l + 3] = (j + vid + 2 - divs)
                    l += 4
                # print i, j
                num_v += 1
            vid = num_v
        

        P = np.array(P).T
        P = vl.transform(vl.translation_matrix(center), P)
        super(TriSphere, self).__init__(npolys, face_index, verts_index, P)

class DiffuseTriangleSphere(TriSphere):
    def __init__(self, center, divs, radius, material):
        super(DiffuseTriangleSphere, self).__init__(center, divs, radius)
        self.material = material


class DiffuseTriangleMesh(TriangleMesh):  # Test Class
    def __init__(self, center, divs, radius, material):
        super(DiffuseTriangleMesh, self).__init__(center, divs, radius)
        self.material = material


if __name__ == "__main__":
    from camera import Camera
    import matplotlib.pyplot as plt

    camera = Camera(500, 500)
    # camera = Camera(10, 10)
    # T = vl.translation_matrix([5, 0, -1])
    T = vl.translation_matrix([0, 0, -5])
    # R = vl.rotation_matrix(np.pi / 4, [1, 0, 0])
    # R = vl.rotation_matrix(np.pi / 4, [0, 1, 0])
    # R = vl.rotation_matrix(np.pi / 10, [0, 1, 0])
    # R = vl.rotation_matrix(np.pi, [0, 1, 0])
    # camera.rotate(R)
    # camera.translate(T)
    # import ipdb; ipdb.set_trace()
    # import ipdb; ipdb.set_trace()

    face_idx = [3, 3]
    vertices = np.array([[1., -1., 1.], [1., 0., 1.], [0., -1., 1.],
                          [1., -1., 2.]]).T
    verts_idx = [0, 1, 2, 0, 3, 1]

    t_sphere = TriSphere([0., 0., 10.], 5, 1.)
    t, m, n = t_sphere.intersection(camera.origin, camera.D)
    # d, t_indices = t_mesh.intersection(camera.origin, camera.D)
    # normal_to = t_mesh.normal_to(None, t_indices)
    
    # plt.imshow((t * (t<global_config.MAX_DISTANCE)).reshape(500, 500)); plt.show()
    # plt.imshow((n * (t<global_config.MAX_DISTANCE)).T.reshape(500, 500, 3)); plt.show()
    # import ipdb; ipdb.set_trace()