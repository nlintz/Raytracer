import vector_lib as vl
import numpy as np
import global_config
from geometry import Geometry


class Polygon(Geometry):
    def __init__(self, vertices, single_sided=1.):
        self.vertices = vertices
        self.normal = self.get_normal()
        self.single_sided = single_sided

    def get_normal(self):
        return -vl.vnorm(vl.triangle_normal(self.vertices))

    def inside_out_test(self, P):
        P_min_V = P[:, None, :] - self.vertices[:, :, None]
        N = self.normal

        edges = [self.vertices[:, i:i+1] - self.vertices[:, i-1:i] for i in range(1, self.vertices.shape[1])]
        edges.append(self.vertices[:, 0:1] - self.vertices[:, -1:])
        edges = np.concatenate(edges, axis=1)
        W = np.cross(edges[:, :, None], P_min_V, axis=0)
        K = vl.vdot(N[:, :, None], W)
        inside = np.prod((K <= 0.), axis=1)
        return inside

    def intersection(self, O, D):
        D_hat = vl.vdot(self.normal, self.vertices[:, :1] - O)

        num = D_hat
        den = vl.vdot(self.normal, D)

        normal_to_d = vl.vdot(self.normal, D)

        # Check if parallel
        is_parallel = (abs(normal_to_d) < global_config.PARALLEL_EPSILON)
        non_parallel_indices = np.where(np.logical_not(is_parallel))

        # Check Back Facing
        back_facing = normal_to_d > 0.
        t_hat = np.zeros([1, den.shape[1]])
        if O.shape[1] > 1:
            num = num[non_parallel_indices]
        t_hat[non_parallel_indices] = num / den[non_parallel_indices]

        P = O + D * t_hat
        iot = self.inside_out_test(P)

        visible_mask = iot & (t_hat > 0.) & (np.logical_not(is_parallel)) & (np.logical_not(back_facing * self.single_sided))

        t = t_hat * visible_mask + global_config.MAX_DISTANCE * np.logical_not(visible_mask)
        t_out = t
        N_out = np.zeros_like(t) + self.normal
        M_out = O + t * D
        return t_out, M_out, N_out


class DiffusePolygon(Polygon):  # Test Class
    def __init__(self, vertices, material):
        self.material = material
        super(DiffusePolygon, self).__init__(vertices)

class ShinyPolygon(Polygon):  # Test Class
    def __init__(self, vertices, material):
        self.material = material
        super(ShinyPolygon, self).__init__(vertices)
        self.transparency = 1.
        self.reflectivity = 0.9


if __name__ == "__main__":
    from camera import Camera
    import matplotlib.pyplot as plt
    
    camera = Camera(500, 500)
    T_ = vl.translation_matrix([0., 0., 0.])
    camera.translate(T_)

    # v = np.array([[-1, -1, 2], [1, -1, 2], [1., 1., 2.], [0., 2., 2.], [-1., 1., 2.]]).T
    # v = np.array([[-0.5, -0.5, 1.5], [0.5, -0.5, 1.5], [0.5, 0.5, 1.5], [-0.5, 0.5, 1.5]]).T
    v = np.array([[-1, -1, 2.], [1, -1, 2.], [0., 1., 2.]]).T
    # v = np.array([[-1, -1, 2], [1, -1, 2], [0., 1., 2.]]).T
    p = Polygon(v)
    N = p.normal

    d = p.intersection(camera.origin, camera.D)
    d = np.where(d<global_config.MAX_DISTANCE, d, 0.)
    plt.imshow(d.reshape(500, 500)); plt.show()
    # import ipdb; ipdb.set_trace()
    print p.intersection(np.array([0, 0, -1.]).reshape(-1, 1), vl.vnorm(np.array([[1., 1., 1.], [-1., -10., 1.], [0., 1., 0.]]).T))
    import ipdb; ipdb.set_trace()