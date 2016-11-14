import vector_lib as vl
import numpy as np
import global_config
from geometry import Geometry
import matplotlib.pyplot as plt


class Triangle(Geometry):
    def __init__(self, vertices, single_sided=1.):
        self.vertices = vertices
        self.normal = self.get_normal()
        self.single_sided = single_sided

        v0 = self.vertices[:, 0:1]
        v1 = self.vertices[:, 1:2]
        v2 = self.vertices[:, 2:3]

        self.v0 = v0
        self.v0v1 = v1 - v0
        self.v0v2 = v2 - v0

    def get_normal(self):
        return -vl.vnorm(vl.triangle_normal(self.vertices))

    def intersection(self, O, D):
        v0 = self.v0
        v0v1 = self.v0v1
        v0v2 = self.v0v2

        p_vec = np.cross(D, v0v2, axis=0)
        det = vl.vdot(v0v1, p_vec)

        # Check Parallel
        is_parallel = abs(det) < global_config.PARALLEL_EPSILON
        non_parallel_indices = np.where(np.logical_not(is_parallel))

        inv_det = np.ones_like(det) * global_config.MAX_DISTANCE
        inv_det[non_parallel_indices] = 1. / det[non_parallel_indices]

        # Check Backfacing
        backfacing = self.single_sided * (det > 0.)

        t_vec = (O - v0)
        u = vl.vdot(t_vec, p_vec) * inv_det
        u_mask = ((u < 0.) | (u > 1.))

        q_vec = np.cross(t_vec, v0v1, axis=0)
        v = vl.vdot(D, q_vec) * inv_det
        w_mask = (v < 0.) | ((u + v) > 1.)

        t = vl.vdot(v0v2, q_vec) * inv_det

        visible_mask = (np.logical_not(u_mask | w_mask)) & (t > 0.) & (np.logical_not(is_parallel)) & (np.logical_not(backfacing)) 
        
        t_out = t * visible_mask + global_config.MAX_DISTANCE * np.logical_not(visible_mask)
        N_out = np.zeros_like(t) + self.normal
        M_out = O + t * D
        if not self.single_sided:
            normal_flip = -1. * (vl.vdot(D*visible_mask, self.normal) > 0.)
            N_out = N_out * normal_flip
            # plt.imshow((N_out).T.reshape(500, 500, 3)); plt.show()
            # import ipdb; ipdb.set_trace()
            # print N_out.max(), N_out.min()
        return t_out, M_out, N_out


class DiffuseTriangle(Triangle):  # Test Class
    def __init__(self, vertices, material):
        self.material = material
        super(DiffuseTriangle, self).__init__(vertices)

class ShinyTriangle(Triangle):  # Test Class
    def __init__(self, vertices, material):
        self.material = material
        super(ShinyTriangle, self).__init__(vertices)
        self.transparency = 1.
        self.reflectivity = 0.9


if __name__ == "__main__":
    from camera import Camera
    import matplotlib.pyplot as plt
    
    camera = Camera(640, 480)
    T_ = vl.translation_matrix([0., 0., 0.])
    R_ = vl.rotation_matrix(np.pi, [0, 1, 0])
    camera.translate(T_)
    # camera.rotate(R_)

    v = np.array([[-0.5, -0.5, 1.5], [0.5, -0.5, 1.5], [0., 0.5, 1.5]]).T # front - front facing
    # v = np.array([[-0.5, -0.5, 1.5], [0., 0.5, 1.5], [0.5, -0.5, 1.5]]).T # front - back facing
    # v = np.array([[-0.5, -0.5, -1.5], [0.5, -0.5, -1.5], [0., 0.5, -1.5]]).T # back - front facing
    # v = np.array([[-0.5, -0.5, -1.5], [0., 0.5, -1.5], [0.5, -0.5, -1.5]]).T # back - back facing

    t = Triangle(v) 
    N = t.normal
    # d = t.intersection(camera.origin, vl.vnorm(np.array([[0., 1., 0.], [0., 0., 1.]]).T))
    d, u_, v_ = t.intersection(camera.origin, camera.D)
    color_channels = np.concatenate([u_, v_, 1-u_-v_])
    img = color_channels * (d<global_config.MAX_DISTANCE)

    # import ipdb; ipdb.set_trace()
    # import ipdb; ipdb.set_trace()
    # plt.imshow(img.T.reshape(480, 640, 3)); plt.show()

    img_out = (img.T.reshape(480, 640, 3) * 255).astype(np.uint8)
    plt.imshow(img_out); plt.show()
    import ipdb; ipdb.set_trace()
    # import ipdb; ipdb.set_trace()
    d = np.where(d<global_config.MAX_DISTANCE, d, 0.)
    plt.imshow(d.reshape(500, 500)); plt.show()
    # import ipdb; ipdb.set_trace()
    print p.intersection(np.array([0, 0, -1.]).reshape(-1, 1), vl.vnorm(np.array([[1., 1., 1.], [-1., -10., 1.], [0., 1., 0.]]).T))
    import ipdb; ipdb.set_trace()