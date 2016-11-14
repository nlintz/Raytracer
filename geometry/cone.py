import vector_lib as vl
import numpy as np
import global_config
from geometry import Geometry, Primitive

class Cone(Geometry, Primitive):
    def __init__(self, center, radius, length=np.inf, closed=True):
        super(Cone, self).__init__()
        self.center = np.reshape(center, (-1, 1))
        self.radius = radius
        self.length = length
        self.closed = closed

        self.z_min = 0
        self.z_max = 1.

        self.T = vl.translation_matrix(center)
        self.S = vl.scale_matrix([radius, radius, length])

    def intersection(self, O, D):
        O_hat = vl.inverse_transform(self.S, vl.inverse_transform(self.R, vl.inverse_transform(self.T, O)))
        D_hat = vl.inverse_transform(self.S, vl.inverse_transform(self.R, D))

        xy_mask = np.array([1., 1., 0.]).reshape(-1, 1)
        z_mask = np.array([0., 0., 1.]).reshape(-1, 1)
        O_xy = O_hat * xy_mask
        O_z = O_hat * z_mask
        D_xy = D_hat * xy_mask
        D_z = D_hat * z_mask

        a = vl.vdot(D_xy, D_xy) - vl.vdot(D_z, D_z)
        b = (2 * vl.vdot(D_xy, O_xy)) - (2 * vl.vdot(D_z, O_z))
        c = vl.vdot(O_xy, O_xy) - vl.vdot(O_z, O_z)

        disc = (b**2) - (4 * a * c)
        t0 = (-b - np.sqrt(np.maximum(disc, 0.))) / (2 * a)
        t1 = (-b + np.sqrt(np.maximum(disc, 0.))) / (2 * a)

        zo = O_hat[2:3]
        zd = D_hat[2:3]
        z0 = (zo + zd*t0)
        z1 = (zo + zd*t1)
        z0_mask = (t0 > 0) & (z0 <= self.z_max) & (z0 >= self.z_min) & (disc > 0.)
        z1_mask = (t1 > 0) & (z1 <= self.z_max) & (z1 >= self.z_min) & (disc > 0.)

        # Compute Cone Intersections
        t_cand0 = np.where(z0_mask, t0, global_config.MAX_DISTANCE)
        t_cand1 = np.where(z1_mask, t1, global_config.MAX_DISTANCE)

        t_cand_cone = np.min(np.array([t_cand0, t_cand1]), axis=0)


        # Compute Endcap Cone Intersections
        if self.closed is True:
            t4 = (self.z_max - zo)/zd
            P_back = (O_xy + D_xy*t4)

            back_cap_hit_mask = (vl.vdot(P_back, P_back) <= 1.)
            hit_mask = (z0_mask | z1_mask)

            t_side = t_cand_cone
            t_back_cap = back_cap_hit_mask * t4
            
            t_back_cap[t_back_cap <= 0.] = global_config.MAX_DISTANCE
            t_side[t_side <= 0.] = global_config.MAX_DISTANCE

            t_out = np.min([t_side, t_back_cap], axis=0)
            t_arg_out = np.argmin([t_side, t_back_cap], axis=0)
            M_out = O_hat + D_hat * t_out 

            M_norm = (M_out * xy_mask)
            M_norm[2, :] = -1.
            normal_cone = M_norm
            normal_back_cap = back_cap_hit_mask * np.array([0, 0, 1]).reshape(-1, 1)
            normals = np.sum([normal_cone * (t_arg_out == 0), normal_back_cap * (t_arg_out == 1)], axis=0)

            M_out = vl.transform(self.T, vl.transform(self.R, vl.transform(self.S, M_out)))
            N_out = vl.vnorm(vl.transform(self.R, vl.inverse_transform(self.S, normals)))
            return t_out, M_out, N_out

        
        else:
            t_out = t_cand_cone
            M = (O_hat + D_hat * t_out)
            M_norm = M * xy_mask
            M_norm[2, :] = -1.

            M_out = vl.transform(self.T, vl.transform(self.R, vl.transform(self.S, M)))
            N_out = vl.vnorm(vl.transform(self.R, vl.inverse_transform(self.S, M_norm)))
            return t_out, M_out, N_out


class DiffuseCone(Cone):  # Test Class
    def __init__(self, center, radius, length, material):
        super(DiffuseCone, self).__init__(center, radius, length, closed=True)
        self.material = material
        # self.R = vl.rotation_matrix(np.pi + 0.1, [0, 1, 0], [0, 0, 0])
        # self.R = vl.rotation_matrix(np.pi/2, [0, 1, 0], [0, 0, 0])
        # self.R = vl.rotation_matrix(-np.pi / 6, [0, 1, 0], [0, 0, 0])
        # self.R = vl.rotation_matrix(-3 * np.pi/4 + 0.1, [1, 0, 0], [0, 0, 0])
        self.R = vl.rotation_matrix(np.pi + np.pi/4, [0, 1, 0], [0, 0, 0])
        # self.R = vl.rotation_matrix(np.pi - np.pi/3, [0, 1, 0], [0, 0, 0])

if __name__ == "__main__":
    from camera import Camera
    import matplotlib.pyplot as plt

    camera = Camera(500, 500)
    camera.translate(vl.translation_matrix([0, 0, 0.]))
    # camera = Camera(10, 10)

    # cone = Cone([0, 0, 5], radius=1., length=1, closed=True)
    # cone = Cone([0, 0, 5], radius=2., length=1, closed=False)
    # cone = Cone([0, 0, 5], radius=2., length=2., closed=True)
    cone = Cone([0, 0, 5], radius=1., length=1., closed=True)

    # R = vl.rotation_matrix(3 * np.pi/4, [0, 1, 0], [0, 0, 0.])
    # R = vl.rotation_matrix(-np.pi/4, [1, 0, 0], [0, 0, 0.])
    # R = vl.rotation_matrix(np.pi/2, [0, 1, 0], [0, 0, 0.])
    # R = vl.rotation_matrix(3 * np.pi/2 - .65, [1, 0, 0], [0, 0, 0.])
    # R = vl.rotation_matrix(-2.9 * np.pi/4, [0, 1, 0], [0, 0, 0.])
    # R = vl.rotation_matrix(3 * np.pi/4, [0, 1, 0], [0, 0, 0])
    R = vl.rotation_matrix(np.pi/2 + np.pi/4 , [0, 1, 0], [0, 0, 0])
    cone.R = R

    # t, m, n = cone.intersection(camera.origin, np.array([0., 0., 1.]).reshape(-1, 1))
    # import ipdb; ipdb.set_trace()
    t, m, n = cone.intersection(camera.origin, camera.D)
    # plt.imshow((d * (d<global_config.MAX_DISTANCE)).reshape(500, 500)); plt.show()
    # plt.imshow(((n * (t<global_config.MAX_DISTANCE)).T).reshape(camera.height, camera.width, 3)); plt.show()
    plt.imshow(((m * (t<global_config.MAX_DISTANCE)).T).reshape(camera.height, camera.width, 3)); plt.show()
    # plt.imshow(((t * (t<global_config.MAX_DISTANCE)).T).reshape(camera.height, camera.width)); plt.show()
    import ipdb; ipdb.set_trace()