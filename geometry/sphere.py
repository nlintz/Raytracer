import vector_lib as vl
import numpy as np
import global_config
from geometry import Geometry, Primitive


class Sphere(Geometry, Primitive):
    def __init__(self, center, radius):
        super(Sphere, self).__init__()
        self.center = np.reshape(center, (-1, 1))
        self.radius = radius
        self.S = vl.scale_matrix([radius, radius, radius])
        self.T = vl.translation_matrix(center)

    def intersection(self, O, D):
        O_hat = vl.inverse_transform(self.S, vl.inverse_transform(self.R, vl.inverse_transform(self.T, O)))
        D_hat = vl.inverse_transform(self.S, vl.inverse_transform(self.R, D))
        a = vl.vdot(D_hat, D_hat)
        b = 2 * vl.vdot(O_hat, D_hat)
        c = vl.vdot(O_hat, O_hat) - 1
        disc = (b**2) - (4 * a * c)

        t0 = (-b - np.sqrt(np.maximum(disc, 0.))) / (2 * a)
        t1 = (-b + np.sqrt(np.maximum(disc, 0.))) / (2 * a)
        t = np.where((t0 > 0) & (t0 < t1), t0, t1)

        pred = (disc > 0.) & (t > 0.)
        t_out = np.where(pred, t, global_config.MAX_DISTANCE)
        M = (O_hat + D_hat * t_out)
        M_out = vl.transform(self.T, vl.transform(self.R, vl.transform(self.S, M)))
        N_out = vl.vnorm(vl.transform(self.R, vl.inverse_transform(self.S, M)))

        return t_out, M_out, N_out

    def normal_to(self, M):
        return vl.vnorm(vl.inverse_transform(self.S, vl.inverse_transform(self.T, M)))


class DiffuseSphere(Sphere):  # Test Class
    def __init__(self, center, radius, material):
        self.material = material
        super(DiffuseSphere, self).__init__(center, radius)


class ShinySphere(Sphere):  # Test Class
    def __init__(self, center, radius, material):
        self.material = material
        super(ShinySphere, self).__init__(center, radius)


if __name__ == "__main__":
    from camera import Camera
    import matplotlib.pyplot as plt
    import cv2

    camera = Camera(500, 500)
    T = vl.translation_matrix([0., 0., 0.])
    camera.translate(T)
    sp_y = 0.
    sp_z = 2.
    sp_r = 1.
    sp = Sphere([0.0, sp_y, sp_z], sp_r)
    # sp.intersection(np.array([0., 2., 0.]).reshape(-1, 1), np.array([0, 0., 1]).reshape(-1, 1))
    t, M, N = sp.intersection(camera.origin, camera.D)
    # plt.imshow(t.reshape(500, 500)); plt.show()
    plt.imshow(N.T.reshape(500, 500, 3)); plt.show()
    import ipdb; ipdb.set_trace()

    D = vl.vnorm(np.array([0, sp_y - 0.5, sp_z]).reshape(-1, 1))
    d = sp.intersection(np.array([0., 0., 0.]).reshape(-1, 1), D)

    _, Y, Z = (d*D + camera.origin).ravel()

    N = vl.vnorm(sp.normal_to(d*D + camera.origin))
    print N
    img = np.zeros((1000, 1000))
    cv2.circle(img, (int(100 * sp_y), int(100 * sp_z)), int(100 * sp_r), (1, 1, 1), thickness=2)
    cv2.line(img, (0, 0), (int(100 * Y), int(100 * Z)), (1, 1, 1), thickness=2)
    print (int(100 * Y), int(100 * Z)), (int(100 * N.ravel()[1]), int(100 * N.ravel()[2]))
    cv2.line(img, (int(100 * Y), int(100 * Z)), (int(100 * Y + 100 * N.ravel()[1]), int(100*Z + 100 * N.ravel()[2])), (1, 0, 1), thickness=2)
    plt.imshow(img); plt.show()
    import ipdb; ipdb.set_trace()

    import ipdb; ipdb.set_trace()

    # print sp.normal_to(d)
    plt.imshow(d.reshape(300, 400)); plt.show()
    # import ipdb; ipdb.set_trace()
    