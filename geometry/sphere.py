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
