import vector_lib as vl
import numpy as np
import global_config
from geometry import Geometry


class Plane(Geometry):
    def __init__(self, center, normal):
        self.center = np.reshape(center, (-1, 1))
        self.normal = np.reshape(normal, (-1, 1))
        self.T = vl.translation_matrix(center)


    def intersection(self, O, D):
        O_hat = vl.inverse_transform(self.T, O)
        D_hat = D
        t = vl.vdot(self.normal, -O_hat) / (vl.vdot(self.normal, D_hat))
        pred = (t > 0.)
        t_out = np.where(pred, t, global_config.MAX_DISTANCE)

        M = (O_hat + D_hat * t_out)
        M_out = vl.transform(self.T, M)
        N_out = np.ones_like(M_out) * self.normal

        return t_out, M_out, N_out
