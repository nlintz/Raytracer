import vector_lib as vl
import numpy as np
import global_config
from geometry import Geometry


class Plane(Geometry):
    def __init__(self, center, normal, reflectivity=0.5, transparency=0.5):
        self.center = np.reshape(center, (-1, 1))
        self.normal = np.reshape(normal, (-1, 1))
        self.T = vl.translation_matrix(center)
        self.reflectivity = reflectivity
        self.transparency = transparency

    def intersection(self, O, D):
        O_hat = vl.inverse_transform(self.T, O)
        D_hat = D
        t = vl.vdot(self.normal, -O_hat) / (vl.vdot(self.normal, D_hat))
        pred = (t > 0.)
        return np.where(pred, t, global_config.MAX_DISTANCE)

    def normal_to(self, M):
        return vl.vnorm(vl.inverse_transform(self.T, self.normal))  # isnt this just the normal?


class DiffusePlane(Plane):  # Test Class
    def __init__(self, center, normal, material):
        self.material = material
        super(DiffusePlane, self).__init__(center, normal, reflectivity=0., transparency=0.)
