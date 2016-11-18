import vector_lib as vl
import numpy as np
import global_config
from geometry import Geometry, Primitive
import matplotlib.pyplot as plt

class Cylinder(Geometry, Primitive):
    def __init__(self, center, radius, length=np.inf, closed=True):
        super(Cylinder, self).__init__()
        self.center = np.reshape(center, (-1, 1))
        self.radius = radius
        self.length = length
        self.closed = closed

        self.z_min = -1.
        self.z_max = 1.

        self.T = vl.translation_matrix(center)
        self.S = vl.scale_matrix([radius, radius, length/2.])

    def intersection(self, O, D):
        O_hat = vl.inverse_transform(self.S, vl.inverse_transform(self.R, vl.inverse_transform(self.T, O)))
        D_hat = vl.inverse_transform(self.S, vl.inverse_transform(self.R, D))

        xy_mask = np.array([1., 1., 0.]).reshape(-1, 1)
        O_xy = O_hat * xy_mask
        D_xy = D_hat * xy_mask

        a = vl.vdot(D_xy, D_xy)
        b = 2 * vl.vdot(D_xy, O_xy)
        c = vl.vdot(O_xy, O_xy) - 1

        disc = (b**2) - (4 * a * c)
        t0 = (-b - np.sqrt(np.maximum(disc, 0.))) / (2 * a)
        t1 = (-b + np.sqrt(np.maximum(disc, 0.))) / (2 * a)

        zo = O_hat[2:3]
        zd = D_hat[2:3]
        z0 = (zo + zd*t0)
        z1 = (zo + zd*t1)

        z0_mask = (t0 > 0) & (z0 <= self.z_max) & (z0 >= self.z_min) & (disc > 0.)
        z1_mask = (t1 > 0) & (z1 <= self.z_max) & (z1 >= self.z_min) & (disc > 0.)

        # # Compute Cylinder Intersections
        t_cand0 = np.where(z0_mask, t0, global_config.MAX_DISTANCE)
        t_cand1 = np.where(z1_mask, t1, global_config.MAX_DISTANCE)
        t_cand_cyl = np.min(np.array([t_cand0, t_cand1]), axis=0)

        # # Compute Endcap Cylinder Intersections
        if self.closed is True:
            t3 = (self.z_min - zo)/zd
            t4 = (self.z_max - zo)/zd

            P_front = (O_xy + D_xy*t3)
            P_back = (O_xy + D_xy*t4)

            front_cap_hit_mask = (vl.vdot(P_front, P_front) <= 1.)
            back_cap_hit_mask = (vl.vdot(P_back, P_back) <= 1.)

            hit_mask = (z0_mask | z1_mask)

            t_front_cap = front_cap_hit_mask * t3
            t_back_cap = back_cap_hit_mask * t4
            t_side = t_cand_cyl
            
            t_front_cap[t_front_cap <= 0.] = global_config.MAX_DISTANCE
            t_back_cap[t_back_cap <= 0.] = global_config.MAX_DISTANCE
            t_side[t_side <= 0.] = global_config.MAX_DISTANCE
            
            t_out = np.min([t_side, t_front_cap, t_back_cap], axis=0)
            t_arg_out = np.argmin([t_side, t_front_cap, t_back_cap], axis=0)

            M_out = O_hat + D_hat * t_out 

            normal_cyl = hit_mask * (M_out * xy_mask)
            normal_front_cap = front_cap_hit_mask * np.array([0, 0, -1]).reshape(-1, 1)
            normal_back_cap = back_cap_hit_mask * np.array([0, 0, 1]).reshape(-1, 1)
            normals = np.sum([normal_cyl * (t_arg_out == 0),
                              normal_front_cap * (t_arg_out == 1),
                              normal_back_cap * (t_arg_out == 2)], axis=0)

            M_out = vl.transform(self.T, vl.transform(self.R, vl.transform(self.S, M_out)))
            N_out = vl.vnorm(vl.transform(self.R, vl.inverse_transform(self.S, normals)))
            return t_out, M_out, N_out
        
        else:
            t_out = t_cand_cyl
            M = (O_hat + D_hat * t_out)
            M_out = vl.transform(self.T, vl.transform(self.R, vl.transform(self.S, M)))
            N_out = vl.vnorm(vl.transform(self.R, vl.inverse_transform(self.S, M * xy_mask)))

            return t_out, M_out, N_out
