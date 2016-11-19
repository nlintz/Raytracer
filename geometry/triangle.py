import vector_lib as vl
import numpy as np
import global_config
from geometry import Geometry


class Triangle(Geometry):
    def __init__(self, vertices, single_sided=1., clockwise=False):
        self.vertices = vertices
        self.single_sided = single_sided
        self.normal = self.get_normal()
        if clockwise:
            self.normal = -self.normal

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
        return t_out, M_out, N_out
