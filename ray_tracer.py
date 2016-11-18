import numpy as np
import vector_lib as vl
import global_config

class RayTracer(object):
    def __init__(self, camera, scene, num_bounces=2, background_color=None):
        self.camera = camera
        self.scene = scene
        self.num_bounces = num_bounces
        if background_color is None:
            background_color = np.zeros((3, 1))
        self.background_color = np.reshape(background_color, (3, 1))

    def light(self, s, O, V, t, M, N, bounces):
        to_O = vl.vnorm(O - M)

        inside = (vl.vdot(V, N) > 0.)

        N_hat = np.where(inside, -N, N)

        ambient_color = self.color_ambient(s)
        diffuse_specular_color = self.color_diffuse_specular(s, O, M, N_hat, V)
        reflected_refracted_color = self.color_reflected_refracted(s, O, M, N_hat, V, inside, bounces)

        return ambient_color + diffuse_specular_color + reflected_refracted_color

    def color_ambient(self, s):
        return s.material.finish.ambient * s.material.surface_color

    def color_diffuse_specular(self, s, O, M, N, V):
        diffuse_color = np.zeros((3, 1))
        specular_color = np.zeros((3, 1))

        to_O = vl.vnorm(O - M)

        for light_source in self.scene.light_sources:
            to_L = vl.vnorm(light_source.geometry.center - M)
            shadow_mask = self.shadow_mask(light_source, M, N, to_L)

            diffuse_color = diffuse_color + self.color_diffuse(s, N, to_L, light_source.material.emission_color) * shadow_mask
            specular_color = specular_color + self.color_specular(s, N, V, to_L, to_O, light_source.material.emission_color) * shadow_mask
        return diffuse_color + specular_color

    def color_reflected_refracted(self, s, O, M, N, V, inside, bounces):
        to_O = vl.vnorm(O - M)

        reflect_color = np.zeros((3, 1))
        refract_color = np.zeros((3, 1))

        if (s.material.finish.transparent == False and s.material.finish.reflection == 0.) or bounces == self.num_bounces:
            return 0.

        reflect_amount = s.material.finish.reflection
        refract_amount = 0.
        
        if s.material.finish.transparent == True:
            reflect_amount = self.reflectance(s, N, V, inside)
            refract_amount = 1. - reflect_amount

        if np.any(reflect_amount > 0.):
            R = self.reflect(to_O, N)
            reflect_color = self.trace(s.geometry.intersection_point(M, N), R, bounces + 1) * reflect_amount

        if np.any(refract_amount > 0.):
            Rf = self.refract(s, V, N, inside)
            refract_color = refract_color + self.trace(s.geometry.intersection_point(M, -N), Rf, bounces + 1) * refract_amount

        return reflect_color + refract_color


    def color_diffuse(self, s, N, to_L, intensity):
        lambert_color = np.maximum(vl.vdot(N, to_L), 0.) * s.material.surface_color
        return lambert_color * intensity * s.material.finish.diffuse

    def color_specular(self, s, N, V, to_L, to_O, intensity):
        if s.material.finish.specular == 0.:
            return 0.
        else:
            R = self.reflect(to_L, N)
            specular_intensity = np.maximum(vl.vdot(to_O, R), 0.)
            specular_color = np.power(specular_intensity, 1./s.material.finish.roughness) * intensity * s.material.finish.specular
            if s.material.metallic == True:
                specular_color = specular_color * vl.vnorm(s.material.surface_color)
            return specular_color

    def reflect(self, V, N):
        return vl.vnorm(N * 2 * vl.vdot(V, N) - V)

    def refract(self, s, V, N, inside):
        n1 = np.where(inside, s.material.finish.ior, 1.)  # TODO don't hardcode air as n1
        n2 = np.where(inside, 1., s.material.finish.ior)
        n = n1 / n2
        cos_i = -vl.vdot(N, V)
        sin_t2 = n * n * (1.0 - cos_i * cos_i)

        cos_t = np.sqrt(1. - sin_t2)

        return vl.vnorm(V * n + N * (n * cos_i - cos_t))

    def reflectance(self, s, N, V, inside):
        n1 = np.where(inside, s.material.finish.ior, 1.)  # TODO don't hardcode air as n1
        n2 = np.where(inside, 1., s.material.finish.ior)
        n = n1 / n2
        cos_i = -vl.vdot(N, V)
        sin_t2 = n * n * (1.0 - cos_i * cos_i)

        cos_t = np.sqrt(1. - sin_t2)

        r_s = (n1 * cos_i - n2 * cos_t) / (n1 * cos_i + n2 * cos_t)
        r_p = (n2 * cos_i - n1 * cos_t) / (n2 * cos_i + n1 * cos_t)

        return np.where(sin_t2 > 1., 1., (r_s * r_s + r_p * r_p) / 2.0)


    def trace(self, O, D, bounces):
        distances_M_N = ([s.geometry.intersection(O, D) for s in self.scene.scene_elements])
        distances = np.array([x[0] for x in distances_M_N])
        M = np.array([x[1] for x in distances_M_N])
        N = np.array([x[2] for x in distances_M_N])

        min_distances = np.min(distances, axis=0)

        color = np.zeros((3, 1))

        all_hits = []
        # for (s, d, m, n) in zip(self.scene.scene_elements, distances, M, N):
        for (s, d, m, n) in zip(self.scene.scene_elements, distances, M, N):
            hits = ((d==min_distances) & (d != global_config.MAX_DISTANCE))
            all_hits.append(hits)
            if np.any(hits):
                _, hit_idxs = np.where(hits)
                V = D[:, hit_idxs]  # origin direction
                t = d[:, hit_idxs]  # distances
                M = m[:, hit_idxs]
                N = n[:, hit_idxs]

                if O.shape[1] > 1:
                    Oc = O[:, hit_idxs]
                else:
                    Oc = O
                lighting_color = self.light(s, Oc, V, t, M, N, bounces)
                color = vl.vplace(hits, lighting_color) + color
        intersections = np.minimum(np.sum(np.concatenate(all_hits),axis=0), 1.)
        color = np.where(intersections, color, self.background_color * np.ones((3, 1)))

        return color

    def shadow_mask(self, s, M, N, to_light):
        visible = 1.
        light_dists = [obj.geometry.intersect_light(M, N, to_light)[0] for obj in self.scene.scene_elements if obj is not s]
        if len(light_dists):
            light_nearest = np.min(light_dists, axis=0)
            point_to_light = np.sqrt(vl.vabs(M - s.geometry.center))
            shadow_mask = ((light_nearest == global_config.MAX_DISTANCE) | (light_nearest > point_to_light))
        else:
            shadow_mask = np.ones((1, M.shape[1])).astype(np.bool)
        return shadow_mask

    def render(self):
        traced = self.trace(self.camera.origin, self.camera.D, bounces=0)
        return traced_to_image(traced, self.camera.width, self.camera.height)


def traced_to_image(traced, w, h):
    # return traced.T.reshape(h, w, 3)
    img = np.minimum(1., traced).T.reshape(h, w, 3)
    img = (img * 255.).astype(np.uint8)
    # img = (img * 255.)
    return img

