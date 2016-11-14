import numpy as np
import vector_lib as vl
import global_config
from shaders import LambertShader, BlinnPhongShader


class RayTracer(object):
    def __init__(self, camera, scene, num_bounces=2, background_color=None):
        self.camera = camera
        self.scene = scene
        self.num_bounces = num_bounces
        if background_color is None:
            background_color = np.ones((3, 1)) * 2.
        self.background_color = np.reshape(background_color, (3, 1))


    def light(self, s, O, V, t, M, N, bounces):
        to_O = vl.vnorm(O - M)
        surface_color = np.zeros((3, 1))

        inside = (vl.vdot(V, N) > 0.)

        N_hat = np.where(inside, -N, N)

        if (s.material.transparency > 0. or s.material.reflectivity > 0.) and bounces < self.num_bounces:
            facing_ratio = -vl.vdot(V, N_hat)
            fresnel_coeff = vl.mix(np.power(1 - facing_ratio, 3), 1, 0.1)
            
            reflection_d = vl.vnorm(V - N_hat * 2 * vl.vdot(V, N_hat))
            reflection_color = self.trace(s.geometry.intersection_point(M, N_hat), reflection_d, bounces + 1)

            refraction_color = np.zeros((3, 1))
            if (s.material.transparency > 0.):
                ior = 1.1  # TODO dont hardcord index of refraction
                eta = np.where(inside, ior, 1./ior)
                cosi = -vl.vdot(N_hat, V)
                k = 1 - eta * eta * (1 - cosi * cosi)
                refraction_d = vl.vnorm(V * eta + N_hat * (eta * cosi - np.sqrt(k)))
                refraction_color = self.trace(s.geometry.intersection_point(M, -N_hat), refraction_d, bounces + 1)

            surface_color = (fresnel_coeff * reflection_color +
                             (1 - fresnel_coeff) * refraction_color * s.material.transparency) * s.material.surface_color

        else:
            for light_source in self.scene.light_sources:
                to_L = vl.vnorm(light_source.geometry.center - M)
                lambert_color = LambertShader()(s.material, N, to_L)
                shadow_mask = self.shadow_mask(light_source, M, N, to_L)
                surface_color = surface_color + lambert_color * light_source.material.emission_color * shadow_mask

                # ambient_color = 0.2 * s.material.surface_color * light_source.material.emission_color
                # surface_color = surface_color + ambient_color
                # surface_color = surface_color + scene_element.material.ambient
                # surface_color = surface_color + 0.2


        return surface_color + s.material.emission_color

    def trace(self, O, D, bounces):
        distances_M_N = ([s.geometry.intersection(O, D) for s in self.scene.scene_elements])
        distances = np.array([x[0] for x in distances_M_N])
        M = np.array([x[1] for x in distances_M_N])
        N = np.array([x[2] for x in distances_M_N])

        min_distances = np.min(distances, axis=0)

        color = np.zeros((3, 1))

        all_hits = []
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

