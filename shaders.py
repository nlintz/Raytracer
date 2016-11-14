import numpy as np
import vector_lib as vl

# Diffuse Shaders

class LambertShader(object):
    def __call__(self, obj, N, to_light):
        return np.maximum(vl.vdot(N, to_light), 0.) * obj.surface_color

class BlinnPhongShader(object):
    def __call__(self, N, to_light, to_origin):
        phong = vl.vdot(N, vl.vnorm(to_light + to_origin))
        return np.ones((3, 1)) * (np.clip(phong, 0., 1.)**50)
