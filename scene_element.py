import numpy as np
import vector_lib as vl

class SceneElement(object):
    def __init__(self, geometry, material):
        self.geometry = geometry
        self.material = material


class LightSource(object):
    def __init__(self, emission_color, intensity):
        self.emission_color = np.array(emission_color).reshape(3, 1)
        self.intensity = intensity


class LightSourcePoint(LightSource):
    def __init__(self, center, emission_color=np.ones((3, 1)), intensity=1.):
        super(LightSourcePoint, self).__init__(emission_color, intensity)
        self.center = np.reshape(center, (3, 1))


class LightSourceDirectional(LightSource):
    def __init__(self, direction, emission_color=np.ones((3, 1)), intensity=1.):
        super(LightSourceDirectional, self).__init__(emission_color, intensity)
        self.direction = vl.vnorm(np.reshape(direction, (3, 1)))
