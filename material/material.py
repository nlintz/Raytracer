import numpy as np

class Finish(object):
    def __init__(self, diffuse=0.6, ambient=0.2, specular=0., roughness=0.0005,
                 ior=1., reflection=0., transparent=False):
        self.diffuse = diffuse
        self.ambient = ambient
        self.specular = specular
        self.roughness = roughness
        self.ior = ior
        self.reflection = reflection
        self.transparent = transparent

class Material(object):
    def __init__(self, surface_color, finish=Finish(),
                 metallic=False):
        self.surface_color = np.reshape(surface_color, (-1, 1))
        self.finish = finish
        self.metallic = metallic


