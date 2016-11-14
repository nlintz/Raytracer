import numpy as np

class Material(object):
    def __init__(self, surface_color, reflectivity, transparency, emission_color=np.zeros((3, 1))):
        self.surface_color = np.reshape(surface_color, (-1, 1))
        self.reflectivity = reflectivity
        self.transparency = transparency
        self.emission_color = emission_color
