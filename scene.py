import vector_lib as vl
from scene_element import LightSourcePoint, LightSourceDirectional


class Scene(object):
    def __init__(self, scene_elements):
        self.scene_elements = []
        self.light_sources = []

        for scene_element in scene_elements:

            if isinstance(scene_element, LightSourcePoint):
                self.light_sources.append(scene_element)
            elif isinstance(scene_element, LightSourceDirectional):
                self.light_sources.append(scene_element)
            else:
                self.scene_elements.append(scene_element)
