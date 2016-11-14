import vector_lib as vl


class Scene(object):
    def __init__(self, scene_elements):
        self.scene_elements = scene_elements
        self.light_sources = []

        for scene_element in self.scene_elements:
            if vl.vabs(scene_element.material.emission_color) > 0.:
                self.light_sources.append(scene_element)
