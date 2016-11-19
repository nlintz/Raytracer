import numpy as np
from camera import Camera
from ray_tracer import RayTracer
from scene_element import SceneElement, LightSourcePoint, LightSourceDirectional
from scene import Scene
from geometry.sphere import Sphere
from geometry.plane import Plane
from material import Material, Finish
from material.finishes import SoftDull
import material.colors as colors
import matplotlib.pyplot as plt


def render():
    (w, h) = (640, 480)
    camera = Camera(w, h, fov=np.pi / 6)

    # Materials
    base_finish = SoftDull
    base_finish.reflection = 0.1
    mat_base = Material(colors.P_Chrome3, finish=base_finish)

    mat_s1 = Material(colors.P_Brass3)

    se_ls = LightSourcePoint([-1., 5., 35.], intensity=200., emission_color=[2., 2., 2.])
    se_base = SceneElement(Plane([0.0, -4.0, 20.], [0., 1., 0.]), mat_base)

    se_s1 = SceneElement(Sphere([0., 0., 40.], 4.), mat_s1)

    scene = Scene([se_ls, se_base, se_s1])
    # scene = Scene([se_ls, se_base])

    # Render
    rt = RayTracer(camera, scene)
    traced = rt.render()
    plt.imshow(traced); plt.show()


if __name__ == "__main__":
    render()