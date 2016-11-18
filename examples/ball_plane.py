import numpy as np
from camera import Camera
from ray_tracer import RayTracer
from scene_element import SceneElement
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
    mat_ls = Material([0., 0., 0.], emission_color=np.array([2., 2., 2.]).reshape(-1, 1))
    base_finish = SoftDull
    base_finish.reflection = 0.5
    mat_base = Material(colors.P_Chrome1, finish=base_finish)

    mat_s1 = Material(colors.P_Brass3)

    se_ls = SceneElement(Sphere([5., 20., 30.], 2.), mat_ls)
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