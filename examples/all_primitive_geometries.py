import numpy as np
from camera import Camera
from ray_tracer import RayTracer
from scene_element import SceneElement
from scene import Scene

from geometry.cone import Cone
from geometry.cylinder import Cylinder
from geometry.sphere import Sphere
from geometry.plane import Plane

from material import Material, Finish
from material.finishes import SoftDull, VeryHardPolished
import material.colors as colors
import matplotlib.pyplot as plt


def render():
    (w, h) = (640, 480)
    camera = Camera(w, h, fov=np.pi / 6)

    # Materials
    mat_ls = Material([0., 0., 0.], emission_color=np.array([3., 3., 3.]).reshape(-1, 1))
    base_finish = SoftDull
    base_finish.reflection = 0.5
    mat_base = Material(colors.P_Chrome1, finish=base_finish)

    mat_s1 = Material(colors.Ruby1, finish=VeryHardPolished)
    mat_s2 = Material(colors.Emerald1, finish=VeryHardPolished)
    mat_s3 = Material(colors.Aquamarine1, finish=VeryHardPolished)

    se_ls = SceneElement(Sphere([0., 20., 30.], 2.), mat_ls)
    se_base = SceneElement(Plane([0.0, -4.0, 20.], [0., 1., 0.]), mat_base)

    cone = Cone([5., 0., 40.], 2., length=4, closed=True)
    cone.rotate(np.pi/2. + np.pi/4, [0, 1, 0])

    cylinder = Cylinder([2., 0., 40.], 2., length=2, closed=True)
    cylinder.rotate(-np.pi/4, [0, 1, 0])
    se_s1 = SceneElement(cone, mat_s1)
    se_s2 = SceneElement(cylinder, mat_s2)
    se_s3 = SceneElement(Sphere([-5., 0., 40.], 2.), mat_s3)

    scene = Scene([se_ls, se_base, se_s1, se_s2, se_s3])
    # scene = Scene([se_ls, se_base, se_s1])
    # scene = Scene([se_ls, se_base, se_s2])

    # Render
    rt = RayTracer(camera, scene)
    traced = rt.render()
    plt.imshow(traced); plt.show()


if __name__ == "__main__":
    render()