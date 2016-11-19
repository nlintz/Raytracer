import numpy as np
from camera import Camera
from ray_tracer import RayTracer
from scene_element import SceneElement, LightSourcePoint, LightSourceDirectional
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
    base_finish = SoftDull
    base_finish.reflection = 0.5
    mat_base = Material(colors.P_Chrome1, finish=base_finish)

    mat_f = VeryHardPolished
    mat_f.reflection = 0.1
    mat_s1 = Material(colors.Ruby1, finish=mat_f)
    mat_s2 = Material(colors.Emerald1, finish=mat_f)
    mat_s3 = Material(colors.Aquamarine1, finish=mat_f)

    # se_ls = LightSourcePoint([-5., 15., 20.], intensity=1000., emission_color=[2., 2., 2.])
    se_ls = LightSourceDirectional([1, -1, 0], intensity=2., emission_color=[1., 1., 1.])

    se_base = SceneElement(Plane([0.0, -4.0, 20.], [0., 1., 0.]), mat_base)

    cone = Cone([5., 0., 40.], 2., length=4, closed=True)
    cone.rotate(np.pi/2. + np.pi/4, [0, 1, 0])

    cylinder = Cylinder([2., 0., 40.], 2., length=2, closed=True)
    cylinder.rotate(-np.pi/4, [0, 1, 0])
    se_s1 = SceneElement(cone, mat_s1)
    se_s2 = SceneElement(cylinder, mat_s2)
    se_s3 = SceneElement(Sphere([-5., 0., 40.], 2.), mat_s3)

    scene = Scene([se_ls, se_base, se_s1, se_s2, se_s3])

    # Render
    rt = RayTracer(camera, scene)
    traced = rt.render()
    plt.imshow(traced); plt.show()


if __name__ == "__main__":
    render()