import numpy as np
from camera import Camera
from ray_tracer import RayTracer
from scene_element import SceneElement
from scene import Scene
from geometry.sphere import Sphere, DiffuseSphere, ShinySphere
from material import Material
import material.colors as colors
import matplotlib.pyplot as plt


def render():
    (w, h) = (640, 480)
    camera = Camera(w, h, fov=np.pi / 6)
    
    # Materials
    mat_ls = Material([0., 0., 0.], reflectivity=0., transparency=0., emission_color=np.array([3., 3., 3.]).reshape(-1, 1))
    mat_base = Material([0.2, 0.2, 0.2], reflectivity=0., transparency=0.)

    mat_s1 = Material(colors.P_Brass3, reflectivity=0., transparency=0.0)
    mat_s2 = Material(colors.CornflowerBlue, reflectivity=0., transparency=0.0)

    se_ls = SceneElement(Sphere([-100., 100., 0.], 3.), mat_ls)
    se_base = SceneElement(Sphere([0.0, -10004., 20.], 10000.), mat_base)

    se_s1 = SceneElement(Sphere([-1., -2., 20.], 2.), mat_s1)
    se_s2 = SceneElement(Sphere([3.25, -2., 22.], 2.), mat_s2)

    scene = Scene([se_ls, se_base, se_s1, se_s2])

    # Render
    rt = RayTracer(camera, scene)
    traced = rt.render()
    plt.imshow(traced); plt.show()


if __name__ == "__main__":
    render()