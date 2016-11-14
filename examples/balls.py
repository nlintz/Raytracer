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

    mat_s1 = Material(colors.P_Brass3, reflectivity=1., transparency=0.5)
    mat_s2 = Material(colors.CornflowerBlue, reflectivity=0.2, transparency=0.0)
    mat_s3 = Material(colors.ForestGreen, reflectivity=0.2, transparency=0.0)
    mat_s4 = Material(colors.GreenCopper, reflectivity=0.2, transparency=0.0)
    # mat_s1 = Material([1.00, 0.32, 0.36], reflectivity=1., transparency=0.5)
    # mat_s2 = Material([0.90, 0.76, 0.46], reflectivity=1., transparency=0.0)
    # mat_s3 = Material([0.65, 0.77, 0.97], reflectivity=1., transparency=0.0)
    # mat_s4 = Material([0.90, 0.90, 0.90], reflectivity=1., transparency=0.0)

    # Scene Elements + Scene
    se_ls = SceneElement(Sphere([0., 20., 30.], 3.), mat_ls)
    se_base = SceneElement(Sphere([0.0, -10004., 20.], 10000.), mat_base)

    se_s1 = SceneElement(Sphere([0., 0., 20.], 4.), mat_s1)
    se_s2 = SceneElement(Sphere([5.0, -1., 15.], 2.), mat_s2)
    se_s3 = SceneElement(Sphere([5.0, 0, 25.], 3.), mat_s3)
    se_s4 = SceneElement(Sphere([-5.5, 0, 15.], 3.), mat_s4)

    scene = Scene([se_ls, se_base, se_s1, se_s2, se_s3, se_s4])

    # Render
    rt = RayTracer(camera, scene)
    traced = rt.render()
    plt.imshow(traced); plt.show()


if __name__ == "__main__":
    render()