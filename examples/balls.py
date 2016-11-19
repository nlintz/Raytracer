import numpy as np
from camera import Camera
from ray_tracer import RayTracer
from scene_element import SceneElement, LightSourcePoint, LightSourceDirectional
from scene import Scene
from geometry.sphere import Sphere
from material import Material, Finish
import material.colors as colors
import matplotlib.pyplot as plt


def render():
    (w, h) = (640, 480)
    camera = Camera(w, h, fov=np.pi / 6)
    
    # Materials
    mat_base = Material([0.2, 0.2, 0.2])

    mat_s1 = Material(colors.P_Brass3, finish=Finish(transparent=True, ior=1.1))
    mat_s2 = Material(colors.CornflowerBlue, finish=Finish(reflection=0.1))
    mat_s3 = Material(colors.ForestGreen, finish=Finish(reflection=0.1))
    mat_s4 = Material(colors.GreenCopper, finish=Finish(reflection=0.1))

    # Scene Elements + Scene
    se_ls = LightSourcePoint([-5., 10., 20.], intensity=1000.)

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