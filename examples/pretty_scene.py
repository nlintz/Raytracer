import numpy as np
from camera import Camera
from ray_tracer import RayTracer
from scene_element import SceneElement, LightSourcePoint, LightSourceDirectional
from scene import Scene
from geometry.sphere import Sphere
from material import Material, Finish
from material.finishes import VerySoftDull, Medium, HardPolished, VeryHardPolished
import material.colors as colors
import matplotlib.pyplot as plt
import scipy.misc as misc


def render():
    (w, h) = (640*2, 480*2)
    camera = Camera(w, h, fov=np.pi / 6)
    
    # Materials
    mat_base = Material([1., 1., 1.], finish=Medium)

    mat_s1 = Material(colors.Sapphire2, finish=HardPolished)
    mat_s2 = Material(colors.Amber3, finish=HardPolished)
    mat_s3 = Material(colors.Tourmaline1, finish=HardPolished)
    mat_s4 = Material(colors.Ruby4, finish=HardPolished)
    mat_s5 = Material(colors.Emerald5, finish=HardPolished)
    mat_s6 = Material(colors.Citrine2, finish=HardPolished)

    mat_s7 = Material(colors.Azurite3, finish=HardPolished)
    mat_s8 = Material(colors.Amethyst3, finish=HardPolished)
    mat_s9 = Material(colors.P_Brass3, finish=HardPolished, metallic=True)


    # Scene Elements + Scene
    se_ls = LightSourcePoint([-5., 15., 15.], intensity=1000., emission_color=[2., 2., 2.])

    se_base = SceneElement(Sphere([0.0, -10004., 20.], 10000.), mat_base)

    se_s1 = SceneElement(Sphere([0., 0., 20.], 4.), mat_s1)
    se_s2 = SceneElement(Sphere([-7., -2., 23.], 2.), mat_s2)
    se_s3 = SceneElement(Sphere([-5., -3., 27.], 1.), mat_s3)
    se_s4 = SceneElement(Sphere([-10., 3., 50.], 7.), mat_s4)
    se_s5 = SceneElement(Sphere([22., 3., 50.], 7.), mat_s5)
    se_s6 = SceneElement(Sphere([15., -3., 35.], 1.), mat_s6)
    se_s7 = SceneElement(Sphere([10., -1., 33.], 3.), mat_s7)
    se_s8 = SceneElement(Sphere([5., -2., 25.], 2.), mat_s8)
    se_s9 = SceneElement(Sphere([4., -3., 15.], 1.), mat_s9)


    scene = Scene([se_ls, se_base, se_s1, se_s2, se_s3, se_s4, se_s5, se_s6, se_s7, se_s8, se_s9])

    # Render
    rt = RayTracer(camera, scene, num_bounces=10)
    traced = rt.render()
    # plt.imshow(traced); plt.show()
    misc.imsave('renders/pretty2.png', traced)


if __name__ == "__main__":
    render()