import numpy as np
from camera import Camera
from ray_tracer import RayTracer
from scene_element import SceneElement
from scene import Scene
from geometry.sphere import Sphere, DiffuseSphere, ShinySphere
from material import Material, Finish
from material.finishes import VerySoftDull, Medium, HardPolished, VeryHardPolished
import material.colors as colors
import matplotlib.pyplot as plt


def render():
    (w, h) = (640, 480)
    camera = Camera(w, h, fov=np.pi / 6)
    
    # Materials
    # mat_ls = Material([0., 0., 0.], emission_color=np.array([2., 2., 2.]).reshape(-1, 1))
    mat_ls = Material([0., 0., 0.], emission_color=np.array([2., 2., 2.]).reshape(-1, 1))
    mat_base = Material([0.2, 0.2, 0.2])

    mat_s1 = Material(colors.P_Silver3, finish=Finish(diffuse=0.7, ambient=0.1, specular=0.8, roughness=1./120, transparent=True, ior=1.5))

    mat_s2 = Material(colors.P_Copper3, finish=Medium, metallic=True)
    mat_s3 = Material(colors.P_Chrome3, finish=Medium, metallic=True)
    mat_s4 = Material(colors.P_Brass3, finish=Medium, metallic=True)


    # Scene Elements + Scene
    se_ls = SceneElement(Sphere([-2., 20., 30.], 3.), mat_ls)
    se_base = SceneElement(Sphere([0.0, -10004., 20.], 10000.), mat_base)

    se_s1 = SceneElement(Sphere([0., 0., 20.], 4.), mat_s1)
    se_s2 = SceneElement(Sphere([5.0, -1., 15.], 2.), mat_s2)
    se_s3 = SceneElement(Sphere([3.0, 0, 22.], 2.), mat_s3)
    se_s4 = SceneElement(Sphere([-5.5, 0, 15.], 3.), mat_s4)

    scene = Scene([se_ls, se_base, se_s1, se_s2, se_s3, se_s4])

    # Render
    rt = RayTracer(camera, scene, num_bounces=5)
    traced = rt.render()
    plt.imshow(traced); plt.show()


if __name__ == "__main__":
    render()