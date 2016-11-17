import numpy as np
from camera import Camera
from ray_tracer import RayTracer
from scene_element import SceneElement
from scene import Scene
from geometry.sphere import Sphere, DiffuseSphere, ShinySphere
from material import Material, Finish
import material.colors as colors
import matplotlib.pyplot as plt


def render():
    (w, h) = (640, 480)
    camera = Camera(w, h, fov=np.pi / 6)
    
    # Materials
    mat_ls = Material([0., 0., 0.], emission_color=np.array([1., 1., 1.]).reshape(-1, 1))
    mat_ls2 = Material([0., 0., 0.], emission_color=np.array([1., 1., 1.]).reshape(-1, 1))
    mat_base = Material(colors.DimGray)

    mat_s1 = Material(colors.CadetBlue, finish=Finish(reflection=0., specular=0.8, roughness=1./20))
    mat_s2 = Material(colors.P_Brass3, finish=Finish(ambient=0.25, diffuse=0.5, transparent=True, specular=0.8, roughness=1./80, ior=1.1), metallic=True)
    # mat_s2 = Material(colors.P_Brass3, finish=Finish(ambient=0.25, diffuse=0.5, transparent=True, specular=0.8, roughness=1./80, metallic=False, ior=1.1))

    se_ls = SceneElement(Sphere([-100., 100., 0.], 3.), mat_ls)
    se_ls2 = SceneElement(Sphere([100., 100., -1.], 3.), mat_ls2)
    se_base = SceneElement(Sphere([0.0, -10004., 20.], 10000.), mat_base)

    se_s1 = SceneElement(Sphere([-1., -2., 20.], 2.), mat_s1)
    se_s2 = SceneElement(Sphere([3.25, -1., 22.], 2.), mat_s2)
    se_s3 = SceneElement(Sphere([3.25, -1., 27.], 2.), mat_s1)

    scene = Scene([se_ls, se_ls2, se_base, se_s1, se_s2, se_s3])

    # Render
    rt = RayTracer(camera, scene)
    traced = rt.render()
    plt.imshow(traced); plt.show()


if __name__ == "__main__":
    render()