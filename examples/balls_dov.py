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
    (w, h) = (640*2, 480*2)
    camera = Camera(w, h, fov=np.pi / 6)
    
    # Materials
    mat_base = Material(colors.DimGray)

    mat_s1 = Material(colors.CadetBlue, finish=Finish(reflection=0., specular=0.8, roughness=1./20))
    mat_s2 = Material(colors.P_Brass3, finish=Finish(ambient=0.25, diffuse=0.5, transparent=True, specular=0.8, roughness=1./80, ior=1.5), metallic=True)

    se_ls = LightSourcePoint([-5., 10., 20.], intensity=1000.)
    se_ls2 = LightSourceDirectional([1., -1., 1.], intensity=0.2)

    se_base = SceneElement(Sphere([0.0, -10004., 20.], 10000.), mat_base)

    se_s1 = SceneElement(Sphere([-3., -1., 20.], 2.), mat_s1)
    se_s2 = SceneElement(Sphere([0, -1., 35.], 2.), mat_s1)
    se_s3 = SceneElement(Sphere([4., -1., 40.], 2.), mat_s1)
    se_s4 = SceneElement(Sphere([9., -1., 55.], 2.), mat_s1)

    scene = Scene([se_ls, se_ls2, se_base, se_s1, se_s2, se_s3, se_s4])

    # Render
    rt = RayTracer(camera, scene)
    # traced = rt.render_dov([0, -1., 35.])
    # traced = rt.render_dov([-3., -1., 20.])
    traced = rt.render_dov([4., -1., 40.])
    # traced = rt.render()
    plt.imshow(traced); plt.show()


if __name__ == "__main__":
    render()