import numpy as np
from geometry.triangle_mesh import TriangleMesh
from camera import Camera
from ray_tracer import RayTracer
from scene_element import SceneElement
from scene import Scene
from material import Material
import material.colors as colors
import matplotlib.pyplot as plt
from geometry.sphere import Sphere

def down_sample(n_polys, fi, V, F):
    fi_out = []
    V_out = []
    F_out = []
    for i in range(0, n_polys, 12):
        fi_out = fi_out + fi[i:i+3]
        F_out = F_out + F[i:i+3]

    F_out2 = []
    for i in range(0, len(F_out), 3):
        F_out2 = F_out2 + F_out[i:i+3]

    return len(F_out)/3, fi_out, V, F_out2


def render():
    # with open('examples/cube.obj') as f:
    with open('examples/teapot.obj') as f:
        lines = f.readlines()

    V = []
    F = []
    for line in lines:
        line = line.replace('  ', ' ')
        if line[0:2] == "v ":
            V.append(map(float, line.strip().split(' ')[1:]))
        elif line[0:2] == "f ":
            f = line.strip().split(' ')[1:]
            f = map(lambda x: int(x.split('/')[0])-1, f)
            F.extend(f)

    n_polys = len(F) / 3
    fi = [3 for _ in range(n_polys)]

    # n_polys, fi, V, F = down_sample(n_polys, fi, V, F)

    tm = TriangleMesh(n_polys, fi, F, np.array(V).T, clockwise=True)

    (w, h) = (640, 480)
    camera = Camera(w, h, fov=np.pi / 2)
    camera.translate([0, 0, -100])
    # camera = Camera(w, h, fov=np.pi / 3)
    
    # Materials
    mat_ls = Material([0., 0., 0.], reflection=0., transparency=0., emission_color=np.array([3., 3., 3.]).reshape(-1, 1))
    mat_base = Material([0.2, 0.2, 0.2], reflection=0., transparency=0.)

    mat_s1 = Material(colors.NeonPink, diffuse=0.7, ambient=0.1, specular=0.8, roughness=1./120, reflection=0.)

    se_ls = SceneElement(Sphere([0., 100., -100.], 3.), mat_ls)
    se_base = SceneElement(Sphere([0.0, -10004., 20.], 10000.), mat_base)

    se_tm = SceneElement(tm, mat_s1)

    scene = Scene([se_ls, se_tm])

    # Render
    rt = RayTracer(camera, scene, num_bounces=0, background_color=0.5*np.ones((3, 1)))
    traced = rt.render()
    plt.imshow(traced); plt.show()

    import ipdb; ipdb.set_trace()

if __name__ == "__main__":
    render()
