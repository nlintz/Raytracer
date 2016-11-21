import numpy as np
from camera import Camera
from geometry.sphere import Sphere
import matplotlib.pyplot as plt
import vector_lib as vl
import scipy.misc as misc
import global_config

def render():
    img = misc.imread('examples/checkered.png')[:, :, :3]
    # img = misc.imread('examples/face.png')[:, :, :3]

    h, w, c = img.shape

    camera = Camera(500, 500)
    sphere = Sphere([0., 0., 0.], 1.)
    camera.translate([0, 0, -2])

    t, m, n = sphere.intersection(camera.origin, camera.D)
    d_hat = vl.vnorm(m - sphere.center)
    d_x = d_hat[0:1]
    d_y = d_hat[1:2]
    d_z = d_hat[2:3]

    u = (0.5 + (np.arctan2(d_z, d_x) / (np.pi * 2)))
    v = 0.5 - (np.arcsin(d_y) / np.pi)

    img_out = img[(v * h-1).astype(np.uint8),
                  (u * w-1).astype(np.uint8)].squeeze()

    mask = (t<global_config.MAX_DISTANCE).T
    plt.imshow((img_out * mask).reshape(camera.height, camera.width, 3)); plt.show()


    import ipdb; ipdb.set_trace()

if __name__ == "__main__":
    render()