import vector_lib as vl
import numpy as np


class Camera(object):
    def __init__(self, width, height, fov=np.pi/2):
        self.origin = np.array([0., 0., 0.]).reshape(-1, 1)
        self.width = width
        self.height = height
        self.fov = fov

        self.screen = self.get_screen()

    def translate(self, T):
        TM = vl.translation_matrix(T)
        self.origin = vl.transform(TM, self.origin)
        self.screen = vl.transform(TM, self.screen)

    def rotate(self, theta, direction, point=None):
        RM = vl.translation_matrix(theta, direction, point)

        self.origin = vl.transform(RM, self.origin)
        self.screen = vl.transform(RM, self.screen)

    def get_screen(self):
        aspect_ratio = float(self.width) / self.height
        scale = np.tan(self.fov / 2.)

        xx = np.arange(0, self.width)
        yy = np.arange(0, self.height)
        x_world = (2 * (xx + 0.5) / self.width - 1) * aspect_ratio * scale
        y_world = (1 - 2 * (yy + 0.5) / self.height) * scale

        Z = np.ones((1, self.width * self.height))
        X, Y = np.meshgrid(x_world, y_world)
        Q = np.concatenate([X.reshape(1, -1), Y.reshape(1, -1), Z], axis=0)
        return Q

    @property
    def D(self):
        return vl.vnorm(self.screen - self.origin)


if __name__ == "__main__":
    camera = Camera(width=400, height=300)
    R = vl.rotation_matrix(np.pi/2, np.array([1, 0, 0]), np.array([0., 5., 5.]))
    T = vl.translation_matrix(np.array([0., 5., 5.]))

    camera.translate(T)

    camera.rotate(R)
    import ipdb; ipdb.set_trace()