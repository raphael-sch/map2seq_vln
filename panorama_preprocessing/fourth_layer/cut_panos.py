import os
import cv2
import numpy as np


def get_slices(pano_path):
    equ = Equirectangular(pano_path)
    h_stride = 45
    nn_heading_angles = list(range(0, 360, h_stride))

    slices = list()
    for heading_angle in nn_heading_angles:
        img = equ.get_perspective(FOV=60,
                                  THETA=heading_angle,
                                  PHI=0,
                                  height=460,
                                  width=800)
        slices.append(img)
    return slices


def render_streetview_image():

    pano_dir = 'panos'
    slices_dir = 'slices'
    pano_id = 'zZQunI0Pn9wxHNdDb6ujFg'
    pano_path = os.path.join(pano_dir, pano_id + '.jpg')

    equ = Equirectangular(pano_path)
    h_stride = 45
    nn_heading_angles = list(range(0, 360, h_stride))

    for heading_angle in nn_heading_angles:
        img = equ.get_perspective(FOV=45,
                                  THETA=heading_angle,
                                  PHI=0,
                                  height=800,
                                  width=460)
        image_name = f'2_{pano_id}_{heading_angle}.jpg'
        image_path = os.path.join(slices_dir, image_name)
        cv2.imwrite(image_path, img)


class Equirectangular:
    # https://github.com/fuenwang/Equirec2Perspec/blob/master/Equirec2Perspec.py
    def __init__(self, img_name):
        self._img = cv2.imread(img_name, cv2.IMREAD_COLOR)
        [self._height, self._width, _] = self._img.shape
        # cp = self._img.copy()
        # w = self._width
        # self._img[:, :w/8, :] = cp[:, 7*w/8:, :]
        # self._img[:, w/8:, :] = cp[:, :7*w/8, :]

    def get_perspective(self, FOV, THETA, PHI, height, width, RADIUS=128):
        #
        # THETA is left/right angle, PHI is up/down angle, both in degree
        #

        equ_h = self._height
        equ_w = self._width
        equ_cx = (equ_w - 1) / 2.0
        equ_cy = (equ_h - 1) / 2.0

        wFOV = FOV
        hFOV = float(height) / width * wFOV

        c_x = (width - 1) / 2.0
        c_y = (height - 1) / 2.0

        wangle = (180 - wFOV) / 2.0
        w_len = 2 * RADIUS * np.sin(np.radians(wFOV / 2.0)) / np.sin(np.radians(wangle))
        w_interval = w_len / (width - 1)

        hangle = (180 - hFOV) / 2.0
        h_len = 2 * RADIUS * np.sin(np.radians(hFOV / 2.0)) / np.sin(np.radians(hangle))
        h_interval = h_len / (height - 1)
        x_map = np.zeros([height, width], np.float32) + RADIUS
        y_map = np.tile((np.arange(0, width) - c_x) * w_interval, [height, 1])
        z_map = -np.tile((np.arange(0, height) - c_y) * h_interval, [width, 1]).T
        D = np.sqrt(x_map ** 2 + y_map ** 2 + z_map ** 2)
        xyz = np.zeros([height, width, 3], np.float)
        xyz[:, :, 0] = (RADIUS / D * x_map)[:, :]
        xyz[:, :, 1] = (RADIUS / D * y_map)[:, :]
        xyz[:, :, 2] = (RADIUS / D * z_map)[:, :]

        y_axis = np.array([0.0, 1.0, 0.0], np.float32)
        z_axis = np.array([0.0, 0.0, 1.0], np.float32)
        [R1, _] = cv2.Rodrigues(z_axis * np.radians(THETA))
        [R2, _] = cv2.Rodrigues(np.dot(R1, y_axis) * np.radians(-PHI))

        xyz = xyz.reshape([height * width, 3]).T
        xyz = np.dot(R1, xyz)
        xyz = np.dot(R2, xyz).T
        lat = np.arcsin(xyz[:, 2] / RADIUS)
        lon = np.zeros([height * width], np.float)
        theta = np.arctan(xyz[:, 1] / xyz[:, 0])
        idx1 = xyz[:, 0] > 0
        idx2 = xyz[:, 1] > 0

        idx3 = ((1 - idx1) * idx2).astype(np.bool)
        idx4 = ((1 - idx1) * (1 - idx2)).astype(np.bool)

        lon[idx1] = theta[idx1]
        lon[idx3] = theta[idx3] + np.pi
        lon[idx4] = theta[idx4] - np.pi

        lon = lon.reshape([height, width]) / np.pi * 180
        lat = -lat.reshape([height, width]) / np.pi * 180
        lon = lon / 180 * equ_cx + equ_cx
        lat = lat / 90 * equ_cy + equ_cy
        # for x in range(width):
        #    for y in range(height):
        #        cv2.circle(self._img, (int(lon[y, x]), int(lat[y, x])), 1, (0, 255, 0))
        # return self._img

        persp = cv2.remap(self._img, lon.astype(np.float32), lat.astype(np.float32), cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_WRAP)
        return persp


if __name__ == '__main__':
    render_streetview_image()
