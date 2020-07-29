import cv2
import random
import numpy as np

# assuming img tensor is in HWC order of rank 3

def with_prob(prob, func, identity=lambda *a, **ka: a[0]): # default identity mapping will discard keyword arguments
    def func_with_prob(*args, **kwargs):
        if random.random() < prob:
            return func(*args, **kwargs)
        else:
            return identity(*args, **kwargs)

    return func_with_prob


def random_crop(img, target_hw=None):
    assert target_hw is not None
    h, w, _ = img.shape
    th, tw = target_hw

    start_h = random.randint(0, h - th + 1)
    start_w = random.randint(0, w - tw + 1)
    return img[start_h:start_h+th, start_w:start_w+tw, :]


class RandomPerspectiveConsts:
    d = np.float32([[1., 1.], [-1., 1.], [1., -1.], [-1., -1.]])
    quad = np.float32([[0., 0.], [1., 0.], [0., 1.], [1., 1.]])

def random_perspective(img, scale=0.1):
    h, w, _ = img.shape
    dim = np.float32([w, h])

    random_noise = scale * (np.random.random((4, 2)) - 0.5)
    random_noise = random_noise.astype(np.float32)

    dst = RandomPerspectiveConsts.quad * dim
    src = dst + random_noise*RandomPerspectiveConsts.d*dim
    trans = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, trans, (h, w))

def random_rotation(img, max_degree=20):
    h, w, _ = img.shape
    degree = random.randint(-max_degree, max_degree)
    rot_mat = cv2.getRotationMatrix2D((h*0.5, w*0.5), degree, 1.0)
    return cv2.warpAffine(img, rot_mat, img.shape[1::-1])


def flip(img):
    return img[:, ::-1, :]
