import numpy as np
import cv2


def make_some_noise():
    # Mean from the ILSVRC dataset
    mean = [103.939, 116.779, 123.68]
    # as (mean-2*sd,mean+2*sd) ~ 95% of the points,
    # we keep sd such that approximately >95% of the samples lie in (0,255)
    sd = [52, 58, 62]
    im = np.zeros((500, 500, 3))
    for i in range(3):
        im[:, :, i] = np.random.normal(
            loc=mean[i], scale=sd[i], size=(500, 500))
    im = np.clip(im, 0, 255)
    cv2.imwrite('gaussian_noise.png', im)


if __name__ == '__main__':
    make_some_noise()
