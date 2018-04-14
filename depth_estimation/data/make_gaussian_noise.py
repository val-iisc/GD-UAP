import numpy as np
import cv2


def make_some_noise():
    # Mean from the ILSVRC dataset

    mean = [104.008, 116.669, 122.68]
    # as (mean-2*sd,mean+2*sd) ~ 95% of the points,
    # we keep sd such that approximately >95% of the samples lie in (0,255)
    sd = [52, 58, 62]
    # Input size is (513,513,3), hence we take a random noise sample of size 2*(513,513,3).
    im = np.zeros((512, 1026, 3))
    for i in range(3):
        im[:, :, i] = np.random.normal(
            loc=mean[i], scale=sd[i], size=(512, 1026))
    im = np.clip(im, 0, 255)
    cv2.imwrite('gaussian_noise.png', im)


if __name__ == '__main__':
    make_some_noise()
