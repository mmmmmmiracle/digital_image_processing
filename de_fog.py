import numpy as np
import cv2
from data_aug import aug, albu, aug_show


def ComputeMinLevel(hist, pnum):
    index = np.add.accumulate(hist)
    return np.argwhere(index>pnum * 8.3 * 0.01)[0][0]


def ComputeMaxLevel(hist, pnum):
    hist_0 = hist[::-1]
    Iter_sum = np.add.accumulate(hist_0)
    index = np.argwhere(Iter_sum > (pnum * 2.2 * 0.01))[0][0]
    return 255-index


def LinearMap(minlevel, maxlevel):
    if (minlevel >= maxlevel):
        return []
    else:
        index = np.array(list(range(256)))
        screenNum = np.where(index<minlevel,0,index)
        screenNum = np.where(screenNum> maxlevel,255,screenNum)
        for i in range(len(screenNum)):
            if screenNum[i]> 0 and screenNum[i] < 255:
                screenNum[i] = (i - minlevel) / (maxlevel - minlevel) * 255
        return screenNum


def DeFog(image, **kwargs):
    h, w, d = image.shape
    newimg = np.zeros([h, w, d])
    for i in range(d):
        imghist = np.bincount(image[:, :, i].reshape(1, -1)[0])
        minlevel = ComputeMinLevel(imghist,  h * w)
        maxlevel = ComputeMaxLevel(imghist, h * w)
        screenNum = LinearMap(minlevel, maxlevel)
        if (screenNum.size == 0):
            continue
        for j in range(h):
            newimg[j, :, i] = screenNum[image[j, :, i]]
    return newimg / 255

def de_fog(**kwargs):
    return DeFog


if __name__ == '__main__':
    # os.chdir(os.path.join(c))
    # img = cv2.imread('input/unwater.jpg')
    # img = aug(img, albu.Resize(270, 480))
    img = cv2.imread('input/heaven.jpg')
    img = aug(img, albu.Resize(256, 256))
    newimg = aug(img, albu.Lambda(image=de_fog()))
    # newimg = DeFog(img)
    cv2.imshow('original_img', img)
    cv2.imshow('new_img', newimg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()