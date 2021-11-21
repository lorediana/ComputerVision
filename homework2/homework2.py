import cv2
import numpy as np
import math


def detectSkin(rgbImage):
    r, g, b = cv2.split(rgbImage)
    mask = (r > 95) & (g > 40) & (b > 20) & ((np.maximum(r, np.maximum(g, b)) - np.minimum(r, np.minimum(g, b))) > 15) & \
        (np.abs(r - g)) > 15 & (r > g) & (r > b)
    mask_arr = np.array(mask, dtype=np.uint8)
    return mask_arr


def gaussProbability(facePixels):
    d1, d2, d3 = facePixels.shape
    x = np.zeros([d1, d2 * d3])
    for i in range(len(facePixels)):
        x[i] = facePixels[i].flatten()

    x -= np.mean(x, axis=0)

    covariance = np.cov(x)
    mean = x.mean(axis=1)

    return mean, covariance


def computeGauss(x, mu, sigma):
    size = len(x)
    print(x.shape)
    print(mu.shape)
    if size == len(mu) and (size, size) == sigma.shape:
        det = np.linalg.det(sigma)
        if det == 0:
            raise NameError("The covariance matrix can't be singular")
        norm_const = 1.0 / (math.pow((2*math.pi), float(size)/2) * math.pow(det, 1.0/2))
        x_mu = math.matrix(x - mu)
        inv = sigma.I
        result = math.pow(math.e, -0.5 * (x_mu * inv * x_mu.T))
        return norm_const * result
    else:
        raise NameError("The dimensions of the input don't match")
        
        
def gaussFaceDetect(rgbImage):
    skinMask = detectSkin(rgbImage)
    skin = cv2.bitwise_and(rgbImage, rgbImage, mask=skinMask)
    cv2.imshow("Detected skin", skin)
    cv2.imwrite("skin.jpg", skin)
    
    mean, cov = gaussProbability(skin)

    mask = computeGauss(skin, mean, cov)
    return mask
  
  
  if __name__ == "__main__":
    img = cv2.imread("detectSkin.png")
    cv2.imshow("Original image", img)

    mask_f = gaussFaceDetect(img)

    cv2.waitKey(0)
