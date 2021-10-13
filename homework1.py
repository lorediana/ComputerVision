import cv2
import numpy as np
from matplotlib import pyplot as plt


def homomorphic_filter(img, d0, gammaL, gammaH):
    img = np.float32(img)
    img = img / 255
    rows, cols, dim = img.shape

    imgYCrCb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(imgYCrCb)

    y_log = np.log(y + 0.01)
    y_fft = np.fft.fft2(y_log)
    y_fft_shift = np.fft.fftshift(y_fft)

    DX = cols / d0
    G = np.ones((rows, cols))
    for i in range(rows):
        for j in range(cols):
            G[i][j] = ((gammaH - gammaL) * (1 - np.exp(-((i - rows / 2) ** 2 + (j - cols / 2) ** 2) / (2 * DX ** 2)))) \
                      + gammaL

    result_filter = G * y_fft_shift
    result_partial = np.real(np.fft.ifft2(np.fft.ifftshift(result_filter)))
    result = np.exp(result_partial)

    return result


if __name__ == "__main__":
    img = cv2.imread("download.jpeg")
    cv2.imshow("Original image", img)
    img_filtered = homomorphic_filter(img, 30, 0.5, 2)
    cv2.imshow("Filtered image", img_filtered)
    cv2.waitKey(0)
