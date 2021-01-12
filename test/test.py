import cv2 as cv
import cv2


def Gamma(img, ga=2.2):
    newimg = img.copy()
    col = img.shape[1]
    row = img.shape[0]
    for i in range(col):
        for j in range(row):
            g1 = (img[j, i] + 0.5) / 256
            g2 = pow(g1, 1 / ga)
            g3 = g2 * 256 - 0.5
            newimg[j, i] = g3

    return newimg


img = cv.imread('56.png',0)
print(img.shape)
img1 = img.copy()
h, w = img.shape
center = (w / 2, h / 2)
M = cv2.getRotationMatrix2D(center, 23, 1)
rotated = cv2.warpAffine(img, M, (w, h))
cv2.imwrite('rotated.png', rotated)
roi = rotated[835:856, 1170:1240]


img3 = Gamma(roi,0.06)
cv2.imwrite('Gamma.png', img3)
ret,img4 = cv.threshold(img3,45,255,cv.THRESH_BINARY)

img5 = cv.medianBlur(img4,3)
cv2.imwrite('res.png', img5)

cv.imshow('img5',img5)

cv.waitKey(0)