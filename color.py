import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# IMAGES
# EDIT THIS TO WORK
img1 = cv.imread("./green/green4.jpg")
img2 = cv.imread("./red/red1.jpg")

def nothing(a):
    pass

def maxImage(w, h, max_res):
	max_ = max(w, h)
	coeff = max_res / max_
	w *= coeff
	h *= coeff
	return (round(w), round(h))

def morph(img):
    kernel = np.ones(shape=(12,12), dtype=np.uint8)
    close = cv.morphologyEx(img, op=cv.MORPH_CLOSE, kernel=kernel)
    return close


def rangeUL(img, lower=[], upper=[]):
    mask = cv.inRange(cv.cvtColor(img, cv.COLOR_RGB2LAB), lower, upper)
    return mask

def rangeMask(img, mask):
    return cv.bitwise_and(src1= img, src2= img, mask = mask)


# Window
cv.namedWindow("LAB")
cv.resizeWindow("LAB", 940, 240)

# Trackbar
cv.createTrackbar("L Lower", "LAB", 0, 255, nothing)
cv.createTrackbar("L Upper", "LAB", 255, 255, nothing)
cv.createTrackbar("A Lower", "LAB", 0, 255, nothing)
cv.createTrackbar("A Upper", "LAB", 255, 255, nothing)
cv.createTrackbar("B Lower", "LAB", 0, 255, nothing)
cv.createTrackbar("B Upper", "LAB", 255, 255, nothing)

while True:
    # LAB Values
    l_down = cv.getTrackbarPos("L Lower", "LAB")
    l_up = cv.getTrackbarPos("L Upper", "LAB")
    a_down = cv.getTrackbarPos("A Lower", "LAB")
    a_up = cv.getTrackbarPos("A Upper", "LAB")
    b_down = cv.getTrackbarPos("B Lower", "LAB")
    b_up = cv.getTrackbarPos("B Upper", "LAB")

    # Lower and Upper Inrange
    lower = np.array([l_down, a_down, b_down], dtype=np.uint8)
    upper = np.array([l_up, a_up, b_up], dtype=np.uint8)

    img1 = cv.resize(img1, maxImage(img1.shape[1], img1.shape[0], 416), interpolation=cv.INTER_CUBIC)
    img2 = cv.resize(img2, maxImage(img2.shape[1], img2.shape[0], 416), interpolation=cv.INTER_CUBIC)

    result1 = rangeMask(img1, morph(rangeUL(img1, lower, upper)))
    result2 = rangeMask(img2, morph(rangeUL(img2, lower, upper)))

    hstack1 = np.hstack([img1, cv.cvtColor(morph(rangeUL(img1, lower, upper)), cv.COLOR_GRAY2BGR), result1])
    hstack2 = np.hstack([img2, cv.cvtColor(morph(rangeUL(img2, lower, upper)), cv.COLOR_GRAY2BGR), result2])
    vstack = np.vstack([hstack1,hstack2])
    cv.imshow("1", vstack)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cv.destroyAllWindows()