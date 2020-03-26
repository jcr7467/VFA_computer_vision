import cv2
import numpy as np
testImagePath = './images/dng/background/tiff-conv/LD163_bkg.tif'
#testImagePath = './images/testimage.jpg'
testImage = cv2.imread(testImagePath)
grayTestImage = testImage[:, :, 1]
grayTestImage= cv2.cvtColor(testImage, cv2.COLOR_BGR2GRAY)
img = cv2.medianBlur(grayTestImage, 5)
circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 2, 200, param1=70, param2=17, minRadius=80, maxRadius=110)
#circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 120, param1=100, param2=30, )
circles = np.uint16(np.around(circles))
listOfCoordinates = []
# hard code a radius because the radiuses will all be the same in the testing
# threshold to binarize the values to find the difference between the edges easier
# if this doesnt work we can have a non linear gain
for i in circles[0, :]:
    thick = 2
    listOfCoordinates.append(i)
    # This circles the outter circle, with a color of (0, 255,0) and a thickness of 2
    cv2.circle(testImage, (i[0], i[1]), i[2], (0, 255, 0), thick)
    # This circles the center of the circle
    cv2.circle(testImage, (i[0], i[1]), 2, (0, 255, 0), 3)
print(listOfCoordinates)

cv2.imshow("circle detection", testImage)
cv2.waitKey(0)
cv2.destroyAllWindows()