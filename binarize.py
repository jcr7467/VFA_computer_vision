import cv2
import numpy as np

from IGMTESTINGCROPPED import cropImage
from IGMTESTINGCROPPED import cropImage2




testImagePath = './images/dng/background/tiff-conv/LD154_bkg.tif'


#########################################################
# Loads up images for manipulation
#########################################################

croppedTestImage = cv2.imread(testImagePath)
testImage = cv2.imread(testImagePath)


#########################################################






#########################################################
# Crop images
#########################################################

testImage = cropImage(testImage)
croppedTestImage = cropImage2(croppedTestImage)


#########################################################


cv2.imshow("super original", testImage)



#########################################################
# Grayscale and binarization of testImage
#########################################################
grayTestImage = testImage[:, :, 1]
grayTestImage= cv2.cvtColor(testImage, cv2.COLOR_BGR2GRAY)
img = cv2.medianBlur(grayTestImage, 5)

#This is the binarized image
bin_image = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 19, 4)
#cv2.imshow('binarized', bin_image)

#########################################################


kernel = np.ones((3,3), np.uint8)


img_dilation = cv2.dilate(bin_image, kernel, iterations=1)
img_erosion = cv2.erode(img_dilation, kernel, iterations=4)

#cv2.imshow('Input', bin_image)

bin_image=cv2.medianBlur(bin_image, 5)

cv2.imshow('Input2', bin_image)

cv2.imshow('Dilation', img_dilation)
cv2.imshow('Erosion', img_erosion)
















#########################################################
# Hough Transform on testImage
#########################################################

toBetested = img_erosion


circles = cv2.HoughCircles(toBetested, cv2.HOUGH_GRADIENT, 2, 200, param1=70, param2=17, minRadius=80, maxRadius=110)
circles = np.uint16(np.around(circles))

listOfCoordinates = []



for i in circles[0, :]:
    thick = 2

    listOfCoordinates.append(i)
    # This circles the outter circle, with a color of (0, 255,0) and a thickness of 2
    cv2.circle(testImage, (i[0], i[1]), i[2], (0, 255, 0), thick)

    # This circles the center of the circle
    cv2.circle(testImage, (i[0], i[1]), 2, (0, 255, 0), 3)

print(listOfCoordinates)

cv2.imshow("circle detection", testImage)

#########################################################










#########################################################
# Grayscale and binarization of cropped image
#########################################################
grayTestImage2 = croppedTestImage[:, :, 1]
grayTestImage2= cv2.cvtColor(croppedTestImage, cv2.COLOR_BGR2GRAY)
img2 = cv2.medianBlur(grayTestImage2, 5)

#This is the binarized image
bin_image2 = cv2.adaptiveThreshold(img2, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 29, 4)
#cv2.imshow('binarized', bin_image2)

#########################################################






#########################################################
# Hough transform on cropped image
#########################################################

circles2 = cv2.HoughCircles(bin_image2, cv2.HOUGH_GRADIENT, 2, 200, param1=70, param2=17, minRadius=80, maxRadius=110)


circles2 = np.uint16(np.around(circles2))

listOfCoordinates = []



for i in circles2[0, :]:
    thick = 2

    listOfCoordinates.append(i)
    # This circles the outter circle, with a color of (0, 255,0) and a thickness of 2
    cv2.circle(croppedTestImage, (i[0], i[1]), i[2], (0, 255, 0), thick)

    # This circles the center of the circle
    cv2.circle(croppedTestImage, (i[0], i[1]), 2, (0, 255, 0), 3)



cv2.imshow("circle detection2", croppedTestImage)

#########################################################








cv2.waitKey(0)
cv2.destroyAllWindows()