import cv2

import numpy as np

import math


def cropImage(cropMe):
    print('its working!')
    return cropMe[1500:3100, 1000:2630]













#testImagePath = './newImages/signal/20200220_155558.tif'
testImagePath = './newImages/signal/20200220_155540.tif'
testImage = cv2.imread(testImagePath)
testImage = cropImage(testImage)
cv2.imshow("Crop", testImage)




# Might be really important ->  grayTestImage2 = testImage[:, :, 1]
grayTestImage = cv2.cvtColor(testImage, cv2.COLOR_BGR2GRAY)


img = cv2.medianBlur(grayTestImage, 5)



cv2.imshow("Crop2", img)




# maximum value to use with the THRESH_BINARY and THRESH_BINARY_INV thresholding types.
maxValue = 250

# Size of a pixel neighborhood that is used to calculate a threshold value for the pixel: 3, 5, 7, and so on.
# binaryBlockSize & con change the image the most
binaryBlockSize = 21

# Constant subtracted from the mean or weighted mean (see the details below).
# Normally, it is positive but may be zero or negative as well.
con = 2

bin_image2 = cv2.adaptiveThreshold(img, maxValue, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, binaryBlockSize, con)
cv2.imshow("Binarized", bin_image2)


kernel = np.ones((2, 1), np.uint8)

img_dilation = cv2.dilate(bin_image2, kernel, iterations=1)
cv2.imshow("Dil", img_dilation)

img_erosion = cv2.erode(img_dilation, kernel, iterations=1)
cv2.imshow("Errosion", img_erosion)


newBlurredImage = cv2.medianBlur(img_erosion, 1)

cv2.imshow("blurrrrrr2", newBlurredImage)




circles = cv2.HoughCircles(bin_image2, cv2.HOUGH_GRADIENT, 2, 200, param1=70, param2=17, minRadius=60, maxRadius=70)
circles = np.uint16(np.around(circles))

coordinatesList = []

for i in circles[0, :]:

    coordinatesList.append(i)
    #cv2.circle(testImage, (i[0], i[1]), i[2], (0, 255, 0), 2)
    #cv2.circle(testImage, (i[0], i[1]), 2, (0, 255, 0), 3)


cv2.imshow("Final", testImage)



circles = cv2.HoughCircles(bin_image2, cv2.HOUGH_GRADIENT, 2, 200, param1=70, param2=17, minRadius=60, maxRadius=70)
circles = np.uint16(np.around(circles))

for i in circles[0, :]:

    coordinatesList.append(i)
    cv2.circle(testImage, (i[0], i[1]), i[2], (0, 255, 0), 2)
    cv2.circle(testImage, (i[0], i[1]), 2, (0, 255, 0), 3)

cv2.imshow("Final2", testImage)


















































































































testImagePath2 = './newImages/signal/20200220_155558.tif'
#testImagePath2 = './newImages/signal/20200220_155540.tif'
testImage2 = cv2.imread(testImagePath2)
testImage2 = cropImage(testImage2)
cv2.imshow("Crop2", testImage2)




# Might be really important ->  grayTestImage2 = testImage[:, :, 1]
grayTestImage2 = cv2.cvtColor(testImage2, cv2.COLOR_BGR2GRAY)


img2 = cv2.medianBlur(grayTestImage2, 5)



cv2.imshow("Crop22", img2)




# maximum value to use with the THRESH_BINARY and THRESH_BINARY_INV thresholding types.
maxValue2 = 250

# Size of a pixel neighborhood that is used to calculate a threshold value for the pixel: 3, 5, 7, and so on.
# binaryBlockSize & con change the image the most
binaryBlockSize2 = 21

# Constant subtracted from the mean or weighted mean (see the details below).
# Normally, it is positive but may be zero or negative as well.
con2 = 2

bin_image22 = cv2.adaptiveThreshold(img2, maxValue2, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, binaryBlockSize2, con2)
cv2.imshow("Binarized2", bin_image2)


kernel2 = np.ones((2, 1), np.uint8)

img_dilation2 = cv2.dilate(bin_image2, kernel2, iterations=1)
cv2.imshow("Dil2", img_dilation2)

img_erosion2 = cv2.erode(img_dilation2, kernel, iterations=1)
cv2.imshow("Errosion2", img_erosion2)




circles2 = cv2.HoughCircles(bin_image22, cv2.HOUGH_GRADIENT, 2, 200, param1=70, param2=17, minRadius=60, maxRadius=70)
circles2 = np.uint16(np.around(circles2))

coordinatesList2 = []

for i in circles2[0, :]:

    coordinatesList2.append(i)
    #cv2.circle(testImage, (i[0], i[1]), i[2], (0, 255, 0), 2)
    #cv2.circle(testImage, (i[0], i[1]), 2, (0, 255, 0), 3)


cv2.imshow("Final", testImage2)



circles2 = cv2.HoughCircles(bin_image22, cv2.HOUGH_GRADIENT, 2, 200, param1=70, param2=17, minRadius=60, maxRadius=70)
circles2 = np.uint16(np.around(circles2))

for i in circles[0, :]:

    coordinatesList2.append(i)
    cv2.circle(testImage2, (i[0], i[1]), i[2], (0, 255, 0), 2)
    cv2.circle(testImage2, (i[0], i[1]), 2, (0, 255, 0), 3)

cv2.imshow("Final2", testImage)






'''








'''




cv2.waitKey(0)

cv2.destroyAllWindows()




