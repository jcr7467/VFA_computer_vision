import cv2
import numpy as np

from matplotlib import pyplot as plt


'''
Questions for Zach:

How should my binarized image look and do you have anything to base it off of
'''




testImagePath = './images/dng/background/tiff-conv/LD180_bkg.tif'
#testImagePath = './images/testimage.jpg'




testImage = cv2.imread(testImagePath)

grayTestImage = testImage[:, :, 1]
grayTestImage= cv2.cvtColor(testImage, cv2.COLOR_BGR2GRAY)





img = cv2.medianBlur(grayTestImage, 5)





bin_image = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 57,3)


#This is the binarized image
cv2.imshow('binarized', bin_image)


#This is the grayscale image
#cv2.imshow("grayscale", img)

#img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


# This is the unchanged image but with a median blur
cv2.imshow("default", testImage)


cv2.waitKey(0)
cv2.destroyAllWindows()



'''
maybe add a blur onto the image that is binarized

keeping connected p

If there is an isolated pixels
'''