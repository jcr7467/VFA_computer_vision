import cv2
import numpy as np
#import matplotlib

from IGMTESTINGCROPPED import cropImage
from IGMTESTINGCROPPED import houghT


pathToImages = './images/'


####################################################################################################


# These variables specify location of signal and background images
pathToSignals = pathToImages + 'dng/signal/'
pathToBackgrounds = pathToImages + 'dng/background/'




# Will detect circles in background images
##########################################
#path to specific testing images(will expand to the rest eventually)
#testImagePath = './images/testimage.jpg'
testImagePath = './images/dng/background/tiff-conv/LD162_bkg.tif'



#loads up image from path
testImage = cv2.imread(testImagePath)


print(type(testImage))


grayImage = cv2.cvtColor(testImage, cv2.COLOR_BGR2GRAY)


print(testImage.shape)


# Specifies the range of x and y coordinates to show of the cropped image
#I am going to assume that we are going to be taking the same size picture
# everytime with roughly the same placement such that the cropping always works
#resizedImage = cropImage(testImage)
#print(resizedImage.shape)


houghT(testImage)


#cv2.imshow('resized', resizedImage)

cv2.imshow("circle detection", testImage)



# Waits for a key to be pressed
# and then destroys windows
cv2.waitKey(0)
cv2.destroyAllWindows()



# Using the coordinates found on background images, I will get the coordinates
# for the rest of the circles with a call to a function written in a seperate file