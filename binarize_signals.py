import cv2
import numpy as np
import math




class SortedCircle:
    def __init__(self, circleIn):
        self.circle = circleIn
        self.xCoord = circleIn[0]
        self.yCoord = circleIn[1]
        self.row = 0
        self.col = 0
        self.radius = circleIn[2]
        self.num_label = 0
        self.dist_from_origin = (circleIn[0]**2 + circleIn[1]**2) ** 0.5














testImagePath = './images/dng/signal/tiff-conv/LD156_sig.tif'


#########################################################
# Loads up images for manipulation
#########################################################

croppedTestImage = cv2.imread(testImagePath)
testImage = cv2.imread(testImagePath)


#########################################################



def cropImage2(cropMe):
    print('its working!')
    return cropMe[2000:3400, 800:2400]

def cropImage(cropMe):
    print('its working!')
    return cropMe[1650:3550, 600:2600]




#########################################################
# Crop images
#########################################################

testImage = cropImage(testImage)
croppedTestImage = cropImage2(croppedTestImage)


#########################################################


cv2.imshow("The original image cropped down", croppedTestImage)



##THIS IS THE BLUR FIRST BINARIZE LATER
#########################################################
# Grayscale and binarization of testImage
#########################################################
grayTestImage2 = testImage[:, :, 1]
grayTestImage2 = cv2.cvtColor(croppedTestImage, cv2.COLOR_BGR2GRAY)
img2 = cv2.medianBlur(grayTestImage2, 13)
#cv2.imshow('Middle2, blur', img2)
#This is the binarized image
bin_image2 = cv2.adaptiveThreshold(img2, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 23, 2)
#cv2.imshow('Final2, binarize', bin_image2)







kernel = np.ones((2,2), np.uint8)

img_erosion = cv2.erode(bin_image2, kernel, iterations=3)

img_dilation = cv2.dilate(bin_image2, kernel, iterations=1)

final_image = cv2.dilate(img_erosion, kernel, iterations=1)
#cv2.imshow('Input', bin_image)
bin_image2=cv2.medianBlur(bin_image2, 3)
cv2.imshow('Input2', bin_image2)
cv2.imshow('Dilation', img_dilation)
cv2.imshow('Erosion', img_erosion)
cv2.imshow('Erosion > dilation', final_image)








toBetested = img_erosion


circles = cv2.HoughCircles(toBetested, cv2.HOUGH_GRADIENT, 2, 200, param1=70, param2=17, minRadius=80, maxRadius=110)
circles = np.uint16(np.around(circles))

listOfCoordinates = []




#cv2.circle(croppedTestImage, (10, 10), 20 , (0,255,0), 2)



for i in circles[0, :]:
    thick = 2

    listOfCoordinates.append(i)
    # This circles the outter circle, with a color of (0, 255,0) and a thickness of 2
    cv2.circle(croppedTestImage, (i[0], i[1]), i[2], (0, 255, 0), thick)

    # This circles the center of the circle
    cv2.circle(croppedTestImage, (i[0], i[1]), 2, (0, 255, 0), 3)

#print(listOfCoordinates)
#print(len(listOfCoordinates))



### Making a list of x values

xCoords = []

for circle in listOfCoordinates:
    xCoords.append(circle[0])
#print(xCoords)
#print("X coordinates\n")


### Making a list of y values

yCoords = []

for circle in listOfCoordinates:
    yCoords.append(circle[1])
#print(yCoords)
#print("Y Coordinates\n")

### Making a list of radius'

radius = []

for circle in listOfCoordinates:
    radius.append(circle[2])

#print(radius)
#print("Radiuses\n")


### Making a list of distances from origin

dist = []

for circle in listOfCoordinates:
    dist.append((circle[0]**2 + circle[1]**2)**.5)

#dist.sort()
#print(dist)
#print("DISTANCES\n")








newList = []

print("###############################################")

### Making a list of SortedCircles

for circleIn in listOfCoordinates:

    newList.append(SortedCircle(circleIn))




for item in newList:
    print(item.dist_from_origin)

newList.sort(key=lambda SortedCircle: SortedCircle.dist_from_origin)

print("#################################")

for item in newList:
    print(item.dist_from_origin)


print("#################################")

for item in newList:
    print(item.dist_from_origin)
#  sortedElements = sorted(elements, key = Dummy.getPrice)


# # To sort the list in place...
# ut.sort(key=lambda x: x.count, reverse=True)
#
# # To return a new list, use the sorted() built-in function...
# newlist = sorted(ut, key=lambda x: x.count, reverse=True)

cv2.imshow("circle detection", croppedTestImage)









cv2.waitKey(0)
cv2.destroyAllWindows()