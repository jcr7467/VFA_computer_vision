import cv2
import numpy as np
import math
from helper_functions import cropImage
from helper_functions import SortedCircle
from helper_functions import binarizeErodeAndDilate
from helper_functions import houghTransform
from helper_functions import rotateAndScale
from helper_functions import rotateAndScale2










testImagePath = './images/dng/signal/tiff-conv/LD180_sig.tif'
#154, 151, 166, dont work, dividing by 0, will check later
# LD011835 doesnt work bc of the cropping there is an extra circle up top
#155 works great



#########################################################
# Loads up images for manipulation
#########################################################

croppedTestImage = cv2.imread(testImagePath)
testImage = cv2.imread(testImagePath)

#########################################################

#########################################################
# Crop images
#########################################################

croppedTestImage = cropImage(croppedTestImage)

#cv2.circle(croppedTestImage, (183, 215), 2, (255, 255, 0), 3)
cv2.imshow("pre", croppedTestImage)
#########################################################




#Binarization


toBetested = binarizeErodeAndDilate(croppedTestImage)

listOfCoordinates = houghTransform(toBetested, croppedTestImage, True)





# Creating a list of SortedCircle Class items
newList = []

print("###############################################")

### Making a list of SortedCircles

for circleIn in listOfCoordinates:

    newList.append(SortedCircle(circleIn))



#PRESORTED
## This function sorts by their distance from the origin
newList.sort(key=lambda SortedCircle: SortedCircle.dist_from_origin)

print("#################################")

#POST SORTED
newList[0].num_label = 1
newList[3].num_label = 7


print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
print(newList[0].xCoord)
print(newList[0].yCoord)
print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
#for j in newList:
 #   croppedTestImage = cv2.putText(croppedTestImage, str(j.num_label), (j.xCoord, j.yCoord), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 2, cv2.LINE_AA)


##FINDING THE TOP RIGHT AND TOP LEFT COORDINATE IN ORDER TO ROTATE

rightMostX = newList[0].xCoord
itsY = newList[0].yCoord
leftCornerX = newList[0].xCoord
leftCornerY = newList[0].yCoord
for imageCoor in newList:
    highestY = newList[0].yCoord - 100
    lowestY = newList[0].yCoord + 100
    if imageCoor.yCoord < lowestY and imageCoor.yCoord > highestY:
        rightMostX = imageCoor.xCoord
        itsY = imageCoor.yCoord


print(rightMostX)
print(itsY)
print("###########################???????######")




deltaX = math.fabs(rightMostX - newList[0].xCoord)
print(deltaX)
deltaY =  math.fabs(newList[0].yCoord - itsY)
print(deltaY)

theta = math.atan(deltaY/deltaX) * (180/math.pi)

print(theta)

angleToRotateCCW = 360 - theta
angleToRotateCCW = math.floor(angleToRotateCCW)

newer = rotateAndScale2(croppedTestImage, 1, 20)

cv2.imshow("Test Rotate2", newer)

croppedTestImage = rotateAndScale(croppedTestImage, 1, angleToRotateCCW)

cv2.imshow("Test Rotate", croppedTestImage)

for coord in listOfCoordinates:

    print(coord)
    theTheta = angleToRotateCCW * (math.pi/180)
    #theTheta = 0.043608
    theX = coord[0] - leftCornerX
    theY =  leftCornerY - coord[1]
    val1 = theX * math.cos(theTheta)
    val2 = theY * math.sin(theTheta)
    newX = val1 + val2

    vals1 = -1 * theX * math.sin(theTheta)
    vals2 = theY * math.cos(theTheta)
    newY = vals1 + vals2

    newX += leftCornerX
    newY += leftCornerY
    coord[0] = newX
    coord[1] = newY

#alignedListOfCoordinates = houghTransform(alignedToBeTested, croppedTestImage, True)



cv2.imshow("Aligned circles", croppedTestImage)

# cv2.circle(drawOnMe, (i[0], i[1]), i[2], (0, 255, 0), thick)

# This circles the center of the circle
# cv2.circle(drawOnMe, (i[0], i[1]), 2, (0, 255, 0), 3)

for coord in listOfCoordinates:
    cv2.circle(croppedTestImage, (coord[0], coord[1]), 2, (0, 255, 0), 3)

cv2.circle(croppedTestImage, (183, 215), 2, (155, 255, 255), 3)

cv2.imshow("Last", croppedTestImage)



cv2.waitKey(0)
cv2.destroyAllWindows()