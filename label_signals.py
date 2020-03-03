import cv2
import numpy as np
import math
from helper_functions import cropImage
from helper_functions import SortedCircle
from helper_functions import binarizeErodeAndDilate
from helper_functions import houghTransform
from helper_functions import rotateAndScale










testImagePath = './images/dng/signal/tiff-conv/LD011862_sig.tif'
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


cv2.imshow("pre", croppedTestImage)
#########################################################




#Binarization


toBetested = binarizeErodeAndDilate(croppedTestImage)

listOfCoordinates = houghTransform(toBetested, croppedTestImage)


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


#print(newList[1].xCoord)
#print(newList[1].yCoord)



leftMostX = newList[0].xCoord
itsY = newList[0].yCoord
for imageCoor in newList:
    highestY = newList[0].yCoord - 100
    lowestY = newList[0].yCoord + 100
    if imageCoor.yCoord < lowestY and imageCoor.yCoord > highestY:
        leftMostX = imageCoor.xCoord
        itsY = imageCoor.yCoord


print(leftMostX)
print(itsY)

print("###########################???????######")

deltaX = math.fabs(leftMostX - newList[0].xCoord)
print(deltaX)
deltaY =  math.fabs(newList[0].yCoord - itsY)
print(deltaY)

theta = math.atan(deltaY/deltaX) * (180/math.pi)


print(theta)

angleToRotateCCW = 360 - theta

angleToRotateCCW = math.floor(angleToRotateCCW)

croppedTestImage = rotateAndScale(croppedTestImage, 1, angleToRotateCCW)

cv2.imshow("circle detection", croppedTestImage)
















cv2.waitKey(0)
cv2.destroyAllWindows()