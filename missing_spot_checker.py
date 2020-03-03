import cv2
import numpy as np
import math
from binarize_and_determine_coords import findCoordinates

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



##POPULATE WITH OUR TYPE OF CLASS OBJECT FROM THE COORDINATES
coordinates = findCoordinates()

sortedCList = []
for circleIn in coordinates:

    sortedCList.append(SortedCircle(circleIn))




sortedCList.sort(key=lambda SortedCircle: SortedCircle.dist_from_origin)

sortedCList[0].num_label = 1



print(sortedCList[0].xCoord)
print(sortedCList[0].yCoord)







cv2.waitKey(0)
cv2.destroyAllWindows()