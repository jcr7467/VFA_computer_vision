import cv2
import numpy as np
import math
import copy

#This is for reading the images that are in the fluorescent/ directory
from os import listdir
from os.path import isfile, join





def cropImage(cropMe):
    '''
    This function simply crops the image that we are working with into the specified
    dimensions that we have hard coded: 1670 x 1600 -> width x height
    :param cropMe: image to be cropped
    :return:
     params of return statement are in [Y, X] cropping ranges
    '''

    return cropMe[1800:3400, 760:2430]






def matchTemplate(image, template):
    '''
    This function finds the alignment markers which our program uses to correctly orient the image to our grid
    If we changed our template to another value, we would simply add the option to our dictionary and the add
    the image to our flurorescent_templates directory

    :param image: image to match template to
    :param template: input option to determine the template to use
    :return:
    '''


    template_dictionary = {
        'template_A': 'top_left_alignment_marker.tif',
        'template_B': 'top_right_alignment_marker.tif',
        'template_C': 'bottom_left_alignment_marker.tif',
        'template_D': 'bottom_right_alignment_marker.tif'
    }


    template = cv2.imread('fluorescent_templates/' + template_dictionary[template], cv2.IMREAD_GRAYSCALE)

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    w,h = template.shape[::-1]

    result = cv2.matchTemplate(gray_image, template, cv2.TM_CCOEFF_NORMED)

    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)


    deltaX = bottom_right[0] - top_left[0]
    deltaY = bottom_right[1] - top_left[1]
    midXPoint = top_left[0] + deltaX//2
    midYPoint = top_left[1] + deltaY//2


    #cv2.rectangle(image, top_left, bottom_right, 255, 2)
    #cv2.circle(image, (midXPoint, midYPoint), 80, (255, 255, 0), 2)
    #cv2.circle(image, (midXPoint, midYPoint), 2, (255, 255, 0), 2)


    return (midXPoint, midYPoint)






def findAngle(alignA, alignB):
    '''

    This function finds the corresponding angle between the two top alignment markers, and also returns which
     direction they should be turned to be on the same axis.
    I could probably remake this to just return either a positive or negative angle, but this works for now
    -----> RETURNS ANGLE IN RADIANS <-----

    :param alignA: alignment marker A
    :param alignB: alignment marker B
    :return: (angleINRADIANS, direction)

    '''

    deltaX = abs(int(alignA[0]) - int(alignB[0]))
    deltaY = int(alignA[1]) - int(alignB[1])

    if (deltaY) >= 0:
        direction = "CW"
    else:
        direction = "CCW"

    deltaY = abs(deltaY)

    return (math.atan(deltaY/deltaX), direction)






def rotateAndScale(img, scaleFactor = 1, degreesCCW = 0):
    '''

    :param img: the image that will get rotated and returned
    :param scaleFactor: option to enlarge, we always use at 1
    :param degreesCCW: DEGREES NOT RADIANS to rotate CCW. Neg value will turn CW
    :return: rotated image
    '''


    (oldY,oldX) = (img.shape[0], img.shape[1]) #note: numpy uses (y,x) convention but most OpenCV functions use (x,y)
    M = cv2.getRotationMatrix2D(center=(oldX/2,oldY/2), angle=degreesCCW, scale=scaleFactor) #rotate about center of image.

    #choose a new image size.
    newX,newY = oldX*scaleFactor,oldY*scaleFactor
    #include this if you want to prevent corners being cut off
    r = np.deg2rad(degreesCCW)
    newX,newY = (abs(np.sin(r)*newY) + abs(np.cos(r)*newX),abs(np.sin(r)*newX) + abs(np.cos(r)*newY))

    #the warpAffine function call, below, basically works like this:
    # 1. apply the M transformation on each pixel of the original image
    # 2. save everything that falls within the upper-left "dsize" portion of the resulting image.

    #So I will find the translation that moves the result to the center of that region.
    (tx,ty) = ((newX-oldX)/2,(newY-oldY)/2)
    M[0,2] += tx #third column of matrix holds translation, which takes effect after rotation.
    M[1,2] += ty

    rotatedImg = cv2.warpAffine(img, M, dsize=(int(newX),int(newY)))

    return rotatedImg






def rotateImage(image, alignA, alignB):
    '''
    This function calls the findAngle function and we use the return values from that function to pass into the
     rotateAndScale function which actually does the rotation
    :param image: image to rotate
    :param alignA: coordinates for alignment marker A
    :param alignB: coordinates for alignment marker B
    :return: rotated image
    '''

    angleToRotate, direction = findAngle(alignA, alignB)

    if direction == "CW":
        angleToRotate = -1 * angleToRotate * (180/math.pi)
    else:
        angleToRotate = angleToRotate * (180/math.pi)

    return rotateAndScale(image, 1, angleToRotate)






def shiftBy(deltaX, deltaY, img):
    '''
    If delta values are negative, the translation matrix will move it the correct direction,
    so we dont have to worry about the negative values
    :param deltaX: shift by this delta x
    :param deltaY: shift by this delta y
    :param img: image to shift
    :return: shifted image
    '''

    num_rows, num_cols = img.shape[:2]

    translation_matrix = np.float32([ [1,0,deltaX], [0,1,deltaY] ])
    img_translation = cv2.warpAffine(img, translation_matrix, (num_cols, num_rows))

    return img_translation






def alignImage(image):
    '''
    This function combines the shiftBy function and the rotateImage function into one.
    Essentially places our image on our predetermined grid. First it rotates the image,
    and then it finds the new coordinates for Alignment Marker A. Using these new coordinates,
    it shifts the entire image such that the new Alignment Marker A coordinates are in the spot we want them to be.

    FOR THIS SETUP, WE WANT ALIGNMENT MARKER A TO BE ON COORDINATES: -> (296, 291)

    :param image: image to be aligned
    :return: shifted and rotated image
    '''


    # Rotates the image
    alignA = matchTemplate(image, "template_A")
    alignB = matchTemplate(image, "template_B")
    rotated_image = rotateImage(image, alignA, alignB)

    # Shifts the image
    new_alignA = matchTemplate(rotated_image, "template_A")
    alignAX = new_alignA[0]
    alignAY = new_alignA[1]

    shifted_and_rotated = shiftBy(296 - alignAX, 291 - alignAY, rotated_image)


    return shifted_and_rotated






def drawCirclesAndLabels(image, pointMap):
    '''
    This function is just to display the image with the labels that we predetermined,
    it has no impact on the resulting calculations
    :param image: The image that is going to be drawn on. This is the image that is ALREADY ALIGNED.
    :param pointMap:
    :return:
    '''

    copyImage = copy.deepcopy(image)

    for key, value in pointMap.items():

        cv2.circle(copyImage, value, 60, (255, 255, 255), 2)

        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 2
        color = (255, 255, 255)
        thickness = 2

        copyImage = cv2.putText(copyImage, key, value, font,
                            fontScale, color, thickness, cv2.LINE_AA)
    return copyImage






def create_circular_mask(h, w, center=None, radius=None):
    '''
    **Note: height and width must be exactly the same as the image we are making a mask for
     bc we multiply the matrices together in the end
    :param h: height of the image we are creating a mask for
    :param w: width of the image we are creating a mask for
    :param center: The center point of the circle
    :param radius: The specified radius that we choose: in our case, we are defaulting to 60px
    :return:
    '''

    if center is None: # use the middle of the image
        center = (int(w/2), int(h/2))
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    return mask






def findAverageLightIntensity(maskedImage, mask):

    '''

    :param maskedImage: This is the original image that has been multiplied with the mask
    :param mask: This is the mask that was made to 'cut out' the
    :return: returns the average, which is found to be the sum of the pixel values divided by the area of the mask
    '''

    sum_of_pixels = np.sum(maskedImage)
    area = np.sum(mask)
    return(sum_of_pixels/area)






def findAllCircleAveragesFor(imagePath, displayCirclesBool):
    pointMap = {
        'A': (296, 291),
        'B': (1374, 291),
        'C': (296, 1308),
        'D': (1374, 1308),
        '1': (690, 440),
        '2': (985, 440),
        '3': (445,665),
        '4': (690,665),
        '5': (985, 665),
        '6': (1225, 665),
        '7': (835, 800),
        '8': (445, 935),
        '9': (690, 935),
        '10': (985, 935),
        '11': (1225, 935),
        '12': (690, 1170),
        '13': (985, 1170)
    }

    #Crops image and aligns it to our grid
    image = cv2.imread(imagePath)
    image = cropImage(image)
    aligned_image = alignImage(image)


    # If we choose, the aligned image with labels will
    # pop up on screen to ensure that circles are on correct points
    if displayCirclesBool == True:
        labeled_image = drawCirclesAndLabels(aligned_image, pointMap)
        cv2.imshow("Labeled Circles for " + imagePath, labeled_image)



    #Prints the path and afterwards displays the average for each circle.
    print("\n\nAverages for image path: " + imagePath + "\n")

    aligned_image = cv2.cvtColor(aligned_image, cv2.COLOR_BGR2GRAY)
    h, w = aligned_image.shape[:2]
    for key, value in pointMap.items():

        #We do not need to print the average intensity for the alignment markers
        if key not in ['A', 'B', 'C', 'D']:

            radius_of_mask = 60
            centerPoint = value
            mask = create_circular_mask(h, w, centerPoint, radius_of_mask)
            maskedImage = np.multiply(aligned_image, mask)

            averageIntensity = findAverageLightIntensity(maskedImage, mask)
            print(key + ": " + str(averageIntensity))






def averagesOfAllImages(displayCirclesBool = False):
    '''
    This function simply runs the findAllCircleAveragesFor every image in our list. The list is compiled by looking
    into the "fluorescent/" directory and selecting the files that begin with "image" as the file name. This is
    to prevent any other types of files to get passed in. This also means that any test image must be named
    as "image*". It just has to start with the word image.

    :param displayCirclesBool:
    :return:
    '''

    mypath = 'fluorescent/'
    imageList = [f for f in listdir(mypath) if (isfile(join(mypath, f)) and ''.join(f[0:5]) == 'image')]
    imageList = sorted(imageList)

    #print(imageList)
    #imageList = ['image_1.tif', 'image_2.tif', 'image_3.tif', 'image_4.tif', 'image_5.tif', 'image_6.tif']

    for image in imageList:
        findAllCircleAveragesFor(mypath + image, displayCirclesBool)






def main():

    averagesOfAllImages(True)
    cv2.waitKey(0)
    cv2.destroyAllWindows()




if __name__ == '__main__':
    main()




'''
This is a list of where all the circles are located in our fixed grid. This is how I originally found them
This is now irrelevant, but here for reference

cv2.circle(aligned_image, (690, 440), 60, (255, 155, 70), 2)
cv2.circle(aligned_image, (985, 440), 60, (255, 155, 70), 2)

cv2.circle(aligned_image, (445, 665), 60, (255, 155, 70), 2)
cv2.circle(aligned_image, (690, 665), 60, (255, 155, 70), 2)
cv2.circle(aligned_image, (985, 665), 60, (255, 155, 70), 2)
cv2.circle(aligned_image, (1225, 665), 60, (255, 155, 70), 2)

cv2.circle(aligned_image, (835, 800), 60, (255, 155, 70), 2)

cv2.circle(aligned_image, (445, 935), 60, (255, 155, 70), 2)
cv2.circle(aligned_image, (690, 935), 60, (255, 155, 70), 2)
cv2.circle(aligned_image, (985, 935), 60, (255, 155, 70), 2)
cv2.circle(aligned_image, (1225, 935), 60, (255, 155, 70), 2)

cv2.circle(aligned_image, (690, 1170), 60, (255, 155, 70), 2)
cv2.circle(aligned_image, (985, 1170), 60, (255, 155, 70), 2)
'''
