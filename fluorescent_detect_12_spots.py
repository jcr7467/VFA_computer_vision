'''
HOW THIS PROGRAM WORKS ON A HIGH LEVEL:
1. First it crops the image to our specified dimensions, to which right now it is set to 2540x2400 -> width x height
2. Then it uses template matching in order to find the top two alignment markers, and using the points
of the alignment markers, it rotates the image and scales the image if found necessary.
3. Then it shifts the image to where the first alignment marker, Alignment Marker A, is at the spot we
need it to be now that the image is upright and oriented correctly. In our case, we want that alignment
marker A to be at the point (591, 528). Once the image is aligned, we then know where the other points are
because of the predefined grid that we made.
4. Once the image is aligned, we make a mask for each individual circle, and multiply (element-wise) it by the original
image to create a new image that outside of the mask, it is completely black (matrix value of 0) We calculate
the average light intensity by taking the sum of the image value (inside the mask, the values will remain
their original values) divided by the sum of the mask
(this is essentially just the area of the circle because inside the mask,
there are only 1's and outside it there are only 0's)
5. We repeat this process for every image in the directory the user specifies, which houses all
 our images that need analysis. As long as our directory specified is inside the datasets directory,
 it will work as expected
'''


import cv2
import numpy as np
import math
import copy
import csv

#This is for reading the images that are in the fluorescent/ directory
from os import listdir, mkdir
from os.path import isfile, join, isdir


MASK_RADIUS = 55
ALIGNMENT_MARKER_A_MAP_LOCATION = (591, 528)
ALIGNMENT_MARKER_B_MAP_LOCATION = (1949, 528)
DISTANCE_FROM_A_TO_B = 1358
#rawpy to turn dng images to tif images

#refining spot map



def cropImage(cropMe):
    '''
    This function simply crops the image that we are working with into the specified
    dimensions that we have hard coded: 2540 x 2400 -> width x height

    cropMe[1200:3600, 560:3100]

    :param cropMe: image to be cropped
    :return:
     params of return statement are in [Y, X] cropping ranges
    '''

    return cropMe[1200:3600, 560:3100]






def matchTemplate(image, template):
    '''
    This function finds the alignment markers which our program uses to correctly orient the image to our grid
    If we changed our template to another value(added new template, modified old one, etc), we would simply add the
    option to our dictionary and then add the image to our alignment_templates directory.

    This function partitions the image into two sections: the right side and the left side
    This is in order to not confuse the templates (mainly due to the fact that alignment marker B & C are identical)

    :param image: image to match template to
    :param template: input option to determine the template to use
    :return:
    '''


    template_dictionary = {
        'template_A': 'alignment_A.tif',
        'template_B': 'alignment_B.tif',
        'template_C': 'alignment_C.tif',
        'template_D': 'alignment_D.tif'
    }

    if template == 'template_A' or template == 'template_C':
        partition = 'A'
    else:
        partition = 'B'


    # Reads the template image from the alignment_templates directory
    template = cv2.imread('alignment_templates/' + template_dictionary[template], cv2.IMREAD_GRAYSCALE)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    ######### Partitioning of image
    if partition == 'B':
        # We are going to look at the right partition
        gray_image = gray_image[0:2400,1225:2450]
    elif partition == 'A':
        # We are going to look at the left partition
        gray_image = gray_image[0:2400, 0:1225]




    ########## Actually completing template match
    w,h = template.shape[::-1]
    result = cv2.matchTemplate(gray_image, template, cv2.TM_CCOEFF_NORMED)


    ##########This section calculates the midpoint of the square that the template matched to
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)
    deltaX = bottom_right[0] - top_left[0]
    deltaY = bottom_right[1] - top_left[1]
    midXPoint = top_left[0] + deltaX//2
    midYPoint = top_left[1] + deltaY//2


    #We run this if we were looking at the right side of the image
    #This is because the midpoint for X would be relative to the cropped image we fed into the template matcher
    # However, we have to add 1225 back to get the x value with respect to the entire image
    if partition == 'B':
        midXPoint = midXPoint + 1225

    return (midXPoint, midYPoint)






def findScaleFactor(alignA, alignB):
    '''
    The purpose of this function is to find the scaling that we should apply to the image during the alignment process.
    We have a set value (based off of our map) that should be the distance between alignment markers A & B.
    This is what DISTANCE_FROM_A_TO_B represents.

    Then, using the coordinates from the template match, we calculate what the actual distance between the two alignment markers are
    and compute a ratio which will signal what we should scale the image by to get it to match our map.

    NOTE: This function uses DISTANCE_FROM_A_TO_B, which is a HORIZONTAL relationship.
    i.e. A & B have the same Y-value (horizontal value).

    If you tried to do this for vertical alignment markers, e.g. A & C, the value would be off


    :param alignA:
    :param alignB:
    :return:
    '''
    deltaX = abs(int(alignA[0]) - int(alignB[0]))
    deltaY = int(alignA[1]) - int(alignB[1])
    deltaY = abs(deltaY)

    distance = math.sqrt(deltaX * deltaX + deltaY * deltaY)

    ratioToScale = DISTANCE_FROM_A_TO_B / distance
    #print("Distance: " + str(distance))
    #print("Ratio: " + str(ratioToScale))


    return ratioToScale






def findAngle(alignA, alignB):
    '''
    This function finds the corresponding angle between the two alignment markers passed in, and also returns which
     direction they should be turned to be on the same axis.

     Also note that the order in which these are passed in is also important.
     We always pass them in from left to right when we look at the image,

     e.g. always alignA, alignB or alignC, alignD

        but never alignB, alignA

    -----> RETURNS ANGLE IN DEGREES <-----

    :param alignA: alignment marker A
    :param alignB: alignment marker B
    :return: angleINDEGREES
    '''

    deltaX = abs(int(alignA[0]) - int(alignB[0]))
    deltaY = int(alignA[1]) - int(alignB[1])

    if (deltaY) >= 0:
        direction = "CW"
    else:
        direction = "CCW"

    deltaY = abs(deltaY)

    angleToRotate = math.atan(deltaY/deltaX)

    if direction == "CW":
        angleToRotate = -1 * angleToRotate * (180/math.pi)
    else:
        angleToRotate = angleToRotate * (180/math.pi)

    return angleToRotate






def rotateAndScale(img, scaleFactor = 1, degreesCCW = 0):
    '''
    :param img: the image that will get rotated and returned
    :param scaleFactor: option to scale image
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






def alignImage(image, image_name):
    '''
    This function combines the shiftBy function and the rotateImage function into one.
    Essentially places our image on our predetermined grid. First it rotates the image (using avg angle between A-B & C-D),
    and then it finds the new coordinates for Alignment Marker A. Using these new coordinates,
    it shifts the entire image such that the new Alignment Marker A coordinates are in the spot we want them to be.

    FOR THIS SETUP, WE WANT ALIGNMENT MARKER A TO BE ON COORDINATES: -> (591, 528)
    This value is a constant up top named ALIGNMENT_MARKER_A_MAP_LOCATION^

    :param image: image to be aligned
    :return: shifted and rotated image
    '''


    ############## Preparing to rotate and scale the image
    alignA = matchTemplate(image, "template_A")
    alignB = matchTemplate(image, "template_B")
    alignC = matchTemplate(image, "template_C")
    alignD = matchTemplate(image, "template_D")

    angle1 = findAngle(alignA, alignB)
    angle2 = findAngle(alignC, alignD)
    avg_angle = (angle1 + angle2)/2

    scaleFactor1 = findScaleFactor(alignA, alignB)
    scaleFactor2 = findScaleFactor(alignC, alignD)
    avg_scale_factor = (scaleFactor1 + scaleFactor2)/2
    #print(avg_scale_factor)



    ############## Basically used as a threshold, given that the test is inserted correctly,
    # it should never be larger than 45 degrees
    if abs(avg_angle) > 45:
        print(image_name + ' is a bad image, it was rotated ' + avg_angle + ' degrees, unexpected amount')


    ########### Actually rotates image
    rotated_image = rotateAndScale(image, avg_scale_factor, avg_angle)




    ############### Shifts the image
    new_alignA = matchTemplate(rotated_image, "template_A")
    alignAX = new_alignA[0]
    alignAY = new_alignA[1]

    shiftBy_x = ALIGNMENT_MARKER_A_MAP_LOCATION[0] - alignAX
    shiftBy_y = ALIGNMENT_MARKER_A_MAP_LOCATION[1] - alignAY

    shifted_and_rotated = shiftBy(shiftBy_x, shiftBy_y, rotated_image)


    return shifted_and_rotated






def drawCirclesAndLabels(already_aligned_image, pointMap):
    '''
    This function is just to display the image with the labels that we predetermined,
    it has no impact on the resulting calculations
    :param image: The image that is going to be drawn on. This is the image that is ALREADY ALIGNED.
    :param pointMap:
    :return:
    '''

    ############This is because of Python's pass by object reference,
    # in order to not modify the version passed in, we make a deep copy
    copyImage = copy.deepcopy(already_aligned_image)


    ##NOTE:
    ##'key' is the name of the circle/alignment marker
    ##'value' is the coordinate of its respective circle/alignment marker
    for key, value in pointMap.items():
        if key not in ['A','B','C','D']:

            cv2.circle(copyImage, value, MASK_RADIUS, (255, 255, 255), 2)

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






def findAllCircleAveragesFor(imagePath, image_name, displayCirclesBool):
    '''
    :param imagePath: the image path for the image that we are going to find all the averages for
    :param image_name: This is the name of the image that we are going to find all the averages for
    :param displayCirclesBool: Determines if the images will be displayed on the screen, or saved to a file

        True: The images will be output to the screen

        False: The images will be saved to a directory named "processed" inside
        the directory the user specified at the beginning


    :return:
    '''

    #### THIS PART IS ESSENTIAL, THIS IS OUR SPOT MAP THAT OUR ENTIRE PROGRAM IS BASED ON
    pointMap = {
        'A': (591, 528),
        'B': (1949, 528),
        'C': (591, 1872),
        'D': (1949, 1872),
        '1': (1100, 700),
        '2': (1430, 700),
        '3': (763, 1030),
        '4': (1100, 1025),
        '5': (1430, 1025),
        '6': (1770, 1032),
        '7': (763, 1365),
        '8': (1100, 1360),
        '9': (1430, 1360),
        '10': (1770, 1360),
        '11': (1100, 1695),
        '12': (1430, 1695)
    }


    ##### This output array will be returned and will be a row in the csv file
    output = []
    output.append(image_name)





    #### Crops image and aligns it to our grid
    full_image_path = imagePath + image_name
    image = cv2.imread(full_image_path)
    image = cropImage(image)
    aligned_image = alignImage(image, image_name)

    ##So that we can create a new image name with _processed appended to it
    image_name = image_name.split('.')[0]





    #### Either displays the result images on the screen or saves them to directory inside calling directory
    if displayCirclesBool == True:
        labeled_image = drawCirclesAndLabels(aligned_image, pointMap)
        cv2.imshow("Labeled Circles for " + image_name, labeled_image)

    else:
        labeled_image = drawCirclesAndLabels(aligned_image, pointMap)
        if not isdir(imagePath + 'processed/'):
            mkdir(imagePath + 'processed/')
        cv2.imwrite(imagePath + 'processed/' + image_name + '_processed.tif', labeled_image)







    ## Grayscale the image to begin masking process
    aligned_image = cv2.cvtColor(aligned_image, cv2.COLOR_BGR2GRAY)
    h, w = aligned_image.shape[:2]

    for key, value in pointMap.items():

        #We do not need to print the average intensity for the alignment markers
        if key not in ['A', 'B', 'C', 'D']:

            radius_of_mask = MASK_RADIUS
            centerPoint = value
            mask = create_circular_mask(h, w, centerPoint, radius_of_mask)
            maskedImage = np.multiply(aligned_image, mask)

            averageIntensity = findAverageLightIntensity(maskedImage, mask)
            output.append(averageIntensity)



    #Prints the path for the image that was just processed
    print("Finished Processing: " + full_image_path)

    return output






def averagesOfAllImages(displayCirclesBool = False):
    '''

    This function simply runs the findAllCircleAveragesFor every image in our list.

    The list is compiled by first prompting the user for the name of the directory they would like to run a test on.
    It then finds all the tif images inside of that directory and adds them to the list.

    NOTE: THE REPOSITORY THE USER ENTERS MUST BE INSIDE 'datasets' DIRECTORY.

    For each image, once it receives the intensity of each spot, it takes the information
    and writes it to a csv file inside of the user-specified directory.

    E.g. Say that we have a directory named 'tiff-conv1' inside the 'datasets' directory
        Once the processing is done, inside 'datasets/tiff-conv1' there will be a csv file named 'tiff-conv1.csv'
        containing all the informatino that we found


    :param displayCirclesBool:
    :return:
    '''


    #### User specified test directory
    test_directory_name = input('Enter directory to test(must be inside of datasets directory, do not include path in name): ')
    if test_directory_name[-1] != '/':
        test_directory_name += '/'
    test_directory_path= 'datasets/' + test_directory_name






    ##Asserting that the directory input by user is valid and has images ending with .tif inside of it
    assert(isdir(test_directory_path)), "Error: Invalid directory"
    imageList = [f for f in listdir(test_directory_path) if (isfile(join(test_directory_path, f))) and f.endswith('.tif')]
    assert (len(imageList) > 0), "Error: an empty directory was passed in, please check the directory"


    imageList = sorted(imageList)
    print(str(len(imageList)) + ' images imported...')






    ##### Writes data acquired from list to our csv file
    i = 0
    matrix = np.ones(13)
    for image in imageList:
        if i == 0:
            matrix = findAllCircleAveragesFor(test_directory_path, image, displayCirclesBool)
            i += 1
            continue
        matrix = np.vstack([matrix,findAllCircleAveragesFor(test_directory_path, image, displayCirclesBool)])
        i += 1
    if not isdir(test_directory_path + 'csv/'):
        mkdir(test_directory_path + 'csv/')
    with open(test_directory_path + 'csv/' + test_directory_name[:-1] + '.csv', 'w+', newline='') as f:
        writer = csv.writer(f, delimiter = ',')
        writer.writerows(matrix)






def main():

    #Change to true to display images with circles drawn on
    averagesOfAllImages(False)
    cv2.waitKey(0)
    cv2.destroyAllWindows()






if __name__ == '__main__':
    main()




'''
This is a list of where all the circles are located in our fixed grid. This is how I originally found them


This is now irrelevant, but here for reference

I first rotated an image from the sample data, and then manually took trial-and-error
 approach to find where each circle's center should be located

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