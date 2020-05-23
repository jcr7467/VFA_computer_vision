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


 VERY IMPORTANT NOTE: to change the map we have, and make
 the program still work, the only things that should be changed should be one of the following:

    1. pointMap
    2. template_dictionary
    3. MASK_RADIUS
    4. YMIN_BOUND
    5. YMAX_BOUND
    6. XMIN_BOUND
    7. XMAX_BOUND

By changing these values, we should be able to make our code work with any changes that we want to make.
i.e. If you have a new image that you have to change the cropping to (maybe bc image is way larger or smaller)
    or you want to change the map, changing the values described above should be the only changes
    you make in order to make program still work

'''

import cv2
import numpy as np
import csv
from helper_functions import create_circular_mask, \
    findAverageLightIntensity, \
    drawCirclesAndLabels, \
    cropImage,\
    alignImage

#This is for reading the images that are in the fluorescent/ directory
from os import listdir, mkdir
from os.path import isfile, join, isdir


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

#Note: if you change this, only change the file name of
# the template, dont change the keys of this dictionary
# e.g. Do not change 'template_A'
template_dictionary = {
    'template_A': 'alignment_A.tif',
    'template_B': 'alignment_B.tif',
    'template_C': 'alignment_C.tif',
    'template_D': 'alignment_D.tif'
}


# CONSTANTS
MASK_RADIUS = 55
ALIGNMENT_MARKER_A_MAP_LOCATION = pointMap['A']
ALIGNMENT_MARKER_B_MAP_LOCATION = pointMap['B']
DISTANCE_FROM_A_TO_B = ALIGNMENT_MARKER_B_MAP_LOCATION[0] - ALIGNMENT_MARKER_A_MAP_LOCATION[0]

#CROP IMAGE DIMENSIONS CONSTANTS
YMIN_BOUND = 1200
YMAX_BOUND = 3600
XMIN_BOUND = 560
XMAX_BOUND = 3100








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




    ##### This output array will be returned and will be a row in the csv file
    output = []
    output.append(image_name)





    #### Crops image and aligns it to our grid
    full_image_path = imagePath + image_name
    image = cv2.imread(full_image_path)
    image = cropImage(image, YMIN_BOUND, YMAX_BOUND, XMIN_BOUND, XMAX_BOUND)
    aligned_image = alignImage(image, image_name, DISTANCE_FROM_A_TO_B, ALIGNMENT_MARKER_A_MAP_LOCATION, template_dictionary)

    ##So that we can create a new image name with _processed appended to it
    image_name = image_name.split('.')[0]





    #### Either displays the result images on the screen or saves them to directory inside calling directory
    if displayCirclesBool == True:
        labeled_image = drawCirclesAndLabels(aligned_image, pointMap, MASK_RADIUS)
        cv2.imshow("Labeled Circles for " + image_name, labeled_image)

    else:
        labeled_image = drawCirclesAndLabels(aligned_image, pointMap, MASK_RADIUS)
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
    assert (len(imageList) > 0), "Error: No tiff images were found, please check the directory"


    imageList = sorted(imageList)
    print(str(len(imageList)) + ' images imported...')






    ##### Writes data acquired from list to our csv file
    i = 0
    matrix = np.ones(13)
    for image in imageList:
        if i == 0:
            matrix = [findAllCircleAveragesFor(test_directory_path, image, displayCirclesBool)]
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