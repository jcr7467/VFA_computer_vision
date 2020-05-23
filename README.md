The most updated files im working on are fluorescent_detect_12.py and fluorescent_detect_12_jpg.py
The difference between is what you'd expect: one works on jpg images and the other works on tiff images.
These are designed to work on test directories which are either entirely jpg/jpeg or entirely tiff images

Notes:
1. This works with fluorescent images
2. The images are stored in a directory named datasets/. 
    - I made it so that one can place a directory full of test images inside 'datasets/' and the program will ask user
    which directory to run test on
3. The program will create a csv file and the resulting images inside of that same directory the user specified
3. The templates are stored in a directory named alignment_templates/