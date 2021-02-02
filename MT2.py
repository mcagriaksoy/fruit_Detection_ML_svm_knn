# Mehmet Cagri Aksoy - SW Eng.
# All codeblock belongs to my work.
# More Info can be seen on my github page: https://github.com/mcagriaksoy/

#Library calls:
import cv2
import numpy as np
import os
import glob
from skimage.feature import hog

#Variable definitions:
trainData = []
labels = []

#Paths:
POSITIVE_PATH = "C:/EE7076_Midterm/Fruits/Training/Banana/*.jpg"
NEGATIVE_PATH = "C:/EE7076_Midterm/Fruits/Training/Mango/*.jpg"
RGB_IMAGE_READ = 1

#This will add images from given path.
# With positive and negative samples.
for filename in glob.glob(POSITIVE_PATH):
    img = cv2.imread(filename, RGB_IMAGE_READ)
    #Histogram of Oriented Gaussian function call
    HoG_Histogram = hog(img)
    #Add Histogram of Oriented Gaussian results to traindData stack.
    trainData.append(HoG_Histogram)
    labels.append(0)

for filename in glob.glob(NEGATIVE_PATH):
    img = cv2.imread(filename, RGB_IMAGE_READ)
    #Histogram of Oriented Gaussian function call
    HoG_Histogram = hog(img)
    #Add Histogram of Oriented Gaussian results to traindData stack.
    trainData.append(HoG_Histogram)
    labels.append(1)

#Casting of traindata and labels.
trainData = np.float32(trainData)
labels = np.array(labels)

print('Starting training process..')
#SVM Configurators
svm = cv2.ml.SVM_create()
svm.setType(cv2.ml.SVM_C_SVC)
svm.setC(2.5)
svm.setGamma(5)
svm.setKernel(cv2.ml.SVM_LINEAR)
svm.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER, int(1e7), 1e-6))
#Train the model.
svm.train(trainData, cv2.ml.ROW_SAMPLE, labels)
print('Finished training process..')

#PREDICT PART:

#We give the unknown data to the system:
IMG_DIR = "C:/EE7076_Midterm/Fruits/Test/Banana/14_100.jpg" #specify path to unclassified images
image_unknown = cv2.imread(IMG_DIR) #read sample.jpg
cv2.imshow("Unknown Image", image_unknown) #show sample.jpg in window on screen
hist = hog(image_unknown)

response = svm.predict(np.float32(hist).reshape(1, -1))

print(response[1].ravel()[0])

if(response[1].ravel()[0] == 1.0):
    print("The tested image is equal to given image")
else:
    print("The tested image is not equal to given image")


