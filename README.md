# fruit_Detection_ML_svm_knn
Simple fruit detection comparison with the different approaches, SVM, HoG, Histogram, KNN

First of all I have worked on Windows environment, python 3.8.6, opencv 4.4.0.46 and used the dataset of: 
https://www.kaggle.com/mariamabouelmaaty/fruits/
In this question solution, 

In first approach, I have used Support Vector Machine solution with my implementation of Histogram of Oriented Gaussian solution. I feed the system with >Rgb images done by HoG method. Then I got good results. Randomly tried some test and train data on the system and results are acceptible.

Second approach, I have used k-Nearest Neighbour with my implementation of Histogram of input image. First I have read the image with grayscale then I would like to do low pass filter on my image to gather different results than first approach and blur image to enhance image and remove some noise like salt-pepper noise. Then I calculated the histogram of each image and feed the system with their(images) histogram values. (On numpy array) 

I created a model and feed it from the dataset. In the nature of SVM is supporting the binary classification and it is OK for my approach. I have used Opencv’s svm function.
In summary, SVM’s results are better than KNN results, KNN is much more simple method so that it’s time and complexity is lower than SVM but accuracy is lower than also. I got 80+ accuracy on SVM solution for the first approach, besides KNN has 70+ accuracy for the second approach. SVM with Histogram of Gaussian works better, as I experimented and read on articles on the internet. My KNN solution uses histogram only that is feed by grayscale images so that if we compare HoG better for this binary image classification task. 

