Question 1
# Question 1: Feature Vector Length Comparison

Setting A produced a longer feature vector than Setting B. This difference occurs because Setting A uses pixels_per_cell=(8,8), which divides each image into a larger number of smaller cells. Since HOG computes gradient histograms for every cell, having more cells results in more histogram values being generated and concatenated into the final feature vector. In contrast, Setting B uses pixels_per_cell=(16,16), creating fewer cells across the image and therefore a shorter feature vector. As a result, Setting A captures finer image details while Setting B captures more coarse-grained information.

# Question 2: Linear SVM vs. k-NN

The Linear SVM achieved an accuracy of 46.67%, while the k-NN classifier achieved an accuracy of 40.00%. Therefore, Linear SVM performed better on this dataset. One possible explanation is that SVM learns an optimal decision boundary in the high-dimensional HOG feature space, allowing it to separate classes more effectively. In contrast, k-NN makes predictions based on the labels of nearby samples, which can be sensitive to noise and variations in the dataset. The relatively small dataset size may also have limited the performance of both classifiers.

# Question 3: Most Confused Pair of Classes

The most confused pair of classes was dogs and pandas. In the Linear SVM confusion matrix, two dog images were classified as pandas and two panda images were classified as dogs. This confusion may occur because both animals share similar body shapes, fur textures, and edge patterns that can produce similar HOG features. One way to reduce this confusion would be to collect more training images showing different poses, backgrounds, and lighting conditions. Additional data would help the classifier learn more distinctive features for each class and improve classification accuracy.

# Question 4: Real-World Application

A lightweight HOG + classical classifier pipeline would be useful in manufacturing quality control. Many factories use low-cost cameras and embedded systems to inspect products moving along a production line. HOG features can efficiently capture shape and edge information while requiring much less computational power than a deep convolutional neural network. This makes the approach suitable for real-time inspection tasks where fast processing, low hardware requirements, and cost efficiency are important. In situations where resources are limited, a classical machine learning pipeline can be more practical than a deep learning solution.
