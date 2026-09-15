Question 1

Setting A produced a longer feature vector than Setting B. This occurred because Setting A uses pixels_per_cell=(8,8), which creates more HOG cells across the image. Since more cells are generated, more gradient histograms are computed and concatenated into the final feature vector. Setting B uses larger cells (16,16), resulting in fewer cells and therefore a shorter feature vector.

Question 2

The Linear SVM achieved higher accuracy than k-NN on this dataset. SVM works by finding an optimal decision boundary that separates classes in the high-dimensional HOG feature space. In contrast, k-NN relies on distance calculations between samples and may be more sensitive to noise and variations in the dataset. This likely explains the performance difference observed.

Question 3

The most confused pair of classes was Bicycle and Motorcycle. Both classes contain wheels and similar edge patterns that can produce similar HOG descriptors. One possible improvement would be to collect more training images from different viewpoints. Additional data would help the classifier learn more discriminative features and reduce confusion between these classes.

Question 4

A HOG + classical classifier pipeline would be useful in manufacturing quality control. Such systems often run on embedded hardware with limited computational resources. HOG features are lightweight and fast to compute, while classifiers such as SVM require far less processing power than deep convolutional neural networks. This makes the approach cost-effective and suitable for real-time inspection tasks.
