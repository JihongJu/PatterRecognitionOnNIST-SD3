# Task 1

#Introduction
The pattern recognition system is trained once, and then applied in the field; **(training data: at least 200 and at most 1000 objects per class)**
#Instruction
Training data: 500 objects per class
Testing: use the function nist_eval
#Classification
##Pixel Representation
###1. Dataset training with classifiers
In this part, we train the dataset with serveral classifiers. The following table shows the cross validation error and test error of each classifier.

| Classifier|Parameter|Cross Validation Error|Testing Error
| -----|:-----------------------:|:------:|:----:|
| Parzen|h = 0.5|7.03%|5.80%|
| KNN|k=3|7.78%|7.50\%|
| Fisher|/|18.68%|16.90%|
| Neural Networks|1 layer, 50 hidden units, 500 iterations|12.43%|10.80%|
| Support Vector|Gaussian Kernel|/|/|
###2. Dataset training with classifiers and PCA feature reduction
In this part, we train the dataset with the same classifiers in part 1. Besides, before training, we first prepocess the data with PCA. The following table shows the cross validation error and test error of each classifier.

| Classifier|Parameter|Cross Validation Error|Testing Error
| -----|:-----------------------:|:------:|:----:|
| Parzen|h = 0.5|6.98%|5.40%|
| KNN|k=3|6.18%|5.00%|
| Fisher|/|16.40%|15.30%|
| Quadratic|/|5.68%|5.40%|
| Neural Networks|1 layer, 50 hidden units, 500 iterations|20.50%|11.90%|
| Support Vector|Gaussian Kernel|/|/|
|Random Forest|/|47.10%|50.50%|
###3. Dataset training with combine classifiers and PCA feature reduction
In this part, we train the dataset with the same classifiers in part 1 and part 2. Besides, this time, we will combine classifiers. Also, before training, we prepocess the data with PCA. The following table shows the cross validation error and test error of each classifier.
##Feature Representation
##Disimilarity Representation