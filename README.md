NIST's Special Database 3 contain binary images of handwritten digits learning techniques and pattern recognition methods on real-world data.
Matlab Pattern Recognition Toolbox PRTools (http://prtools.org/) was utilized to implement various classification alogrithms.

##Two tasks
1. The pattern recognition system is trained once, and then applied in the field; **(training data: at least 200 and at most 1000 objects per class)**
2. The pattern recognition system is trained for each.**(training data: at most 10 objects per class)**

##Choice of representation
1. Pixels
2. Features
3. Dissimilarities

##Log (Important!)
1. pca_frac.csv used to plot performance vs. fraction;
2. combined_knnc_qdc.csv compares combined classifier vs. single classifier;
3. pca_all_classifiers.csv shows performance of all classifiers (except svm) with e1-cv error, e-test error, e2-test error using lab testset, e3-live test(using handwritten digits images example_digits.png)
