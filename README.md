# Random Forest 0.1

## Introduction

This is a library for Decision Tree (DT) and Random Forest (RF), with a MATLAB (mex) wrapper. 
Note it's not a standard DT/RF library which uses axis-aligned classifiers, while the splitting nodes use linear classifiers.

I don't have plans to make it a highly reusable library with excellent documentation in the near future.
And it may need some effort to adapt it to your project, but wouldn't be too much. 

## Features

 * Optimized (or different) algorithms based on standard DT/RF algorithms
     * Linear classifiers as weak classifiers in the decision nodes
     * Optional preprocessing included in the decision nodes (0-mean, 1-std)
     * Balance factor considered in the training process (i.e. the training algorithm encourages balanced nodes)
 * Efficient implementation in C++
 * Parallel training and testing with OpenMP
 * MATLAB (`mex`) wrapper provided (currently only prediction stage has `mex` wrappers) 
 * Compatible with Windows and Linux, with VS solution files and Makefile provided.
 * Simple mechanism for distributed training of RFs

## Usage

### Configurations

The class number (`LabelNum`) and feature dimensions (`dim`) are specified in `/include/config.h`.
Change these two values to adapt the code to your program.

### mex

The `mex` wrappers are provided for the prediction stage. 
Assume the features are `d` dimensions, `C` classes, and there are `N` feature points to classifier. 
The usage is like

* `dist = DTClassifyDist(feature)`: read in the decision tree model file named `tree.dat` and classifier `feature`, which is a dxN matrix contains features to classify, every column a feature. The returned `dist` is a CxN matrix with each column containing the probability score for each feature and each class.
* `dist = RFClassifyDist(feature)`: read in the random forest model file named `forest.dat` and classifier `feature`. The parameter and return value specification are the same as `DTClassifyDist`. But there is one trick (and a possible pitfall for the users) we did to improve performance: the target random forest will _NOT_ be automatically released after classification, like `DTClassifyDist`. This may save a huge amount of time when you have fairly large random forests (say, several gigs), but may also cause memory leakage if you're not careful. To release the memory, do `RFClassifyDist([]);`. Or `clean mex` will also work.

We don't provide precompiled `mex` wrappers because you really need to configure the parameters like `d` and `C` beforehand, otherwise it wouldn't work. To compile the `mex` wrappers (after checking the `Configurations`):

* For Windows: Change the configurations and then run `/mex/make.m`.
* For Linux: Change the configurations. Change the MATLAB path in `/mex/Makefile` (the first line). And then `make` in `/mex`. 

### C++

We provide both the training and testing interfaces in C++.
There is nothing to compile to use the code. Just `#include "DT.h"` and `#include "config.h"`.

Check `/src/main.cpp` for the code to train the DT/RF.
`merger.cpp` is to provide some naive distributed training capability, which I found especially useful to be used with AWS's Spot Request and `rsync`.
To compile the code, use the provided VS solution file in `/src/VS` or Makefile in `/src`.

To switch on and off the feature normalization, define (switched on) or not define (switched off) the macro `_WHITENING` in the VS solution or the Makefile. This feature is turned on by default.

## Known issues

* Using the Makefile with g++ in cygwin (Windows) may result in degenerated performance. The training speed will be affected as the CPU utilization may not reach 100%. Use the VS solution file instead to avoid this problem.
