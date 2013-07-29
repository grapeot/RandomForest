Introduction
========

This is a library for decision tree (DT) and random forest (RF), with a MATLAB (mex) wrapper. 
Note it's not a standard DT/RF library which uses axis-aligned classifiers.
More detailed explanations are available in Section Features.

Currently this version only supports Windows, but the only platform-dependent part is PPL.
It should be trivial to transplant it into Linux (if you don't care about parallel training and testing).

It's not a general library (yet). 
It may need some effort to adapt it to your project, but wouldn't too much. 
And it's still under development to make the code more general and clean. 

Features
========
 * Optimized (or different) algorithms based on standard DT/RF algorithms
     * Linear classifiers as weak classifiers in the decision nodes
     * Preprocessing included in the decision nodes (0-mean, 1-std)
     * Balance factor considered in the training process (i.e. the training algorithm encourages balanced nodes)
 * Efficient implementation in C++
 * Parallel training and testing in Windows with Microsoft Parallel Patterns Library (PPL)
 * MATLAB (mex) wrapper provided

Compilation
=======
There is nothing to compile to use the code. Just `#include <DT.h>`.
Note the code extensively uses lambda expressions in C++.
Therefore make sure the compiler supports C++0x at least. (VS2012 and VS2010 tested. If using g++, which hasn't been tested yet, use flag `std=c++11`)
