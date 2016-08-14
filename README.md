---
title: Timing Comparisons for Supervised Learning of a Classifier
author:
- name: Joshua Cook
  affiliation: Udacity, Machine Learning Nanodegree
- name: Other Authors
  affiliation: Udacity, Machine Learning Nanodegree
abstract:
    - This paper describes $n$ methods for fitting a binary supervised learning classifier on a single large dataset with multiple-typed typed features. Timing and accuracy are for the methods with analysis on the results in terms of the structure of the data set. Fits were performed using the popular open-source machine learning library `scikit-learn`. Additionally, a code repository including all necessary infrastructure has been developed and shared for reproducibility of results. 

# System Design
For portability and reproducibility of results, we have elected to use the Docker system and its `Dockerfile` syntax to prepare. As this work is done using Python and its `scikit-learn` libraries we have elected to use a system built via the Anaconda package manager. Furthermore, leveraging images designed by and for using the Jupyter system, which is built via Anaconda, allows a single container to be both for running a script and for interactive analysis of the data. The following `Dockerfile` completely describes the system used for this work. Note that it inherits from a Docker image designed and maintained by the [Jupyter team](https://hub.docker.com/r/jupyter/scipy-notebook/). 

## `mlnd/tcsl Dockerfile` 
```
FROM jupyter/scipy-notebook
VOLUMES .:/home/jovyan/work
```

Via the above, image analysis can be run on a single classifier,

```
$ docker run -e CLASSIFIER='decision tree' mlnd/tcsl python project.py
```

all classifiers,

```
$ docker run mlnd/tcsl python project.py
```

or via an interactive notebook server

```
$ docker run mlnd/tcsl
```
 
Note that the last leverages a built-in launch script inherited from the original notebook definition.

# Data Set
Select a dataset
Proposed requirements:
- Large but not too large i.e. can fit on a single system running Docker
- lends itself to binary classification
- many different types of feature parameters
- from UCI Machine Learning Dataset Library

# Data Visualization


# Feature Engineering
one-hot encode classification parameters
convert all booleans to numeric values

# Split Data Set
- training
- test
- use seed for reproducibility

# Models
For each model complete the following:
Copy and paste this template to add a new model.
name:
brief description:
time complexity, training:
time complexity, prediction:
strengths:
Weaknesses:

# Metrics
What metrics should be used for timing, for accuracy, others?

# Pipeline
1. raw fit of classifier
1. raw prediction of classifier
1. gridsearchCV fit
1. prediction on tuned model

# Analysis
Highest performing model
What this says about the data set chosen


