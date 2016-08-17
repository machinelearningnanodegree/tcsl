---
title: Timing Comparisons for Supervised Learning of a Classifier
author:
- Author One, Udacity, Machine Learning Nanodegree
- Author Two, Udacity, Machine Learning Nanodegree
- Author Three, Udacity, Machine Learning Nanodegree
- Author Four, Udacity, Machine Learning Nanodegree
abstract:
    - This paper describes $n$ methods for fitting a binary supervised learning classifier on a single large dataset with multiple-typed features. Timing and accuracy metrics are presented for each method, with analysis on the results in terms of the structure of the data set. Fits were performed using the popular open-source machine learning library `scikit-learn`. Additionally, a code repository including all necessary infrastructure has been developed and shared for reproducibility of results.
---    

# Project Structure

based on this work: http://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1000424

```
/data     <-- data
/doc      <-- write up, ipynb, latex
/lib      <-- code
/results  <-- output
```

# System Design
For portability and reproducibility of results, we have elected to use the Docker system and its `Dockerfile` syntax to prepare. As this work is done using Python and its `scikit-learn` libraries we have elected to use a system built via the Anaconda package manager. Furthermore, leveraging images designed by and for using the Jupyter system, which is built via Anaconda, allows a single container to be used both for running the analysis script and for interactive analysis of the data via Jupyter. The following `Dockerfile` completely describes the system used for this work. Note that it inherits from a Docker image designed and maintained by the [Jupyter team](https://hub.docker.com/r/jupyter/scipy-notebook/).

## `mlnd/tcsl Dockerfile`
```
FROM jupyter/scipy-notebook
VOLUMES .:/home/jovyan/work
```

Via the above, fit analysis can be run on a single classifier,

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

Note that the last leverages a built-in launch script inherited from the original notebook definition, in that no explicit command was passed to the container.

# Data Set
Select a dataset
Proposed requirements:
- Large but not too large i.e. can fit on a single system running Docker
- lends itself to binary classification
- many different types of feature parameters
- from [UCI Machine Learning Dataset Library](https://archive.ics.uci.edu/ml/datasets.html)

## Dataset
https://archive.ics.uci.edu/ml/datasets/Adult (proposed)
Abstract: Predict whether income exceeds $50K/yr based on census data. Also known as "Census Income" dataset.
Data Set Information:

Extraction was done by Barry Becker from the 1994 Census database. A set of reasonably clean records was extracted using the following conditions: ((AAGE>16) && (AGI>100) && (AFNLWGT>1)&& (HRSWK>0)) 

Prediction task is to determine whether a person makes over 50K a year. 


Attribute Information:

Listing of attributes: 

>50K, <=50K. 

age: continuous. 
workclass: Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked. 
fnlwgt: continuous. 
education: Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool. 
education-num: continuous. 
marital-status: Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse. 
occupation: Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces. 
relationship: Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried. 
race: White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black. 
sex: Female, Male. 
capital-gain: continuous. 
capital-loss: continuous. 
hours-per-week: continuous. 
native-country: United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands.


Relevant Papers:

Ron Kohavi, "Scaling Up the Accuracy of Naive-Bayes Classifiers: a Decision-Tree Hybrid", Proceedings of the Second International Conference on Knowledge Discovery and Data Mining, 1996 

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
**PUT YOUR NAME NEXT TO ONE YOU WOULD LIKE TO IMPLEMENT**

name
:  \ \  
: brief description
: time complexity, training
: time complexity, prediction
: strengths
: Weaknesses  
 \ \   


Support Vector Machines (Matt)
:  \ \    
:  brief description
:  time complexity, training
:  time complexity, prediction
:  strengths
:  Weaknesses

Decision Trees
:  \ \  
: brief description
: time complexity, training
: time complexity, prediction
: strengths
: Weaknesses  
 \ \   


Naive Bayes
:  \ \  
: brief description
: time complexity, training
: time complexity, prediction
: strengths
: Weaknesses  
 \ \   


Ridge Regression
:  \ \  
: brief description
: time complexity, training
: time complexity, prediction
: strengths
: Weaknesses  
 \ \   


Stochastic Gradient Descent (Joshua)
:  \ \  
: brief description
`sklearn.linear_model.SGDClassifier`

```
{loss:
penalty:
alpha:
l1_ratio:
fit_intercept:
n_iter:
shuffle:
random_state:
verbose:
epsilon:
learning_rate:
eta0:
class_weight:
warm_start:
average:}
```
: time complexity, training
: time complexity, prediction
: strengths
: Weaknesses  
 \ \   


Adaptive Moment Estimation (ADAM)
:  \ \  
: brief description
: time complexity, training
: time complexity, prediction
: strengths
: Weaknesses  
 \ \   


Linear/Logistic Regression
:  \ \  
: brief description
: time complexity, training
: time complexity, prediction
: strengths
: Weaknesses  
 \ \   


K-nearest Neighbors (Matt)
:  \ \  
: brief description
: time complexity, training
: time complexity, prediction
: strengths
: Weaknesses  
 \ \   


Random Forests
:  \ \  
: brief description
: time complexity, training
: time complexity, prediction
: strengths
: Weaknesses  
 \ \   


XGBoost (may require additional lib) (Matt)
:  \ \  
: brief description
: time complexity, training
: time complexity, prediction
: strengths
: Weaknesses  
 \ \   


Linear Discriminant Analysis
:  \ \  
: brief description
: time complexity, training
: time complexity, prediction
: strengths
: Weaknesses  
 \ \   


Quadratic Discriminant Analysis (Joshua)
:  \ \  
: brief description
`sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis`

```
{priors:
reg_param:}
```
: time complexity, training
: time complexity, prediction
: strengths
: Weaknesses  
 \ \   


Gaussian Processes
:  \ \  
: brief description
: time complexity, training
: time complexity, prediction
: strengths
: Weaknesses  
 \ \   


Elastic Lasso
:  \ \  
: brief description
: time complexity, training
: time complexity, prediction
: strengths
: Weaknesses  
 \ \   


AdaBoost (David)
:  \ \  
: Adaptive Boosting uses a large number of "weak" learners to make predictions
with high accuracy. These learners are all weighted and their collective output
is used to make a classification.
: time complexity, training - TBD
: time complexity, prediction - TBD
: Strengths - It doesn't require high-accuracy classifiers.
: Weaknesses - More complicated than a single classifier.
 \ \   


Gradient Tree Boost (Nash)
:  \ \  
: brief description
: time complexity, training
: time complexity, prediction
: strengths
: Weaknesses  
 \ \   

Perceptron (Maxime)
:  \ \  
: brief description
: time complexity, training
: time complexity, prediction
: strengths
: Weaknesses  
 \ \   

Neural Network (Maxime)
:  \ \  
: brief description
: time complexity, training
: time complexity, prediction
: strengths
: Weaknesses  
 \ \  


List of Supervised Learning Models:


http://scikit-learn.org/stable/supervised_learning.html




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
