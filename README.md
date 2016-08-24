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
**PUT YOUR NAME NEXT TO ONE YOU WOULD LIKE TO IMPLEMENT**

```
- calibration.CalibratedClassifierCV                   (JOSHUA)
- discriminant_analysis.LinearDiscriminantAnalysis
- discriminant_analysis.QuadraticDiscriminantAnalysis  (JOSHUA)
- dummy.DummyClassifier                                (JOSHUA)
- ensemble.AdaBoostClassifier                          (DAVID)
- ensemble.BaggingClassifier                           (BHARAT)
- ensemble.ExtraTreesClassifier
- ensemble.GradientBoostingClassifier                  (NASH)
- ensemble.RandomForestClassifier                      (MATT)
- ensemble.RandomTreesEmbedding
- ensemble.RandomTreesEmbedding
- ensemble.VotingClassifier                            (BHARAT)
- linear_model.LogisticRegression                      (JOSHUA)
- linear_model.PassiveAggressiveClassifier
- linear_model.RidgeClassifier                         (BHARAT)
- linear_moder.SGDClassifier                           (JOSHUA)
- multiclass.OneVsOneClassifier
- multiclass.OneVsRestClassifier
- multiclass.OutputCodeClassifier
- naive_bayes.BernoulliNB                              (ANDREY)
- naive_bayes.GaussianNB                               (ANDREY)
- naive_bayes.MultinomialNB
- neighbors.KNeighborsClassifier                       (MATT)
- neighbors.NearestCentroid
- neighbors.RadiusNeighborsClassifier
- neural_network.BernoulliRBM                          (MAXIME)
- semi_supervised.LabelPropagation
- svm.LinearSVC
- svm.NuSVC                                            (ANDREY)
- svm.SVC                                              (MATT)
- tree.DecisionTreeClassifier                          (MATT)
- tree.ExtraTreeClassifier
```

## Discriminant Analysis
TODO: General description.

### Linear discriminant analysis
#### Brief description

#### Strengths

#### Weaknesses


### Quadratic discriminant analysis
#### Brief description

#### Strengths

#### Weaknesses



## Ensemble
Ensemble methods are not learning algorithms themselves, in the sense that they map features to some output. Rather, the ensemble technique is a *meta-algorithm* that combines many learners together to create one single learner. These *base learners* (those being combined) are typically either constructed to be high bias (i.e. boosting) or high variance (i.e. bagging). When combined, whether additively or through voting or otherwise, these base learners come together to produce one strong, regularized model. There are countless ensemble meta-algorithms; what follows is an analysis of most of the common ensemble methods.

### AdaBoost
#### Brief description
Adaptive Boosting uses a large number of "weak" learners to make predictions with high accuracy. These learners are all weighted and their collective output is used to make a classification.
#### Strengths
- It doesn't require high-accuracy classifiers
#### Weaknesses
- More complicated than a single classifier

### Gradient Boosting
#### Brief description
In any boosting algorithm, the shortcomings of the existing model are what the next learner focuses on. In gradient boosting, those shortcomings are identified the gradient of the cost function, $r_i=\frac{\partial{L(y_i,F(x_i))}}{\partial{F(x_i)}}$ for $i=1,...,n$. Once this is computed, the next learner is fit to a new dataset constructed from those gradients (or "residuals", but gradients is the more general term) as $D=\{(x_i,r_i)\}_{i=1}^{n}$. The model can then be updated by adding this new weak learner, and the process begins again. Gradient boosting is most commonly used with decision trees as the base learners. Gradient boosting can be shown to be a more general case of AdaBoost, one that is able to handle any differentiable cost function.
#### Strengths
- as with most ensemble methods, gradient boosting tends to do better than individual trees because intuitively, it is taking the best that each tree has to offer and adding it all up
- with the ability to generalize to any cost function, gradient boosting has the potential to be robust to outliers; this and similar properties can be obtained by the selection of an appropriate cost function
#### Weaknesses
- to a point, the strength of the model is proportional to its computational cost; the more trees added, the more expensive the training process
- overfitting is quite easy and effective regularization is necessary; this is controllable by the hyper-parameters, most importantly n_estimators, the number of trees

### Random Forest
#### Brief description

#### Strengths

#### Weaknesses


### Bagging Classifier
#### Brief description

#### Strengths

#### Weaknesses


### Extra Trees Classifier
#### Brief description

#### Strengths

#### Weaknesses


### Random Trees Embedding
#### Brief description

#### Strengths

#### Weaknesses


### Voting Classifier
#### Brief description

#### Strengths

#### Weaknesses



## Linear Model
TODO: General description.

### Logistic Regression
#### Brief description

#### Strengths

#### Weaknesses


### Ridge Classifier
#### Brief description

#### Strengths

#### Weaknesses


### SGD Classifier
#### Brief description

#### Strengths

#### Weaknesses


### Passive Aggressive Classifier
#### Brief description

#### Strengths

#### Weaknesses



## Multiclass
TODO: General description.

### One VS One Classifier
#### Brief description

#### Strengths

#### Weaknesses


### One VS Rest Classifier
#### Brief description

#### Strengths

#### Weaknesses


### Output Code Classifier
#### Brief description

#### Strengths

#### Weaknesses



## Naive Bayes
TODO: General description.

### Gaussian NB
#### Brief description

#### Strengths

#### Weaknesses


### Bernoulli NB
#### Brief description

#### Strengths

#### Weaknesses


### Multinomial NB
#### Brief description

#### Strengths

#### Weaknesses



## Neighbors
TODO: General description.

### K Neighbors Classifier
#### Brief description

#### Strengths

#### Weaknesses


### Nearest Centroid
#### Brief description

#### Strengths

#### Weaknesses


### Radius Neighbours Classifier
#### Brief description

#### Strengths

#### Weaknesses



## SVM
TODO: General description.

### Support Vector Classifier
#### Brief description

#### Strengths

#### Weaknesses


### Linear Support Vector Classifier
#### Brief description

#### Strengths

#### Weaknesses


### Nu Support Vector Classifier
#### Brief description

#### Strengths

#### Weaknesses



## Tree
TODO: General description.

### Decision Tree Classifier
#### Brief description

#### Strengths

#### Weaknesses


### Extra Trees Classifier
#### Brief description

#### Strengths

#### Weaknesses



## Misc
TODO: General description.

### Calibrated Classifier CV
#### Brief description

#### Strengths

#### Weaknesses


### Dummy Classifier
#### Brief description

#### Strengths

#### Weaknesses


### Bernoulli Restricted Boltzmann Machine
#### Brief description

#### Strengths

#### Weaknesses


### Label Propagation
#### Brief description

#### Strengths

#### Weaknesses


List of Supervised Learning Models [here](http://scikit-learn.org/stable/supervised_learning.html).


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
