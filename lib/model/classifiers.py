from sklearn import calibration
from sklearn import discriminant_analysis
from sklearn import dummy
from sklearn import ensemble
from sklearn import linear_model
from sklearn import multiclass
from sklearn import naive_bayes
from sklearn import neighbors
from sklearn import neural_network
from sklearn import semi_supervised
from sklearn import svm
from sklearn import tree

CLASSIFIERS = [
    # calibration.CalibratedClassifierCV,
    (discriminant_analysis.LinearDiscriminantAnalysis, {}, [{ ####HOLY WARNINGS!!!!
        'solver' : ['svd'],
        'priors' : [None], # what else can go here?
        'n_components' : [None, 1],
        'store_covariance' : [True, False],
        'tol' : [1e-7, 1e-6, 1e-5, 1e-4],},
        {
        'solver' : ['lsqr', 'eigen'],
        'shrinkage' : [None, 'auto', 0.001, 0.01, 0.1, 0.2, 0.5, 0.9, 1.0],
        'priors' : [None],
        'n_components' : [None, 1],
        'store_covariance' : [True, False],}
    ]),
    # discriminant_analysis.QuadraticDiscriminantAnalysis,
    # dummy.DummyClassifier,
    # ensemble.AdaBoostClassifier,
    # ensemble.BaggingClassifier,
    # ensemble.ExtraTreesClassifier,
    # (ensemble.GradientBoostingClassifier, {'random_state' : 42}, {
    #     'n_estimators' : [50,75,100,125,150],
    #     'max_leaf_nodes' : list(range(4,9)),
    # }),
    # ensemble.RandomForestClassifier,
    # ensemble.RandomTreesEmbedding,
    # ensemble.RandomTreesEmbedding,
    # ensemble.VotingClassifier,
    # linear_model.LogisticRegression,
    # linear_model.PassiveAggressiveClassifier,
    # linear_model.RidgeClassifier,
    # linear_model.SGDClassifier,
    # multiclass.OneVsOneClassifier,
    # multiclass.OneVsRestClassifier,
    # multiclass.OutputCodeClassifier,
    # naive_bayes.BernoulliNB,
    # naive_bayes.GaussianNB,
    # naive_bayes.MultinomialNB,
    # neighbors.KNeighborsClassifier,
    # neighbors.NearestCentroid,
    # neighbors.RadiusNeighborsClassifier,
    # neural_network.BernoulliRBM,
    # semi_supervised.LabelPropagation,
    # svm.LinearSVC,
    # svm.NuSVC,
    # (svm.SVC, {}, {}),
    # (tree.DecisionTreeClassifier, {'random_state' : 42}, {})
    # tree.ExtraTreeClassifier
    ]
