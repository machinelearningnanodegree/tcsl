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
    # discriminant_analysis.LinearDiscriminantAnalysis,
    # discriminant_analysis.QuadraticDiscriminantAnalysis,
    # dummy.DummyClassifier,
    # ensemble.AdaBoostClassifier,
    # ensemble.BaggingClassifier,
    # ensemble.ExtraTreesClassifier,
    # ensemble.GradientBoostingClassifier,
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
    (svm.SVC, {}),
    (tree.DecisionTreeClassifier, {'random_state' : 42})
    # tree.ExtraTreeClassifier
    ]
