import time
from sklearn.metrics import f1_score
from sklearn.grid_search import GridSearchCV

class Model:
    def __init__(self, classifier, parameters={}, gs_params={}):
        self.classifier = classifier
        self.model = classifier(**parameters)
        self.parameters = parameters
        self.gs_model = GridSearchCV(self.model, gs_params)
        self.gs_params = gs_params
        self.optimal_model = None
        self.optimal_params = {}

        self.training_time = None
        self.length_training_set = None
        self.f1_train, self.train_prediction_time = None, None
        self.f1_test, self.test_prediction_time = None, None

    def train_classifier(self, X_train, y_train):
        print("Training {}...".format(self.classifier.__name__))

         # start = np.datetime64(datetime.datetime.now(),"us")
        start = time.time()
        self.model.fit(X_train, y_train)
        # end = np.datetime64(datetime.datetime.now(),"us")
        end = time.time()
        self.training_time = end - start

        print("Training {} with grid search...".format(self.classifier.__name__))
        # Find optimal parameters with grid search.
        start = time.time()
        self.gs_model.fit(X_train, y_train)
        end = time.time()
        self.gs_time = end - start

        # Create an "optimal" model with this classifier.
        self.optimal_model = self.classifier(**self.gs_model.best_params_)
        self.optimal_params = self.gs_model.best_params_

        print("Training {} with optimal parameters...".format(self.classifier.__name__))
        # Fit optimal model independant of grid-serach
        start = time.time()
        self.optimal_model.fit(X_train, y_train)
        end = time.time()
        self.optimal_training_time = end - start

    def predict_labels(self, features, target):
        print("Predicting labels using {}...".format(self.classifier.__name__))
         # start = np.datetime64(datetime.datetime.now(),"us")
        start = time.time()
        y_pred = self.model.predict(features)
        # end = np.datetime64(datetime.datetime.now(),"us")
        end = time.time()
        prediction_time = end - start

        f1_score_output = f1_score(target, y_pred, average="macro")

        print("Predicting labels using {} with optimal parameters...".format(self.classifier.__name__))
        
        start = time.time()
        y_pred = self.optimal_model.predict(features)
        end = time.time()
        optimal_prediction_time = end - start

        f1_optimal_score_output = f1_score(target, y_pred, average="macro")

        return f1_score_output, prediction_time, \
            f1_optimal_score_output, optimal_prediction_time

    def __str__(self):
        return """
Model(classifer: {}
      length training set: {}
      training time: {}
      train f1/pred. time: {} {}
      test f1/pred. time: {} {})
               """.format(self.classifier.__class__.__name__,
                          self.length_training_set,
                          self.training_time, self.f1_train, self.train_prediction_time,
                          self.f1_test, self.test_prediction_time)

    def __repr__(self):
        return """
Model(classifer: {}
      length training set: {}
      training time: {}
      train f1/pred. time: {} {}
      test f1/pred. time: {} {})

Detailed Classifier Description:
{}
               """.format(self.classifier.__class__.__name__,
                          self.length_training_set,
                          self.training_time, self.f1_train, self.train_prediction_time,
                          self.f1_test, self.test_prediction_time, self.classifier)

    def __call__(self, X_train, y_train, X_test, y_test):
        self.length_training_set = len(X_train)

        # Train all classifiers
        self.train_classifier(X_train, y_train)

        # Store times and scores for training and testing vanilla and
        # optimized classifiers.
        self.f1_train, self.train_prediction_time, \
            self.f1_optimal_train, self.optimal_train_prediction_time = self.predict_labels(X_train, y_train)
        self.f1_test, self.test_prediction_time, \
            self.f1_optimal_test, self.optimal_test_prediction_time = self.predict_labels(X_test, y_test)
